"""Evaluation integrity: held-out sampling, pairing, invalid output and safe resume."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_projector_checkpoint import select_balanced, word_counts, run_case
from scripts.report_projector_validation import verified_case, likelihood, paired_nll, read_json_lines
from scripts.projector_data import read_json
from scripts.published_feedback_metrics import summarize
from scripts.validation_records import word_counts as unicode_word_counts


class ProjectorValidationTests(unittest.TestCase):
    def rows(self):
        return [{"utterance_id": f"{speaker}-{i}", "speaker_id": speaker, "source_row_sha256": f"hash-{speaker}-{i}",
                 "reference_text": "The cat sat.", "feedback": {}} for speaker in ("A", "B", "C") for i in range(10)]

    def test_balanced_selection_is_fixed_without_looking_at_feedback(self):
        rows = self.rows()
        selected = select_balanced(rows, 2, 42, "feedback")
        mutated = deepcopy(rows)
        for row in mutated:
            row["feedback"] = {"DIFFERENT": "Do not inspect"}
        self.assertEqual([r["utterance_id"] for r in selected],
                         [r["utterance_id"] for r in select_balanced(mutated, 2, 42, "feedback")])
        self.assertEqual([r["speaker_id"] for r in selected], ["A", "A", "B", "B", "C", "C"])
        with self.assertRaises(ValueError):
            select_balanced(rows, 11, 42, "feedback")

    def test_wer_normalizes_case_and_punctuation_and_preserves_insertions(self):
        self.assertEqual(word_counts("The cat sat.", "THE cat, sat!")["word_errors"], 0)
        counts = word_counts("cat", "dog rat bat")
        self.assertEqual(counts["word_errors"], 3)
        self.assertEqual(counts["reference_words"], 1)
        self.assertFalse(counts["exact_sentence"])
        self.assertEqual(unicode_word_counts("cat", "猫 狗")["word_errors"], 2)

    def test_resume_continues_failed_prefix_then_rejects_changed_identity(self):
        rows = self.rows()[:3]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def failing(index, row):
                if index == 1:
                    raise RuntimeError("Simulated interruption")
                return {"reference_text": row["reference_text"]}
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_case(root, "case", {"dataset": "frozen"}, rows, failing)
            seen = []
            result = run_case(root, "case", {"dataset": "frozen"}, rows,
                              lambda i, r: seen.append(i) or {"reference_text": r["reference_text"]})
            self.assertEqual(seen, [1, 2])
            self.assertEqual(len(result), 3)
            self.assertEqual(read_json(root / "case/metadata.json")["status"], "completed")
            verified_case(root / "case", rows, "frozen")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run_case(root, "case", {"dataset": "other"}, rows, failing)
            path = root / "case/predictions.jsonl"
            with path.open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                run_case(root, "case", {"dataset": "frozen"}, rows, failing)
            with self.assertRaises(ValueError):
                verified_case(root / "case", rows, "frozen")

    def test_pilot_cannot_be_reported_as_full_split(self):
        rows = self.rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_case(root, "pilot", {"dataset": "frozen", "purpose": "balanced_generation_pilot"}, rows[:2],
                     lambda i, r: {"reference_text": r["reference_text"]})
            with self.assertRaisesRegex(ValueError, "unpaired"):
                verified_case(root / "pilot", rows, "frozen")

    def test_token_weighted_loss_and_paired_cluster_delta(self):
        before = [{"utterance_id": "a", "speaker_id": "A", "nll_sum": 10., "target_tokens": 2},
                  {"utterance_id": "b", "speaker_id": "B", "nll_sum": 2., "target_tokens": 1}]
        after = [{**r, "nll_sum": r["nll_sum"] - r["target_tokens"]} for r in before]
        self.assertEqual(likelihood(before)["target_token_nll"], 4.)
        result = paired_nll(before, after, repetitions=100)
        self.assertEqual(result["delta_nll_after_minus_before"], -1.)
        self.assertEqual(result["ci95"], [-1., -1.])
        with self.assertRaises(ValueError):
            paired_nll(before, list(reversed(after)))

    def test_invalid_feedback_stays_in_denominator(self):
        reference = {"cat": [{"issue": "Missing T", "suggestion": "Say T"}]}
        observations = [{"utterance_id": str(i), "speaker_id": "A", "word_order": ["cat"],
                         "reference_feedback": reference, "prediction": reference if i == 0 else None,
                         "valid_json": i == 0, "generation_sec": 1.} for i in range(2)]
        summary, _ = summarize(observations)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["json_valid_percent"], 50.)
        self.assertAlmostEqual(summary["feedback_word_recall_percent"], 50.)

    def test_generated_unicode_line_separators_do_not_split_json_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "generations.jsonl"
            rows = [{"raw_text": "one\u0085two\u2028three\u2029four"}, {"raw_text": "normal"}]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(read_json_lines(path), rows)
            source = self.rows()[:2]
            result = run_case(Path(temp), "unicode", {"dataset": "frozen"}, source, lambda i, r: rows[i])
            cached = run_case(Path(temp), "unicode", {"dataset": "frozen"}, source,
                              lambda i, r: self.fail("Completed case must not generate again"))
            self.assertEqual(result, cached)


if __name__ == "__main__":
    unittest.main()
