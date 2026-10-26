"""Evidence gates for the read-only comparison renderer; no neural packages."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_model_comparison import (
    GRAMMAR, PHONEME, PANELS, digest, object_sha, load_bundle,
    grammar_rows, pronunciation_rows, case_cards,
    key_findings,
)
from mdd.metrics import score_utterance, aggregate_scores


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(directory):
    g, p = directory / "grammar", directory / "pronunciation"
    sources = [{"id": task, "task": task, "source": "<script>alert('data')</script>",
                "cohort": "published_heldout", "references": ["A good sentence."]}
               for task in ("grammar_corrected", "natural")]
    gm = {"rows": sources}
    save(g / "manifest.json", gm)
    summary = {"manifest_sha256": digest(g / "manifest.json"), "models": {}, "copy_baseline": {}}
    quality = {"rows": 1, "chrf": 50., "exact_reference_rate": 0., "gleu": {"score": 40.}}
    for task in ("grammar_corrected", "natural"):
        summary["copy_baseline"][task] = deepcopy(quality)
    for arm in GRAMMAR:
        path = g / arm
        path.mkdir()
        observations = [{**r, "source_row_sha256": object_sha(r), "raw": "not valid JSON",
                         "prediction": None, "error": "invalid_json", "seconds": .2} for r in sources]
        (path / "predictions.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in observations), encoding="utf-8")
        metadata = {"status": "completed", "completed_rows": 2, "manifest_sha256": digest(g / "manifest.json"),
                    "predictions_sha256": digest(path / "predictions.jsonl")}
        save(path / "metadata.json", metadata)
        summary["models"][arm] = {"metadata": metadata, "tasks": {
            task: {"rows": 1, "available_rate": 0., "quality_on_valid_only": None,
                   "quality_with_explicit_source_fallback": deepcopy(quality), "unchanged_controls": 0,
                   "unchanged_controls_preserved_exact": 0,
                   "unchanged_controls_preserved_lexical": 0, "mean_seconds": .2}
            for task in ("grammar_corrected", "natural")}}
    save(g / "summary.json", summary)
    annotation = {"native_language": "Vietnamese"}
    pm = {"cases": [{"id": "TLV-one", "speaker_id": "TLV", "group": "expert_test", "reference_text": "A.",
                      "audio_sha256": "audiohash", "annotation": annotation}]}
    save(p / "manifest.json", pm)
    score = score_utterance(["AA"], ["AA"], ["AA"])
    reduced = deepcopy(score)
    reduced["removed_indistinguishable_expert_events"] = []
    review = {"counts": {"reference_words": 1, "gold_error_words": 0, "available_words": 1,
                         "hint_tp": 0, "hint_fp": 0, "hint_fn": 0, "hint_tn": 1, "calibrated_decided_words": 0}}
    prediction = {"status": "completed", "audio_sha256": "audiohash", "reference_supplied_to_model": False,
                  "phonemes": ["AA"], "runtime": {"device": "cpu"}, "inference_sec": .3,
                  "research_original_inventory": deepcopy(score), "research_full39": deepcopy(score),
                  "research_common37": reduced, "variant_review_full39": review, "variant_review_common37": review}
    pr = {"status": "completed", "manifest_sha256": object_sha(pm, compact=True),
          "predictions": {a: {"TLV-one": deepcopy(prediction)} for a in PHONEME},
          "summary": {"paired": {"completed_expert_cases": 1}, "models": {
              a: {panel: aggregate_scores([prediction[panel]]) for panel in PANELS} for a in PHONEME}}}
    save(p / "results.json", pr)


class ComparisonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_invalid_native_is_unavailable_not_missing_arm(self):
        bundle = load_bundle(self.root)
        qwen = next(r for r in grammar_rows(bundle) if r["model"] == "qwen-base")
        self.assertEqual(qwen["native_available_rate"], 0.)
        self.assertEqual(qwen["valid_only_rows"], 0)
        self.assertIsNone(qwen["valid_only_gleu"])
        self.assertEqual(qwen["delivery_gleu"], 40.)

    def test_running_grammar_rejected_even_when_rows_present(self):
        path = self.root / "grammar/qwen-base/metadata.json"
        metadata = json.loads(path.read_text()); metadata["status"] = "running"; save(path, metadata)
        with self.assertRaisesRegex(ValueError, "completed grammar"):
            load_bundle(self.root)

    def test_running_pronunciation_rejected(self):
        path = self.root / "pronunciation/results.json"
        result = json.loads(path.read_text()); result["status"] = "running"; save(path, result)
        with self.assertRaisesRegex(ValueError, "every pronunciation"):
            load_bundle(self.root)

    def test_changed_prediction_bytes_rejected(self):
        path = self.root / "grammar/qwen-base/predictions.jsonl"
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, "identity differs"):
            load_bundle(self.root)

    def test_invented_aggregate_rejected(self):
        path = self.root / "pronunciation/results.json"
        result = json.loads(path.read_text())
        result["summary"]["models"]["cascade-B"]["research_full39"]["counts"]["TA"] += 1
        save(path, result)
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            load_bundle(self.root)

    def test_speaker_undefined_f1_not_coerced_to_perfect(self):
        rows = pronunciation_rows(load_bundle(self.root), by_speaker=True)
        self.assertTrue(all(r["native_languages"] == "Vietnamese" for r in rows))
        self.assertTrue(all(r["f1"] is None for r in rows))
        self.assertTrue(all(r["review_coverage"] == 1. for r in rows))

    def test_raw_source_is_escaped_in_html(self):
        markup = case_cards(load_bundle(self.root))
        self.assertNotIn("<script>alert", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_available_only_recall_distinct_from_all_expert_error_recall(self):
        bundle = load_bundle(self.root)
        counts = {"reference_words": 4, "gold_error_words": 2, "available_words": 2,
                  "hint_tp": 1, "hint_fp": 1, "hint_fn": 0, "hint_tn": 0, "calibrated_decided_words": 0}
        for arm in PHONEME:
            row = bundle["pronunciation_results"]["predictions"][arm]["TLV-one"]
            for key in ("variant_review_full39", "variant_review_common37"):
                row[key]["counts"] = deepcopy(counts)
        rows = pronunciation_rows(bundle)
        for row in rows:
            self.assertEqual(row["review_recall_available_only"], 1.)
            self.assertEqual(row["review_recall_all_expert_errors"], .5)
            self.assertEqual(row["review_false_positive_rate_available_only"], 1.)
            self.assertEqual(row["review_coverage"], .5)
            self.assertEqual(row["review_unavailable_error_words"], 1)
            self.assertEqual(row["review_unavailable_correct_words"], 1)

    def test_leading_findings_use_matched_metrics_and_keep_scope(self):
        bundle = load_bundle(self.root)
        task = bundle["grammar_summary"]["models"]["coedit-large"]["tasks"]["grammar_corrected"]
        task["quality_with_explicit_source_fallback"]["gleu"]["score"] = 47.5
        text = '\n'.join(key_findings(bundle))
        self.assertIn("+7.50 điểm", text)
        self.assertIn("cùng 1 câu", text)
        self.assertIn("FPR trên phần khả dụng", text)
        self.assertIn("chưa đổi model mặc định", text)

    def test_known_reference_matching_guard_blocks_not_all_called_correct(self):
        bundle = load_bundle(self.root)
        bundle["delivery_audit"] = {"models": {"qwen-grammar": {"summary": {"published_heldout": {
            "grammar_corrected": {"native_available": 100, "reference_matching_outputs_withheld": 7,
                                  "reasons": {"lexical_guard_withheld": 32}},
            "natural": {"native_available": 100, "reference_matching_outputs_withheld": 7,
                        "reasons": {"lexical_guard_withheld": 55}},
        }}}}}
        text = '\n'.join(key_findings(bundle))
        self.assertIn("chặn 87/200", text)
        self.assertIn("14 đầu ra khớp", text)
        self.assertIn("không chứng minh tất cả 87", text)


if __name__ == "__main__":
    unittest.main()
