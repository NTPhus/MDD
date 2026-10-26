"""Meaningful contracts for the frozen phoneme comparison, without neural models."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

from scripts import evaluate_pronunciation_models as evaluation
from scripts import setup_phoneme_baseline as setup


def annotation(identifier="NJS/example", expected="AO", perceived="AA"):
    return {"utterance_id": identifier, "speaker_id": identifier.split("/")[0],
            "canonical_phones": ["F", expected], "perceived_phones": ["F", perceived],
            "canonical_word_indices": [0, 0], "word_intervals": [{"index": 0, "text": "for"}],
            "gold_error_events": [{"kind": "substitution", "position": 1, "expected_phone": expected,
                                    "observed_phone": perceived, "word_index": 0}]}


class FrozenSelectionTests(unittest.TestCase):
    def fixtures(self):
        rows = [annotation(speaker + "/" + str(index))
                for speaker in ("NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA") for index in range(23)]
        split = {"cascade_test_ids": [row["utterance_id"] for row in rows],
                 "joint_train_ids": ["train/id"], "joint_validation_ids": ["validation/id"]}
        return rows, split

    def test_selection_is_120_and_independent_of_error_labels_and_row_order(self):
        rows, split = self.fixtures()
        original = evaluation.selected_cases(rows, split)
        self.assertEqual(len(original), 120)
        for row in rows:
            row["gold_error_events"] = []
            row["perceived_phones"] = ["Z"] * 100
        random.Random(82).shuffle(rows)
        selected = evaluation.selected_cases(rows, split)
        self.assertEqual([r["utterance_id"] for r in original], [r["utterance_id"] for r in selected])
        self.assertEqual(set(sum(r["speaker_id"] == speaker for r in selected)
                             for speaker in {r["speaker_id"] for r in rows}), {20})

    def test_overlap_rejected_before_selection(self):
        rows, split = self.fixtures()
        split["joint_validation_ids"].append(rows[0]["utterance_id"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation.selected_cases(rows, split)


class InventoryTests(unittest.TestCase):
    def test_ctc_blank_separates_repeated_phones_and_unknown_is_retained(self):
        tokens, phones = evaluation.decode_ipa_ids([1, 1, 41, 1, 16, 40, 40, 42, 43],
                                                   {1: "ɑ", 41: "[PAD]", 16: " ", 40: "[UNK]", 42: "<s>", 43: "</s>"}, 41)
        self.assertEqual(tokens, ["ɑ", "ɑ", "[UNK]"])
        self.assertEqual(phones, ["AA", "AA", "[UNK]"])

    def test_native_timit_map_preserves_diphthongs_and_affricates(self):
        tokens, phones = evaluation.decode_ipa_ids([4, 5, 7, 20, 10],
                                                   {4: "aʊ", 5: "aɪ", 7: "ʧ", 20: "ʤ", 10: "ɾ"}, 41)
        self.assertEqual(phones, ["AW", "AY", "CH", "JH", "T"])
        self.assertEqual(len(tokens), 5, "Do not split multi-character IPA tokens into characters")

    def test_full39_retains_expert_contrast_common37_counts_removed(self):
        row = annotation()
        before = deepcopy(row)
        full = evaluation.scored(row, ["F", "AA"], normalized=True)
        common = evaluation.common_scored(row, ["F", "AA"])
        self.assertEqual(full["counts"]["TR"], 1)
        self.assertEqual(common["counts"]["TR"], 0)
        self.assertEqual(common["counts"]["TA"], 2)
        self.assertEqual(len(common["removed_indistinguishable_expert_events"]), 1)
        self.assertEqual(row, before, "Never modify frozen expert annotations")

    def test_unknown_labels_do_not_become_equivalence_evidence(self):
        row = annotation(expected="ERR", perceived="UNK")
        normalized, removed = evaluation.common_reference(row)
        self.assertEqual(removed, [])
        self.assertEqual(len(normalized["gold_error_events"]), 1)
        self.assertEqual(normalized["gold_error_events"][0]["expected_phone"], "[UNK]")

    def test_common_map_is_symmetric_for_all_supported_merges(self):
        self.assertEqual([evaluation.common_phone(value) for value in ["AO", "AA", "ZH", "SH", "DX", "T", "AX", "AH*", None]],
                         ["AA", "AA", "SH", "SH", "T", "T", "AH", "AH", None])
        row = annotation(expected="ZH", perceived="SH")
        normalized, removed = evaluation.common_reference(row)
        self.assertEqual(normalized["canonical_phones"], normalized["perceived_phones"])
        self.assertEqual(len(removed), 1)

    def test_addition_and_deletion_events_kept_without_moving_positions(self):
        row = annotation()
        row["gold_error_events"] = [
            {"kind": "addition", "position": 1, "expected_phone": None, "observed_phone": "AO", "word_index": 0},
            {"kind": "deletion", "position": 1, "expected_phone": "AO", "observed_phone": None, "word_index": 0}]
        normalized, removed = evaluation.common_reference(row)
        self.assertEqual(removed, [])
        self.assertEqual([e["position"] for e in normalized["gold_error_events"]], [1, 1])
        self.assertEqual(normalized["gold_error_events"][0]["observed_phone"], "AA")


class EvidenceTests(unittest.TestCase):
    def test_unavailable_review_is_not_scored_as_a_correct_negative(self):
        row = annotation()
        counts = evaluation.review_counts(row, [], available=False)
        self.assertEqual(counts["available_words"], 0)
        self.assertEqual(counts["hint_fn"], 0)
        self.assertEqual(counts["hint_tn"], 0)
        self.assertEqual(counts["calibrated_decided_words"], 0)
        self.assertEqual(counts["unavailable_error_words"], 1)
        self.assertEqual(counts["unavailable_correct_words"], 0)

    def test_frozen_source_change_aborts_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scorer.py"
            source.write_text("before", encoding="utf-8")
            manifest = {"source_sha256": {"scorer.py": evaluation.digest(source)}, "models": {}, "cases": []}
            source.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                evaluation.verify(manifest, root)

    def test_weight_pin_rejects_corruption_without_deserializing(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.safetensors"
            path.write_bytes(b"safe bytes")
            expected = (10, hashlib.sha256(b"safe bytes").hexdigest())
            self.assertEqual(setup.file_record(path, expected)["bytes"], 10)
            path.write_bytes(b"unsafe byt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                setup.file_record(path, expected)


if __name__ == "__main__":
    unittest.main()
