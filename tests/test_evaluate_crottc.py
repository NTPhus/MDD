"""Offline pilot selection, phoneme scoring and reference-isolation contracts."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_crottc as pilot


def annotation(identifier="NJS/example", *, error=False):
    return {"utterance_id": identifier, "speaker_id": identifier.split("/")[0],
            "canonical_phones": ["F", "AH"], "perceived_phones": ["F", "AA"] if error else ["F", "AH"],
            "canonical_word_indices": [0, 0], "word_intervals": [{"index": 0, "text": "for"}],
            "gold_error_events": [{"kind": "substitution", "position": 1, "expected_phone": "AH", "observed_phone": "AA", "word_index": 0}] if error else []}


class SelectionTests(unittest.TestCase):
    def fixtures(self):
        rows = [annotation(speaker + "/" + str(index), error=index >= 4)
                for speaker in ("NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA") for index in range(8)]
        return rows, {"cascade_test_ids": [row["utterance_id"] for row in rows], "joint_train_ids": ["train/id"], "joint_validation_ids": ["validation/id"]}

    def test_stratified_24_is_reproducible_without_inference(self):
        rows, splits = self.fixtures()
        selected = pilot.selected_cases(rows, splits)
        self.assertEqual(len(selected), 24)
        random.Random(7).shuffle(rows)
        self.assertEqual(pilot.selected_cases(rows, splits), selected)
        for speaker in {row["speaker_id"] for row in selected}:
            actual = [row for row in selected if row["speaker_id"] == speaker]
            self.assertEqual(sum(row["stratum"] == "lower_error_density_half" for row in actual), 2)
            self.assertEqual(sum(row["stratum"] == "upper_error_density_half" for row in actual), 2)

    def test_protocol_overlap_is_rejected(self):
        rows, splits = self.fixtures()
        splits["joint_train_ids"].append(rows[0]["utterance_id"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            pilot.selected_cases(rows, splits)


class PhoneScoringTests(unittest.TestCase):
    def test_shared_inventory_policy_is_symmetric_and_explicit(self):
        self.assertEqual([pilot.shared_phone(value) for value in ["AH*", "AX", "IY1", "err", "DX", None]],
                         ["AH", "AH", "IY", "[UNK]", "[UNK]", None])

    def test_normalized_distortion_retains_expert_event_not_fake_correct(self):
        row = annotation(error=True)
        row["perceived_phones"] = ["F", "AH*"]
        row["gold_error_events"][0]["observed_phone"] = "AH*"
        raw = pilot.scored(row, ["F", "AH"])
        normalized = pilot.scored(row, ["F", "AH"], normalized=True)
        self.assertEqual(raw["counts"]["FA"], 1)
        self.assertEqual(normalized["counts"]["FA"], 1)
        self.assertEqual(normalized["counts"]["PER_S"], 0)
        self.assertEqual(row["perceived_phones"], ["F", "AH*"], "Scoring must not mutate the frozen annotation")

    def test_unknown_model_label_retained_instead_of_silently_dropped(self):
        phones = pilot.raw_decoded({"decoded_labels": ["<bos>", "f", "ERR", "sil", "<eos>"]}, "crottc-if")
        self.assertEqual(phones, ["F", "[UNK]"])
        result = pilot.scored(annotation(), phones)
        self.assertEqual(result["counts"]["PER_S"], 1)

    def test_ui_hint_list_is_not_a_calibrated_error_decision(self):
        row = annotation(error=True)
        fields = {"sound_candidates": {"status": "available", "words": [{"index": 0}]},
                  "assessment": {"words": [{"status": "uncertain"}], "calibration": {"applied": False}}}
        result = pilot.review_counts(row, fields)
        self.assertEqual(result["hint_tp"], 1)
        self.assertEqual(result["calibrated_decided_words"], 0)
        self.assertFalse(result["calibration_applied"])
        unavailable = pilot.review_counts(row, {"sound_candidates": {"status": "unavailable"}})
        self.assertEqual(unavailable["unavailable_words"], 1)
        self.assertEqual(unavailable["hint_fn"], 0, "Do not score unavailable assessment as a negative decision")


class RunContractTests(unittest.TestCase):
    def test_audio_only_inference_then_scoring_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"frozen")
            case = {"id": "case", "group": "expert_test_pilot", "audio": str(audio), "audio_sha256": pilot.digest(audio),
                    "reference_text": "for", "annotation": annotation()}
            manifest = {"source_sha256": {}, "models": {model: {"files": []} for model in pilot.MODELS}, "cases": [case]}
            calls, loaded, completed = [], [], []

            class Engine:
                def close(self):
                    completed.append("closed")

            def load(model, configured_root):
                loaded.append(model)

                def infer(waveform):
                    self.assertIsInstance(waveform, np.ndarray)
                    calls.append("audio-only")
                    return {"phonemes": [{"phone": "F"}, {"phone": "AH"}], "decoded_labels": ["f", "ah"]}

                return Engine(), infer

            def assess(configured_root, text, evidence, *, confirmed):
                self.assertGreater(len(calls), 0)
                self.assertEqual(text, "for")
                self.assertTrue(confirmed)
                return {"sound_candidates": {"status": "available", "words": []}, "assessment": {"words": [{"status": "uncertain"}]}}

            with patch.object(pilot, "load_backend", side_effect=load), patch.object(pilot, "load_audio", return_value=np.zeros(16000, dtype=np.float32)), patch("mdd.chat_assessment.assessment_fields", side_effect=assess):
                result = pilot.run(manifest, root, root=root)
                resumed = pilot.run(manifest, root, root=root, resume=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(loaded, list(pilot.MODELS))
            self.assertEqual(len(calls), 2)
            self.assertEqual(completed, ["closed", "closed"])
            manifest["cases"][0]["reference_text"] = "changed"
            with self.assertRaisesRegex(ValueError, "Resume identity differs"):
                pilot.run(manifest, root, root=root, resume=True)

    def test_source_hash_change_aborts_before_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scorer.py"
            source.write_text("before", encoding="utf-8")
            manifest = {"source_sha256": {"scorer.py": pilot.digest(source)}, "models": {}, "cases": []}
            source.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                pilot.verify(manifest, root)


if __name__ == "__main__":
    unittest.main()
