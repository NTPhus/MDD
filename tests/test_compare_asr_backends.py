"""Metric, split-leakage, paired-denominator and resumable offline contract tests."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import compare_asr_backends as comparison
from scripts.asr_vi_data import prompt_split


def example_parts():
    parts = {}
    for index, name in enumerate(("train", "validation", "test")):
        text = next("sample word " + suffix for suffix in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
                    if prompt_split("sample word " + suffix) == name)
        parts[name] = [{"speaker_id": "speaker" + str(index), "audio_sha256": str(index),
                        "utterance_id": str(index), "text": text, "split": name}]
    return parts


class MetricTests(unittest.TestCase):
    def test_substitution_deletion_insertion_separately(self):
        result = comparison.alignment("one two three", "one four three five")
        self.assertEqual((result["substitutions"], result["deletions"], result["insertions"]), (1, 0, 1))
        self.assertEqual(result["word_errors"], 2)
        self.assertAlmostEqual(result["wer"], 2 / 3)
        self.assertEqual(comparison.alignment("one two", "one")["deletions"], 1)

    def test_punctuation_case_but_not_numbers_normalized(self):
        self.assertTrue(comparison.alignment("I don't know!", "i don't know.")["exact_match"])
        self.assertEqual(comparison.alignment("six spoons", "6 spoons")["word_errors"], 1)

    def test_empty_hypothesis_and_empty_reference(self):
        self.assertEqual(comparison.alignment("three test words", "")["deletions"], 3)
        self.assertIsNone(comparison.alignment("", "hello")["wer"])


class SplitAuditTests(unittest.TestCase):
    def test_independent_speakers_audio_and_prompt_buckets(self):
        result = comparison.audit_splits(example_parts())
        self.assertTrue(result["passed"])
        self.assertEqual(result["counts"], {"train": 1, "validation": 1, "test": 1})

    def test_same_speaker_is_rejected(self):
        parts = example_parts()
        parts["test"][0]["speaker_id"] = parts["train"][0]["speaker_id"]
        with self.assertRaisesRegex(ValueError, "split leakage"):
            comparison.audit_splits(parts)

    def test_same_audio_is_rejected(self):
        parts = example_parts()
        parts["test"][0]["audio_sha256"] = parts["train"][0]["audio_sha256"]
        with self.assertRaisesRegex(ValueError, "split leakage"):
            comparison.audit_splits(parts)

    def test_same_normalized_prompt_is_rejected(self):
        parts = example_parts()
        parts["test"][0]["text"] = parts["train"][0]["text"].lower() + "."
        with self.assertRaisesRegex(ValueError, "split leakage"):
            comparison.audit_splits(parts)

    def test_wrong_split_bucket_is_rejected(self):
        parts = example_parts()
        parts["test"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "bucket"):
            comparison.audit_splits(parts)


class AggregationTests(unittest.TestCase):
    def test_partial_failure_uses_same_paired_denominator(self):
        cases = [{"id": "a", "group": "heldout", "duration_sec": 2.},
                 {"id": "b", "group": "heldout", "duration_sec": 3.}]
        predictions = {}
        for index, model in enumerate(comparison.MODEL_IDS):
            predictions[model] = {"a": {"status": "completed", "strict": comparison.alignment("one two", "one two" if index else "one"),
                                          "number_normalized": comparison.alignment("one two", "one two" if index else "one"),
                                          "inference_sec": 1.},
                                  "b": {"status": "completed", "strict": comparison.alignment("one two", ""),
                                          "number_normalized": comparison.alignment("one two", ""), "inference_sec": 1.}}
        predictions[comparison.MODEL_IDS[-1]]["b"] = {"status": "error"}
        result = comparison.aggregate(cases, predictions)["heldout"]
        self.assertEqual(result["paired_complete_cases"], 1)
        self.assertEqual(result["excluded_incomplete_ids"], ["b"])
        self.assertTrue(all(model["strict"]["reference_words"] == 2 for model in result["models"].values()))
        pair = result["paired_comparisons"][comparison.MODEL_IDS[0] + " -> " + comparison.MODEL_IDS[1]]
        self.assertEqual(pair["improved_cases"], 1)
        self.assertEqual(pair["word_error_delta"], -1)


class FrozenAndRunTests(unittest.TestCase):
    def test_frozen_source_change_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("before", encoding="utf-8")
            manifest = {"source_sha256": {"source.py": comparison.digest(source)}, "models": {}, "cases": []}
            source.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                comparison.verify_frozen(manifest, root)

    def test_audio_only_call_raw_scoring_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "test.audio"
            audio.write_bytes(b"frozen_audio")
            case = {"id": "case", "group": "heldout", "audio": str(audio),
                    "audio_sha256": comparison.digest(audio), "duration_sec": 1., "reference_text": "six spoons"}
            manifest = {"cases": [case], "models": {model: {"files": {}} for model in comparison.MODEL_IDS}, "source_sha256": {}}
            loaded = []
            calls = []

            def load(identifier, metadata, configured_root):
                loaded.append(identifier)

                def predict(waveform):
                    self.assertIsInstance(waveform, np.ndarray)
                    self.assertEqual(waveform.shape, (16000,))
                    calls.append(identifier)
                    return {"text": "six spoons", "raw_text": "6 spoons"} if identifier == "whisper-small.en" else {"text": "six spoons"}

                return object(), predict

            with patch.object(comparison, "load_backend", side_effect=load), patch.object(comparison, "load_audio", return_value=np.zeros(16000, dtype=np.float32)):
                result = comparison.run(manifest, root, root=root)
                resumed = comparison.run(manifest, root, resume=True, root=root)
            self.assertEqual(loaded, list(comparison.MODEL_IDS))
            self.assertEqual(calls, list(comparison.MODEL_IDS))
            self.assertEqual(resumed["status"], "completed")
            whisper = result["predictions"]["whisper-small.en"]["case"]
            self.assertEqual(whisper["text"], "6 spoons")
            self.assertEqual(whisper["production_text"], "six spoons")
            self.assertEqual(whisper["strict"]["word_errors"], 1)
            self.assertEqual(whisper["number_normalized"]["word_errors"], 0)
            manifest["cases"][0]["reference_text"] = "a changed reference"
            with self.assertRaisesRegex(ValueError, "Resume manifest identity differs"):
                comparison.run(manifest, root, resume=True, root=root)


if __name__ == "__main__":
    unittest.main()
