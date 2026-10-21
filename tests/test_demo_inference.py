"""Audio validation and honest CTC output; synthetic tests never train a model."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from src.mdd.demo_inference import LocalInference, canonical_comparison, decode_ctc


def audio_bytes(seconds=0.4, rate=16000, silent=False):
    samples = np.zeros(int(seconds * rate), dtype=np.float32)
    if not silent:
        samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(len(samples)) / rate)).astype(np.float32)
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


class ForbiddenReference(dict):
    def get(self, key, default=None):
        if key in {"perceived_phones", "gold_error_events", "target_phones", "phone_intervals"}:
            raise AssertionError("Inference must never read gold/perceived labels")
        return super().get(key, default)


class DemoInferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.engine = LocalInference(self.root)

    def test_ctc_repeat_collapse_precedes_blank_removal_and_times_are_bounded(self):
        phones = decode_ctc([1, 1, 0, 1, 2, 2, 0], [.9] * 7,
                            {0: "[PAD]", 1: "AA", 2: "T"}, 0, 1.4)
        self.assertEqual([item["phone"] for item in phones], ["AA", "AA", "T"])
        self.assertEqual([(item["start_sec"], item["end_sec"]) for item in phones],
                         [(0, 0.4), (0.6, 0.8), (0.8, 1.2)])
        self.assertTrue(all(item["confidence"] == .9 for item in phones))

    def test_silence_rejected_before_loading_any_model(self):
        with patch.object(self.engine, "_get_bundle") as loader:
            with self.assertRaisesRegex(ValueError, "yên lặng"):
                self.engine.analyze({}, audio_bytes(silent=True), "hello")
        loader.assert_not_called()

    def test_duration_header_is_validated_before_decode_allocation(self):
        header = SimpleNamespace(samplerate=16000, frames=16000 * 31, channels=1)
        with patch("soundfile.SoundFile") as stream:
            stream.return_value.__enter__.return_value = header
            with self.assertRaisesRegex(ValueError, "30 giây"):
                self.engine._audio(b"bad wav with an exaggerated header")
        self.assertFalse(hasattr(header, "read"))

    def test_real_long_audio_and_nonfinite_audio_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "30 giây"):
            self.engine._audio(audio_bytes(seconds=30.01, rate=8000))
        stream = io.BytesIO()
        sf.write(stream, np.full(6400, np.nan, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "không hữu hạn"):
            self.engine._audio(stream.getvalue())

    def test_resampling_and_envelope_bound(self):
        waveform, info = self.engine._audio(audio_bytes(rate=44100))
        self.assertEqual(len(waveform), 6400)
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(len(info["waveform"]), 160)
        self.assertTrue(any(value < 0 for value in info["waveform"]))

    def test_canonical_comparison_requires_matching_prompt_and_never_reads_gold(self):
        reference = ForbiddenReference(prompt_text="Hello world", canonical_phones=["HH", "W"],
                                       canonical_word_indices=[0, 1], perceived_phones=["GOLD"])
        self.assertIsNone(canonical_comparison("hello there", reference, ["HH", "W"]))
        result = canonical_comparison("  HELLO  world ", reference, ["HH", "R"])
        self.assertEqual(result["counts"], {"substitution": 1, "deletion": 0, "insertion": 0, "match": 1})
        self.assertEqual([word["status"] for word in result["words"]], ["match", "difference"])
        self.assertEqual(result["events"][0]["word_index"], 1)

    def test_experimental_checkpoint_suppresses_comparison_even_with_matching_reference(self):
        class TinyModel:
            def __call__(self, **inputs):
                self.inputs = set(inputs)
                self.grad_enabled = torch.is_grad_enabled()
                return SimpleNamespace(logits=torch.tensor([[[0., 8., 0.], [0., 8., 0.], [8., 0., 0.]]]))

        network = TinyModel()
        bundle = {"network": network, "extractor": lambda wave, **kw: {"input_values": torch.as_tensor(wave)[None]},
                  "vocabulary": {"[PAD]": 0, "HH": 1, "W": 2}, "experimental": True}
        model = {"id": "smoke", "label": "Smoke", "path": self.root, "kind": "phoneme", "experimental": False}
        reference = ForbiddenReference(prompt_text="hello", canonical_phones=["HH"], canonical_word_indices=[0])
        with patch.object(self.engine, "_get_bundle", return_value=bundle), \
                patch("src.mdd.demo_inference.canonical_comparison") as compare:
            result = self.engine.analyze(model, audio_bytes(), "hello", reference)
        compare.assert_not_called()
        self.assertIsNone(result["comparison"])
        self.assertIsNone(result["asr_text"])
        self.assertEqual(result["analysis_kind"], "experimental_phonemes")
        self.assertTrue(result["model"]["experimental"])
        self.assertFalse(network.grad_enabled)
        self.assertEqual(network.inputs, {"input_values"})
        self.assertEqual([phone["phone"] for phone in result["phonemes"]], ["HH"])
        json.dumps(result, allow_nan=False)

    def test_misaligned_corpus_word_tier_does_not_flag_the_wrong_prompt_word(self):
        reference = ForbiddenReference(prompt_text="A low bush", canonical_phones=["L", "B"],
                                       canonical_word_indices=[0, 1],
                                       word_intervals=[{"index": 0, "text": "low"}, {"index": 1, "text": "bush"}])
        result = canonical_comparison("A low bush", reference, ["R", "B"])
        self.assertEqual(result["counts"]["substitution"], 1)
        self.assertEqual([word["status"] for word in result["words"]], ["unavailable"] * 3)
        self.assertIsNone(result["events"][0]["word_index"])

    def test_failed_model_replacement_keeps_previous_cached_model(self):
        model = {"path": self.root, "kind": "backbone", "fingerprint": "one"}
        previous = {"network": object()}
        with patch.object(self.engine, "_load_bundle", side_effect=[previous, ValueError("broken checkpoint")]) as load:
            self.assertIs(self.engine._get_bundle(model), previous)
            with self.assertRaisesRegex(ValueError, "broken checkpoint"):
                self.engine._get_bundle(dict(model, fingerprint="two"))
            self.assertIs(self.engine._get_bundle(model), previous)
        self.assertEqual(load.call_count, 2)

    def test_missing_ctc_weights_are_rejected_instead_of_using_random_head(self):
        (self.root / "vocab.json").write_text('{"[PAD]": 0, "HH": 1}', encoding="utf-8")
        (self.root / "metadata.json").write_text('{"purpose": "scientific_training"}', encoding="utf-8")
        network = SimpleNamespace(config=SimpleNamespace(vocab_size=2, pad_token_id=0))
        with patch("transformers.Wav2Vec2FeatureExtractor.from_pretrained", return_value=object()), \
                patch("transformers.Wav2Vec2ForCTC.from_pretrained",
                      return_value=(network, {"missing_keys": ["lm_head.weight"]})):
            with self.assertRaisesRegex(ValueError, "ngẫu nhiên"):
                self.engine._load_bundle(self.root, "phoneme")


if __name__ == "__main__":
    unittest.main()
