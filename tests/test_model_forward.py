"""Portable forward checks use tiny tensors and local audio, never BASE or network calls."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from scripts import check_mdd_forward as forward


class TinyForward(torch.nn.Module):
    def __init__(self, invalid=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.invalid = invalid
        self.calls = []

    def _get_feat_extract_output_lengths(self, samples):
        return samples // 400

    def forward(self, input_values, **kwargs):
        self.calls.append(kwargs)
        frames = self._get_feat_extract_output_lengths(input_values.shape[-1])
        logits = torch.zeros((1, frames, 3), device=input_values.device)
        for index in range(frames):
            logits[0, index, [1, 1, 0, 1, 2, 2][index % 6]] = 1
        features = torch.zeros((1, frames, 2), device=input_values.device)
        loss = torch.tensor(float("nan") if self.invalid == "loss" else 1.0,
                            device=input_values.device) if "labels" in kwargs else None
        if self.invalid == "logits":
            logits[0, 0, 0] = float("nan")
        if self.invalid == "features":
            features[0, 0, 0] = float("inf")
        return SimpleNamespace(logits=logits, hidden_states=(features,), loss=loss)


class ModelForwardTests(unittest.TestCase):
    def test_cli_preserves_corpus_cpu_defaults_and_supports_external_audio(self):
        defaults = forward.build_parser().parse_args([])
        self.assertIsNone(defaults.audio)
        self.assertEqual(defaults.device, "cpu")
        args = forward.build_parser().parse_args(["--audio", "sample.wav", "--device", "auto"])
        self.assertEqual(args.audio, Path("sample.wav"))
        self.assertEqual(args.device, "auto")

    def test_stereo_audio_downmixes_and_resamples_to_16khz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.full((4410, 2), 0.1, dtype=np.float32), 44100)
            audio, source_rate = forward.load_audio(path)
            self.assertEqual(source_rate, 44100)
            self.assertEqual(audio.shape, (1600,))
            self.assertEqual(audio.dtype, np.float32)
            self.assertTrue(np.isfinite(audio).all())
            self.assertAlmostEqual(float(audio.mean()), 0.1, places=3)

    def test_missing_empty_and_nonfinite_audio_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with self.assertRaises(FileNotFoundError):
                forward.load_audio(path)
            for samples in (np.empty(0, dtype=np.float32), np.array([0, np.nan], dtype=np.float32)):
                with self.subTest(samples=len(samples)):
                    sf.write(path, samples, 16000, subtype="FLOAT")
                    with self.assertRaisesRegex(ValueError, "nonempty.*finite"):
                        forward.load_audio(path)

    def test_default_sample_selection_excludes_held_out_and_out_of_range_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data/processed/l2arctic-v1"
            protocol = root / "data/protocol/l2arctic-v1"
            processed.mkdir(parents=True)
            protocol.mkdir(parents=True)
            rows = [
                {"utterance_id": "source/short", "duration_sec": 1.0},
                {"utterance_id": "source/long", "duration_sec": 2.0},
                {"utterance_id": "source/too-short", "duration_sec": 0.25},
                {"utterance_id": "held-out", "duration_sec": 0.75},
            ]
            (processed / "annotations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (protocol / "splits.json").write_text(
                json.dumps({"source_train_ids": [row["utterance_id"] for row in rows[:3]]}), encoding="utf-8")
            with patch.object(forward, "ROOT", root):
                self.assertEqual(forward.source_training_sample()["utterance_id"], "source/short")

    def run_tiny_check(self, *, supervised=False, invalid=None, seconds=0.15, device="cpu"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data/processed/l2arctic-v1"
            processed.mkdir(parents=True)
            (processed / "vocab.json").write_text(json.dumps({"[PAD]": 0, "AA": 1, "T": 2}), encoding="utf-8")
            base = root / "models/pretrained/wav2vec2-base"
            base.mkdir(parents=True)
            (base / "download_revision.txt").write_text("test-revision", encoding="utf-8")
            audio_path = root / "input.wav"
            sf.write(audio_path, np.full(int(16000 * seconds), 0.1, dtype=np.float32), 16000)
            model = TinyForward(invalid)
            extractor = lambda audio, **kwargs: {"input_values": torch.as_tensor(audio).unsqueeze(0)}
            extractor_factory = Mock(return_value=extractor)
            model_factory = Mock(return_value=model)
            fake_transformers = SimpleNamespace(
                Wav2Vec2FeatureExtractor=SimpleNamespace(from_pretrained=extractor_factory),
                Wav2Vec2ForCTC=SimpleNamespace(from_pretrained=model_factory),
            )
            row = {"audio_path": "input.wav", "utterance_id": "source/sample", "perceived_phones": ["AA", "T"]}
            args = argparse.Namespace(audio=None if supervised else audio_path, device=device)
            with patch.object(forward, "ROOT", root), \
                    patch.object(forward, "source_training_sample", return_value=row) as choose_sample, \
                    patch.object(forward, "memory_report", return_value={}), \
                    patch.dict(sys.modules, {"transformers": fake_transformers}), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                report = forward.run_check(args)
            for factory in (extractor_factory, model_factory):
                self.assertTrue(factory.call_args.kwargs["local_files_only"])
            self.assertEqual(choose_sample.call_count, int(supervised))
            return report, model

    def test_external_audio_decodes_without_labels_or_loss_or_corpus(self):
        report, model = self.run_tiny_check(device="auto")
        self.assertEqual(report["input_split"], "external_unlabeled")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["requested_device"], "auto")
        self.assertNotIn("labels", model.calls[0])
        self.assertNotIn("finite_ctc_loss", report)
        self.assertNotIn("ctc_target_length", report)
        self.assertFalse(report["trained"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(report["finite_logits"])
        self.assertTrue(report["finite_hidden_features"])
        self.assertEqual(report["predicted_phones_untrained"], ["AA", "AA", "T"])

    def test_corpus_mode_keeps_supervised_ctc_loss(self):
        report, model = self.run_tiny_check(supervised=True)
        self.assertEqual(report["input_split"], "source_train")
        self.assertEqual(model.calls[0]["labels"].tolist(), [[1, 2]])
        self.assertEqual(report["finite_ctc_loss"], 1.0)
        self.assertEqual(report["ctc_target_length"], 2)

    def test_nonfinite_model_outputs_and_short_audio_are_rejected(self):
        for invalid in ("logits", "features", "loss"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "nonfinite|invalid CTC loss"):
                self.run_tiny_check(supervised=True, invalid=invalid)
        with self.assertRaisesRegex(ValueError, "too short"):
            self.run_tiny_check(seconds=0.01)

    def test_explicit_cuda_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            self.run_tiny_check(device="cuda")

    def test_windows_peak_rss_and_other_platform_current_rss_are_distinguished(self):
        for memory, peak in ((SimpleNamespace(rss=1024**2, peak_wset=2 * 1024**2), 2.0),
                             (SimpleNamespace(rss=1024**2), None)):
            with self.subTest(peak=peak), patch.dict(sys.modules, {
                "psutil": SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: memory)),
            }):
                report = forward.memory_report()
            self.assertEqual(report["process_rss_mib"], 1.0)
            self.assertEqual(report["peak_process_rss_mib"], peak)


if __name__ == "__main__":
    unittest.main()
