"""Training plumbing tests use a two-parameter synthetic model, never BASE training."""
import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd import runtime


class TinyExtractor:
    def __call__(self, audio, **kwargs):
        maximum = max(map(len, audio))
        values = torch.zeros((len(audio), maximum))
        mask = torch.zeros((len(audio), maximum), dtype=torch.long)
        for index, waveform in enumerate(audio):
            values[index, :len(waveform)] = torch.as_tensor(waveform)
            mask[index, :len(waveform)] = 1
        return {"input_values": values, "attention_mask": mask}

    def save_pretrained(self, path):
        (path / "preprocessor_config.json").write_text("{}")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(1, 3, bias=False)
        self.numpy_draws = []

    def _get_feat_extract_output_lengths(self, samples):
        return samples // 320

    def forward(self, input_values, attention_mask, labels):
        self.numpy_draws.append(float(np.random.random()))
        values = input_values.mean(dim=-1).view(-1, 1, 1)
        logits = self.lm_head(values).expand(-1, 5, -1)
        loss = (self.lm_head.weight - 0.25).square().mean()
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path, **kwargs):
        # Synthetic state only; no downloaded model is loaded or modified.
        torch.save(self.state_dict(), path / "model.safetensors")
        (path / "config.json").write_text("{}")


class RetryOnceScaler:
    """Simulate one FP16 overflow on CPU without invoking CUDA or BASE."""
    def __init__(self):
        self.scale_value = 8.0
        self.calls = 0
        self.skipped = False

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.calls += 1
        self.skipped = self.calls == 1
        if not self.skipped:
            optimizer.step()

    def update(self):
        if self.skipped:
            self.scale_value /= 2

    def state_dict(self):
        return {}


def row(utterance_id, speaker, duration=1.0):
    return {"utterance_id": utterance_id, "speaker_id": speaker, "audio_path": "synthetic.wav",
            "duration_sec": duration, "perceived_phones": ["AA"], "canonical_phones": ["AA"],
            "gold_error_events": [], "canonical_word_indices": [0]}


class RuntimeTests(unittest.TestCase):
    def test_greedy_collapse_happens_before_blank_removal(self):
        phones = {0: "[PAD]", 1: "AA", 2: "T"}
        self.assertEqual(runtime.greedy_decode([1, 1, 0, 1, 2, 2, 0], phones), ["AA", "AA", "T"])

    def test_repeated_target_requires_an_extra_ctc_frame(self):
        self.assertEqual(runtime.ctc_required_frames([2, 2, 3, 3, 3]), 8)

    def test_duration_filter_drops_whole_sample(self):
        short = row("short", "one", 1)
        long = row("long", "two", 9)
        chosen, report = runtime.select_records({"short": short, "long": long}, ["short", "long"], max_seconds=3)
        self.assertEqual(chosen, [short])
        self.assertEqual(report["excluded"], [{"utterance_id": "long", "reason": "duration_filter_no_audio_truncation"}])
        self.assertEqual(long["duration_sec"], 9)

    def test_invalid_training_configuration_is_rejected(self):
        for key, value in (("batch_size", 0), ("max_steps", -1), ("mixed_precision", "half"),
                           ("source_replay_fraction", 2), ("fold", 5)):
            with self.subTest(key=key):
                config = dict(runtime.DEFAULTS, **{key: value})
                with self.assertRaises(ValueError):
                    runtime.validate_config(config)

    def test_audio_resamples_mono_and_verifies_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.ones((4410, 2), dtype=np.float32) * 0.1, 44100)
            record = dict(row("synthetic", "one"), audio_path=str(path), audio_sha256=runtime.sha256(path))
            waveform = runtime.load_audio(record)
            self.assertEqual(waveform.shape, (1600,))
            self.assertEqual(waveform.dtype, np.float32)
            record["audio_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "frozen manifest"):
                runtime.load_audio(record)

    def test_collator_pads_targets_with_minus100_and_rejects_impossible_ctc(self):
        collator = runtime.PhoneCollator(TinyExtractor(), {"[PAD]": 0, "[UNK]": 1, "AA": 2}, TinyModel())
        one, two = row("one", "s"), row("two", "s")
        two["perceived_phones"] = ["AA", "AA"]
        with patch.object(runtime, "load_audio", return_value=np.zeros(1600, dtype=np.float32)):
            batch = collator([one, two])
            self.assertEqual(batch["labels"].tolist(), [[2, -100], [2, 2]])
        with patch.object(runtime, "load_audio", return_value=np.zeros(640, dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "CTC-infeasible"):
                collator([two])

    def test_source_adapt_and_control_select_disjoint_speakers(self):
        records = {item["utterance_id"]: item for item in
                   [row("src", "source"), row("src_dev", "source_dev"), row("vi", "vi_train"), row("vi_dev", "vi_dev")]}
        splits = {"source_train_ids": ["src"], "source_validation_ids": ["src_dev"],
                  "folds": [{"fold": 1, "train_ids": ["vi"], "validation_ids": ["vi_dev"]}]}
        for stage in "ABC":
            config = dict(runtime.DEFAULTS, stage=stage)
            source, target, validation, report = runtime.training_selections(config, records, splits)
            self.assertEqual(source[0]["utterance_id"], "src")
            self.assertEqual(bool(target), stage == "B")
            self.assertEqual(validation[0]["utterance_id"], "src_dev" if stage == "A" else "vi_dev")
            self.assertFalse(report["test_used_for_training"])

    def test_amp_retry_checkpoint_state_and_resume_with_synthetic_model(self):
        """Exercise optimizer/RNG persistence on three toy parameters, not BASE."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory / "config.json"
            config_path.write_text(json.dumps({
                "stage": "A", "purpose": "technical_smoke_only", "device": "cpu", "mixed_precision": "no",
                "max_steps": 1, "eval_every": 1, "checkpoint_every": 1,
                "gradient_accumulation_steps": 1, "freeze_backbone": True,
                "gradient_checkpointing": False, "selection_metric": "per",
            }))
            records = {"s": row("s", "source"), "v": row("v", "validation"), "t": row("t", "vi")}
            vocab = {"[PAD]": 0, "[UNK]": 1, "AA": 2}
            splits = {"source_train_ids": ["s"], "source_validation_ids": ["v"],
                      "folds": [{"fold": 1, "train_ids": ["t"], "validation_ids": ["v"]}]}
            fake_hashes = {"scoring_protocol": "test", "files": {
                name: {"path": "synthetic", "sha256": name}
                for name in ("annotations", "vocab", "normalization", "manifest", "splits", "protocol", "scorer")}}
            loaded_models = []

            def load_toy(config, phone_vocab, checkpoint=None):
                model = TinyModel()
                if checkpoint:
                    model.load_state_dict(torch.load(checkpoint / "model.safetensors", weights_only=True))
                loaded_models.append(model)
                return model, TinyExtractor()

            arguments = runtime.train_parser().parse_args([
                "--config", str(config_path), "--run-id", "toy", "--runs-root", str(directory)])
            with patch.object(runtime, "load_records", return_value=(records, vocab, splits)), \
                 patch.object(runtime, "provenance", return_value=fake_hashes), \
                 patch.object(runtime, "load_model", side_effect=load_toy), \
                 patch.object(runtime, "load_audio", return_value=np.ones(1600, dtype=np.float32)), \
                 patch.object(runtime, "sha256", return_value="synthetic-checksum"):
                fake_scaler = RetryOnceScaler()
                with patch.object(runtime.torch.amp, "GradScaler", return_value=fake_scaler):
                    run_dir = runtime.run_train(arguments)
                self.assertEqual(fake_scaler.calls, 2)
                self.assertEqual(loaded_models[0].numpy_draws[0], loaded_models[0].numpy_draws[1])
                overflows = [json.loads(line) for line in (run_dir / "amp_overflows.jsonl").read_text().splitlines()]
                self.assertEqual(len(overflows), 1)
                self.assertEqual(overflows[0]["optimizer_steps_completed"], 0)
                state = torch.load(run_dir / "checkpoint/training_state.pt", weights_only=False)
                self.assertEqual(state["step"], 1)
                self.assertEqual(state["seen_samples"], {"source": 1, "target": 0})
                self.assertEqual(state["seen_audio_seconds"], {"source": 1.0, "target": 0.0})
                # A killed process can have logged a step after its last complete checkpoint.
                with (run_dir / "training_log.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": 2, "loss": 999, "note": "uncheckpointed attempt"}) + "\n")
                arguments.resume = str(run_dir / "checkpoint")
                arguments.max_steps = 2
                runtime.run_train(arguments)
                state = torch.load(run_dir / "checkpoint/training_state.pt", weights_only=False)
                self.assertEqual(state["step"], 2)
                self.assertEqual(state["seen_samples"], {"source": 2, "target": 0})
                logs = [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines()]
                self.assertEqual([item["step"] for item in logs], [1, 2])
                self.assertNotIn(999, [item["loss"] for item in logs])
                self.assertEqual(len(list(run_dir.glob("training_log.uncheckpointed-*.jsonl"))), 1)
            self.assertTrue((run_dir / "best_checkpoint/model.safetensors").is_file())
            completed = runtime.read_json(run_dir / "completed.json")
            self.assertFalse(completed["test_evaluated"])
            self.assertEqual(completed["purpose"], "technical_smoke_only")
            best_metadata = runtime.read_json(run_dir / "best_checkpoint/metadata.json")
            self.assertEqual(best_metadata["effective_config"]["max_steps"], 2)
            self.assertEqual(best_metadata["completed_training"]["optimizer_steps"], 2)
            predictions = [json.loads(line) for line in (run_dir / "predictions_validation.jsonl").read_text().splitlines()]
            self.assertEqual([item["utterance_id"] for item in predictions], ["v"])

    def test_interrupted_checkpoint_directory_rename_recovers_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            desired = Path(directory) / "checkpoint"
            previous = Path(directory) / "checkpoint.previous"
            previous.mkdir()
            (previous / "metadata.json").write_text("{}")
            self.assertEqual(runtime.resolve_checkpoint(desired), previous)
            desired.mkdir()
            self.assertEqual(runtime.resolve_checkpoint(desired), desired)

    def test_resume_archives_uncheckpointed_validation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runtime.write_json(directory / "metrics_validation.json", {"step": 9})
            (directory / "predictions_validation.jsonl").write_text('{"example": true}\n')
            (directory / "validation_log.jsonl").write_text('{"step": 3}\n{"step": 9}\n')
            runtime.archive_uncheckpointed_logs(directory, step=3)
            self.assertFalse((directory / "metrics_validation.json").exists())
            self.assertEqual(len(list(directory.glob("metrics_validation.json.uncheckpointed-*"))), 1)
            retained = [json.loads(line) for line in (directory / "validation_log.jsonl").read_text().splitlines()]
            self.assertEqual(retained, [{"step": 3}])

    def test_control_has_same_optimizer_budget_and_rates_as_adaptation(self):
        configs = runtime.ROOT / "configs"
        adaptation = runtime.read_json(configs / "adapt_vi.json")
        control = runtime.read_json(configs / "control.json")
        for key in ("seed", "fold", "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every",
                    "backbone_learning_rate", "head_learning_rate", "freeze_backbone", "selection_metric"):
            self.assertEqual(adaptation[key], control[key], key)


if __name__ == "__main__":
    unittest.main()
