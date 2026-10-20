"""Verify local CPU settings preserve the scientific experiment and trainable scope."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdd import runtime


def config(name):
    supplied = json.loads((ROOT / "configs" / f"local_{name}.json").read_text())
    unknown = supplied.keys() - runtime.DEFAULTS.keys()
    if unknown:
        raise AssertionError(f"Unrecognized runtime settings: {unknown}")
    merged = dict(runtime.DEFAULTS, **supplied)
    runtime.validate_config(merged)
    return merged


def record(utterance_id, speaker, duration):
    return {"utterance_id": utterance_id, "speaker_id": speaker,
            "duration_sec": duration, "audio_path": "not-loaded.wav",
            "perceived_phones": ["AA"], "canonical_phones": ["AA"]}


class ScopeModel(torch.nn.Module):
    """Three tiny layers stand in for CNN, Transformer, and the phoneme head."""
    def __init__(self):
        super().__init__()
        self.wav2vec2 = torch.nn.ModuleDict({
            "feature_extractor": torch.nn.Linear(1, 1),
            "encoder": torch.nn.Linear(1, 1),
        })
        self.lm_head = torch.nn.Linear(1, 3)
        self.checkpointing = None

    def freeze_feature_encoder(self):
        self.wav2vec2["feature_extractor"].requires_grad_(False)

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = kwargs

    def gradient_checkpointing_disable(self):
        self.checkpointing = False


class LocalTrainingConfigTests(unittest.TestCase):
    def test_local_configs_select_cpu_without_querying_cuda(self):
        for name in ("source", "adapt_vi", "control", "smoke"):
            with self.subTest(name=name):
                current = config(name)
                self.assertEqual(current["device"], "cpu")
                self.assertEqual(current["mixed_precision"], "no")
                self.assertEqual(current["cpu_threads"], 4)
                with patch.object(runtime.torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
                    self.assertEqual(runtime.configure_device(current), torch.device("cpu"))
                with runtime.autocast_context(current):
                    self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_scientific_configs_keep_gpu_budgets_and_all_utterance_durations(self):
        # A long synthetic utterance must remain in scientific data selection;
        # short audio caps are permitted only for the technical check.
        rows = {"long_source": record("long_source", "source", 30),
                "source_dev": record("source_dev", "source_dev", 25),
                "long_vi": record("long_vi", "vi_train", 20),
                "vi_dev": record("vi_dev", "vi_dev", 15)}
        splits = {"source_train_ids": ["long_source"], "source_validation_ids": ["source_dev"],
                  "folds": [{"fold": 1, "train_ids": ["long_vi"], "validation_ids": ["vi_dev"]}]}
        for name, gpu_name in (("source", "source"), ("adapt_vi", "adapt_vi"), ("control", "control")):
            with self.subTest(name=name):
                local = config(name)
                gpu = json.loads((ROOT / "configs" / f"{gpu_name}.json").read_text())
                self.assertEqual(local["purpose"], "scientific_training")
                for key in ("max_steps", "gradient_accumulation_steps", "backbone_learning_rate", "head_learning_rate"):
                    self.assertEqual(local[key], gpu[key], key)
                for key in ("max_train_samples", "max_eval_samples", "max_audio_seconds"):
                    self.assertIsNone(local[key], key)
                source, target, validation, audit = runtime.training_selections(local, rows, splits)
                self.assertEqual([row["utterance_id"] for row in source], ["long_source"])
                self.assertEqual([row["utterance_id"] for row in target], ["long_vi"] if name == "adapt_vi" else [])
                self.assertEqual([row["utterance_id"] for row in validation],
                                 ["source_dev"] if name == "source" else ["vi_dev"])
                self.assertFalse(audit["test_used_for_training"])
                self.assertEqual(local["protocol_dir"], "data/protocol/l2arctic-v1")
                self.assertEqual(local["model_path"], "models/pretrained/wav2vec2-base")

    def test_adaptation_and_control_have_identical_update_and_selection_settings(self):
        adaptation, control = config("adapt_vi"), config("control")
        for key in runtime.DEFAULTS:
            if key != "stage":
                self.assertEqual(adaptation[key], control[key], key)
        self.assertEqual(adaptation["source_replay_fraction"], 0.5)
        self.assertEqual(adaptation["max_steps"], 400)
        self.assertEqual(config("source")["max_steps"], 2000)
        for current in (config("source"), adaptation, control):
            self.assertEqual(current["batch_size"] * current["gradient_accumulation_steps"], 4)
            self.assertLessEqual(current["checkpoint_every"], 50)
            self.assertTrue(current["save_optimizer"])

    def test_smoke_exercises_same_trainable_layers_without_becoming_a_baseline(self):
        smoke = config("smoke")
        self.assertEqual(smoke["purpose"], "technical_smoke_only")
        self.assertEqual(smoke["max_steps"], 1)
        self.assertEqual(smoke["max_train_samples"], 2)
        self.assertEqual(smoke["max_eval_samples"], 2)
        self.assertEqual(smoke["max_audio_seconds"], 3)
        for name in ("source", "adapt_vi", "control", "smoke"):
            with self.subTest(name=name):
                model = ScopeModel()
                with patch.object(runtime.Wav2Vec2ForCTC, "from_pretrained", return_value=model), \
                     patch.object(runtime.Wav2Vec2FeatureExtractor, "from_pretrained", return_value=object()):
                    loaded, _ = runtime.load_model(config(name), {"[PAD]": 0, "[UNK]": 1, "AA": 2})
                self.assertFalse(any(p.requires_grad for p in loaded.wav2vec2["feature_extractor"].parameters()))
                self.assertTrue(all(p.requires_grad for p in loaded.wav2vec2["encoder"].parameters()))
                self.assertTrue(all(p.requires_grad for p in loaded.lm_head.parameters()))
                self.assertEqual(loaded.checkpointing, {"gradient_checkpointing_kwargs": {"use_reentrant": False}})


if __name__ == "__main__":
    unittest.main()
