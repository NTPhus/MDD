"""Exercise the real trainer/checkpoint loop on a tiny CPU Qwen, including resume."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from scripts import train_projector as trainer
from scripts import projector_data as data
from tests.test_projector import tiny_model


class TinyDataset:
    def __init__(self, rows, *unused):
        self.rows, self.lengths = rows, [14] * len(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Deterministic audio per sample; training RNG is reserved for the real model.
        return {"features": torch.randn((1, 5, 8), generator=torch.Generator().manual_seed(index + 100)),
                "prefix_ids": torch.tensor([[1, 2]]), "suffix_ids": torch.tensor([[3, 4, 5]]),
                "target_ids": torch.tensor([[6, 7, 8, 40]])}


class ProjectorTrainerTests(unittest.TestCase):
    def test_real_cpu_encoder_cache_resumes_and_feature_corruption_is_rejected(self):
        import numpy as np
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        from tests.test_projector import SimpleTokenizer
        torch.set_num_threads(2)
        encoder = Wav2Vec2ForCTC(Wav2Vec2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, vocab_size=8, conv_dim=(8, 8, 8), conv_kernel=(3, 3, 3), conv_stride=(2, 2, 2),
            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2, mask_time_prob=0., apply_spec_augment=False))
        extractor = Wav2Vec2FeatureExtractor(return_attention_mask=True)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            config = {'cpu_threads':2, 'audio_hidden_size':8, 'feature_stride':1, 'feature_dir':temp,
                      'encoder_checkpoint':str(Path(temp)/'encoder'), 'max_length':3072}
            row = {'utterance_id':'TEST/a0001', 'audio_path':'unused.wav', 'audio_sha256':'frozen-audio-hash',
                   'reference_text':'A short sentence.', 'feedback':{}}
            parts = {'train':[row], 'validation':[], 'test':[]}
            stack.enter_context(patch.object(data, 'load', return_value=(parts, {'fixture':'tiny-encoder'})))
            stack.enter_context(patch.object(data, 'encoder_identity', return_value={'fixture':'CPU-frozen'}))
            stack.enter_context(patch('transformers.Wav2Vec2ForCTC.from_pretrained', return_value=encoder))
            stack.enter_context(patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained', return_value=extractor))
            # Import path used by cache_features; no actual dataset audio is read.
            import sys
            sys.path.insert(0, str(data.ROOT/'src'))
            with patch('mdd.runtime.load_audio', return_value=np.sin(np.arange(800)/10).astype(np.float32)) as audio:
                first = data.cache_features(config)
                second = data.cache_features(config)
                self.assertEqual(audio.call_count, 1)
            self.assertEqual(first, second)
            dataset = data.LatentDataset([row], SimpleTokenizer(), config, 'feedback')
            self.assertEqual(dataset[0]['features'].shape[2], 8)
            self.assertEqual(dataset[0]['target_ids'][0, -1].item(), 0)
            path = Path(temp)/'TEST/a0001.npy'
            path.write_bytes(b'corrupt feature')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                dataset[0]

    def test_real_loop_resume_matches_uninterrupted_and_completed_run_is_idempotent(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            folder = Path(temp)
            model_path = folder / "model"
            feature_path = folder / "features"
            model_path.mkdir()
            feature_path.mkdir()
            for name in ("model.safetensors", "tokenizer.json"):
                (model_path / name).write_bytes(b"test-only identity; tiny model injected")
            (feature_path / "manifest.json").write_text('{}')
            config = {"model_path": str(model_path), "feature_dir": str(feature_path), "run_root": str(folder / "full"),
                      "seed": 42, "cpu_threads": 2, "text_hidden_size": 16,
                      "gradient_accumulation_steps": 2, "eval_steps": 2, "alignment": {"epochs": 2, "learning_rate": .001}}
            rows = {"train": [{"id": i} for i in range(4)], "validation": [{"id": 0}]}
            stack.enter_context(patch.object(trainer, "load", return_value=(rows, {"test": "tiny-fixture"})))
            stack.enter_context(patch.object(trainer, "LatentDataset", TinyDataset))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=object()))
            stack.enter_context(patch("scripts.projector_model.build_model", side_effect=lambda *a, **kw: tiny_model()))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self))
            stack.enter_context(patch("torch.autocast", side_effect=lambda *a, **kw: nullcontext()))
            stack.enter_context(patch("torch.cuda.is_bf16_supported", return_value=True))
            stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch("torch.cuda.get_rng_state_all", return_value=[]))
            stack.enter_context(patch("torch.cuda.set_rng_state_all"))
            full = trainer.train(config, "alignment")
            self.assertEqual(full["optimizer_steps"], 4)
            self.assertEqual(full["parameter_counts"]["llm_trainable"], 0)
            config2 = deepcopy(config)
            config2["run_root"] = str(folder / "resumed")
            real_save = trainer.save

            def stop_after_checkpoint(path, value):
                if path.name == "progress.json" and value["step"] == 2:
                    raise RuntimeError("simulated interruption after optimizer checkpoint")
                return real_save(path, value)

            with patch.object(trainer, "save", side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    trainer.train(config2, "alignment")
            state = torch.load(folder / "resumed/alignment/checkpoint/state.pt", weights_only=False)
            self.assertEqual(state["step"], 2)
            resumed = trainer.train(config2, "alignment")
            self.assertEqual(resumed["optimizer_steps"], 4)
            self.assertEqual(resumed["best_validation_loss"], full["best_validation_loss"])
            first = load_file(str(folder / "full/alignment/best/projector.safetensors"))
            second = load_file(str(folder / "resumed/alignment/best/projector.safetensors"))
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertEqual(trainer.train(config2, "alignment"), resumed)
            config2["alignment"]["learning_rate"] = .002
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                trainer.train(config2, "alignment")


if __name__ == "__main__":
    unittest.main()
