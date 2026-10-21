"""Local backend routing, offline generation limits and model caching."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.llm_client import Completion, LLMClient, LLMError, ROOT, is_configured
from scripts import local_feedback_model as local
from src.mdd.demo_feedback import FeedbackService


class Tokens(dict):
    def to(self, device):
        return self


class LocalFeedbackTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "config.json").write_text('{"model_type":"qwen3"}')
        (self.root / "model.safetensors").write_bytes(b"fixture")
        env = patch.dict(os.environ, {"LLM_BACKEND": "local", "LLM_LOCAL_MODEL_PATH": str(self.root)}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(lambda: setattr(local, "_CACHE", None))
        self.addCleanup(lambda: setattr(local, "_CACHE_KEY", None))
        local._CACHE = local._CACHE_KEY = None

    def test_local_feedback_needs_no_api_key_and_uses_existing_validation(self):
        client = LLMClient()
        assessment = json.loads((ROOT / "contracts/examples/assessment.json").read_text())
        feedback = json.loads((ROOT / "contracts/examples/feedback.json").read_text())
        original = deepcopy(assessment)
        completion = Completion(json.dumps(feedback), "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", None, {}, "stop", .1)
        with patch.object(local, "complete", return_value=completion.__dict__) as generate:
            result = client.generate_feedback(assessment)
        self.assertEqual(result.source, "llm")
        self.assertEqual(original, assessment)
        self.assertTrue(is_configured())
        self.assertEqual(json.loads(generate.call_args.args[0][1]["content"]), assessment)
        self.assertTrue(FeedbackService(self.root).capabilities()["llm_configured"])

    def test_local_failure_never_calls_remote_gateway(self):
        os.environ["LLM_API_KEY"] = "unused-legacy-key"
        client = LLMClient()
        assessment = json.loads((ROOT / "contracts/examples/assessment.json").read_text())
        with patch.object(local, "complete", side_effect=RuntimeError("local error")), patch("scripts.llm_client.build_opener") as remote:
            result = client.generate_feedback(assessment)
        self.assertEqual(result.source, "template")
        remote.assert_not_called()

    def test_local_feedback_without_tts_is_repaired_then_rejected(self):
        client = LLMClient()
        assessment = json.loads((ROOT / "contracts/examples/assessment.json").read_text())
        feedback = json.loads((ROOT / "contracts/examples/feedback.json").read_text())
        feedback["tts_segments"] = []
        completion = Completion(json.dumps(feedback), "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", None, {}, "stop", .1)
        with patch.object(local, "complete", return_value=completion.__dict__) as generate:
            result = client.generate_feedback(assessment)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.source, "template")
        self.assertEqual(result.fallback_reason, "invalid_feedback_after_repair")

    def test_missing_adapter_or_weights_is_unconfigured(self):
        os.environ["LLM_LOCAL_ADAPTER_PATH"] = str(self.root / "missing")
        self.assertFalse(is_configured())
        with self.assertRaises(LLMError):
            LLMClient()
        os.environ.pop("LLM_LOCAL_ADAPTER_PATH")
        (self.root / "model.safetensors").unlink()
        self.assertFalse(is_configured())

    def bundle(self, ids):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = Tokens(input_ids=torch.tensor([[4, 5]]))
        tokenizer.decode.return_value = '{"result":"fixture"}'
        tokenizer.eos_token_id = 2
        network = Mock(config=SimpleNamespace(max_position_embeddings=4096),
                       generation_config=SimpleNamespace(eos_token_id=[2]))
        network.generate.return_value = torch.tensor([[4, 5] + ids])
        return network, tokenizer

    def test_generation_disables_thinking_and_reuses_frozen_local_model(self):
        bundle = self.bundle([6, 2])
        with patch.object(local, "_load", return_value=bundle) as load:
            for _ in range(2):
                result = local.complete([{"role": "user", "content": "hello"}], local.settings(), 16, 3)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(result["usage"], {"prompt_tokens": 2, "completion_tokens": 2})
        self.assertFalse(bundle[1].apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(bundle[0].generate.call_args.kwargs["max_time"], 3)

    def test_overflow_and_unfinished_generations_are_rejected(self):
        bundle = self.bundle([6, 7])
        with patch.object(local, "_load", return_value=bundle):
            with self.assertRaisesRegex(ValueError, "context budget"):
                local.complete([], local.settings(), 4096, 3)
            bundle[0].generate.assert_not_called()
            with self.assertRaisesRegex(ValueError, "token/time limit"):
                local.complete([], local.settings(), 16, 3)


if __name__ == "__main__":
    unittest.main()
