"""Conversation decoder routing, without model weights or native tensor loads."""
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import local_chat_process
from scripts.local_chat_worker import QwenChatEngine
from scripts.local_speech_engine import dialogue_request


class ChatSamplingTests(unittest.TestCase):
    def test_ipc_forwards_only_the_supported_internal_profile(self):
        messages = [{"role": "user", "content": "Hello"}]
        for profile in (None, "conversation"):
            with self.subTest(profile=profile), patch.object(local_chat_process, "_request", return_value={}) as send:
                local_chat_process.complete(messages, {"device": "cpu", "adapter": None,
                                                      "generation_profile": profile},
                                            80, 30., on_text=lambda _: None, cancelled=threading.Event())
                payload = send.call_args.args[0]
                self.assertEqual(payload["generation_profile"], profile)
                self.assertEqual(payload["messages"], messages)
                self.assertEqual(payload["model_id"], "qwen-base")
        with patch.object(local_chat_process, "_request") as send, self.assertRaises(ValueError):
            local_chat_process.complete(messages, {"device": "cpu", "adapter": None,
                                                  "generation_profile": "client-temperature"},
                                        80, 30., on_text=lambda _: None, cancelled=threading.Event())
        send.assert_not_called()

    def test_only_dialogue_requests_opt_in_to_conversation_sampling(self):
        request = dialogue_request({"model_id": "qwen-trained", "history": []}, "I like pizza.")
        self.assertEqual(request["model_id"], "qwen-base")
        self.assertEqual(request["generation_profile"], "conversation")
        from scripts.sentence_correction import correction_request
        self.assertNotIn("generation_profile", correction_request("I like pizza.", []))

    def test_worker_passes_sampling_options_to_model_and_keeps_structured_default_greedy(self):
        class Streamer:
            def __init__(self, *args, **kwargs):
                pass

        class Output:
            def __getitem__(self, index):
                return [12, 9]

        native_doubles = {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "transformers": SimpleNamespace(TextStreamer=Streamer, StoppingCriteria=object,
                                            StoppingCriteriaList=list),
        }
        for profile in (None, "conversation"):
            with self.subTest(profile=profile):
                engine = QwenChatEngine()
                engine.load = Mock()
                engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096),
                                               generation_config=SimpleNamespace(eos_token_id=9),
                                               generate=Mock(return_value=Output()))
                engine.tokenizer = SimpleNamespace(eos_token_id=9)
                inputs = {"input_ids": SimpleNamespace(shape=(1, 4))}
                request = {"model_id": "qwen-base", "messages": [], "max_tokens": 30,
                           "timeout": 10., "generation_profile": profile}
                with patch.dict(sys.modules, native_doubles), patch("scripts.local_chat_worker.bounded_inputs", return_value=(inputs, 0)):
                    metadata = engine.generate(request, threading.Event(), lambda _: None)
                options = engine.model.generate.call_args.kwargs
                self.assertEqual(options["do_sample"], profile == "conversation")
                if profile == "conversation":
                    for key, value in {"temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0.}.items():
                        self.assertEqual(options[key], value)
                else:
                    self.assertNotIn("temperature", options)
                    self.assertNotIn("top_p", options)
                self.assertEqual(metadata["generation_profile"], profile)
                self.assertEqual(metadata["sampling"]["do_sample"], profile == "conversation")
                self.assertEqual(metadata["finish_reason"], "stop")

    def test_worker_rejects_unknown_profile_before_loading_a_model(self):
        engine = QwenChatEngine()
        engine.load = Mock()
        with self.assertRaises(ValueError):
            engine.generate({"generation_profile": "other"}, threading.Event(), lambda _: None)
        engine.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
