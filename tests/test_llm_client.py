"""LLM gateway tests with fake HTTP responses; no API credits used."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.llm_client import Completion, LLMClient, LLMError, ROOT, load_env
from scripts.validate_contracts import read_json, validate_pair


class LLMTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"LLM_API_KEY": "unit-test-key"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.client = LLMClient()
        self.assessment = read_json(ROOT / "contracts/examples/assessment.json")
        self.feedback = read_json(ROOT / "contracts/examples/feedback.json")

    def completion(self, feedback=None):
        return Completion(
            content=json.dumps(self.feedback if feedback is None else feedback),
            requested_model="claude-sonnet-5", response_model="gateway-reported-model",
            request_id="example-request", usage={"total_tokens": 100},
            finish_reason="stop", latency_sec=0.1,
        )

    def test_chat_sends_configured_gateway_model_and_bearer_header(self):
        response = {
            "id": "example-id", "model": "gateway-reported-model",
            "choices": [{"message": {"role": "assistant", "content": "Xin chào!"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }
        with patch.object(self.client.api, "open", return_value=io.BytesIO(json.dumps(response).encode())) as api:
            result = self.client.complete([{"role": "user", "content": "Hello"}])
        request = api.call_args.args[0]
        self.assertEqual(request.full_url, "https://llm.mrdnd.dev/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")
        self.assertEqual(json.loads(request.data)["model"], "claude-sonnet-5")
        self.assertEqual(result.content, "Xin chào!")
        self.assertEqual(result.requested_model, "claude-sonnet-5")
        self.assertEqual(result.response_model, "gateway-reported-model")

    def test_invalid_http_response_is_safe_and_not_retried(self):
        error = HTTPError("https://llm.mrdnd.dev", 401, "unit-test-key", {}, io.BytesIO(b"unit-test-key"))
        with patch.object(self.client.api, "open", side_effect=error) as api:
            with self.assertRaises(LLMError) as caught:
                self.client.complete([{"role": "user", "content": "Hello"}])
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("unit-test-key", str(caught.exception))
        self.assertEqual(api.call_count, 1)

    def test_bad_gateway_envelopes_and_truncated_output_are_rejected(self):
        values = [
            b"not-json", b"[]", b'{"choices": []}',
            b'{"choices": [{"message": {"content": null}}]}',
            b'{"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}',
        ]
        for value in values:
            with self.subTest(value=value), patch.object(self.client.api, "open", return_value=io.BytesIO(value)):
                with self.assertRaises(LLMError):
                    self.client.complete([{"role": "user", "content": "Hello"}])

    def test_feedback_validated_and_acoustic_assessment_preserved(self):
        original = deepcopy(self.assessment)
        with patch.object(self.client, "complete", return_value=self.completion()) as complete:
            result = self.client.generate_feedback(self.assessment)
        self.assertEqual(result.source, "llm")
        validate_pair(self.assessment, result.feedback)
        self.assertEqual(self.assessment, original)
        messages = complete.call_args.args[0]
        self.assertEqual(json.loads(messages[1]["content"]), self.assessment)
        self.assertIn('"required"', messages[0]["content"])
        metadata = result.metadata(self.assessment)
        self.assertEqual(metadata["data_kind"], "synthetic_example")
        self.assertNotIn("content", metadata["completions"][0])

    def test_invented_error_is_repaired_once(self):
        invalid = deepcopy(self.feedback)
        invalid["tips"][0]["error_id"] = "invented-error"
        with patch.object(self.client, "complete", side_effect=[self.completion(invalid), self.completion()]) as complete:
            result = self.client.generate_feedback(self.assessment)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(result.source, "llm")
        self.assertEqual(len(result.completions), 2)
        self.assertEqual(result.validation_errors, ["LLM invented an error ID"])
        self.assertIn("LLM invented an error ID", complete.call_args.args[0][-1]["content"])
        validate_pair(self.assessment, result.feedback)

    def test_changed_reference_after_two_attempts_uses_template(self):
        invalid = deepcopy(self.feedback)
        for segment in invalid["tts_segments"]:
            if segment["purpose"] == "reference":
                segment["text"] = "A changed reference."
        with patch.object(self.client, "complete", return_value=self.completion(invalid)) as complete:
            result = self.client.generate_feedback(self.assessment)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(result.source, "template")
        self.assertEqual(result.fallback_reason, "invalid_feedback_after_repair")
        self.assertEqual(result.feedback["tips"], [])
        self.assertEqual(result.feedback["tts_segments"], [])
        validate_pair(self.assessment, result.feedback)

    def test_network_error_returns_template_without_retry(self):
        with patch.object(self.client, "complete", side_effect=LLMError("LLM HTTP 401")) as complete:
            result = self.client.generate_feedback(self.assessment)
        self.assertEqual(complete.call_count, 1)
        self.assertEqual(result.source, "template")
        self.assertIn("401", result.fallback_reason)
        validate_pair(self.assessment, result.feedback)

    def test_uncertain_audio_does_not_call_llm(self):
        self.assessment.update(status="uncertain", errors=[])
        with patch.object(self.client, "complete") as complete:
            result = self.client.generate_feedback(self.assessment)
        complete.assert_not_called()
        self.assertEqual(result.fallback_reason, "assessment_not_ok")
        validate_pair(self.assessment, result.feedback)

    def test_invalid_acoustic_output_rejected_before_network(self):
        self.assessment["errors"][0]["confidence"] = 0.99
        with patch.object(self.client, "complete") as complete:
            with self.assertRaises(ValueError):
                self.client.generate_feedback(self.assessment)
        complete.assert_not_called()

    def test_config_overrides_and_invalid_urls(self):
        with patch.dict(os.environ, {"LLM_MODEL": "chosen-model", "LLM_BASE_URL": "https://example.test/v1/"}):
            client = LLMClient()
            self.assertEqual(client.model, "chosen-model")
            self.assertEqual(client.base_url, "https://example.test/v1")
        for url in ("http://example.test/v1", "https://key@example.test/v1", "https://example.test/v1?key=secret"):
            with self.subTest(url=url), patch.dict(os.environ, {"LLM_BASE_URL": url}):
                with self.assertRaises(LLMError):
                    LLMClient()

    def test_env_loader_preserves_existing_key_and_does_not_execute_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("LLM_API_KEY='file-key'\nLLM_MODEL='$(do-not-execute)'\nAI33_API_KEY=unrelated\n")
            load_env(path)
        self.assertEqual(os.environ["LLM_API_KEY"], "unit-test-key")
        self.assertEqual(os.environ["LLM_MODEL"], "$(do-not-execute)")
        self.assertNotIn("AI33_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
