"""Feedback grounding, request idempotency and voice allowlisting; no live APIs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
from http.client import HTTPConnection
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mdd.demo import DemoApplication, DemoError, DemoServer, MAX_FEEDBACK_BODY_BYTES
from mdd.demo_feedback import FeedbackService, FeedbackNotFound, VoiceNotFound, build_assessment
from scripts.llm_client import Completion, FeedbackResult, fallback_feedback
from scripts.validate_contracts import validate_pair


def inference():
    return {"model_id": "run:B:best", "reference_text": "I think this useful.",
            "model": {"id": "run:B:best", "kind": "phoneme", "experimental": False},
            "phonemes": [{"phone": "T"}, {"phone": "IH"}],
            "comparison": {
                "words": [{"text": word, "status": "difference" if index else "match"}
                          for index, word in enumerate("I think this useful.".split())],
                "events": [
                    {"kind": "substitution", "expected_phone": "TH", "observed_phone": "T", "word_index": 1},
                    {"kind": "deletion", "expected_phone": "AH", "observed_phone": None, "word_index": 2},
                    {"kind": "insertion", "expected_phone": None, "observed_phone": "R*", "word_index": 3}]}}


def grounded_feedback(assessment):
    return {"schema_version": "1.0", "utterance_id": assessment["utterance_id"], "language": "vi",
            "summary": "Model gợi ý luyện lại các từ được đánh dấu.",
            "tips": [{"error_id": error["id"], "explanation": "Đây là sai khác âm vị do model ước lượng.",
                      "instruction": "Đọc chậm từ được đánh dấu, rồi thử lại cả câu.",
                      "practice_words": []} for error in assessment["errors"][:2]],
            "tts_segments": [{"language": "en-US", "purpose": "reference", "text": assessment["reference_text"]},
                             {"language": "vi-VN", "purpose": "feedback", "text": "Hãy đọc chậm từ được đánh dấu rồi thử lại cả câu."}]}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_feedback(self, assessment):
        self.calls.append(deepcopy(assessment))
        time.sleep(0.005)
        return SimpleNamespace(feedback=grounded_feedback(assessment), source="llm")


class FakeTTS:
    def __init__(self, cache):
        self.cache = cache
        self.calls = []

    def synthesize(self, text, locale="en-US"):
        self.calls.append((text, locale))
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / (hashlib.sha256((locale + text).encode()).hexdigest() + ".wav")
        path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio")
        return path


class DemoFeedbackTests(unittest.TestCase):
    def setUp(self):
        # Default-mode Windows tempfile permissions can be restricted by a
        # sandbox. Normal inherited ACLs keep these isolated fixtures portable.
        self.fixture_parent = Path(tempfile.gettempdir()).resolve()
        self.root = self.fixture_parent / ("mdd-feedback-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.clean_fixture)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.llm = FakeLLM()
        self.tts = FakeTTS(self.root / "data/generated/tts/ai33")
        self.service = FeedbackService(self.root, llm_factory=lambda: self.llm, tts_factory=lambda: self.tts)

    def clean_fixture(self):
        resolved = self.root.resolve()
        if resolved.parent != self.fixture_parent or not resolved.name.startswith("mdd-feedback-tests-"):
            raise RuntimeError("Refusing cleanup outside the feedback fixture")
        shutil.rmtree(resolved)

    def test_assessment_maps_all_events_without_calibration_or_invented_ipa(self):
        result = inference()
        result["perceived_phones"] = ["NEVER_SEND_GOLD"]
        result["gold_error_events"] = [{"kind": "DO_NOT_SEND"}]
        assessment, _ = build_assessment(result, "a" * 32)
        self.assertEqual(assessment["status"], "ok")
        self.assertEqual([error["type"] for error in assessment["errors"]], ["substitution", "deletion", "addition"])
        self.assertEqual(assessment["errors"][0]["expected_ipa"], "θ")
        self.assertEqual(assessment["errors"][0]["observed_ipa"], "t")
        self.assertIsNone(assessment["errors"][1]["expected_ipa"])
        self.assertIsNone(assessment["errors"][2]["observed_ipa"])
        for error in assessment["errors"]:
            self.assertIsNone(error["confidence"])
            self.assertIsNone(error["start_sec"])
            self.assertIsNone(error["end_sec"])
        self.assertNotIn("NEVER_SEND_GOLD", json.dumps(assessment))
        validate_pair(assessment, fallback_feedback(assessment))

    def test_unknown_word_mapping_never_assigns_an_error_to_another_word(self):
        for index in (None, -1, 4, True, "1"):
            with self.subTest(index=index):
                result = inference()
                result["comparison"]["events"][1]["word_index"] = index
                assessment, notices = build_assessment(result, "a" * 32)
                self.assertEqual(assessment["status"], "uncertain")
                self.assertEqual(assessment["errors"], [])
                self.assertTrue(notices)

    def test_base_smoke_blank_and_missing_reference_never_call_providers(self):
        variants = []
        result = inference(); result["model"]["kind"] = "backbone"; variants.append(result)
        result = inference(); result["model"]["experimental"] = True; variants.append(result)
        result = inference(); result["phonemes"] = []; variants.append(result)
        result = inference(); result["comparison"] = None; variants.append(result)
        result = inference(); result["comparison"]["words"][1]["status"] = "unavailable"; variants.append(result)
        for result in variants:
            with self.subTest(result=result):
                registered = self.service.register(result)
                self.assertFalse(registered["feedback_available"])
                response = self.service.feedback(registered["analysis_id"])
                self.assertEqual(response["status"], "unavailable")
                self.assertEqual(response["assessment"]["errors"], [])
                self.assertEqual(response["feedback"]["tips"], [])
                self.assertEqual(response["audio"], [])
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.tts.calls, [])

    def test_unconfigured_services_preserve_assessment_without_provider_calls(self):
        service = FeedbackService(self.root)
        registered = service.register(inference())
        with patch("scripts.llm_client.LLMClient") as llm, patch("scripts.ai33_tts.AI33TTS") as tts:
            response = service.feedback(registered["analysis_id"])
        llm.assert_not_called(); tts.assert_not_called()
        self.assertTrue(registered["feedback_available"])
        self.assertEqual(response["status"], "unavailable")
        self.assertEqual(len(response["assessment"]["errors"]), 3)
        self.assertEqual(response["feedback_source"], "template")
        self.assertTrue(all(isinstance(value, bool) for value in service.capabilities().values()))

    def test_settings_can_be_filled_after_an_unconfigured_request(self):
        service = FeedbackService(self.root)
        registered = service.register(inference())
        (self.root / ".env").write_text("LLM_API_KEY=\nAI33_API_KEY=\n", encoding="utf-8")
        self.assertEqual(service.feedback(registered["analysis_id"], False)["status"], "unavailable")
        (self.root / ".env").write_text("LLM_API_KEY=FAKE_TEST_KEY\nAI33_API_KEY=\n", encoding="utf-8")
        with patch("scripts.llm_client.LLMClient", return_value=self.llm):
            response = service.feedback(registered["analysis_id"], False)
        self.assertEqual(response["status"], "ready")
        self.assertEqual(len(self.llm.calls), 1)

    def test_repeated_requests_reuse_one_llm_and_three_voice_calls(self):
        result = inference()
        registered = self.service.register(result)
        result["comparison"]["events"][0]["observed_phone"] = "MUTATED"
        registered["assessment"]["errors"].clear()
        first = self.service.feedback(registered["analysis_id"])
        second = self.service.feedback(registered["analysis_id"])
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "ready")
        self.assertEqual(first["voice_status"], "ready")
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(len(self.tts.calls), 3)
        self.assertEqual([item["purpose"] for item in first["audio"]], ["feedback", "reference", "practice"])
        self.assertEqual(first["audio"][1]["text"], "I think this useful.")
        self.assertEqual(first["audio"][2]["text"], "think. this. useful.")
        self.assertEqual(self.llm.calls[0]["errors"][0]["observed_phone"], "T")
        validate_pair(first["assessment"], first["feedback"])

    def test_text_only_request_can_upgrade_to_voice_without_repeating_llm(self):
        identifier = self.service.register(inference())["analysis_id"]
        response = self.service.feedback(identifier, False)
        self.assertEqual(response["voice_status"], "not_requested")
        self.assertEqual(self.tts.calls, [])
        self.service.feedback(identifier, True)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(len(self.tts.calls), 3)

    def test_no_detected_error_produces_only_instruction_and_reference_voice(self):
        result = inference()
        result["comparison"]["events"] = []
        for word in result["comparison"]["words"]:
            word["status"] = "match"
        response = self.service.feedback(self.service.register(result)["analysis_id"])
        self.assertEqual(response["assessment"]["errors"], [])
        self.assertEqual([item["purpose"] for item in response["audio"]], ["feedback", "reference"])
        self.assertEqual(len(self.tts.calls), 2)

    def test_concurrent_requests_are_idempotent(self):
        identifier = self.service.register(inference())["analysis_id"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: self.service.feedback(identifier), range(3)))
        self.assertEqual(responses[0], responses[1])
        self.assertEqual(responses[1], responses[2])
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(len(self.tts.calls), 3)

    def test_invalid_llm_output_falls_back_without_exposing_untrusted_text(self):
        bad = Mock()
        bad.generate_feedback.side_effect = RuntimeError("SECRET https://cdn.invalid/voice?signature=PRIVATE C:\\hidden\\key")
        service = FeedbackService(self.root, llm_factory=lambda: bad)
        identifier = service.register(inference())["analysis_id"]
        response = service.feedback(identifier, False)
        self.assertEqual(response["feedback_source"], "template")
        self.assertEqual(response["status"], "partial")
        self.assertNotIn("SECRET", json.dumps(response))
        self.assertNotIn("signature=", json.dumps(response))
        self.assertNotIn("hidden", json.dumps(response))
        service.feedback(identifier, False)
        self.assertEqual(bad.generate_feedback.call_count, 1)

    def test_private_diagnostics_store_metadata_without_completion_content_or_keys(self):
        identifier = self.service.register(inference())["analysis_id"]
        assessment, _ = build_assessment(inference(), identifier)
        os.environ["LLM_API_KEY"] = "PRIVATE_LLM_KEY"
        os.environ["AI33_API_KEY"] = "PRIVATE_TTS_KEY"
        generated = FeedbackResult(
            feedback=grounded_feedback(assessment), source="llm", fallback_reason=None,
            completions=[Completion(content="RAW_PRIVATE_COMPLETION", requested_model="test-model",
                                    response_model="gateway-model", request_id="test-request",
                                    usage={"total_tokens": 100, "content": "RAW_PRIVATE_COMPLETION"},
                                    finish_reason="stop", latency_sec=0.1)],
            validation_errors=["Gateway echoed PRIVATE_LLM_KEY and PRIVATE_TTS_KEY"])
        self.llm.generate_feedback = Mock(return_value=generated)
        response = self.service.feedback(identifier, False)
        path = self.root / "data/generated/demo/feedback" / f"{identifier}.meta.json"
        text = path.read_text(encoding="utf-8")
        metadata = json.loads(text)
        self.assertEqual(metadata["feedback_source"], "llm")
        self.assertEqual(metadata["completions"][0]["usage"], {"total_tokens": 100})
        self.assertEqual(metadata["completions"][0]["finish_reason"], "stop")
        self.assertNotIn("RAW_PRIVATE_COMPLETION", text)
        self.assertNotIn("PRIVATE_LLM_KEY", text)
        self.assertNotIn("PRIVATE_TTS_KEY", text)
        self.assertNotIn("completions", response)
        self.assertNotIn("validation_errors", response)

    def test_diagnostic_write_failure_does_not_break_valid_feedback_or_voice(self):
        identifier = self.service.register(inference())["analysis_id"]
        assessment, _ = build_assessment(inference(), identifier)
        generated = FeedbackResult(grounded_feedback(assessment), "llm", None, [])
        self.llm.generate_feedback = Mock(return_value=generated)
        with patch.object(Path, "write_text", side_effect=OSError("private storage error")):
            response = self.service.feedback(identifier)
        self.assertEqual(response["feedback_source"], "llm")
        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["voice_status"], "ready")
        self.assertEqual(len(self.tts.calls), 3)

    def test_raised_generation_diagnostics_contain_only_error_category(self):
        identifier = self.service.register(inference())["analysis_id"]
        self.llm.generate_feedback = Mock(side_effect=RuntimeError("PRIVATE_RAW_ERROR_TEXT"))
        self.service.feedback(identifier, False)
        path = self.root / "data/generated/demo/feedback" / f"{identifier}.meta.json"
        text = path.read_text(encoding="utf-8")
        metadata = json.loads(text)
        self.assertEqual(metadata["error_type"], "RuntimeError")
        self.assertEqual(metadata["fallback_reason"], "feedback_generation_failed")
        self.assertNotIn("PRIVATE_RAW_ERROR_TEXT", text)

    def test_llm_cannot_invent_error_ids_or_change_reference_tts(self):
        for modification in ("error", "reference"):
            with self.subTest(modification=modification):
                client = Mock()
                def bad_feedback(assessment):
                    feedback = grounded_feedback(assessment)
                    if modification == "error": feedback["tips"][0]["error_id"] = "INVENTED"
                    else: feedback["tts_segments"][0]["text"] = "INVENTED"
                    return SimpleNamespace(feedback=feedback, source="llm")
                client.generate_feedback.side_effect = bad_feedback
                service = FeedbackService(self.root, llm_factory=lambda: client)
                response = service.feedback(service.register(inference())["analysis_id"], False)
                self.assertEqual(response["feedback_source"], "template")
                self.assertNotIn("INVENTED", json.dumps(response))

    def test_partial_tts_failure_preserves_valid_feedback_and_never_resubmits(self):
        synthesize = self.tts.synthesize
        def fail_vietnamese(text, locale="en-US"):
            if locale == "vi-VN":
                self.tts.calls.append((text, locale))
                raise RuntimeError("SECRET signed URL https://cdn.invalid/?signature=PRIVATE")
            return synthesize(text, locale)
        self.tts.synthesize = fail_vietnamese
        identifier = self.service.register(inference())["analysis_id"]
        response = self.service.feedback(identifier)
        self.assertEqual(response["feedback_source"], "llm")
        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["voice_status"], "partial")
        self.assertEqual([item["status"] for item in response["audio"]], ["error", "ready", "ready"])
        self.assertNotIn("SECRET", json.dumps(response))
        self.assertEqual(self.service.feedback(identifier), response)
        self.assertEqual(len(self.tts.calls), 3)

    def test_voice_registry_rejects_paths_outside_cache_and_unknown_ids(self):
        secret = self.root / ".env"
        secret.write_bytes(b"RIFF\x24\x00\x00\x00WAVESECRET")
        self.tts.synthesize = Mock(return_value=secret)
        response = self.service.feedback(self.service.register(inference())["analysis_id"])
        self.assertEqual(response["voice_status"], "unavailable")
        self.assertTrue(all(item["url"] is None for item in response["audio"]))
        self.assertNotIn("SECRET", json.dumps(response))
        with self.assertRaises(VoiceNotFound): self.service.voice("a" * 32)

    def test_voice_mime_and_size_are_checked_on_registered_assets(self):
        response = self.service.feedback(self.service.register(inference())["analysis_id"])
        item = response["audio"][0]
        audio, mime = self.service.voice(item["id"])
        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertEqual(mime, "audio/wav")
        with patch("mdd.demo_feedback.MAX_VOICE_BYTES", 10):
            with self.assertRaises(VoiceNotFound): self.service.voice(item["id"])

    def test_cache_eviction_removes_analysis_and_voice_registration(self):
        service = FeedbackService(self.root, llm_factory=lambda: self.llm, tts_factory=lambda: self.tts, max_analyses=1)
        first = service.register(inference())["analysis_id"]
        voice = service.feedback(first)["audio"][0]["id"]
        second = service.register(inference())["analysis_id"]
        self.assertNotEqual(first, second)
        with self.assertRaises(FeedbackNotFound): service.feedback(first)
        with self.assertRaises(VoiceNotFound): service.voice(voice)

    def test_application_rejects_client_assessment_and_nonboolean_options(self):
        app = DemoApplication(self.root, feedback_service=self.service)
        identifier = self.service.register(inference())["analysis_id"]
        for body in ({"analysis_id": identifier, "assessment": {}},
                     {"analysis_id": identifier, "errors": []},
                     {"analysis_id": identifier, "include_voice": "true"},
                     {"analysis_id": "../../.env"}, []):
            with self.subTest(body=body), self.assertRaises(DemoError): app.feedback(body)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.tts.calls, [])

    def test_analyze_returns_valid_server_owned_assessment_before_feedback(self):
        checkpoint = self.root / "runs/B/best_checkpoint"
        checkpoint.mkdir(parents=True)
        for path, payload in ((checkpoint / "metadata.json", {"stage": "B", "purpose": "scientific_training", "step": 300}),
                              (checkpoint / "config.json", {}),
                              (self.root / "runs/B/config.json", {"stage": "B", "purpose": "scientific_training"})):
            path.write_text(json.dumps(payload), encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"synthetic model")
        engine = Mock()
        engine.analyze.return_value = inference()
        app = DemoApplication(self.root, engine=engine, feedback_service=self.service)
        response = app.analyze({"model_id": "run:B:best", "reference_text": "I think this useful.",
                                "audio_base64": base64.b64encode(b"synthetic audio").decode(),
                                "assessment": {"errors": [{"id": "FORGED"}]}})
        self.assertTrue(response["feedback_available"])
        self.assertEqual(response["assessment"]["utterance_id"], response["analysis_id"])
        self.assertEqual(len(response["assessment"]["errors"]), 3)
        self.assertNotIn("FORGED", json.dumps(response["assessment"]))
        validate_pair(response["assessment"], fallback_feedback(response["assessment"]))
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.tts.calls, [])

    def test_http_feedback_voice_and_origin_guards(self):
        app = DemoApplication(self.root, feedback_service=self.service)
        identifier = self.service.register(inference())["analysis_id"]
        server = DemoServer(("127.0.0.1", 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(method, path, body=None, headers=None):
            connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                connection.request(method, path, body, headers or {})
                response = connection.getresponse()
                return response.status, response.read(), dict(response.getheaders())
            finally: connection.close()
        try:
            body = json.dumps({"analysis_id": identifier, "include_voice": True})
            headers = {"Content-Type": "application/json"}
            status, raw, _ = request("POST", "/api/feedback", body, headers)
            self.assertEqual(status, 200)
            response = json.loads(raw)
            voice = response["audio"][0]
            status, audio, audio_headers = request("GET", voice["url"])
            self.assertEqual(status, 200)
            self.assertTrue(audio.startswith(b"RIFF"))
            self.assertEqual(audio_headers["Content-Type"], "audio/wav")
            self.assertEqual(request("GET", "/api/voice/../../.env")[0], 404)
            self.assertEqual(request("GET", "/api/voice/" + "f" * 32)[0], 404)
            self.assertEqual(request("POST", "/api/feedback", body,
                                     {**headers, "Origin": "https://evil.invalid"})[0], 403)
            self.assertEqual(request("POST", "/api/feedback", "{}",
                                     {**headers, "Content-Length": str(MAX_FEEDBACK_BODY_BYTES + 1)})[0], 413)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
