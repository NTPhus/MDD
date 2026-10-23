"""Offline measurement-contract tests; no model, network or voice API."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from scripts.evaluate_vi_voice_pipeline import (EXPECTED_MODELS, RecordingClient, aggregate,
    frozen_config, load_manifest, record_case, verify_job_models)
from scripts.benchmark_demo_latency import run_once


class ViVoicePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_parent = Path(__file__).resolve().parents[1] / "data/generated/test-temp"
        cls.temp_parent.mkdir(parents=True, exist_ok=True)

    def case(self):
        return {"id": "TLV-1", "corpus": "L2-ARCTIC", "speaker_id": "TLV", "audio": "clip.wav",
            "audio_sha256": hashlib.sha256(b"audio").hexdigest(), "reference_text": "It was difficult to get there.",
            "reference_kind": "published_reading_prompt", "source_url": "https://example.org/corpus", "split_notes": "held-out"}

    def test_manifest_checks_all_audio_before_inference(self):
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as folder:
            root = Path(folder)
            (root / "clip.wav").write_bytes(b"audio")
            path = root / "manifest.json"
            path.write_text(json.dumps({"cases": [self.case()]}))
            manifest, sha = load_manifest(path, root)
            self.assertEqual(manifest["cases"][0]["speaker_id"], "TLV")
            self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())
            (root / "clip.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_manifest(path, root)

    def test_manifest_rejects_duplicate_ids_and_escaping_paths(self):
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as folder:
            root = Path(folder)
            (root / "clip.wav").write_bytes(b"audio")
            path = root / "manifest.json"
            path.write_text(json.dumps({"cases": [self.case(), self.case()]}))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(path, root)
            case = self.case(); case["audio"] = "../outside.wav"
            path.write_text(json.dumps({"cases": [case]}))
            with self.assertRaisesRegex(ValueError, "workspace"):
                load_manifest(path, root)

    def test_config_requires_current_models_and_readiness(self):
        config = {**EXPECTED_MODELS, "speech_ready": True, "device": "auto"}
        self.assertEqual(frozen_config(config)["asr_model_id"], "wav2vec2-vi-en-v1")
        config["grammar_model_id"] = "qwen-base"
        with self.assertRaisesRegex(ValueError, "grammar_model_id"):
            frozen_config(config)

    def test_micro_metrics_and_manual_semantic_review(self):
        first = record_case(self.case(), {"status": "ready", "text": "Where was it?",
            "speech": {"transcript": "It was difficult to get there", "normalization": {"text": "It was difficult to get there."}}})
        case = self.case(); case["id"] = "second"; case["reference_text"] = "Hello"
        second = record_case(case, {"status": "ready", "text": "Where was it?", "speech": {"transcript": "world"}})
        failed = deepcopy(first); failed.update(id="failed", status="error")
        metrics = aggregate([first, second, failed])
        self.assertEqual(metrics["transcript"]["reference_words"], 7)
        self.assertEqual(metrics["transcript"]["word_errors"], 1)
        self.assertAlmostEqual(metrics["transcript"]["wer"], 1 / 7)
        self.assertEqual(metrics["transcript"]["exact_matches"], 1)
        self.assertEqual(metrics["literal_repeated_reply_count"], 1)
        self.assertEqual(first["human_review"]["status"], "pending")
        self.assertNotIn("semantic_accuracy", metrics)

    def test_job_proves_models_instead_of_trusting_config_only(self):
        job = {"speech": {"asr_model": "wav2vec2-vi-en-v1", "rewrite": {"model_id": "qwen-grammar"}},
               "metadata": {"reply_model_id": "qwen-base"}}
        verify_job_models(job)
        job["speech"]["asr_model"] = "old-asr"
        with self.assertRaisesRegex(ValueError, "frozen pipeline"):
            verify_job_models(job)

    def test_actual_worker_runtime_key_is_preserved(self):
        runtime = {"device": "cuda", "dtype": "bfloat16"}
        row = record_case(self.case(), {"status": "ready", "metadata": {"qwen_runtime": runtime}})
        self.assertEqual(row["qwen_runtime"], runtime)

    def test_interpretation_is_recorded_separately_and_scored_as_conversation(self):
        case = self.case()
        raw, recovered = 'It was divicote to get there', 'It was difficult to get there.'
        recovery = {'status': 'accepted', 'original': raw, 'text': recovered,
                    'requires_confirmation': True, 'reference_confirmed': False}
        row = record_case(case, {'status': 'ready', 'speech': {
            'transcript': raw, 'conversation_text': recovered, 'interpretation': recovery,
            'conversation_source': 'asr_interpreted', 'reference_source': 'asr'}})
        self.assertEqual(row['transcript'], raw)
        self.assertEqual(row['interpretation'], recovery)
        metrics = aggregate([row])
        self.assertGreater(metrics['transcript']['wer'], metrics['conversation_text']['wer'])
        self.assertEqual(metrics['interpretation_statuses'], {'accepted': 1})
        self.assertEqual(metrics['inferred_interpretations_requiring_confirmation'], 1)
        self.assertEqual(metrics['conversation_wording_changed'], 1)
        self.assertNotIn('semantic_accuracy', metrics)

    def test_actual_submission_has_no_reference_or_history(self):
        calls = []
        class Client:
            def request(self, route, payload=None):
                calls.append((route, payload))
                return {"id": "a" * 32, "status": "ready", "speech": {"reply_ready": True},
                        "voice": {"status": "not_requested"}, "text": "Hello"}
        run_once(RecordingClient(Client()), "encoded", 1, .15, 20)
        payload = calls[0][1]
        self.assertEqual(payload["history"], [])
        self.assertNotIn("reference_text", payload)
        self.assertNotIn("dialogue_text", payload)
        self.assertEqual([route for route, _ in calls], ["/api/chat/speech"])

    def test_only_chat_busy_is_retried_without_cancelling_user_work(self):
        calls, delays = [], []
        class Client:
            def request(self, route, payload=None):
                calls.append(route)
                if len(calls) == 1:
                    raise urllib.error.HTTPError("local", 409, "busy", {}, io.BytesIO(b'{"error":"chat_busy"}'))
                return {"id": "b" * 32, "status": "ready"}
        client = RecordingClient(Client(), clock=lambda: 1., sleep=delays.append)
        self.assertEqual(client.request("/api/chat/speech", {})["status"], "ready")
        self.assertEqual(delays, [1.])
        self.assertEqual(calls, ["/api/chat/speech", "/api/chat/speech"])


if __name__ == "__main__":
    unittest.main()
