"""Offline integration tests: no account, network traffic or credits required."""

from email.message import Message
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from scripts.ai33_tts import AI33Error, AI33TTS, load_env


class Response(io.BytesIO):
    def __init__(self, data, content_type="application/json"):
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


class AI33Tests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "AI33_API_KEY": "unit-test-secret",
            "AI33_VOICE_EN": "example-english",
            "AI33_VOICE_VI": "example-vietnamese",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.client = AI33TTS(self.cache)

    def audio_response(self):
        return Response(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"test-audio", "audio/wav")

    def cached_receipt(self, text, speed, *, audio=True):
        """Build a receipt with the original unnormalized JSON cache key."""
        payload = {"text": text, "voice_id": "example-english", "speed": speed}
        cache_input = {"base_url": self.client.base_url, "locale": "en-US", "payload": payload}
        digest = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
        receipt = {"task_id": "existing-task", "voice_id": "example-english", "speed": speed}
        path = self.cache / (digest + ".json")
        if audio:
            data = b"ID3cached-audio"
            file = self.cache / (digest + ".mp3")
            file.write_bytes(data)
            receipt.update(audio_file=file.name, audio_sha256=hashlib.sha256(data).hexdigest())
        self.client.save_receipt(path, receipt)
        return path, self.cache / (digest + ".mp3")

    def test_equivalent_integer_and_float_speeds_share_one_submission_and_cache(self):
        responses = [{"task_id": "one-task"}, {"status": "done", "audio_url": "https://cdn.example/a.wav"}]
        with patch.object(self.client.api, "open", side_effect=[Response(json.dumps(value).encode()) for value in responses]) as api, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()) as download:
            first = self.client.synthesize("Hello.", speed=1)
            self.assertEqual(self.client.synthesize("Hello.", speed=1.0), first)
            self.assertEqual(self.client.synthesize("Hello."), first)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(download.call_count, 1)
        self.assertIs(type(json.loads(api.call_args_list[0].args[0].data)['speed']), float)
        self.assertTrue(self.client.last_timing['cache_hit'])
        self.assertEqual(self.client.last_timing['new_tasks_submitted'], 0)

    def test_legacy_integer_cache_is_reused_without_network_or_receipt_migration(self):
        legacy, audio = self.cached_receipt("Hello.", 1)
        original = legacy.read_bytes()
        with patch.object(self.client, "request") as request, patch.object(self.client.downloads, "open") as download:
            for speed in (1, 1.0, None):
                self.assertEqual(self.client.synthesize("Hello.", speed=speed), audio)
        request.assert_not_called()
        download.assert_not_called()
        self.assertEqual(legacy.read_bytes(), original)
        self.assertEqual(list(self.cache.glob('*.json')), [legacy])
        self.assertTrue(self.client.last_timing['cache_hit'])
        _, different, _, _ = self.client._cache_request("Hello.", "en-US", speed=.85)
        self.assertNotEqual(different, legacy)

    def test_canonical_float_receipt_is_preferred_when_both_legacy_receipts_exist(self):
        legacy, _ = self.cached_receipt("Hello.", 1)
        canonical, audio = self.cached_receipt("Hello.", 1.0)
        originals = {path: path.read_bytes() for path in (legacy, canonical)}
        with patch.object(self.client, "request") as request:
            self.assertEqual(self.client.synthesize("Hello.", speed=1), audio)
            self.assertEqual(self.client.synthesize("Hello.", speed=1.0), audio)
        request.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in (legacy, canonical)}, originals)

    def test_legacy_integer_pending_task_resumes_with_float_speed_using_get_only(self):
        _, canonical, _, _ = self.client._cache_request("Hello.", "en-US", speed=1.0)
        legacy, _ = self.cached_receipt("Hello.", 1, audio=False)
        self.assertFalse(canonical.exists())
        self.assertTrue(self.client.can_resume("Hello.", speed=1.0))
        with patch.object(self.client, "request", return_value={"status": "done", "audio_url": "https://cdn.example/a.wav"}) as request, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()):
            self.assertTrue(self.client.resume("Hello.", speed=1.0).exists())
        self.assertEqual([call.args[0] for call in request.call_args_list], ['GET'])
        self.assertTrue(legacy.exists())
        self.assertFalse(canonical.exists())
        self.assertEqual(self.client.last_timing['new_tasks_submitted'], 0)
        self.assertTrue(self.client.last_timing['resumed_task'])

    def test_submit_poll_download_cache_and_key_isolation(self):
        responses = [
            {"data": {"task_id": "example-task"}},
            {"status": "processing"},
            {"status": "done", "metadata": {"audio_url": "https://cdn.example/audio.wav?signature=private"}},
        ]
        with patch.object(self.client.api, "open") as api, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()) as download, \
                patch("scripts.ai33_tts.time.sleep"):
            api.side_effect = [Response(json.dumps(value).encode()) for value in responses]
            audio = self.client.synthesize("I think this is useful.")
            self.assertTrue(audio.is_file())
            self.assertEqual(audio.suffix, ".wav")
            self.assertEqual(self.client.synthesize("I think this is useful."), audio)
            self.assertEqual(api.call_count, 3)
            self.assertEqual(download.call_count, 1)
            submitted = api.call_args_list[0].args[0]
            self.assertEqual(submitted.get_method(), "POST")
            self.assertEqual(submitted.get_header("Xi-api-key"), "unit-test-secret")
            self.assertEqual(json.loads(submitted.data), {
                "text": "I think this is useful.", "voice_id": "example-english", "speed": 1.0,
            })
            downloaded = download.call_args.args[0]
            self.assertEqual(downloaded.get_header("User-agent"), "Mozilla/5.0")
            self.assertIsNone(downloaded.get_header("Xi-api-key"))
            self.assertIsNone(downloaded.get_header("Authorization"))
        receipt_text = next(self.cache.glob("*.json")).read_text()
        self.assertNotIn("unit-test-secret", receipt_text)
        self.assertNotIn("signature=private", receipt_text)
        self.assertNotIn("I think this is useful.", receipt_text)

    def test_stage_timings_measure_submit_poll_and_download_then_cache_without_network(self):
        responses = [
            {"task_id": "private-task"}, {"status": "processing"},
            {"status": "done", "audio_url": "https://cdn.example/a.wav?signature=private"},
        ]
        with patch.object(self.client.api, "open", side_effect=[Response(json.dumps(value).encode()) for value in responses]) as api, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()) as download, \
                patch("scripts.ai33_tts.time.sleep"), \
                patch("scripts.ai33_tts.time.perf_counter", side_effect=[0, 1, 3, 4, 9, 10, 25, 26, 29, 30]):
            audio = self.client.synthesize("Do not put learner text in timings.")
        timing = self.client.last_timing.copy()
        self.assertEqual({key: timing[key] for key in ("total_sec", "cache_lookup_sec", "submit_sec", "wait_sec", "download_sec")},
                         {"total_sec": 30, "cache_lookup_sec": 2, "submit_sec": 5, "wait_sec": 15, "download_sec": 3})
        self.assertFalse(timing["cache_hit"])
        self.assertFalse(timing["resumed_task"])
        self.assertEqual((timing["new_tasks_submitted"], timing["request_count"], timing["poll_count"], timing["download_attempts"]), (1, 3, 2, 1))
        self.assertTrue(all(type(value) in (int, float, bool) for value in timing.values()))
        self.assertNotIn("private", json.dumps(timing))
        self.assertEqual(api.call_count, 3)
        self.assertEqual(download.call_count, 1)
        with patch.object(self.client.api, "open") as api, patch.object(self.client.downloads, "open") as download, \
                patch("scripts.ai33_tts.time.perf_counter", side_effect=[100, 101, 102, 103]):
            self.assertEqual(self.client.synthesize("Do not put learner text in timings."), audio)
        api.assert_not_called()
        download.assert_not_called()
        self.assertTrue(self.client.last_timing["cache_hit"])
        self.assertEqual(self.client.last_timing["total_sec"], 3)
        self.assertEqual(self.client.last_timing["cache_lookup_sec"], 1)
        for key in ("submit_sec", "wait_sec", "download_sec", "new_tasks_submitted", "request_count", "poll_count", "download_attempts"):
            self.assertEqual(self.client.last_timing[key], 0)
        self.assertFalse(self.client.last_timing["resumed_task"])

    def test_failed_submission_retains_timing_without_retry_or_sensitive_details(self):
        with patch.object(self.client.api, "open", side_effect=URLError("unit-test-secret")) as api:
            with self.assertRaises(AI33Error):
                self.client.synthesize("Private learner text.")
        self.assertEqual(api.call_count, 1)
        timing = self.client.last_timing
        self.assertEqual(timing["new_tasks_submitted"], 1)
        self.assertEqual(timing["request_count"], 1)
        self.assertEqual(timing["poll_count"], 0)
        self.assertGreaterEqual(timing["total_sec"], timing["submit_sec"])
        self.assertGreaterEqual(timing["submit_sec"], 0)
        self.assertNotIn("unit-test-secret", json.dumps(timing))
        self.assertNotIn("Private learner", json.dumps(timing))
        self.assertIsNone(self.client._timing)

    def test_timing_distinguishes_task_polls_from_retried_api_attempts(self):
        unavailable = HTTPError("https://api.ai33.pro/v1/task/existing", 503, "private", {}, None)
        with patch.object(self.client.api, "open", side_effect=[
                Response(b'{"task_id":"existing"}'), unavailable,
                Response(b'{"status":"done","audio_url":"https://cdn.example/a.wav"}')]), \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()), \
                patch("scripts.ai33_tts.time.sleep"):
            self.assertTrue(self.client.synthesize("Hello.").exists())
        self.assertEqual(self.client.last_timing['poll_count'], 1)
        self.assertEqual(self.client.last_timing['request_count'], 3)
        self.assertEqual(self.client.last_timing['new_tasks_submitted'], 1)

    def test_poll_failure_resumes_without_second_submission(self):
        done = {"data": {"status": "completed", "audio_url": "https://cdn.example/a.wav"}}
        with patch.object(self.client, "request") as request, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()):
            request.side_effect = [{"task_id": "keep-this-task"}, AI33Error("Temporary connection failure")]
            with self.assertRaises(AI33Error):
                self.client.synthesize("Hello.", "vi-VN")
            self.assertEqual(json.loads(next(self.cache.glob("*.json")).read_text())["task_id"], "keep-this-task")
            request.side_effect = [done]
            result = self.client.synthesize("Hello.", "vi-VN")
            self.assertTrue(result.is_file())
            self.assertEqual([call.args[0] for call in request.call_args_list], ["POST", "GET", "GET"])
            self.assertEqual(request.call_args_list[0].args[2]["voice_id"], "example-vietnamese")

    def test_cache_corruption_redownloads_existing_task(self):
        with patch.object(self.client, "request", return_value={"task_id": "reuse"}) as request, \
                patch.object(self.client, "wait", return_value="https://cdn.example/a.wav"), \
                patch.object(self.client.downloads, "open", side_effect=[self.audio_response(), self.audio_response()]) as download:
            audio = self.client.synthesize("Hello.")
            audio.write_bytes(b"corrupted")
            self.client.synthesize("Hello.")
            self.assertEqual(request.call_count, 1)
            self.assertEqual(download.call_count, 2)

    def test_resume_without_receipt_never_submits_a_new_task(self):
        with patch.object(self.client, "request") as request:
            self.assertFalse(self.client.can_resume("Not submitted."))
            with self.assertRaisesRegex(AI33Error, "No cached task"):
                self.client.resume("Not submitted.")
            request.assert_not_called()

    def test_explicit_resume_recovers_existing_task_using_only_get(self):
        with patch.object(self.client, "request", side_effect=[{"task_id": "existing"}, AI33Error("temporary")]):
            with self.assertRaises(AI33Error):
                self.client.synthesize("Hello.")
        self.assertTrue(self.client.can_resume("Hello."))
        with patch.object(self.client, "request", return_value={"status": "done", "audio_url": "https://cdn.example/a.wav"}) as request, \
                patch.object(self.client.downloads, "open", return_value=self.audio_response()):
            self.assertTrue(self.client.resume("Hello.").exists())
            self.assertEqual([call.args[0] for call in request.call_args_list], ["GET"])
        self.assertTrue(self.client.last_timing["resumed_task"])
        self.assertFalse(self.client.last_timing["cache_hit"])
        self.assertEqual(self.client.last_timing["new_tasks_submitted"], 0)
        self.assertEqual(self.client.last_timing["submit_sec"], 0)
        self.assertEqual(self.client.last_timing["poll_count"], 1)

    def test_transient_task_get_retries_but_authentication_failure_does_not(self):
        transient = HTTPError("https://api.ai33.pro/v1/task/existing", 503, "secret", {}, None)
        with patch.object(self.client.api, "open", side_effect=[transient, Response(b'{"status":"done"}')]) as api, \
                patch("scripts.ai33_tts.time.sleep"):
            self.assertEqual(self.client.request("GET", "/v1/task/existing"), {"status": "done"})
            self.assertEqual([call.args[0].get_method() for call in api.call_args_list], ["GET", "GET"])
        unauthorized = HTTPError("https://api.ai33.pro/v1/task/existing", 401, "secret", {}, None)
        with patch.object(self.client.api, "open", side_effect=unauthorized) as api:
            with self.assertRaises(AI33Error):
                self.client.request("GET", "/v1/task/existing")
            self.assertEqual(api.call_count, 1)

    def test_english_voice_required_before_submission(self):
        del os.environ["AI33_VOICE_EN"]
        with patch.object(self.client, "request") as request:
            with self.assertRaisesRegex(AI33Error, "AI33_VOICE_EN"):
                self.client.synthesize("Hello.")
            request.assert_not_called()

    def test_error_body_does_not_expose_key_and_post_is_not_retried(self):
        error = HTTPError("https://api.ai33.pro", 401, "unit-test-secret", {}, None)
        with patch.object(self.client.api, "open", side_effect=error) as api:
            with self.assertRaises(AI33Error) as caught:
                self.client.request("POST", "/v3/text-to-speech", {"text": "Hello"})
            self.assertNotIn("unit-test-secret", str(caught.exception))
            self.assertEqual(api.call_count, 1)

    def test_failed_and_unrecognized_tasks_fail_clearly(self):
        for response in ({"status": "failed"}, {"status": "done"}, {"other": "unknown"}):
            with self.subTest(response=response), patch.object(self.client, "request", return_value=response):
                with self.assertRaises(AI33Error):
                    self.client.wait("example-task")

    def test_pending_task_times_out(self):
        self.client.timeout = 0.002
        with patch.object(self.client, "request", return_value={"status": "queued"}):
            with self.assertRaisesRegex(AI33Error, "timed out"):
                self.client.wait("example-task")

    def test_processing_task_url_is_not_used_before_completed_status(self):
        url = "https://cdn.example/future.mp3"
        with patch.object(self.client, "request", side_effect=[
                {"status": "processing", "audio_url": url},
                {"status": "completed", "audio_url": url}]) as request, \
                patch("scripts.ai33_tts.time.sleep") as sleep:
            self.assertEqual(self.client.wait("still-rendering"), url)
        self.assertEqual(request.call_count, 2)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["GET", "GET"])
        sleep.assert_called_once()

    def test_transient_cdn_404_download_retries_get_without_auth_or_submission(self):
        url = "https://cdn.example/propagating.mp3?signature=private"
        error = HTTPError(url, 404, "not propagated", {}, None)
        audio = b"ID3\x04test-mp3-audio"
        with patch.object(self.client.downloads, "open", side_effect=[error, Response(audio, "audio/mpeg")]) as download, \
                patch.object(self.client.api, "open") as api, patch("scripts.ai33_tts.time.sleep") as sleep:
            path, _ = self.client.download(url, self.cache / "propagating")
        self.assertEqual(path.suffix, ".mp3")
        self.assertEqual(path.read_bytes(), audio)
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(1)
        api.assert_not_called()
        for call in download.call_args_list:
            request = call.args[0]
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("User-agent"), "Mozilla/5.0")
            self.assertIsNone(request.get_header("Xi-api-key"))
            self.assertIsNone(request.get_header("Authorization"))

    def test_cdn_connection_failure_is_retried_as_read_only_get(self):
        with patch.object(self.client.downloads, "open", side_effect=[URLError("temporary"), self.audio_response()]) as download, \
                patch.object(self.client.api, "open") as api, patch("scripts.ai33_tts.time.sleep") as sleep:
            path, _ = self.client.download("https://cdn.example/a.wav", self.cache / "temporary")
        self.assertTrue(path.is_file())
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(1)
        api.assert_not_called()

    def test_persistent_transient_cdn_failure_is_bounded_to_three_gets(self):
        error = HTTPError("https://cdn.example/a.wav", 503, "not ready", {}, None)
        with patch.object(self.client.downloads, "open", side_effect=error) as download, \
                patch("scripts.ai33_tts.time.sleep") as sleep:
            with self.assertRaises(AI33Error):
                self.client.download("https://cdn.example/a.wav", self.cache / "unavailable")
        self.assertEqual(download.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_cdn_authentication_failure_is_not_retried(self):
        error = HTTPError("https://cdn.example/a.wav", 401, "denied", {}, None)
        with patch.object(self.client.downloads, "open", side_effect=error) as download, \
                patch("scripts.ai33_tts.time.sleep") as sleep:
            with self.assertRaises(AI33Error):
                self.client.download("https://cdn.example/a.wav", self.cache / "denied")
        self.assertEqual(download.call_count, 1)
        sleep.assert_not_called()

    def test_html_error_page_is_not_saved_as_audio(self):
        with patch.object(self.client.downloads, "open", return_value=Response(b"<html>Error</html>", "text/html")) as download, \
                patch("scripts.ai33_tts.time.sleep") as sleep:
            with self.assertRaises(AI33Error):
                self.client.download("https://cdn.example/a.mp3", self.cache / "example")
        self.assertEqual(download.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_list_voices_sends_required_provider_and_encoded_search(self):
        with patch.object(self.client, "request", return_value={"voices": []}) as request:
            self.client.list_voices("edge", "en-US", 2)
            self.assertEqual(request.call_args.args[0], "GET")
            self.assertEqual(parse_qs(urlsplit(request.call_args.args[1]).query), {
                "provider": ["edge"], "search": ["en-US"], "page": ["2"],
            })

    def test_invalid_url_and_speed_rejected(self):
        for value in ("http://api.ai33.pro", "https://key@api.ai33.pro", "https://api.ai33.pro?key=secret"):
            with self.subTest(value=value), patch.dict(os.environ, {"AI33_BASE_URL": value}):
                with self.assertRaises(AI33Error):
                    AI33TTS(self.cache)
        for value in ("nan", "inf", "2", "0", "abc"):
            with self.subTest(speed=value), patch.dict(os.environ, {"AI33_SPEED": value}):
                with self.assertRaises(AI33Error):
                    AI33TTS(self.cache)

    def test_env_loader_preserves_environment_and_does_not_execute_values(self):
        env = self.cache / ".env"
        env.write_text("AI33_API_KEY='file-secret'\nAI33_VOICE_EN='$(do-not-execute)'\n", encoding="utf-8")
        del os.environ["AI33_VOICE_EN"]
        load_env(env)
        self.assertEqual(os.environ["AI33_API_KEY"], "unit-test-secret")
        self.assertEqual(os.environ["AI33_VOICE_EN"], "$(do-not-execute)")


if __name__ == "__main__":
    unittest.main()
