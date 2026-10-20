"""Offline integration tests: no account, network traffic or credits required."""

from email.message import Message
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
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
            self.assertIsNone(downloaded.get_header("Xi-api-key"))
            self.assertIsNone(downloaded.get_header("Authorization"))
        receipt_text = next(self.cache.glob("*.json")).read_text()
        self.assertNotIn("unit-test-secret", receipt_text)
        self.assertNotIn("signature=private", receipt_text)
        self.assertNotIn("I think this is useful.", receipt_text)

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

    def test_html_error_page_is_not_saved_as_audio(self):
        with patch.object(self.client.downloads, "open", return_value=Response(b"<html>Error</html>", "text/html")):
            with self.assertRaises(AI33Error):
                self.client.download("https://cdn.example/a.mp3", self.cache / "example")
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
