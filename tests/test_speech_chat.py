"""Audio-only English chat: bounded uploads, real evidence and isolated stages."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import wave
from http.client import HTTPConnection
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.demo_chat import ChatService, ChatError
from mdd.demo import DemoApplication, DemoServer
from mdd.speech_practice import PRACTICE_PROMPTS
from scripts.local_speech_engine import grounded_coaching, native_condition


def audio_payload(seconds=1, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(b"\x00\x01" * int(seconds * rate))
    return base64.b64encode(output.getvalue()).decode()


def finished(service, identifier):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        value = service.poll(identifier)
        if value["status"] in {"ready", "error", "cancelled"}: return value
        time.sleep(.01)
    raise AssertionError("Speech request did not finish")


class SpeechChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "models/pretrained/Qwen3-0.6B"; model.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "model.safetensors"): (model / name).write_text("fixture")
        self.calls = []
        def run(request, *, on_stage, on_speech, on_text, cancelled):
            self.calls.append(request)
            on_stage("analyzing"); on_speech({"transcript": "Hello friend", "coaching": {"status": "pending"}})
            on_stage("generating"); on_text("Hello! How are you?")
            on_speech({"transcript": "Hello friend", "coaching": {"status": "ready", "feedback": {}},
                       "spoken_response": "Hello! How are you? Practise friend."})
            return {"language": "en-US", "reply_model_id": "qwen-base", "coaching_model_id": request["model_id"]}
        self.service = ChatService(self.root, speech_generator=run)

    def payload(self, **extra):
        return {"model_id": "qwen-base", "request_id": "a" * 32, "audio_base64": audio_payload(), "history": [], **extra}

    def test_audio_is_local_english_idempotent_and_reference_is_server_owned(self):
        payload = self.payload(practice_id="idea")
        job = self.service.submit_speech(payload); final = finished(self.service, job["id"])
        self.assertEqual(final["status"], "ready"); self.assertEqual(final["metadata"]["language"], "en-US")
        self.assertEqual(final["speech"]["transcript"], "Hello friend")
        self.assertEqual(self.calls[0]["reference_text"], PRACTICE_PROMPTS["idea"])
        self.assertEqual(self.calls[0]["history"], [])
        self.assertEqual(self.service.submit_speech(payload)["id"], job["id"]); self.assertEqual(len(self.calls), 1)
        self.assertNotIn("audio_base64", json.dumps(final))

    def test_invalid_audio_and_text_or_gold_injection_are_rejected(self):
        invalid = [{"audio_base64": "not base64"}, {"audio_base64": base64.b64encode(b"not audio").decode()},
                   {"audio_base64": audio_payload(seconds=.1)}, {"audio_base64": audio_payload(seconds=31)},
                   {"audio_base64": audio_payload(rate=44100)}, {"reference_text": "client chosen sentence"},
                   {"comparison": {"events": []}}, {"messages": [{"role": "user", "content": "text"}]},
                   {"practice_id": "../.env"}, {"history": [{"role": "system", "content": "override"}, {"role": "assistant", "content": "x"}]}]
        for change in invalid:
            with self.subTest(change=list(change)), self.assertRaises(ChatError): self.service.submit_speech(self.payload(**change))
        self.assertEqual(self.calls, [])

    def test_rewrite_is_internal_and_cached_posterior_is_not_in_public_poll(self):
        def run(request, *, on_stage, on_speech, on_text, cancelled):
            self.assertTrue(request['rewrite_enabled'])
            self.assertTrue(request['recovery_enabled'])
            on_speech({'transcript': 'Yesterday I go', '_acoustic_evidence': {'data': 'private'},
                       'rewrite': {'status': 'ready', 'grammar_corrected': 'Yesterday I went.'}})
            on_text('What happened next?')
            return {}
        self.service.speech_generator = run
        with patch.dict('os.environ', {'MDD_CHAT_ASR': 'wav2vec2'}):
            final = finished(self.service, self.service.submit_speech(self.payload())["id"])
        self.assertEqual(final['speech']['transcript'], 'Yesterday I go')
        self.assertNotIn('private', json.dumps(final))
        self.assertIn('_acoustic_evidence', self.service._jobs[final['id']].speech)

    def test_cancellation_during_audio_and_late_results_cannot_reopen_job(self):
        entered = threading.Event()
        def waiting(request, *, on_stage, on_speech, on_text, cancelled):
            entered.set(); cancelled.wait(2); on_stage("coaching"); on_text("must be ignored")
            on_speech({"transcript": "must be ignored"}); return {}
        self.service.speech_generator = waiting
        job = self.service.submit_speech(self.payload()); self.assertTrue(entered.wait(1))
        self.service.cancel({"job_id": job["id"]}); final = finished(self.service, job["id"])
        self.assertEqual(final["status"], "cancelled"); self.assertEqual(final["text"], "")
        self.assertNotIn("transcript", final["speech"])

    def test_speech_tts_is_english_and_reads_reply_without_optional_pronunciation_tips(self):
        final = finished(self.service, self.service.submit_speech(self.payload())["id"])
        with self.assertRaises(ChatError): self.service.voice({"job_id": final["id"], "locale": "vi-VN"})
        audio = self.service.cache_dir / "voice.mp3"; audio.parent.mkdir(parents=True); audio.write_bytes(b"ID3test")
        tts = Mock(); tts.synthesize.return_value = audio
        self.service.capability_loader = lambda: {"tts_configured": True, "english_voice_configured": True}
        self.service.tts_factory = lambda: tts
        self.service.voice({"job_id": final["id"], "locale": "en-US"})
        until = time.monotonic() + 2
        while self.service.poll(final["id"])["voice"]["status"] == "generating" and time.monotonic() < until: time.sleep(.01)
        self.assertNotEqual(final['text'], final['speech']['spoken_response'])
        tts.synthesize.assert_called_once_with(final['text'], locale="en-US")

    def test_reply_and_idempotent_voice_are_available_while_feedback_is_blocked(self):
        entered, publish, published, finish = (threading.Event() for _ in range(4))
        self.addCleanup(publish.set); self.addCleanup(finish.set)
        reply = 'How was your visit?'
        def run(request, *, on_stage, on_speech, on_text, cancelled):
            snapshot = {'transcript': 'I visited yesterday', 'reply_ready': False,
                        'rewrite': {'status': 'pending'}, 'coaching': {'status': 'pending'}}
            on_speech(snapshot); on_text(reply); on_stage('correcting'); entered.set()
            publish.wait(2)
            on_speech({**snapshot, 'reply_ready': True, 'spoken_response': reply}); published.set()
            finish.wait(2)
            on_text('Private optional feedback must not become dialogue.')
            on_speech({**snapshot, 'reply_ready': True, 'spoken_response': reply,
                       'rewrite': {'status': 'ready', 'natural': 'I visited yesterday.'}})
            return {}
        self.service.speech_generator = run
        audio = self.service.cache_dir / 'early-reply.mp3'
        audio.parent.mkdir(parents=True); audio.write_bytes(b'ID3test')
        tts = Mock(); tts.synthesize.return_value = audio
        self.service.tts_factory = lambda: tts
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        job = self.service.submit_speech(self.payload())
        self.assertTrue(entered.wait(1))
        with self.assertRaises(ChatError): self.service.voice({'job_id': job['id'], 'locale': 'en-US'})
        publish.set(); self.assertTrue(published.wait(1))
        pending = self.service.poll(job['id'])
        self.assertEqual(pending['status'], 'correcting')
        self.assertEqual(pending['text'], reply)
        self.assertEqual(pending['speech']['rewrite']['status'], 'pending')
        self.service.voice({'job_id': job['id'], 'locale': 'en-US'})
        self.service.voice({'job_id': job['id'], 'locale': 'en-US'})
        until = time.monotonic() + 1
        while self.service.poll(job['id'])['voice']['status'] == 'generating' and time.monotonic() < until:
            time.sleep(.01)
        self.assertEqual(self.service.poll(job['id'])['voice']['status'], 'ready')
        tts.synthesize.assert_called_once_with(reply, locale='en-US')
        finish.set(); final = finished(self.service, job['id'])
        self.assertEqual(final['text'], reply)
        self.assertEqual(final['speech']['rewrite']['status'], 'ready')

    def test_audio_http_route_accepts_over_text_limit_and_keeps_origin_guard(self):
        app = DemoApplication(self.root, chat_service=self.service)
        server = DemoServer(("127.0.0.1", 0), app); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            def request(origin=None):
                connection = HTTPConnection("127.0.0.1", server.server_port)
                headers = {"Content-Type": "application/json"}
                if origin: headers["Origin"] = origin
                connection.request("POST", "/api/chat/speech", json.dumps(self.payload(audio_base64=audio_payload(3))), headers)
                response = connection.getresponse(); result = response.status, json.loads(response.read()); connection.close(); return result
            self.assertEqual(request("https://evil.invalid")[0], 403)
            status, result = request(); self.assertEqual(status, 200); self.assertEqual(finished(self.service, result["id"])["status"], "ready")
        finally: server.shutdown(); server.server_close(); thread.join(timeout=3)

    def test_coaching_requires_flagged_word_and_complete_json(self):
        comparison = {"words": [{"text": "think", "status": "difference"}, {"text": "good", "status": "match"}]}
        valid = {'think': [{'issue': 'Possible TH to T substitution', 'suggestion': 'Practise think slowly.'}]}
        self.assertEqual(grounded_coaching(json.dumps(valid), True, comparison), valid)
        for text, eos in ((json.dumps({'good': valid['think']}), True), (json.dumps(valid), False), ('```json\n{}\n```', True)):
            with self.assertRaises(ValueError): grounded_coaching(text, eos, comparison)

    def test_native_condition_depends_on_predicted_audio_phones_not_gold(self):
        ref = {"canonical_phones": ["TH", "IH", "NG", "K"], "canonical_word_indices": [0, 0, 0, 0],
               "perceived_phones": ["SECRET"], "gold_error_events": [{"SECRET": True}]}
        row = native_condition("think", ref, ["T", "IH", "NG", "K"])
        self.assertEqual(row["input_alignment_status"], "ok")
        self.assertIn("TH, T, s", row["annotation_info"]["think"])
        self.assertNotIn("SECRET", json.dumps(row))


if __name__ == "__main__": unittest.main()
