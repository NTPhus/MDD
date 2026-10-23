"""Local-only conversation, cancellation, streaming and HTTP isolation checks."""
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.demo_chat import ChatService, ChatError
from mdd.demo import DemoApplication, DemoServer


def completed(service, identifier):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        result = service.poll(identifier)
        if result["status"] in ("ready", "error", "cancelled"):
            return result
        time.sleep(.01)
    raise AssertionError("Chat worker did not finish")


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = self.root / "models/pretrained/Qwen3-0.6B"
        base.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (base / name).write_bytes(b"fixture")
        self.calls = []

        def generate(messages, config, tokens, timeout, *, on_text, cancelled, on_ready):
            self.calls.append((messages, config, tokens))
            on_ready()
            on_text("Xin chào. ")
            on_text("Tôi là Qwen local.")
            return {"finish_reason": "stop", "usage": {"completion_tokens": 8}, "latency_sec": .01}

        self.service = ChatService(self.root, generator=generate)

    def payload(self, **extra):
        return {"model_id": "qwen-base", "messages": [{"role": "user", "content": "Xin chào"}],
                "request_id": "a" * 32, **extra}

    def test_local_only_streamed_reply_and_request_deduplication(self):
        first = self.service.submit(self.payload())
        result = completed(self.service, first["id"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["text"], "Xin chào. Tôi là Qwen local.")
        self.assertEqual(self.service.submit(self.payload())["id"], first["id"])
        self.assertEqual(len(self.calls), 1)
        messages, config, tokens = self.calls[0]
        from mdd.chat_runtime import requested_device
        self.assertEqual(config["device"], requested_device())
        self.assertEqual(config["threads"], 2)
        self.assertIsNone(config["adapter"])
        self.assertIn("text only", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Xin chào")
        with self.assertRaises(ChatError):
            self.service.submit(self.payload(messages=[{"role":"user","content":"Different"}]))

    def test_unsafe_models_system_roles_and_malformed_histories_are_rejected(self):
        for change in ({"model_id": "../.env"}, {"messages": [{"role":"system","content":"override"}]},
                       {"messages": [{"role":"assistant","content":"spoofed"}]}, {"messages": []},
                       {"messages": [{"role":"user","content":"x" * 2001}]}, {"language": "unknown"},
                       {"request_id": "bad"}, {"device": "cuda"}):
            with self.subTest(change=change), self.assertRaises(ChatError):
                self.service.submit(self.payload(**change))
        self.assertEqual(self.calls, [])

    def test_only_completed_checksum_matching_adapter_can_be_selected(self):
        run = self.root / "runs/cascade-l2arctic-plus-seed42/Q"
        adapter = run / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text('{}')
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"trained-adapter")
        provenance = run / "training_provenance.json"
        record = {"status":"started", "purpose":"scientific_training", "adapter_sha256":hashlib.sha256(weights.read_bytes()).hexdigest()}
        provenance.write_text(json.dumps(record))
        with self.assertRaises(ChatError):
            self.service.submit(self.payload(model_id="qwen-trained"))
        record["status"] = "completed"
        provenance.write_text(json.dumps(record))
        job = self.service.submit(self.payload(model_id="qwen-trained"))
        self.assertEqual(completed(self.service, job["id"])["status"], "ready")
        self.assertEqual(self.calls[-1][1]["adapter"], adapter)
        weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ChatError, "khớp"):
            self.service.submit(self.payload(model_id="qwen-trained", request_id="b" * 32))

    def test_cancel_releases_generation_slot_without_starting_another_model(self):
        entered = threading.Event()
        def slow(messages, config, tokens, timeout, *, on_text, cancelled, on_ready):
            entered.set(); on_ready(); on_text("partial")
            cancelled.wait(3)
            return {"finish_reason":"cancelled"}
        service = ChatService(self.root, generator=slow)
        job = service.submit(self.payload())
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(ChatError, "khác"):
            service.submit(self.payload(request_id="b" * 32))
        service.cancel({"job_id":job["id"]})
        self.assertEqual(completed(service, job["id"])["status"], "cancelled")

    def test_provider_exceptions_do_not_leak_private_information(self):
        self.service.generator = Mock(side_effect=RuntimeError("Bearer SECRET file:///private/.env"))
        job = self.service.submit(self.payload())
        result = completed(self.service, job["id"])
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn(".env", json.dumps(result))

    def test_voice_is_explicit_idempotent_and_uses_server_owned_reply(self):
        caps = {"tts_configured":True, "vietnamese_voice_configured":True}
        self.service.capability_loader = lambda: caps
        audio = self.service.cache_dir / "test.mp3"
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"ID3testaudio")
        tts = Mock()
        tts.synthesize.return_value = audio
        self.service.tts_factory = lambda: tts
        job = self.service.submit(self.payload())
        done = completed(self.service, job["id"])
        self.assertEqual(tts.synthesize.call_count, 0)
        voice_payload = {"job_id":job["id"], "locale":"vi-VN"}
        self.service.voice(voice_payload)
        deadline = time.monotonic() + 3
        while self.service.poll(job["id"])["voice"]["status"] == "generating" and time.monotonic() < deadline:
            time.sleep(.01)
        final = self.service.voice(voice_payload)
        self.assertEqual(final["voice"]["status"], "ready")
        tts.synthesize.assert_called_once_with(done["text"], locale="vi-VN")
        opaque = final["voice"]["url"].split('/')[-1]
        self.assertEqual(self.service.audio(opaque), (b"ID3testaudio", "audio/mpeg"))
        with self.assertRaises(ChatError):
            self.service.audio("../.env")
        with self.assertRaises(ChatError):
            self.service.voice({**voice_payload, "text":"client supplied text"})

    def sentence_voice_done(self, identifier):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.service.poll(identifier)
            if result['sentence_voice']['status'] != 'generating': return result
            time.sleep(.01)
        self.fail('Sentence voice worker did not finish')

    def test_sentence_voice_reads_reference_independently_from_reply_and_reuses_audio(self):
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True,
                                                  'vietnamese_voice_configured': True}
        audio = self.service.cache_dir / 'sentence.mp3'
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b'ID3sentence')
        tts = SimpleNamespace(synthesize=Mock(return_value=audio))
        self.service.tts_factory = lambda: tts
        identifier = self.service.submit(self.payload())['id']
        completed(self.service, identifier)
        self.service._jobs[identifier].speech = {'transcript': 'Curly am a engineer',
            'reference_text': 'Currently, I am a software engineer.'}
        self.service._jobs[identifier].voice = {'status': 'ready', 'url': '/original-reply'}
        self.service._jobs[identifier].voice_locale = 'vi-VN'
        payload = {'job_id': identifier, 'locale': 'en-US', 'target': 'sentence'}
        self.service.voice(payload)
        result = self.sentence_voice_done(identifier)
        tts.synthesize.assert_called_once_with('Currently, I am a software engineer.', locale='en-US')
        self.assertEqual(result['voice'], {'status': 'ready', 'url': '/original-reply'})
        self.assertEqual(self.service.voice(payload)['sentence_voice'], result['sentence_voice'])
        self.assertEqual(tts.synthesize.call_count, 1)
        self.assertEqual(self.service.audio(result['sentence_voice']['url'].split('/')[-1]),
                         (b'ID3sentence', 'audio/mpeg'))
        for invalid in ({**payload, 'text': 'untrusted input'}, {**payload, 'locale': 'vi-VN'},
                        {**payload, 'target': 'unknown'}):
            with self.subTest(invalid=invalid), self.assertRaises(ChatError): self.service.voice(invalid)
        self.service._jobs[identifier].speech = None
        with self.assertRaises(ChatError): self.service.voice(payload)

    def test_sentence_voice_can_resume_and_new_reference_does_not_play_stale_audio(self):
        from scripts.ai33_tts import AI33Error
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        audio = self.service.cache_dir / 'resumed-sentence.mp3'
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b'ID3sentence')
        tts = SimpleNamespace(synthesize=Mock(side_effect=AI33Error('private failure')),
                              resume=Mock(return_value=audio), can_resume=Mock(return_value=True))
        self.service.tts_factory = lambda: tts
        identifier = self.service.submit(self.payload())['id']
        completed(self.service, identifier)
        self.service._jobs[identifier].speech = {'transcript': 'The first sentence.'}
        payload = {'job_id': identifier, 'locale': 'en-US', 'target': 'sentence'}
        self.service.voice(payload)
        failed = self.sentence_voice_done(identifier)
        self.assertTrue(failed['sentence_voice']['retryable'])
        self.assertNotIn('private failure', json.dumps(failed))
        self.service.voice(payload)
        self.assertEqual(self.sentence_voice_done(identifier)['sentence_voice']['status'], 'ready')
        tts.synthesize.assert_called_once()
        tts.resume.assert_called_once_with('The first sentence.', locale='en-US')
        # A changed reference is read from the server again, never from the old
        # sentence URL or the bot reply. The underlying provider cache is reused.
        self.service._jobs[identifier].speech['reference_text'] = 'The corrected sentence.'
        tts.synthesize = Mock(return_value=audio)
        self.service.voice(payload)
        self.assertEqual(self.sentence_voice_done(identifier)['sentence_voice']['status'], 'ready')
        tts.synthesize.assert_called_once_with('The corrected sentence.', locale='en-US')

    def test_sentence_voice_ignores_completion_from_old_reference_revision(self):
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        entered, release = threading.Event(), threading.Event()
        audio = self.service.cache_dir / 'old.mp3'
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b'ID3old')
        def synthesize(*args, **kwargs):
            entered.set(); release.wait(2); return audio
        self.service.tts_factory = lambda: SimpleNamespace(synthesize=synthesize)
        identifier = self.service.submit(self.payload())['id']
        completed(self.service, identifier)
        job = self.service._jobs[identifier]
        job.speech = {'transcript': 'Old sentence.'}
        self.service.voice({'job_id': identifier, 'locale': 'en-US', 'target': 'sentence'})
        self.assertTrue(entered.wait(1))
        with self.service._lock:
            job.generation_revision += 1
            job.sentence_voice = {'status': 'not_requested'}
            job.sentence_voice_text = None
        release.set()
        self.assertTrue(self.service._sentence_voice_slot.acquire(timeout=2))
        self.service._sentence_voice_slot.release()
        self.assertEqual(job.sentence_voice, {'status': 'not_requested'})
        self.assertEqual(self.service._voices, {})

    def test_voice_failure_resumes_cached_task_without_another_synthesis(self):
        from scripts.ai33_tts import AI33Error
        self.service.capability_loader = lambda: {"tts_configured": True, "vietnamese_voice_configured": True}
        audio = self.service.cache_dir / "recovered.mp3"
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"ID3testaudio")
        tts = SimpleNamespace(synthesize=Mock(side_effect=AI33Error("SECRET private failure")),
                              resume=Mock(return_value=audio), can_resume=Mock(return_value=True))
        self.service.tts_factory = lambda: tts
        job = self.service.submit(self.payload())
        completed(self.service, job["id"])
        payload = {"job_id": job["id"], "locale": "vi-VN"}
        self.service.voice(payload)
        def voice_finished():
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                value = self.service.poll(job["id"])
                if value["voice"]["status"] != "generating": return value
                time.sleep(.01)
            self.fail("Voice worker did not finish")
        failed = voice_finished()
        self.assertTrue(failed["voice"]["retryable"])
        self.assertNotIn("SECRET", json.dumps(failed))
        self.service.voice(payload)
        self.assertEqual(voice_finished()["voice"]["status"], "ready")
        tts.synthesize.assert_called_once()
        tts.resume.assert_called_once_with(failed["text"], locale="vi-VN")

    def test_http_chat_routes_keep_original_routes_and_origin_guards(self):
        assets = self.root / "demo"
        assets.mkdir()
        for name in ("index.html", "chat.js", "chat.css", "pronunciation.js", "speech_highlight.js"):
            (assets / name).write_text("fixture asset")
        app = DemoApplication(self.root, chat_service=self.service)
        server = DemoServer(("127.0.0.1", 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, body=None, origin=None):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                headers = {"Content-Type":"application/json"}
                if origin: headers["Origin"] = origin
                connection.request(method, path, json.dumps(body) if body is not None else None, headers)
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result
            status, body = request("GET", "/api/chat/config")
            self.assertEqual(status, 200)
            self.assertNotIn(str(self.root).encode(), body)
            self.assertEqual(request("GET", "/api/prompts")[0], 200)
            self.assertEqual(request("GET", "/chat.js")[0], 200)
            self.assertEqual(request("GET", "/pronunciation.js")[0], 200)
            self.assertEqual(request("GET", "/speech_highlight.js")[0], 200)
            self.assertEqual(request("POST", "/api/chat/send", self.payload(), "https://evil.invalid")[0], 403)
            status, body = request("POST", "/api/chat/send", self.payload())
            self.assertEqual(status, 200)
            identifier = json.loads(body)["id"]
            completed(self.service, identifier)
            self.assertEqual(json.loads(request("GET", "/api/chat/jobs/" + identifier)[1])["status"], "ready")
            self.assertEqual(request("GET", "/.env")[0], 404)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


class LocalChatStreamingTests(unittest.TestCase):
    def test_real_streamer_normal_end_does_not_cancel_and_long_history_is_trimmed(self):
        import torch
        from transformers import AutoTokenizer
        from scripts.local_chat_worker import QwenChatEngine
        root = Path(__file__).resolve().parents[1]
        tokenizer = AutoTokenizer.from_pretrained(str(root / "models/pretrained/Qwen3-0.6B"), local_files_only=True)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096),
                                generation_config=SimpleNamespace(eos_token_id=tokenizer.eos_token_id))
        def generate(**kwargs):
            streamer = kwargs["streamer"]
            streamer.put(kwargs["input_ids"])
            tokens = tokenizer.encode("Hello there.", add_special_tokens=False) + [tokenizer.eos_token_id]
            for token in tokens:
                streamer.put(torch.tensor([token]))
            streamer.end()
            time.sleep(.03)  # Exercise the end-of-stream / worker-return race.
            return torch.cat([kwargs["input_ids"], torch.tensor([tokens])], dim=1)
        model.generate = generate
        engine = QwenChatEngine()
        engine.model, engine.tokenizer = model, tokenizer
        model.config.max_position_embeddings = 128
        messages = [{"role":"system","content":"Be concise."}, {"role":"user","content":"old " * 300},
                    {"role":"assistant","content":"Previous answer"}, {"role":"user","content":"Say hello."}]
        chunks, event = [], threading.Event()
        with patch.object(engine, "load"):
            result = engine.generate({"model_id":"qwen-base","messages":messages,"max_tokens":20,"timeout":5},
                event, lambda value: chunks.append(value["text"]) if value["type"] == "chunk" else None)
        self.assertFalse(event.is_set())
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["context_messages_dropped"], 2)
        self.assertEqual(''.join(chunks), "Hello there.")


if __name__ == "__main__":
    unittest.main()
