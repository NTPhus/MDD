"""Isolated local Qwen conversation jobs. Never calls an LLM API or starts training."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import base64
import binascii
import io
import importlib
import json
import math
import logging
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
import unicodedata
import wave

from .speech_practice import PRACTICE_PROMPTS
from .speech_topics import TOPICS, public_topics, guidance
from .chat_voices import load_voices, preview_stem

ID = re.compile(r"[0-9a-f]{32}\Z")
SYSTEM = ("You are a helpful conversational assistant. Answer the user's actual question clearly and concisely. "
          "You receive text only. Do not claim to hear audio, diagnose pronunciation from recordings, access files, "
          "execute commands, or change training. When asked about pronunciation without acoustic evidence, "
          "give general guidance and do not invent errors. Use plain text. Do not output hidden reasoning.")


def spoken_text_key(text):
    """Compare spoken words; case, spacing and punctuation alone are not an edit."""
    if not isinstance(text, str):
        return ()
    text = unicodedata.normalize('NFKC', text).replace('’', "'").lower()
    return tuple(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text))


class ChatError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


@dataclass
class ChatJob:
    model_id: str
    identifier: str = field(default_factory=lambda: secrets.token_hex(16))
    cancelled: threading.Event = field(default_factory=threading.Event)
    status: str = "loading"
    text: str = ""
    created: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)
    error: dict | None = None
    voice: dict = field(default_factory=lambda: {"status": "not_requested"})
    voice_locale: str | None = None
    sentence_voice: dict = field(default_factory=lambda: {"status": "not_requested"})
    sentence_voice_text: str | None = None
    voice_options: dict = field(default_factory=dict)
    sentence_voice_options: dict = field(default_factory=dict)
    speech: dict | None = None
    speech_context: dict | None = None
    generation_revision: int = 0
    reply_revisions: list = field(default_factory=list)


class ChatService:
    def __init__(self, root, *, capability_loader=None, generator=None, tts_factory=None, speech_generator=None, reference_generator=None):
        self.root = Path(root).resolve()
        self.capability_loader = capability_loader or (lambda: {})
        self.generator, self.tts_factory = generator, tts_factory
        self.speech_generator = speech_generator
        self.reference_generator = reference_generator
        self._lock = threading.RLock()
        self._generation_slot = threading.BoundedSemaphore(1)
        self._voice_slot = threading.BoundedSemaphore(1)
        self._jobs = OrderedDict()
        self._requests = {}
        self._voices = {}
        self._runner = None
        self.cache_dir = self.root / "data/generated/tts/ai33/chat"

    def _module(self, name):
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        return importlib.import_module("scripts." + name)

    def _models(self):
        base = self.root / "models/pretrained/Qwen3-0.6B"
        run = self.root / "runs/cascade-l2arctic-plus-seed42/Q"
        adapter = run / "adapter"
        available = all((base / name).is_file() for name in ("config.json", "tokenizer.json", "model.safetensors"))
        try:
            provenance = json.loads((run / "training_provenance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            provenance = {}
        trained = available and provenance.get("status") == "completed" and provenance.get("purpose") == "scientific_training" and all(
            (adapter / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors"))
        return {"qwen-base": {"id": "qwen-base", "label": "Qwen3-0.6B · Bản gốc", "ready": available,
                              "description": "Mô hình gốc cho hội thoại văn bản.", "adapter": None},
                "qwen-trained": {"id": "qwen-trained", "label": "Qwen3-0.6B · Sau train", "ready": bool(trained),
                                 "description": "LoRA đã học hướng dẫn phát âm; chất lượng chat tự do cần kiểm chứng.",
                                 "adapter": adapter, "expected_sha256": provenance.get("adapter_sha256")}}

    def config(self):
        capabilities = self.capability_loader()
        english_voices = self.english_voices()
        voice_name = os.environ.get("AI33_VOICE_EN_NAME", "").strip()[:120] if capabilities.get("english_voice_configured") else ""
        models = [{key: item[key] for key in ("id", "label", "ready", "description")} for item in self._models().values()]
        return {"models": models, "default_model_id": "qwen-base", "device": "cpu", "threads": 2,
                "max_message_chars": 2000, "max_new_tokens": 384,
                "ai33_available": bool(capabilities.get("tts_configured")),
                "english_voice_name": voice_name,
                "english_voices": english_voices,
                "default_voice_id": os.environ.get('AI33_VOICE_EN', '').strip() or (english_voices[0]['id'] if english_voices else ''),
                "voice_speeds": [0.85, 1.0, 1.15],
                "default_voice_provider": "ai33" if capabilities.get("tts_configured") and (voice_name or english_voices) else "browser",
                "voice_locales": [locale for locale, key in (("vi-VN", "vietnamese_voice_configured"), ("en-US", "english_voice_configured"))
                                  if capabilities.get(key) or (locale == 'en-US' and english_voices and capabilities.get('tts_configured'))],
                "speech_ready": (self.root / "models/pretrained/wav2vec2-base-960h/model.safetensors").is_file()
                    and (self.root / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint/model.safetensors").is_file(),
                "practice_prompts": [{"id": key, "text": value} for key, value in PRACTICE_PROMPTS.items()],
                "topics": public_topics(),
                "max_audio_seconds": 30, "speech_language": "en-US", "reply_model_id": "qwen-base",
                "sentence_mode": "grammar_and_natural", "rewrite_mode": "grammar_and_natural", "dialogue_policy": "natural_conversation",
                "pronunciation_method": "ctc_variant_posterior_margin_v1", "pronunciation_uncertainty": True,
                "pronunciation_feedback": "collapsed", "voice_content": "reply_only",
                "reference_confirmation": True,
                "reference_regenerates_reply": True,
                "reference_regenerates_only_on_change": True,
                "pronunciation_review": "immediate_provisional",
                "notice": "Audio is processed locally on CPU. Speech replies and pronunciation practice use English."}

    def english_voices(self):
        voices = load_voices(self.root)
        default = os.environ.get('AI33_VOICE_EN', '').strip()
        if default and re.fullmatch(r'[\w-]{1,160}', default) and not any(v['id'] == default for v in voices):
            voices.insert(0, {'id': default, 'name': os.environ.get('AI33_VOICE_EN_NAME', 'English voice')[:120],
                             'locale': 'en-US', 'gender': '', 'provider': 'ai33', 'preview_url': ''})
        return [{key: value for key, value in voice.items() if key != 'preview_url'} |
                {'preview_url': '/api/chat/voice-preview/' + voice['id'] if voice['preview_url'] else None}
                for voice in voices]

    def voice_preview(self, identifier):
        voice = next((v for v in load_voices(self.root) if v['id'] == identifier), None)
        if voice is None or not voice['preview_url']:
            raise ChatError(404, 'chat_voice_not_found', 'Chưa có bản nghe thử cho giọng này.')
        stem = preview_stem(self.cache_dir, identifier)
        path = next((stem.with_suffix(ext) for ext in ('.mp3', '.wav', '.ogg') if stem.with_suffix(ext).is_file()), None)
        if path is None:
            try:
                self.capability_loader()  # Load local provider configuration before a read-only download.
                client = self._module('ai33_tts').AI33TTS(cache_dir=self.cache_dir)
                stem.parent.mkdir(parents=True, exist_ok=True)
                path, _ = client.download(voice['preview_url'], stem)
            except Exception:
                raise ChatError(503, 'chat_preview_unavailable', 'Chưa tải được bản nghe thử. Hãy thử lại.') from None
        return self._read_voice(path)

    def _validate(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"model_id", "messages", "language", "request_id"}:
            raise ChatError(400, "invalid_chat", "Nội dung chat không hợp lệ.")
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or model_id not in self._models():
            raise ChatError(400, "invalid_chat_model", "Chọn mô hình Qwen trong danh sách.")
        language = payload.get("language", "vi-VN")
        if language not in ("vi-VN", "en-US"):
            raise ChatError(400, "invalid_language", "Chọn tiếng Việt hoặc tiếng Anh.")
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not ID.fullmatch(request_id):
            raise ChatError(400, "invalid_request_id", "Mã lượt chat không hợp lệ.")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 21 or len(messages) % 2 != 1:
            raise ChatError(400, "invalid_chat_history", "Lịch sử chat phải kết thúc bằng câu của bạn.")
        cleaned = []
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message.get("role") != expected or not isinstance(message.get("content"), str)
                    or not message["content"].strip() or len(message["content"]) > (2000 if expected == "user" else 12000)):
                raise ChatError(400, "invalid_chat_history", "Tin nhắn hoặc vai trò trong lịch sử chat không hợp lệ.")
            cleaned.append({"role": expected, "content": message["content"].strip()})
        if sum(len(m["content"]) for m in cleaned) > 16000:
            raise ChatError(413, "chat_history_too_long", "Cuộc trò chuyện đã dài; hãy bắt đầu cuộc trò chuyện mới.")
        system = SYSTEM + (" Reply in Vietnamese unless the user requests another language." if language == "vi-VN" else " Reply in English unless the user requests another language.")
        return model_id, request_id, [{"role": "system", "content": system}, *cleaned]

    def submit(self, payload):
        model_id, request_id, messages = self._validate(payload)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            if request_id in self._requests:
                prior, job_id = self._requests[request_id]
                if prior != fingerprint:
                    raise ChatError(409, "chat_request_conflict", "Lượt gửi này đã có nội dung khác.")
                return self.poll(job_id)
            model = self._models()[model_id]
            if not model["ready"]:
                raise ChatError(503, "chat_model_unavailable", "Mô hình đã chọn chưa có trọng số hoàn tất trên máy.")
            if model["adapter"]:
                actual = hashlib.sha256((model["adapter"] / "adapter_model.safetensors").read_bytes()).hexdigest()
                if not model.get("expected_sha256") or actual != model["expected_sha256"]:
                    raise ChatError(503, "chat_adapter_changed", "Adapter sau train không khớp bản đã lưu.")
            if not self._generation_slot.acquire(blocking=False):
                raise ChatError(409, "chat_busy", "Qwen đang trả lời một lượt khác. Hãy chờ rồi gửi lại.")
            job = ChatJob(model_id)
            self._jobs[job.identifier] = job
            self._requests[request_id] = (fingerprint, job.identifier)
            self._prune()
        threading.Thread(target=self._generate, args=(job, model, messages), daemon=True, name="local-chat-job").start()
        return self.poll(job.identifier)

    def start_topic(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"model_id", "request_id", "topic_id"}:
            raise ChatError(400, "invalid_topic_start", "Choose a speaking topic.")
        topic_id = payload.get("topic_id")
        if not isinstance(topic_id, str) or topic_id not in TOPICS:
            raise ChatError(400, "invalid_topic", "Choose a topic from the list.")
        model_id, request_id, _ = self._validate({"model_id": payload.get("model_id"), "request_id": payload.get("request_id"),
            "language": "en-US", "messages": [{"role": "user", "content": "Start topic"}]})
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if request_id in self._requests:
                previous, identifier = self._requests[request_id]
                if previous != fingerprint:
                    raise ChatError(409, "chat_request_conflict", "This start request already has different content.")
                return self.poll(identifier)
            if not self._models()[model_id]['ready']:
                raise ChatError(503, "chat_model_unavailable", "The selected coaching model is unavailable.")
            job = ChatJob(model_id, status='ready', text=TOPICS[topic_id]['opening'], metadata={
                'input': 'topic', 'language': 'en-US', 'source': 'topic_catalog', 'topic_id': topic_id,
                'guidance': guidance(topic_id)})
            self._jobs[job.identifier] = job
            self._requests[request_id] = (fingerprint, job.identifier)
            self._prune()
            return self.poll(job.identifier)

    def submit_speech(self, payload):
        allowed = {"model_id", "request_id", "audio_base64", "history", "practice_id", "topic_id", "correction_job_id"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ChatError(400, "invalid_speech", "Send a recording or audio file, not a text message.")
        history = payload.get("history", [])
        if not isinstance(history, list) or len(history) % 2:
            raise ChatError(400, "invalid_speech_history", "The conversation history is invalid.")
        model_id, request_id, cleaned = self._validate({"model_id": payload.get("model_id"), "request_id": payload.get("request_id"),
            "language": "en-US", "messages": [*history, {"role": "user", "content": "Audio recording"}]})
        practice_id = payload.get("practice_id")
        if practice_id is not None and (not isinstance(practice_id, str) or practice_id not in PRACTICE_PROMPTS):
            raise ChatError(400, "invalid_practice_prompt", "Choose a reading sentence from the list.")
        topic_id = payload.get('topic_id')
        if topic_id is not None and (not isinstance(topic_id, str) or topic_id not in TOPICS or practice_id is not None):
            raise ChatError(400, "invalid_topic", "Choose a speaking topic or a reading sentence.")
        corrected_reference = None
        correction_job_id = payload.get('correction_job_id')
        if correction_job_id is not None:
            if not isinstance(correction_job_id, str) or not ID.fullmatch(correction_job_id) or practice_id is not None:
                raise ChatError(400, 'invalid_correction_target', 'Choose one completed sentence to practise.')
            with self._lock:
                source = self._job(correction_job_id)
                correction = (source.speech or {}).get('correction') or {}
                if (source.status != 'ready' or source.model_id != model_id or source.metadata.get('topic_id') != topic_id
                        or correction.get('status') != 'ready' or correction.get('mode') != 'completion_only'
                        or not isinstance(correction.get('corrected'), str)
                        or not 0 < len(correction['corrected'].strip()) <= 1000):
                    raise ChatError(409, 'correction_target_unavailable', 'That completed sentence is not available in this conversation.')
                corrected_reference = correction['corrected'].strip()
        encoded = payload.get("audio_base64")
        if not isinstance(encoded, str) or not 0 < len(encoded) <= 1400000:
            raise ChatError(413, "speech_too_large", "Use a recording of up to 30 seconds.")
        try:
            audio = base64.b64decode(encoded, validate=True)
            if len(audio) > 1024 * 1024 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                raise ValueError("Not a bounded WAV")
            with wave.open(io.BytesIO(audio)) as stream:
                if (stream.getframerate() != 16000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2
                        or not 4000 <= stream.getnframes() <= 480000 or stream.getcomptype() != "NONE"):
                    raise ValueError("Invalid WAV format")
                if len(stream.readframes(stream.getnframes())) != stream.getnframes() * 2:
                    raise ValueError("Incomplete WAV")
        except (binascii.Error, ValueError, wave.Error, EOFError):
            raise ChatError(400, "invalid_speech_audio", "Use a nonempty 16 kHz mono PCM WAV, between 0.25 and 30 seconds.") from None
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if request_id in self._requests:
                previous, identifier = self._requests[request_id]
                if previous != fingerprint:
                    raise ChatError(409, "chat_request_conflict", "This recording ID already has different content.")
                return self.poll(identifier)
            model = self._models()[model_id]
            if not model["ready"] or (model["adapter"] and hashlib.sha256((model["adapter"] / "adapter_model.safetensors").read_bytes()).hexdigest() != model.get("expected_sha256")):
                raise ChatError(503, "speech_model_unavailable", "The selected coaching model is unavailable.")
            if not self.speech_generator and not self.config()["speech_ready"]:
                raise ChatError(503, "speech_unavailable", "The local speech recognition models are not ready.")
            if not self._generation_slot.acquire(blocking=False):
                raise ChatError(409, "chat_busy", "Another recording is being processed. Please wait.")
            job = ChatJob(model_id, status="transcribing", speech={"coaching": {"status": "pending"}},
                          metadata={'reference_revision': 0},
                          speech_context={'history': deepcopy(cleaned[1:-1]), 'topic_id': topic_id,
                                          'model_id': model_id, 'rewrite_enabled': True})
            self._jobs[job.identifier] = job
            self._requests[request_id] = (fingerprint, job.identifier)
            self._prune()
        request = {"audio_base64": encoded, "history": cleaned[1:-1],
                   'rewrite_enabled': True,
                   "reference_text": corrected_reference or PRACTICE_PROMPTS.get(practice_id), "model_id": model_id, "topic_id": topic_id,
                   'reference_source': 'corrected_sentence' if corrected_reference else 'reading_prompt' if practice_id else 'asr'}
        threading.Thread(target=self._generate_speech, args=(job, request), daemon=True, name="local-speech-job").start()
        return self.poll(job.identifier)

    def reference(self, payload):
        """Reassess the recording; replace its reply only if spoken words change."""
        if not isinstance(payload, dict) or set(payload) != {'job_id', 'reference_text'}:
            raise ChatError(400, 'invalid_chat_reference', 'Gửi mã bản thu và câu tiếng Anh cần đối chiếu.')
        with self._lock:
            job = self._job(payload['job_id'])
            if job.status != 'ready' or not isinstance(job.speech, dict):
                raise ChatError(409, 'chat_reference_not_ready', 'Chờ bản thu xử lý xong rồi xác nhận câu đối chiếu.')
            previous = job.speech
            original = deepcopy(previous)
        from .chat_reference import reassess_reference
        try:
            speech = reassess_reference(self.root, original, payload['reference_text'])
        except ValueError as error:
            raise ChatError(400, 'invalid_chat_reference', str(error)) from None
        with self._lock:
            if self._job(job.identifier).speech is not previous:
                raise ChatError(409, 'chat_reference_changed', 'Câu đối chiếu đã thay đổi; hãy thử lại.')
            if job.status != 'ready':
                raise ChatError(409, 'chat_reference_not_ready', 'Chờ lượt hiện tại xử lý xong rồi đối chiếu lại.')
            speech['conversation_text'] = speech['reference_text']
            previous_text = original.get('conversation_text') or original.get('reference_text') or original.get('transcript')
            changed = spoken_text_key(speech['reference_text']) != spoken_text_key(previous_text)
            if not changed:
                # Confirmation alone can update pronunciation provenance. It
                # must not interrupt playback, discard a useful reply or spend
                # another Qwen generation on the same spoken words.
                job.speech = speech
                sentence = str(speech.get('reference_text') or speech.get('transcript') or '').strip()
                if job.sentence_voice_text != sentence:
                    job.sentence_voice = {'status': 'not_requested'}
                    job.sentence_voice_text = None
                    job.sentence_voice_options = {}
                job.metadata['pronunciation_reference_source'] = 'user_confirmed'
                response = self.poll(job.identifier)
                response['reference_update'] = {'changed': False, 'regenerated_reply': False}
                return response
            if not self._generation_slot.acquire(blocking=False):
                raise ChatError(409, 'chat_busy', 'Đang xử lý một lượt nói khác. Bạn chờ xong rồi thử lại.')
            for key in ('spoken_response', 'follow_up', 'guidance'):
                speech.pop(key, None)
            speech['rewrite'] = {'status': 'pending', 'mode': 'grammar_and_natural', 'original': speech['conversation_text']}
            speech['correction'] = {'status': 'not_needed', 'mode': 'completion_only', 'original': speech['conversation_text']}
            speech['transcript_clarity'] = {'needs_clarification': False, 'reason': None}
            revision = job.generation_revision + 1
            job.reply_revisions.append({'reference_revision': job.generation_revision, 'text': job.text,
                                        'metadata': deepcopy(job.metadata)})
            job.reply_revisions = job.reply_revisions[-4:]
            job.generation_revision = revision
            job.speech = speech
            job.text = ''
            job.error = None
            job.cancelled = threading.Event()
            job.status = 'generating'
            job.created = time.monotonic()
            job.voice = {'status': 'not_requested'}
            job.voice_locale = None
            job.sentence_voice = {'status': 'not_requested'}
            job.sentence_voice_text = None
            job.voice_options = {}
            job.sentence_voice_options = {}
            self._voices = {key: value for key, value in self._voices.items() if value[0] != job.identifier}
            context = deepcopy(job.speech_context or {'history': [], 'topic_id': job.metadata.get('topic_id'),
                                                      'model_id': job.model_id, 'rewrite_enabled': True})
            job.metadata = {'reference_revision': revision, 'pronunciation_reference_source': 'user_confirmed',
                            'regenerated_from_reference': True, 'input': 'audio', 'language': 'en-US',
                            'topic_id': context.get('topic_id')}
            request = {**context, 'operation': 'regenerate_reference', 'dialogue_text': speech['conversation_text'],
                       'reference_source': 'user_confirmed', 'reference_revision': revision,
                       'speech': {key: deepcopy(value) for key, value in speech.items() if not key.startswith('_')}}
            # Return the accepted empty-reply revision even when the worker is
            # fast. The client resets its old voice/text before polling chunks.
            accepted = self.poll(job.identifier)
            accepted['reference_update'] = {'changed': True, 'regenerated_reply': True}
        try:
            threading.Thread(target=self._generate_speech, args=(job, request), daemon=True,
                             name='local-reference-reply').start()
        except Exception:
            self._generation_slot.release()
            with self._lock:
                job.status = 'error'
                job.error = {'code': 'reference_reply_failed', 'message': 'Chưa khởi động được câu trả lời mới. Bạn thử một lượt nói mới.'}
            failed = self.poll(job.identifier)
            failed['reference_update'] = {'changed': True, 'regenerated_reply': True}
            return failed
        return accepted

    def _generate_speech(self, job, request):
        revision = job.generation_revision
        cancelled = job.cancelled
        def current():
            return job.generation_revision == revision and job.cancelled is cancelled
        try:
            def stage(name):
                with self._lock:
                    if current() and not cancelled.is_set() and name in {"transcribing", "analyzing", "correcting", "generating", "coaching"}:
                        job.status = name
            def speech(value):
                with self._lock:
                    if current() and not cancelled.is_set():
                        private = {key: item for key, item in (job.speech or {}).items() if key.startswith('_')}
                        job.speech = {**value, **private}
            def text(chunk):
                with self._lock:
                    if current() and not cancelled.is_set():
                        job.text = (job.text + chunk)[:12000]
            regenerate = request.get('operation') == 'regenerate_reference'
            runner = self.reference_generator if regenerate else self.speech_generator
            if runner is None:
                self._runner = self._module("local_chat_process")
                runner = self._runner.regenerate_reference if regenerate else self._runner.speech
            metadata = runner(request, on_stage=stage, on_speech=speech, on_text=text, cancelled=cancelled)
            with self._lock:
                if not current(): return
                stable = {key: value for key, value in job.metadata.items() if key in {
                    'reference_revision', 'pronunciation_reference_source', 'regenerated_from_reference'}}
                job.metadata = {**metadata, **stable}
                job.status = "cancelled" if cancelled.is_set() else "ready" if job.text.strip() else "error"
                if job.status == "error":
                    job.error = {"code": "empty_speech_reply", "message": "No reply was generated. Please try a shorter recording."}
        except Exception:
            with self._lock:
                if current():
                    job.status = "cancelled" if cancelled.is_set() else "error"
                    job.error = {"code": "speech_failed", "message": "Could not finish this reply. Check the displayed sentence or try another recording."}
        finally:
            self._generation_slot.release()

    def _prune(self):
        while len(self._jobs) > 32:
            key = next((key for key, j in self._jobs.items() if j.status in {"ready", "error", "cancelled"}
                        and j.voice.get("status") != "generating" and j.sentence_voice.get("status") != "generating"), None)
            if key is None:
                break
            self._jobs.pop(key)
            self._requests = {k: v for k, v in self._requests.items() if v[1] != key}
            self._voices = {k: v for k, v in self._voices.items() if v[0] != key}

    def _generate(self, job, model, messages):
        try:
            settings = {"path": self.root / "models/pretrained/Qwen3-0.6B", "adapter": model["adapter"],
                        "device": "cpu", "threads": 2, "context": 4096}
            def ready():
                with self._lock:
                    job.status = "generating"
            def text(chunk):
                with self._lock:
                    if not job.cancelled.is_set():
                        job.text = (job.text + chunk)[:12000]
            if self.generator:
                generator = self.generator
            else:
                self._runner = self._module("local_chat_process")
                generator = self._runner.complete
            result = generator(messages, settings, 384, 120., on_text=text, cancelled=job.cancelled, on_ready=ready)
            with self._lock:
                job.metadata = result
                job.status = "cancelled" if job.cancelled.is_set() else "ready" if job.text.strip() else "error"
                if job.status == "error":
                    job.error = {"code": "empty_reply", "message": "Qwen chưa tạo được câu trả lời. Bạn có thể thử một câu ngắn hơn."}
        except Exception as error:
            with self._lock:
                job.status = "cancelled" if job.cancelled.is_set() else "error"
                message = "Tin nhắn vượt ngữ cảnh của Qwen; hãy rút ngắn hoặc tạo cuộc trò chuyện mới." if str(error) == "chat_context_too_long" else "Chưa chạy được Qwen local. Kiểm tra trọng số và bộ nhớ của máy chủ rồi thử lại."
                job.error = {"code": "local_chat_failed", "message": message}
        finally:
            self._generation_slot.release()

    def _job(self, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier) or identifier not in self._jobs:
            raise ChatError(404, "chat_not_found", "Lượt chat đã hết hạn hoặc không tồn tại.")
        return self._jobs[identifier]

    def poll(self, identifier):
        with self._lock:
            job = self._job(identifier)
            # The server retains compressed posteriors to reassess a confirmed
            # reference. They are never part of the public poll/API response.
            speech = None if job.speech is None else {key: value for key, value in job.speech.items() if not key.startswith('_')}
            return {"id": job.identifier, "model_id": job.model_id, "status": job.status, "text": job.text,
                    "elapsed_sec": round(time.monotonic() - job.created, 2), "metadata": deepcopy(job.metadata),
                    "error": deepcopy(job.error), "voice": deepcopy(job.voice),
                    "sentence_voice": deepcopy(job.sentence_voice), "speech": deepcopy(speech)}

    def cancel(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"job_id"}:
            raise ChatError(400, "invalid_chat_cancel", "Mã lượt chat không hợp lệ.")
        with self._lock:
            job = self._job(payload["job_id"])
            if job.status in {"loading", "generating", "transcribing", "analyzing", "correcting", "coaching"}:
                job.cancelled.set()
                job.status = "cancelling"
            return self.poll(job.identifier)

    def voice(self, payload):
        if (not isinstance(payload, dict) or set(payload) - {"job_id", "locale", "target", "voice_id", "speed"}
                or not {"job_id", "locale"} <= set(payload) or payload.get("locale") not in ("vi-VN", "en-US")
                or payload.get("target", "reply") not in ("reply", "sentence")):
            raise ChatError(400, "invalid_chat_voice", "Yêu cầu giọng đọc không hợp lệ.")
        target = payload.get("target", "reply")
        field = "sentence_voice" if target == "sentence" else "voice"
        option_field = 'sentence_voice_options' if target == 'sentence' else 'voice_options'
        options = {}
        if 'voice_id' in payload:
            if payload['locale'] != 'en-US' or not any(v['id'] == payload['voice_id'] for v in self.english_voices()):
                raise ChatError(400, 'invalid_chat_voice', 'Chọn một giọng tiếng Anh trong danh sách.')
            options['voice_id'] = payload['voice_id']
        if 'speed' in payload:
            speed = payload['speed']
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not .5 <= speed <= 1.5:
                raise ChatError(400, 'invalid_chat_voice', 'Tốc độ đọc phải trong khoảng 0.5–1.5.')
            options['speed'] = speed
        with self._lock:
            job = self._job(payload["job_id"])
            if (target == "sentence" or job.speech is not None or job.metadata.get('input') == 'topic') and payload["locale"] != "en-US":
                raise ChatError(400, "speech_voice_language", "Speech practice uses an English voice.")
            if job.status != "ready":
                raise ChatError(409, "chat_not_ready", "Chờ Qwen trả lời xong rồi tạo giọng đọc.")
            spoken = job.text
            if target == "sentence":
                speech = job.speech or {}
                spoken = str(speech.get("reference_text") or speech.get("transcript") or "").strip()
                if not spoken or len(spoken) > 5000:
                    raise ChatError(400, "sentence_voice_unavailable", "Chưa có câu để đọc mẫu.")
                if job.sentence_voice_text != spoken:
                    job.sentence_voice = {"status": "not_requested"}
                    job.sentence_voice_text = spoken
            voice = getattr(job, field)
            resuming = voice["status"] == "error" and voice.get("retryable") is True
            if target == "reply" and job.voice_locale is not None and job.voice_locale != payload["locale"]:
                raise ChatError(409, "chat_voice_locale", "Lượt này đã chọn ngôn ngữ giọng đọc khác.")
            if voice['status'] != 'not_requested' and getattr(job, option_field) != options:
                raise ChatError(409, 'chat_voice_selection', 'Bản đọc này đã dùng giọng và tốc độ khác. Lựa chọn mới áp dụng cho lượt tiếp theo.')
            if voice["status"] != "not_requested" and not resuming:
                return self.poll(job.identifier)
            caps = self.capability_loader()
            key = "vietnamese_voice_configured" if payload["locale"] == "vi-VN" else "english_voice_configured"
            if not caps.get("tts_configured") or not (caps.get(key) or options.get('voice_id')):
                raise ChatError(503, "chat_voice_unavailable", "Giọng ai33 cho ngôn ngữ này chưa được cấu hình.")
            if not self._voice_slot.acquire(blocking=False):
                raise ChatError(409, "chat_voice_busy", "Đang tạo một giọng đọc khác; hãy thử lại sau.")
            locale = payload["locale"]
            if target == "reply":
                job.voice_locale = locale
            revision = job.generation_revision
            setattr(job, option_field, options)
            setattr(job, field, {"status": "generating", 'reference_revision': revision, **options})
        threading.Thread(target=self._synthesize, args=(job, revision, spoken, locale, resuming, target, options), daemon=True, name="chat-ai33-voice").start()
        return self.poll(job.identifier)

    def _synthesize(self, job, revision, spoken, locale, resuming=False, target="reply", voice_options=None):
        client = None
        voice_options = voice_options or {}
        field = "sentence_voice" if target == "sentence" else "voice"
        def current():
            return (revision == job.generation_revision
                    and (target != "sentence" or job.sentence_voice_text == spoken))
        try:
            client = self.tts_factory() if self.tts_factory else self._module("ai33_tts").AI33TTS(cache_dir=self.cache_dir)
            action = client.resume if resuming else client.synthesize
            path = Path(action(spoken[:5000], locale=locale, **voice_options)).resolve()
            self._read_voice(path)
            with self._lock:
                if not current():
                    return
                identifier = secrets.token_hex(16)
                self._voices[identifier] = (job.identifier, path)
                setattr(job, field, {"status": "ready", "url": "/api/chat/audio/" + identifier, 'reference_revision': revision, **voice_options})
        except Exception as error:
            code = getattr(error, "code", "voice_local_error")
            if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,60}", code):
                code = "voice_local_error"
            retryable = (code != "ai33_task_failed" and client is not None
                         and callable(getattr(client, "can_resume", None))
                         and callable(getattr(client, "resume", None))
                         and client.can_resume(spoken[:5000], locale=locale, **voice_options) is True)
            # Only typed error identifiers are logged, never raw provider errors,
            # credentials, signed CDN URLs or learner/conversation text.
            logging.getLogger(__name__).warning("Chat voice failed: job=%s code=%s exception=%s cached_resume=%s",
                job.identifier, code, type(error).__name__, retryable)
            with self._lock:
                if current():
                    setattr(job, field, {"status": "error", "code": code, "retryable": retryable, **voice_options,
                                 "message": "Chưa tải được giọng đọc. Bấm Thử lại giọng đọc để tải tiếp tác vụ đã có."
                                    if retryable else "Chưa tạo được giọng đọc. Bạn có thể chọn giọng trình duyệt trong Settings."})
        finally:
            self._voice_slot.release()

    def _read_voice(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.cache_dir.resolve()) or not path.is_file() or not 0 < path.stat().st_size <= 25 * 1024 * 1024:
            raise ChatError(404, "chat_voice_not_found", "Không tìm thấy giọng đọc.")
        data = path.read_bytes()
        if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 255 and data[1] & 224 == 224):
            mime = "audio/mpeg"
        elif data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            mime = "audio/wav"
        elif data.startswith(b"OggS"):
            mime = "audio/ogg"
        else:
            raise ChatError(404, "chat_voice_not_found", "Định dạng giọng đọc chưa được hỗ trợ.")
        return data, mime

    def audio(self, identifier):
        with self._lock:
            if not isinstance(identifier, str) or not ID.fullmatch(identifier) or identifier not in self._voices:
                raise ChatError(404, "chat_voice_not_found", "Không tìm thấy giọng đọc.")
            _, path = self._voices[identifier]
        return self._read_voice(path)

    def close(self):
        with self._lock:
            for job in self._jobs.values(): job.cancelled.set()
        if self._runner:
            self._runner.shutdown()
