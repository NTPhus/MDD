"""Grounded, bounded feedback and voice delivery for the local demo.

Only server-owned inference snapshots enter this adapter. Provider failures are
represented by fixed public messages; credentials, receipts and CDN URLs stay
inside the provider adapters and their local cache.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import threading
from typing import Any


MAX_ANALYSES = 64
MAX_VOICE_BYTES = 25 * 1024 * 1024
OPAQUE_ID = re.compile(r"[0-9a-f]{32}\Z")
_SETTINGS_LOCK = threading.Lock()
_PROJECT = Path(__file__).resolve().parents[2]

ARPABET_IPA = {
    "AA": "ɑ", "AE": "æ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def _module(name):
    # Resolve project adapters from this installed source tree, never a path
    # supplied by an HTTP request or an uploaded audio file.
    if str(_PROJECT) not in sys.path:
        sys.path.insert(0, str(_PROJECT))
    return importlib.import_module("scripts." + name)


def _phone(value):
    return value is None or (isinstance(value, str) and bool(value) and value.strip() == value)


def build_assessment(result: dict, analysis_id: str) -> tuple[dict, list[str]]:
    """Translate canonical comparison events, without reading gold perception.

    If any event cannot be assigned to a known word, retain the comparison in
    the original result and withhold a word-level diagnosis from the LLM.
    """
    text = str(result.get("reference_text", "")).strip()
    words = text.split()
    if not words:
        raise ValueError("A reference sentence is required")
    assessment = {
        "schema_version": "1.0", "data_kind": "inference", "utterance_id": analysis_id,
        "model_id": str(result.get("model_id") or result.get("model", {}).get("id") or "unavailable"),
        "status": "uncertain", "target_locale": "en-US", "reference_text": text,
        "asr_text": None, "confidence_calibrated": False,
        "words": [{"index": index, "text": word} for index, word in enumerate(words)], "errors": [],
    }
    unavailable = "Chưa đủ đối chiếu âm vị theo từ để tạo hướng dẫn; hãy dùng checkpoint đã huấn luyện và câu mẫu có âm chuẩn."
    model, comparison = result.get("model", {}), result.get("comparison")
    if (model.get("kind") != "phoneme" or model.get("experimental") is not False
            or not result.get("phonemes") or not isinstance(comparison, dict)):
        return assessment, [unavailable]
    compared_words, events = comparison.get("words"), comparison.get("events")
    if (not isinstance(compared_words, list) or len(compared_words) != len(words)
            or not isinstance(events, list)
            or any(not isinstance(word, dict) or word.get("text") != text_word
                   or word.get("status") not in {"match", "difference"}
                   for word, text_word in zip(compared_words, words))):
        return assessment, ["Chưa xác định chắc vị trí từ của đối chiếu; hệ thống không tự gán lỗi sang từ khác."]
    errors = []
    for event in events:
        if not isinstance(event, dict):
            return assessment, [unavailable]
        index = event.get("word_index")
        kind = "addition" if event.get("kind") == "insertion" else event.get("kind")
        expected, observed = event.get("expected_phone"), event.get("observed_phone")
        valid_phones = (_phone(expected) and _phone(observed)
                        and ((kind == "substitution" and expected is not None and observed is not None)
                             or (kind == "deletion" and expected is not None and observed is None)
                             or (kind == "addition" and expected is None and observed is not None)))
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(words)
                or not valid_phones):
            return assessment, ["Có sai khác âm vị chưa gắn được chắc chắn với từ; chưa gửi chẩn đoán này cho dịch vụ ngoài."]
        errors.append({"id": f"e{len(errors) + 1}", "word_index": index, "type": kind,
                       "expected_phone": expected, "observed_phone": observed,
                       "expected_ipa": ARPABET_IPA.get(expected), "observed_ipa": ARPABET_IPA.get(observed),
                       "start_sec": None, "end_sec": None, "confidence": None})
    assessment.update(status="ok", errors=errors)
    return assessment, ["Sai khác âm vị do model ước lượng; hướng dẫn chưa thay thế đánh giá của giáo viên."]


class FeedbackNotFound(LookupError):
    pass


class VoiceNotFound(LookupError):
    pass


@dataclass
class _Analysis:
    identifier: str
    result: dict
    assessment: dict
    notices: list[str]
    lock: Any = field(default_factory=threading.Lock)
    feedback: dict | None = None
    source: str = "template"
    voice_attempted: bool = False
    audio: list[dict] = field(default_factory=list)


class FeedbackService:
    def __init__(self, root: Path, *, llm_factory=None, tts_factory=None, max_analyses=MAX_ANALYSES):
        if not isinstance(max_analyses, int) or isinstance(max_analyses, bool) or not 1 <= max_analyses <= MAX_ANALYSES:
            raise ValueError("The analysis cache must contain 1 to 64 results")
        self.root = Path(root).resolve()
        self.cache_dir = self.root / "data/generated/tts/ai33"
        self.llm_factory, self.tts_factory = llm_factory, tts_factory
        self.max_analyses = max_analyses
        self._guard = threading.Lock()
        self._voice_lock = threading.Lock()
        self._analyses: OrderedDict[str, _Analysis] = OrderedDict()
        self._voices: dict[str, tuple[str, Path]] = {}

    def capabilities(self) -> dict:
        valid_settings = True
        with _SETTINGS_LOCK:
            try:
                # A server may start before .env has been filled in. Empty
                # placeholders must not prevent later configuration from loading.
                for key, value in list(os.environ.items()):
                    if key.startswith(("LLM_", "AI33_")) and not value.strip():
                        os.environ.pop(key, None)
                _module("llm_client").load_env(self.root / ".env")
                _module("ai33_tts").load_env(self.root / ".env")
            except (OSError, ValueError, RuntimeError):
                valid_settings = False
            return {
                "llm_configured": bool(self.llm_factory or (valid_settings and _module("llm_client").is_configured())),
                "tts_configured": bool(self.tts_factory or (valid_settings and os.environ.get("AI33_API_KEY", "").strip())),
                "english_voice_configured": bool(self.tts_factory or os.environ.get("AI33_VOICE_EN", "").strip()),
                "vietnamese_voice_configured": bool(self.tts_factory or os.environ.get("AI33_VOICE_VI", "").strip()),
            }

    def register(self, result: dict) -> dict:
        identifier = secrets.token_hex(16)
        snapshot = deepcopy(result)
        assessment, notices = build_assessment(snapshot, identifier)
        llm = _module("llm_client")
        _module("validate_contracts").validate_pair(assessment, llm.fallback_feedback(assessment))
        entry = _Analysis(identifier, snapshot, assessment, notices)
        with self._guard:
            self._analyses[identifier] = entry
            while len(self._analyses) > self.max_analyses:
                removed, _ = self._analyses.popitem(last=False)
                self._voices = {key: value for key, value in self._voices.items() if value[0] != removed}
        eligible = assessment["status"] == "ok"
        return {"analysis_id": identifier, "assessment": deepcopy(assessment), "feedback_available": eligible,
                "feedback_unavailable_reason": None if eligible else notices[0]}

    def _entry(self, identifier):
        with self._guard:
            entry = self._analyses.get(identifier)
        if entry is None:
            raise FeedbackNotFound("Analysis not found")
        return entry

    def feedback(self, identifier: str, include_voice: bool = True) -> dict:
        entry = self._entry(identifier)
        with entry.lock:
            llm = _module("llm_client")
            validate = _module("validate_contracts").validate_pair
            fallback = llm.fallback_feedback(entry.assessment)
            caps = self.capabilities()
            notices = list(entry.notices)
            if entry.assessment["status"] != "ok":
                return self._response(entry, fallback, "unavailable", "unavailable" if include_voice else "not_requested", [], notices)
            if entry.feedback is None:
                if not caps["llm_configured"]:
                    notices.append("Hướng dẫn tự động chưa sẵn sàng; kết quả phân tích âm vị vẫn được giữ.")
                    return self._response(entry, fallback, "unavailable", "unavailable" if include_voice else "not_requested", [], notices)
                try:
                    client = self.llm_factory() if self.llm_factory else llm.LLMClient()
                    generated = client.generate_feedback(deepcopy(entry.assessment))
                    candidate = generated.feedback
                    validate(entry.assessment, candidate)
                    if candidate["language"] != "vi":
                        raise ValueError("Vietnamese feedback is required")
                    entry.feedback = deepcopy(candidate)
                    entry.source = "llm" if generated.source == "llm" else "template"
                    self._record_diagnostics(entry, generated=generated)
                except Exception as error:
                    # Never return provider exception strings: they may contain
                    # local paths, credentials or signed download URLs.
                    entry.feedback, entry.source = fallback, "template"
                    self._record_diagnostics(entry, error=error)
                if entry.source != "llm":
                    entry.notices.append("Đang dùng hướng dẫn mẫu vì dịch vụ giải thích chưa trả về phản hồi hợp lệ.")
                notices = list(entry.notices)
            audio, voice_status = [], "not_requested"
            if include_voice:
                if not caps["tts_configured"]:
                    voice_status = "unavailable"
                    notices.append("Giọng đọc chưa sẵn sàng; bạn vẫn có thể xem hướng dẫn bằng chữ.")
                else:
                    if not entry.voice_attempted:
                        entry.voice_attempted = True
                        self._create_audio(entry, caps)
                    audio = deepcopy(entry.audio)
                    ready = sum(item["status"] == "ready" for item in audio)
                    voice_status = "ready" if audio and ready == len(audio) else "partial" if ready else "unavailable"
                    if voice_status != "ready":
                        notices.append("Một phần giọng đọc chưa tạo được; hướng dẫn bằng chữ vẫn được giữ. Tác vụ không tự gửi lại để tránh tính phí trùng.")
            status = "ready" if entry.source == "llm" and voice_status in {"ready", "not_requested"} else "partial"
            return self._response(entry, entry.feedback, status, voice_status, audio, notices)

    def _record_diagnostics(self, entry, *, generated=None, error=None):
        """Save provider metadata locally without affecting feedback or playback."""
        try:
            if error is not None:
                metadata = {"feedback_source": "template", "fallback_reason": "feedback_generation_failed",
                            "error_type": type(error).__name__, "validation_errors": [], "completions": []}
            else:
                getter = getattr(generated, "metadata", None)
                if not callable(getter):
                    return
                source = getter(deepcopy(entry.assessment))
                if not isinstance(source, dict):
                    return
                metadata = {key: source.get(key) for key in ("feedback_source", "fallback_reason", "validation_errors")}
                allowed = {"requested_model", "response_model", "request_id", "finish_reason", "latency_sec"}
                completions = []
                for item in source.get("completions", []):
                    if not isinstance(item, dict):
                        continue
                    completion = {key: item[key] for key in allowed if key in item}
                    # Usage metadata contains counts; never copy arbitrary text
                    # or content-bearing extensions from a gateway envelope.
                    if isinstance(item.get("usage"), dict):
                        completion["usage"] = {key: value for key, value in item["usage"].items()
                                               if isinstance(value, (int, float)) and not isinstance(value, bool)
                                               and math.isfinite(value)}
                    completions.append(completion)
                metadata["completions"] = completions
            metadata.update(analysis_id=entry.identifier, data_kind=entry.assessment["data_kind"])
            private_values = [os.environ.get(name, "").strip() for name in ("LLM_API_KEY", "AI33_API_KEY")]

            def redact(value):
                if isinstance(value, str):
                    for private in private_values:
                        if private:
                            value = value.replace(private, "[redacted]")
                    return value
                if isinstance(value, dict):
                    return {key: redact(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [redact(item) for item in value]
                return value

            directory = self.root / "data/generated/demo/feedback"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{entry.identifier}.meta.json").write_text(
                json.dumps(redact(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except Exception:
            # Diagnostic metadata must never turn a successful result into a
            # fallback or block voice generation when local storage is full.
            pass

    @staticmethod
    def _response(entry, feedback, status, voice_status, audio, notices):
        return {"analysis_id": entry.identifier, "assessment": deepcopy(entry.assessment),
                "feedback": deepcopy(feedback), "feedback_source": entry.source,
                "status": status, "voice_status": voice_status, "audio": audio,
                "notices": list(dict.fromkeys(notices))}

    @staticmethod
    def _voice_segments(entry):
        feedback, assessment = entry.feedback, entry.assessment
        instruction = next((item["text"] for item in feedback["tts_segments"]
                            if item["language"] == "vi-VN" and item["purpose"] == "feedback"), None)
        if instruction is None:
            instruction = " ".join([feedback["summary"], *[tip["instruction"] for tip in feedback["tips"][:1]]])
        instruction = " ".join(instruction.split())
        if len(instruction) > 1200:
            instruction = instruction[:1200].rsplit(" ", 1)[0]
        segments = [{"language": "vi-VN", "purpose": "feedback", "text": instruction},
                    {"language": assessment["target_locale"], "purpose": "reference", "text": assessment["reference_text"]}]
        practice = []
        for error in assessment["errors"]:
            word = assessment["words"][error["word_index"]]["text"].strip(".,!?;:\"'()[]{}")
            if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word) and word.lower() not in {item.lower() for item in practice}:
                practice.append(word)
            if len(practice) == 3:
                break
        if practice:
            segments.append({"language": assessment["target_locale"], "purpose": "practice", "text": ". ".join(practice) + "."})
        return segments

    def _create_audio(self, entry, caps):
        segments = self._voice_segments(entry)
        try:
            client = self.tts_factory() if self.tts_factory else _module("ai33_tts").AI33TTS(cache_dir=self.cache_dir)
        except Exception:
            client = None
        # The provider's disk cache is single-writer. Serializing all voice
        # submissions also protects reuse of the same text across analyses.
        with self._voice_lock:
            for segment in segments:
                item = {"id": secrets.token_hex(16), **segment, "url": None, "status": "error"}
                configured = caps["vietnamese_voice_configured"] if segment["language"] == "vi-VN" else caps["english_voice_configured"]
                if client is not None and configured:
                    try:
                        path = Path(client.synthesize(segment["text"], locale=segment["language"]))
                        self._read_audio(path)
                        with self._guard:
                            if self._analyses.get(entry.identifier) is not entry:
                                raise VoiceNotFound("Analysis expired")
                            self._voices[item["id"]] = (entry.identifier, path.resolve())
                        item.update(status="ready", url="/api/voice/" + item["id"])
                    except Exception:
                        item["message"] = "Chưa tạo được đoạn âm thanh này; kết quả chữ vẫn được giữ."
                else:
                    item["message"] = "Giọng đọc này hiện chưa sẵn sàng."
                entry.audio.append(item)

    def _read_audio(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.cache_dir.resolve()) or not path.is_file():
            raise VoiceNotFound("Unknown voice asset")
        with path.open("rb") as stream:
            data = stream.read(MAX_VOICE_BYTES + 1)
        if not data or len(data) > MAX_VOICE_BYTES:
            raise VoiceNotFound("Invalid voice asset")
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            mime = "audio/wav"
        elif data.startswith(b"ID3") or (len(data) > 1 and data[0] == 255 and data[1] & 224 == 224):
            mime = "audio/mpeg"
        elif data.startswith(b"OggS"):
            mime = "audio/ogg"
        elif data.startswith(b"fLaC"):
            mime = "audio/flac"
        elif data[4:8] == b"ftyp":
            mime = "audio/mp4"
        else:
            raise VoiceNotFound("Unsupported voice asset")
        return data, mime

    def voice(self, identifier):
        with self._guard:
            registered = self._voices.get(identifier)
        if registered is None:
            raise VoiceNotFound("Unknown voice asset")
        try:
            return self._read_audio(registered[1])
        except (OSError, ValueError):
            raise VoiceNotFound("Unknown voice asset") from None
