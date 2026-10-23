"""Loopback-only MDD service with optional grounded LLM and voice feedback."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import platform
import re
import threading
import time
from typing import Any
import unicodedata
from urllib.parse import quote, unquote, urlsplit

from .demo_feedback import FeedbackNotFound, FeedbackService, OPAQUE_ID, VoiceNotFound
from .demo_chat import ChatError, ChatService
from .demo_realtime import practice_reference, prefix_comparison, preview_assessment
from .demo_reference import dictionary_reference


MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_FEEDBACK_BODY_BYTES = 4096
STATIC_FILES = {"/": "index.html", "/index.html": "index.html",
                "/app.js": "app.js", "/styles.css": "styles.css",
                "/chat.js": "chat.js", "/chat.css": "chat.css",
                "/pronunciation.js": "pronunciation.js",
                "/speech_highlight.js": "speech_highlight.js",
                "/recorder-worklet.js": "recorder-worklet.js",
                "/assets/mezon-wordmark.png": "assets/mezon-wordmark.png",
                "/assets/fonts/InterVariable.woff2": "assets/fonts/InterVariable.woff2",
                "/assets/fonts/InterVariable-Italic.woff2": "assets/fonts/InterVariable-Italic.woff2",
                **{f"/assets/topics/{name}.webp": f"assets/topics/{name}.webp" for name in
                   ("daily-life", "travel", "food", "work", "hobbies", "technology", "learning", "city")}}
RFC1918_NETWORKS = tuple(ipaddress.IPv4Network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def validate_lan_hosts(allow_hosts=(), *, lan=False) -> frozenset[str]:
    """Require explicitly listed RFC1918 IPv4 addresses for optional LAN access."""
    if not isinstance(lan, bool):
        raise ValueError("LAN mode must be explicitly enabled or disabled")
    if isinstance(allow_hosts, str):
        raise ValueError("Pass allowed LAN addresses separately, not as one host string")
    try:
        values = tuple(allow_hosts)
    except TypeError:
        raise ValueError("Allowed LAN hosts must be a collection of IPv4 addresses") from None
    if not lan:
        if values:
            raise ValueError("--allow-host requires --lan")
        return frozenset()
    if not values:
        raise ValueError("--lan requires at least one explicit --allow-host RFC1918 IPv4 address")
    addresses = set()
    for value in values:
        try:
            if not isinstance(value, str):
                raise ValueError()
            address = ipaddress.IPv4Address(value)
        except (ipaddress.AddressValueError, ValueError):
            raise ValueError("Each allowed LAN host must be a plain RFC1918 IPv4 address") from None
        if not any(address in network for network in RFC1918_NETWORKS):
            raise ValueError("Allowed LAN hosts must be RFC1918 IPv4 addresses (10/8, 172.16/12 or 192.168/16)")
        addresses.add(str(address))
    return frozenset(addresses)


class DemoError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _json(path: Path) -> dict:
    """A checkpoint directory can be absent briefly while being replaced."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _tail_jsonl(path: Path, limit: int = 180) -> list[dict]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 512 * 1024))
            if stream.tell():
                stream.readline()
            content = stream.read()
        # An in-flight append is not yet an observation.
        lines = content.splitlines(keepends=True)
        rows = []
        for line in lines[-limit:]:
            if not line.endswith(b"\n"):
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except (ValueError, UnicodeError):
                continue
        return rows
    except OSError:
        return []


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _step(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _iso(timestamp: float) -> str | None:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class DemoApplication:
    def __init__(self, root: Path, engine: Any = None, threads: int = 2, *, feedback_service=None, chat_service=None):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
            raise ValueError("Inference threads must be between 1 and 4")
        self.root = Path(root).resolve()
        self.engine = engine
        self.threads = threads
        self.inference_lock = threading.Lock()
        self.feedback_service = feedback_service if feedback_service is not None else FeedbackService(self.root)
        self.chat_service = chat_service if chat_service is not None else ChatService(self.root, capability_loader=self.feedback_service.capabilities)
        self.samples = self._catalog()

    def _catalog(self) -> dict[str, dict]:
        splits = _json(self.root / "data/protocol/l2arctic-v1/splits.json")
        source_ids = set(splits.get("source_train_ids", []))
        manifest = {}
        try:
            with (self.root / "data/protocol/l2arctic-v1/manifest.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("utterance_id") in source_ids:
                        manifest[row["utterance_id"]] = row
            candidates = []
            with (self.root / "data/processed/l2arctic-v1/annotations.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    frozen = manifest.get(row.get("utterance_id"))
                    if not frozen or frozen.get("audio", {}).get("path") != row.get("audio_path"):
                        continue
                    path = self.root / row["audio_path"]
                    text = row.get("prompt_text", "")
                    duration = _finite(row.get("duration_sec"))
                    if (isinstance(text, str) and 3 <= len(text.split()) <= 12
                            and duration is not None and 1 <= duration <= 6):
                        row["_path"] = path
                        row["audio_sha256"] = frozen["audio"].get("sha256")
                        candidates.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            return {}
        candidates.sort(key=lambda row: (abs(row["duration_sec"] - 2.8), row["utterance_id"]))
        selected, speakers, texts = {}, set(), set()
        for row in candidates:
            if row.get("speaker_id") in speakers or row["prompt_text"].lower() in texts:
                continue
            if not _inside(row["_path"], self.root / "l2arctic_release_v5.0") or not row["_path"].is_file():
                continue
            selected[row["utterance_id"]] = row
            speakers.add(row.get("speaker_id"))
            texts.add(row["prompt_text"].lower())
            if len(selected) == 8:
                break
        return selected

    def prompts(self) -> dict:
        def available(row):
            try:
                practice_reference(row, row["prompt_text"], "sentence")
                return True
            except ValueError:
                return False
        return {"prompts": [{"id": row["utterance_id"], "text": row["prompt_text"],
                             "live_available": available(row),
                             "speaker": row.get("speaker_id"), "duration_sec": row["duration_sec"],
                             "audio_url": "/api/samples/" + quote(row["utterance_id"], safe="")}
                            for row in self.samples.values()]}

    def _run_dirs(self) -> list[Path]:
        runs = self.root / "runs"
        # Existing launchers also use runs/local-checks/<run-id>.
        result = []
        for pattern in ("*", "*/*"):
            for path in runs.glob(pattern):
                if (_inside(path, runs) and path.is_dir() and (path / "config.json").is_file()
                        and _json(path / "config.json").get("stage") in {"A", "B", "C"}):
                    result.append(path)
        return sorted(set(result))

    def _models(self, run_dirs: list[Path]) -> dict[str, dict]:
        models = {}
        base = self.root / "models/pretrained/wav2vec2-base"
        if any((base / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
            models["base"] = {"id": "base", "label": "BASE · Trước fine-tune MDD", "kind": "backbone",
                              "experimental": False, "purpose": "self_supervised_backbone", "path": base,
                              "training_phase": "pretrained", "phase_label": "Trước fine-tune MDD · Chỉ trích xuất đặc trưng",
                              "can_extract_features": True, "can_recognize_phonemes": False, "can_compare_phonemes": False,
                              "ready": True, "warning": "BASE chỉ tạo đặc trưng audio; chưa có head phoneme MDD."}
        for run in run_dirs:
            run_id = run.relative_to(self.root / "runs").as_posix()
            run_stage = _json(run / "config.json").get("stage")
            for suffix, name in (("best", "best_checkpoint"), ("latest", "checkpoint")):
                checkpoint = run / name
                if not _inside(checkpoint, self.root / "runs"):
                    continue
                metadata = _json(checkpoint / "metadata.json")
                weights = next((checkpoint / item for item in ("model.safetensors", "pytorch_model.bin")
                                if (checkpoint / item).is_file()), None)
                if not metadata or weights is None or not (checkpoint / "config.json").is_file():
                    continue
                try:
                    fingerprint = f"{weights.stat().st_mtime_ns}:{weights.stat().st_size}:{_mtime(checkpoint / 'metadata.json')}"
                except OSError:
                    continue
                purpose = metadata.get("purpose", "unknown")
                stage = metadata.get("stage", run_stage)
                scientific = purpose == "scientific_training"
                phases = {"A": ("source", "A · Trước thích ứng Việt · Đã train MDD"),
                          "B": ("adapted_vi", "B · Sau thích ứng Việt"),
                          "C": ("control", "C · Đối chứng học tiếp source")}
                phase, phase_label = phases.get(stage, ("source", "Đã train MDD")) if scientific else (
                    "technical", "Kiểm tra kỹ thuật · Chưa dùng đánh giá")
                identifier = f"run:{run_id}:{suffix}"
                models[identifier] = {"id": identifier, "label": f"{phase_label} · {run_id} · {suffix}",
                                      "kind": "phoneme", "experimental": not scientific,
                                      "purpose": purpose, "step": _step(metadata.get("step")),
                                      "run_id": run_id, "stage": stage,
                                      "training_phase": phase, "phase_label": phase_label,
                                      "can_extract_features": False, "can_recognize_phonemes": True,
                                      "can_compare_phonemes": scientific,
                                      "fingerprint": fingerprint, "path": checkpoint, "ready": True,
                                      "warning": ("Checkpoint smoke kiểm tra kỹ thuật; chưa dùng để chấm phát âm."
                                                  if purpose != "scientific_training" else
                                                  "Dự đoán từ checkpoint local; sai khác phoneme là gợi ý cần kiểm chứng.")}
        return models

    def status(self) -> dict:
        run_dirs = self._run_dirs()
        models = self._models(run_dirs)
        training = []
        for run in run_dirs:
            config, completed = _json(run / "config.json"), _json(run / "completed.json")
            rows = _tail_jsonl(run / "training_log.jsonl")
            validation = _tail_jsonl(run / "validation_log.jsonl", 1)
            last = rows[-1] if rows else {}
            checkpoint = _json(run / "checkpoint/metadata.json")
            updated = max(_mtime(run / name) for name in ("training_log.jsonl", "completed.json", "config.json"))
            recent = rows and 0 <= time.time() - _mtime(run / "training_log.jsonl") < 120
            status = "completed" if completed.get("status") == "completed" else (
                "recent_activity" if recent else "checkpointed" if checkpoint else "initialized")
            best_f1 = None
            if config.get("selection_metric", "f1") == "f1":
                best_f1 = _finite(completed.get("best_validation_value", checkpoint.get("best_validation_value")))
                if best_f1 is None and validation:
                    best_f1 = _finite(validation[-1].get("best_value"))
            training.append({"run_id": run.relative_to(self.root / "runs").as_posix(),
                             "stage": config.get("stage"), "purpose": config.get("purpose"), "status": status,
                             "step": max(_step(last.get("step")), _step(completed.get("optimizer_steps")), _step(checkpoint.get("step"))),
                             "max_steps": _step(config.get("max_steps")), "loss": _finite(last.get("loss")),
                             "best_f1": best_f1, "updated_at": _iso(updated),
                             "history": [{"step": _step(row.get("step")), "loss": _finite(row.get("loss"))} for row in rows]})
        training.sort(key=lambda row: row["updated_at"] or "", reverse=True)
        phoneme = [model for model in models.values() if model["kind"] == "phoneme"]
        # An adapted model serves the pronunciation demo; optimizer-step
        # numbers from different stages do not measure comparable quality.
        stage_priority = {"B": 3, "A": 2, "C": 1}
        phoneme.sort(key=lambda model: (not model["experimental"], stage_priority.get(model.get("stage"), 0),
                                        model["id"].endswith(":best"), model["step"], model["id"]), reverse=True)
        default = phoneme[0]["id"] if phoneme else "base" if "base" in models else None
        public = [{key: value for key, value in model.items() if key not in {"path", "fingerprint"}}
                  for model in models.values()]
        gpu_name = next((_json(run / "environment.json").get("gpu") for run in run_dirs
                         if isinstance(_json(run / "environment.json").get("gpu"), str)), None)
        if gpu_name is None:
            probe = _json(self.root / "data/generated/setup/windows-cuda.json")
            gpu = probe.get("torch", {}).get("gpu", {}) if isinstance(probe.get("torch"), dict) else {}
            gpu_name = gpu.get("name") if isinstance(gpu, dict) else None
        return {"models": public, "default_model_id": default, "inference_device": "cpu", "training": training,
                "capabilities": {**self.feedback_service.capabilities(), "realtime": True,
                                 "practice_modes": ["sentence", "word"], "live_interval_ms": 1200},
                "system": {"platform": platform.system(), "python": platform.python_version(),
                           "offline": True, "inference_threads": self.threads, "sample_count": len(self.samples),
                           "gpu_name": gpu_name}}

    def sample_audio(self, identifier: str) -> bytes:
        row = self.samples.get(identifier)
        if row is None:
            raise DemoError(404, "sample_not_found", "Mẫu audio không có trong danh sách demo.")
        try:
            path = row["_path"]
            if not _inside(path, self.root / "l2arctic_release_v5.0"):
                raise OSError("Sample changed")
            audio = path.read_bytes()
            if len(audio) > MAX_AUDIO_BYTES:
                raise OSError("Sample too large")
            return audio
        except OSError:
            raise DemoError(503, "sample_unavailable", "Audio mẫu tạm thời không đọc được.") from None

    def reference_info(self, body: Any) -> dict:
        if not isinstance(body, dict) or set(body) != {"reference_text"}:
            raise DemoError(400, "invalid_reference", "Chỉ gửi câu tham chiếu.")
        text = body["reference_text"]
        if (not isinstance(text, str) or not text.strip() or len(text) > 1000 or len(text.split()) > 100
                or any(ord(c) < 32 and c not in "\t\r\n" for c in text)):
            raise DemoError(400, "invalid_reference", "Câu tham chiếu cần 1–1.000 ký tự, tối đa 100 từ.")
        row, missing = self._reference(text)
        available = False
        if row:
            try:
                practice_reference(row, text, "sentence")
                available = True
            except ValueError:
                pass
        return {"reference_text": text, "available": available, "missing_words": missing,
                "source": row.get("reference_source", "dataset") if row else None}

    def _reference(self, text: str) -> tuple[dict | None, list[str]]:
        row = next((row for row in self.samples.values()
                    if _normalize_text(text) == _normalize_text(row["prompt_text"])), None)
        return (row, []) if row is not None else dictionary_reference(self.root, text)

    def analyze(self, body: Any, *, live=False) -> dict:
        if not isinstance(body, dict):
            raise DemoError(400, "invalid_request", "Body phải là JSON object.")
        model_id, text = body.get("model_id"), body.get("reference_text")
        mode, final, word_index = "sentence", True, 0
        if live:
            if set(body) - {"model_id", "reference_text", "audio_base64", "mode", "final", "word_index"}:
                raise DemoError(400, "invalid_live_request", "Yêu cầu realtime có trường không được hỗ trợ.")
            mode, final, word_index = body.get("mode", "sentence"), body.get("final", False), body.get("word_index", 0)
            if not isinstance(mode, str) or mode not in {"sentence", "word"} or type(final) is not bool or type(word_index) is not int:
                raise DemoError(400, "invalid_live_request", "Chế độ, vị trí từ hoặc trạng thái chốt không hợp lệ.")
        if not isinstance(model_id, str) or len(model_id) > 300:
            raise DemoError(400, "invalid_model", "Hãy chọn một model local.")
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise DemoError(400, "invalid_reference", "Câu tham chiếu phải có 1–1.000 ký tự.")
        if len(text.split()) > 100 or any(ord(char) < 32 and char not in "\t\n\r" for char in text):
            raise DemoError(400, "invalid_reference", "Câu tham chiếu không hợp lệ hoặc quá dài.")
        has_sample, has_audio = "sample_id" in body, "audio_base64" in body
        if has_sample == has_audio:
            raise DemoError(400, "invalid_audio", "Chọn đúng một nguồn: audio mẫu hoặc WAV tải lên.")
        reference = None
        if has_sample:
            identifier = body["sample_id"]
            if not isinstance(identifier, str):
                raise DemoError(400, "invalid_sample", "ID audio mẫu không hợp lệ.")
            audio = self.sample_audio(identifier)
        else:
            encoded = body["audio_base64"]
            if not isinstance(encoded, str) or len(encoded) > (MAX_AUDIO_BYTES * 4 // 3 + 4):
                raise DemoError(413, "audio_too_large", "WAV tối đa 12 MiB.")
            try:
                audio = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise DemoError(400, "invalid_audio", "Audio base64 không hợp lệ.") from None
            if not audio or len(audio) > MAX_AUDIO_BYTES:
                raise DemoError(400, "invalid_audio", "Audio trống hoặc quá lớn.")
        # A browser recording/upload gets the same canonical prompt lookup as a
        # corpus sample. No perceived labels or gold error events enter inference.
        row, missing = self._reference(text)
        if row is not None:
            reference = {key: row[key] for key in ("prompt_text", "canonical_phones", "canonical_word_indices", "word_intervals") if key in row}
        parent_text = text.strip()
        if live:
            if row is None:
                raise DemoError(409, "reference_unavailable", "Chưa có âm chuẩn cho mọi từ. Hãy chọn câu mẫu hoặc sửa từ chưa có trong từ điển.")
            try:
                text, reference = practice_reference(row, text, mode, word_index)
            except ValueError as error:
                raise DemoError(400, "invalid_word_reference", str(error)) from None
        models = self._models(self._run_dirs())
        model = models.get(model_id)
        if model is None:
            raise DemoError(409, "model_unavailable", "Model đang được cập nhật hoặc không còn sẵn; tải lại danh sách và thử lại.")
        if live and (model["kind"] != "phoneme" or model["experimental"]):
            raise DemoError(409, "live_model_unavailable", "Realtime cần checkpoint A/B/C đã huấn luyện; BASE chỉ trích xuất đặc trưng.")
        if not self.inference_lock.acquire(blocking=False):
            raise DemoError(409, "busy", "Model đang xử lý audio khác; thử lại sau.")
        try:
            if self.engine is None:
                from mdd.demo_inference import LocalInference
                self.engine = LocalInference(self.root, threads=self.threads)
            result = self.engine.analyze(model, audio, text.strip(), reference=reference)
            result_model = dict(result.get("model") or {})
            result_model.update(id=model_id, label=model["label"], kind=model["kind"],
                                experimental=bool(model["experimental"] or result_model.get("experimental", False)))
            if model["kind"] == "backbone":
                # BASE has no trained phone head. Bind the public result to
                # the selected catalog capability even if an engine regresses.
                result = {**result, "comparison": None, "phonemes": [], "asr_text": None,
                          "analysis_kind": "backbone_features"}
            elif result_model["experimental"]:
                result = {**result, "comparison": None}
            result = {**result, "model": result_model}
            result = {**result, "model_id": model_id, "reference_text": text.strip()}
            result["reference_source"] = row.get("reference_source", "dataset") if row else None
            if result["reference_source"] == "cmudict":
                result["notice"] = result.get("notice", "") + " Âm chuẩn từ CMUdict tiếng Anh Mỹ, dùng cách đọc đầu tiên; biến thể phát âm có thể khác."
            if live:
                result["live"] = {"mode": mode, "is_final": final, "word_index": word_index if mode == "word" else None,
                                  "parent_reference_text": parent_text, "tentative": not final}
                if not final:
                    result["comparison"] = prefix_comparison(text.strip(), reference, [phone["phone"] for phone in result.get("phonemes", [])])
                    result["assessment"] = preview_assessment(result)
                    result.update(analysis_kind="phoneme_comparison", feedback_available=False, analysis_id=None,
                                  notice="Kết quả tạm thời theo phần đã đọc; có thể thay đổi khi nhận thêm âm thanh. Phần chưa đọc đang chờ.")
                    if result["reference_source"] == "cmudict":
                        result["notice"] += " Âm chuẩn CMUdict tiếng Anh Mỹ, cách đọc đầu tiên."
                    return result
            return {**result, **self.feedback_service.register(result)}
        except DemoError:
            raise
        except (ValueError, RuntimeError, OSError) as error:
            # Inference errors are rendered as text in the UI; never return tracebacks or local paths.
            message = str(error)
            if str(self.root) in message or re.search(r"[A-Za-z]:\\|/Users/|/home/", message):
                message = "Không đọc được model/audio local. Nếu checkpoint đang lưu, hãy thử lại sau."
            raise DemoError(422, "inference_failed", message[:500] or "Không phân tích được audio.") from None
        finally:
            self.inference_lock.release()

    def feedback(self, body: Any) -> dict:
        if not isinstance(body, dict) or set(body) - {"analysis_id", "include_voice"}:
            raise DemoError(400, "invalid_feedback_request", "Chỉ gửi mã lượt phân tích và lựa chọn giọng đọc.")
        identifier = body.get("analysis_id")
        include_voice = body.get("include_voice", True)
        if not isinstance(identifier, str) or not OPAQUE_ID.fullmatch(identifier):
            raise DemoError(400, "invalid_analysis", "Mã lượt phân tích không hợp lệ.")
        if not isinstance(include_voice, bool):
            raise DemoError(400, "invalid_voice_option", "Lựa chọn giọng đọc phải là true hoặc false.")
        try:
            return self.feedback_service.feedback(identifier, include_voice)
        except FeedbackNotFound:
            raise DemoError(404, "analysis_not_found", "Lượt phân tích đã hết hạn hoặc không có trong phiên này; hãy phân tích lại.") from None

    def voice_audio(self, identifier: str) -> tuple[bytes, str]:
        if not OPAQUE_ID.fullmatch(identifier):
            raise DemoError(404, "voice_not_found", "Đoạn giọng đọc không còn trong phiên này.")
        try:
            return self.feedback_service.voice(identifier)
        except (VoiceNotFound, OSError, ValueError):
            raise DemoError(404, "voice_not_found", "Đoạn giọng đọc không còn trong phiên này.") from None


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], application: DemoApplication, *, lan=False, allow_hosts=()):
        trusted_lan_hosts = validate_lan_hosts(allow_hosts, lan=lan)
        if address[0] != "127.0.0.1" and not (lan and address[0] == "0.0.0.0"):
            raise ValueError("The demo binds to 127.0.0.1 by default; 0.0.0.0 requires explicit LAN mode")
        self.application = application
        self.lan = lan
        self.allowed_hostnames = frozenset({"127.0.0.1", "localhost", *trusted_lan_hosts})
        super().__init__(address, DemoHandler)

    def server_close(self):
        self.application.chat_service.close()
        super().server_close()


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "MDDLocal/1.0"
    sys_version = ""

    def _guard(self) -> None:
        port = self.server.server_address[1]
        allowed = {f"{hostname}:{port}" for hostname in self.server.allowed_hostnames}
        hosts = self.headers.get_all("Host", [])
        host = hosts[0] if len(hosts) == 1 else ""
        if host not in allowed:
            raise DemoError(403, "invalid_host", "Địa chỉ truy cập không nằm trong danh sách demo được cho phép.")
        origins = self.headers.get_all("Origin", [])
        if origins and (len(origins) != 1 or origins[0] != f"http://{host}"):
            raise DemoError(403, "invalid_origin", "Yêu cầu phải đến từ đúng trang demo đang mở.")
        if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
            raise DemoError(403, "cross_site", "Không chấp nhận yêu cầu từ trang khác.")

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(content)

    def _send_json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _dispatch(self, method: str) -> None:
        try:
            self._guard()
            path = urlsplit(self.path).path
            app = self.server.application
            if method == "GET":
                if path == "/api/status":
                    self._send_json(200, app.status())
                elif path == "/api/prompts":
                    self._send_json(200, app.prompts())
                elif path == "/api/chat/config":
                    self._send_json(200, app.chat_service.config())
                elif path.startswith('/api/chat/voice-preview/'):
                    audio, content_type = app.chat_service.voice_preview(unquote(path[len('/api/chat/voice-preview/'):]))
                    self._send(200, audio, content_type)
                elif path.startswith("/api/chat/jobs/"):
                    self._send_json(200, app.chat_service.poll(unquote(path[len("/api/chat/jobs/"):])))
                elif path.startswith("/api/chat/audio/"):
                    audio, content_type = app.chat_service.audio(unquote(path[len("/api/chat/audio/"):]))
                    self._send(200, audio, content_type)
                elif path.startswith("/api/samples/"):
                    self._send(200, app.sample_audio(unquote(path[len("/api/samples/"):])), "audio/wav")
                elif path.startswith("/api/voice/"):
                    audio, content_type = app.voice_audio(unquote(path[len("/api/voice/"):]))
                    self._send(200, audio, content_type)
                elif path in STATIC_FILES:
                    file = app.root / "demo" / STATIC_FILES[path]
                    if not _inside(file, app.root / "demo") or not file.is_file():
                        raise DemoError(404, "not_found", "Trang demo chưa có sẵn.")
                    types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                             ".css": "text/css; charset=utf-8", ".png": "image/png", ".woff2": "font/woff2",
                             ".webp": "image/webp"}
                    self._send(200, file.read_bytes(), types[file.suffix])
                else:
                    raise DemoError(404, "not_found", "Không tìm thấy tài nguyên.")
            elif method == "POST" and path in {"/api/analyze", "/api/analyze-live", "/api/feedback", "/api/reference", "/api/chat/send", "/api/chat/speech", "/api/chat/topic", "/api/chat/cancel", "/api/chat/voice", "/api/chat/reference"}:
                if self.headers.get("Transfer-Encoding") is not None:
                    raise DemoError(400, "invalid_body", "Không hỗ trợ transfer encoding.")
                if self.headers.get_content_type() != "application/json":
                    raise DemoError(415, "invalid_content_type", "Sử dụng application/json.")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise DemoError(400, "invalid_body", "Content-Length không hợp lệ.")
                length = int(lengths[0])
                maximum = 2 * 1024 * 1024 if path == "/api/chat/speech" else 65536 if path.startswith("/api/chat/") else MAX_FEEDBACK_BODY_BYTES if path in {"/api/feedback", "/api/reference"} else MAX_BODY_BYTES
                if not 0 < length <= maximum:
                    raise DemoError(413, "body_too_large", "Nội dung yêu cầu vượt quá giới hạn của API.")
                self.connection.settimeout(20)
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Incomplete body")
                    body = json.loads(raw)
                except (OSError, UnicodeError, ValueError):
                    raise DemoError(400, "invalid_json", "JSON không hợp lệ hoặc chưa tải đủ.") from None
                if path.startswith("/api/chat/"):
                    handler = {"/api/chat/send": app.chat_service.submit, "/api/chat/speech": app.chat_service.submit_speech, "/api/chat/topic": app.chat_service.start_topic, "/api/chat/cancel": app.chat_service.cancel, "/api/chat/voice": app.chat_service.voice, "/api/chat/reference": app.chat_service.reference}[path]
                    result = handler(body)
                else:
                    result = app.feedback(body) if path == "/api/feedback" else app.reference_info(body) if path == "/api/reference" else app.analyze(body, live=path == "/api/analyze-live")
                self._send_json(200, result)
            else:
                raise DemoError(404, "not_found", "Không tìm thấy API.")
        except (DemoError, ChatError) as error:
            self._send_json(error.status, {"error": error.code, "message": error.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._send_json(500, {"error": "internal_error", "message": "Demo gặp lỗi local; kiểm tra cửa sổ server và thử lại."})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, format: str, *args: Any) -> None:
        # Paths/IDs suffice for local diagnostics; never log uploaded bodies.
        super().log_message(format, *args)
