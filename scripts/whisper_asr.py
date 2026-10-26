"""Pinned, offline English Whisper ASR for the local demo.

The optional runtime is isolated under .tools, and inference never downloads
weights, reads prompts from the conversation, or trains either speech model.
Word times are Whisper predictions, not human annotations or forced alignment.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

MODEL_ID = "Systran/faster-whisper-small.en"
REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
MODEL_RELATIVE = Path("models/pretrained/faster-whisper-small.en")
DEPS_RELATIVE = Path(".tools/whisper-asr/python")
PACKAGES = {"faster-whisper": "1.2.1", "ctranslate2": "4.8.3", "av": "19.0.2"}
_DLL_HANDLES = []
_CUDA_DLLS_PREPARED = False
FILES = {
    "config.json": {"bytes": 2657, "sha256": "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae"},
    "model.bin": {"bytes": 483545366, "sha256": "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a"},
    "tokenizer.json": {"bytes": 2128466, "sha256": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"},
    "vocabulary.txt": {"bytes": 422309, "sha256": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"},
}
OPTIONS = {
    "language": "en", "task": "transcribe", "beam_size": 5, "best_of": 1,
    "temperature": 0., "vad_filter": False, "condition_on_previous_text": False,
    "initial_prompt": None, "prefix": None, "hotwords": None,
    "word_timestamps": True, "without_timestamps": False,
}
CARDINALS = dict(enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                          "sixteen", "seventeen", "eighteen", "nineteen")))
CARDINALS.update({20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
                  70: "seventy", 80: "eighty", 90: "ninety"})


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("Whisper transcription cancelled")


def file_record(path, expected, *, checksum=True):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing or wrong-sized Whisper file: {path.name}")
    record = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
    if checksum:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Whisper SHA256 mismatch: {path.name}")
    return record


def package_versions(path):
    distributions = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(path)])
    }
    for name, version in PACKAGES.items():
        if distributions.get(name) != version:
            raise ValueError(f"Whisper optional runtime requires {name}=={version}")
    return dict(PACKAGES)


def verify(root, *, checksum=True):
    """Verify pinned metadata; checksum=True is required before loading weights."""
    root = Path(root).resolve()
    model_dir = root / MODEL_RELATIVE
    record = json.loads((model_dir / "setup_manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("status") not in {"downloaded", "ready"}
            or record.get("model_id") != MODEL_ID or record.get("revision") != REVISION
            or record.get("packages") != PACKAGES):
        raise ValueError("Whisper manifest does not match the pinned model/runtime")
    for name, expected in FILES.items():
        if record.get("files", {}).get(name) != expected:
            raise ValueError(f"Whisper manifest checksum mismatch: {name}")
        file_record(model_dir / name, expected, checksum=checksum)
    package_versions(root / DEPS_RELATIVE)
    return record


def whisper_selection(root):
    """Cheap readiness check: metadata and sizes, not a repeated 484 MB hash."""
    root = Path(root).resolve()
    result = {"id": "whisper-small-en", "model_id": MODEL_ID, "revision": REVISION,
              "path": str(root / MODEL_RELATIVE), "device": "cpu", "compute_type": "int8",
              "sha256": FILES["model.bin"]["sha256"], "files": FILES,
              "ready": False, "available": False, "selection": "whisper-small-en"}
    try:
        verify(root, checksum=False)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["reason"] = str(error)
        return result
    result.update(ready=True, available=True)
    return result


def _load_runtime(root):
    deps = (Path(root) / DEPS_RELATIVE).resolve()
    package_versions(deps)
    if str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    # Do not silently reuse a different optional runtime imported by another tool.
    for name, version in (("faster_whisper", PACKAGES["faster-whisper"]),
                          ("ctranslate2", PACKAGES["ctranslate2"]), ("av", PACKAGES["av"])):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(deps) or getattr(module, "__version__", None) != version:
            raise ValueError(f"Whisper runtime import is not the pinned optional package: {name}")
    return importlib.import_module("faster_whisper").WhisperModel


def _prepare_cuda(root):
    """Reuse installed CUDA 12/cuDNN 9 DLLs; no download or global PATH edit."""
    global _CUDA_DLLS_PREPARED
    if os.name != 'nt' or _CUDA_DLLS_PREPARED:
        return
    import ctypes
    folder = Path(root).resolve() / '.venv/Lib/site-packages/torch/lib'
    if folder.is_dir():
        _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
        # cuDNN loads its companion DLLs lazily; retain this process-local PATH.
        os.environ['PATH'] = str(folder) + os.pathsep + os.environ.get('PATH', '')
        names = ['cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll', 'cudnn64_9.dll']
        names += [p.name for p in sorted(folder.glob('cudnn*64_9.dll')) if p.name not in names]
        for name in names:
            _DLL_HANDLES.append(ctypes.WinDLL(str(folder / name)))
    _CUDA_DLLS_PREPARED = True


def _cuda_available(root):
    _prepare_cuda(root)
    runtime = importlib.import_module('ctranslate2')
    return (runtime.get_cuda_device_count() > 0
            and 'int8_float16' in runtime.get_supported_compute_types('cuda', 0))


def _cuda_error(error):
    """Only recover from accelerator failures, never corrupt weights/input errors."""
    message = str(error).lower()
    return isinstance(error, (RuntimeError, OSError)) and any(
        token in message for token in ('cuda', 'cudnn', 'cublas', 'out of memory', 'not enough memory'))


def _intervals(text, words, duration):
    """Map native word pieces to exact whitespace tokens used by phonetic_candidates.

    Some tokenizers split contractions into more than one timed piece. Match
    their lexical characters in order, and abstain if any characters disagree.
    There is no reuse of wav2vec CTC token positions for a Whisper transcript.
    """
    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches or not words:
        return []
    normalized = lambda value: "".join(re.findall(r"[a-z0-9']", value.lower().replace("’", "'")))
    tokens = [normalized(match.group()) for match in token_matches]
    pieces = []
    for word in words:
        value = normalized(word["word"])
        if not value:
            continue
        start, end = word["start"], word["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration + .1):
            return []
        pieces.append((value, start, min(end, duration)))
    if not all(tokens) or "".join(tokens) != "".join(piece[0] for piece in pieces):
        return []
    result, cursor = [], 0
    for index, token in enumerate(tokens):
        stop = cursor + len(token)
        offset, overlap = 0, []
        for value, start, end in pieces:
            if offset < stop and offset + len(value) > cursor:
                overlap.append((start, end))
            offset += len(value)
        if not overlap:
            return []
        start, end = overlap[0][0], overlap[-1][1]
        # A native piece spanning multiple whitespace tokens cannot supply
        # distinct word times. Leave sound-neighbour retrieval unavailable.
        if result and start < result[-1]["end_sec"] - .001:
            return []
        result.append({"index": index, "text": token_matches[index].group(),
                       "start_sec": start, "end_sec": end})
        cursor = stop
    return result


def _format_cardinals(text):
    """Spell only unambiguous one-word cardinals for the pronunciation dictionary.

    This preserves values and token count. Leave dates, leading-zero IDs,
    decimals, signed values, ranges and clock times in the raw representation.
    """
    matches = list(re.finditer(r"\S+", text))
    changes, replacements = [], []
    for index, match in enumerate(matches):
        token = match.group()
        number = re.fullmatch(r'''([\(\["']?)(0|[1-9][0-9]?)([\)\],.!?;"']*)''', token)
        after = matches[index + 1].group().lower().strip(".,") if index + 1 < len(matches) else ""
        before = matches[index - 1].group().lower().strip(".,") if index else ""
        # A split minus sign is still a signed value; am/pm marks clock time.
        if (not number or int(number[2]) not in CARDINALS or before in {"-", "+", "−"}
                or after in {"am", "pm", "a.m", "p.m"}):
            replacements.append(token)
            continue
        replacement = number[1] + CARDINALS[int(number[2])] + number[3]
        replacements.append(replacement)
        changes.append({"index": index, "original": token, "replacement": replacement})
    # ASR output is already joined into single-spaced native segment text.
    return " ".join(replacements), changes


class WhisperASR:
    def __init__(self, root, cpu_threads=2, *, device='cpu'):
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 16:
            raise ValueError("Whisper cpu_threads must be an integer between 1 and 16")
        self.root = Path(root).resolve()
        self.cpu_threads = cpu_threads
        if device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('Whisper device must be cpu, cuda or auto')
        self.requested_device = device
        self.device = 'cpu'
        self.compute_type = 'int8'
        self.fallback_reason = None
        self.model = None
        self.load_sec = None
        self._lock = threading.RLock()

    def load(self, cancelled=None):
        _check_cancelled(cancelled)
        with self._lock:
            if self.model is not None:
                return self
            started = time.perf_counter()
            verify(self.root)
            _check_cancelled(cancelled)
            model_class = _load_runtime(self.root)
            target = 'cpu'
            if self.requested_device != 'cpu':
                try:
                    if _cuda_available(self.root):
                        target = 'cuda'
                    elif self.requested_device == 'cuda':
                        raise RuntimeError('Whisper CUDA int8_float16 is unavailable')
                    else:
                        self.fallback_reason = 'cuda_unavailable'
                except (RuntimeError, OSError) as error:
                    if self.requested_device == 'cuda':
                        raise
                    self.fallback_reason = 'cuda_probe_failed'
            try:
                model = self._construct(model_class, target)
            except (RuntimeError, OSError) as error:
                if target != 'cuda' or self.requested_device != 'auto' or not _cuda_error(error):
                    raise
                self.fallback_reason = 'cuda_load_failed'
                model = self._construct(model_class, 'cpu')
                target = 'cpu'
            _check_cancelled(cancelled)
            self.model = model
            self.device = target
            self.compute_type = 'int8_float16' if target == 'cuda' else 'int8'
            self.load_sec = round(time.perf_counter() - started, 4)
        return self

    def _construct(self, model_class, device):
        return model_class(str(self.root / MODEL_RELATIVE), device=device,
                           compute_type='int8_float16' if device == 'cuda' else 'int8',
                           cpu_threads=self.cpu_threads, num_workers=1, local_files_only=True)

    def close(self):
        with self._lock:
            model, self.model = self.model, None
            if model is not None:
                unload = getattr(getattr(model, 'model', None), 'unload_model', None)
                if callable(unload):
                    unload()

    def _decode(self, audio, cancelled):
        segments, info = self.model.transcribe(audio, **OPTIONS)
        rows, native_words = [], []
        for segment in segments:
            _check_cancelled(cancelled)
            rows.append({'start_sec': float(segment.start), 'end_sec': float(segment.end),
                         'text': str(segment.text), 'avg_logprob': float(segment.avg_logprob),
                         'no_speech_prob': float(segment.no_speech_prob)})
            for word in segment.words or ():
                native_words.append({'word': str(word.word), 'start': float(word.start),
                                     'end': float(word.end), 'probability': float(word.probability)})
        _check_cancelled(cancelled)
        return rows, native_words, info

    def transcribe(self, waveform, cancelled=None):
        import numpy as np
        _check_cancelled(cancelled)
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Whisper expects finite, nonempty mono 16 kHz samples")
        # The caller decodes/resamples once. Never resample a second time here.
        audio = np.ascontiguousarray(audio)
        duration = audio.size / 16000.
        self.load(cancelled)
        started = time.perf_counter()
        with self._lock:
            _check_cancelled(cancelled)
            try:
                rows, native_words, info = self._decode(audio, cancelled)
            except (RuntimeError, OSError) as error:
                if self.device != 'cuda' or self.requested_device != 'auto' or not _cuda_error(error):
                    raise
                # Discard every partial segment and retry the SAME complete audio.
                error.__traceback__ = None
                self.close()
                import gc
                gc.collect()
                _check_cancelled(cancelled)
                self.model = self._construct(_load_runtime(self.root), 'cpu')
                self.device, self.compute_type = 'cpu', 'int8'
                self.fallback_reason = 'cuda_inference_failed'
                rows, native_words, info = self._decode(audio, cancelled)
        raw_text = " ".join(row["text"].strip() for row in rows).strip()
        intervals = _intervals(raw_text, native_words, duration)
        text, changes = _format_cardinals(raw_text)
        formatted_tokens = text.split()
        for interval in intervals:
            interval["text"] = formatted_tokens[interval["index"]]
        return {"text": text, "raw_text": raw_text, "word_intervals": intervals,
                "text_normalization": {"kind": "english_single_token_cardinal_formatting", "changes": changes},
                "word_timing": "whisper_native" if intervals else "unavailable",
                "model_id": MODEL_ID, "revision": REVISION, "device": self.device, "compute_type": self.compute_type,
                'requested_device': self.requested_device, 'fallback_reason': self.fallback_reason,
                "language": info.language, "language_probability": float(info.language_probability),
                "inference_sec": round(time.perf_counter() - started, 4), "load_sec": self.load_sec,
                "segments": rows, "native_words": native_words, "options": dict(OPTIONS),
                "sample_rate": 16000, "audio_duration_sec": duration,
                "reference_supplied_to_asr": False, "word_timing_is_forced_alignment": False}
