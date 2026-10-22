"""Read-only CPU inference for the local demo; ML imports are deferred until use."""

from __future__ import annotations

import io
import json
import hashlib
from functools import lru_cache
import math
from pathlib import Path
import time
import unicodedata
from typing import Any

from .metrics import align_phones


SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MIN_AUDIO_SECONDS = 0.25


@lru_cache(maxsize=8)
def _file_digest(path, modified, size):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize_text(text: str) -> str:
    """Match canonical prompts exactly after case/Unicode/whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def decode_ctc(frame_ids, frame_confidences, id_to_phone, blank_id, duration_sec):
    """Collapse adjacent repeats before removing blanks; times are approximate."""
    if len(frame_ids) != len(frame_confidences):
        raise ValueError("CTC frame IDs and confidence lengths differ")
    result = []
    frame_count = len(frame_ids)
    index = 0
    while index < frame_count:
        end = index + 1
        while end < frame_count and frame_ids[end] == frame_ids[index]:
            end += 1
        token = frame_ids[index]
        if token != blank_id:
            if token not in id_to_phone:
                raise ValueError("CTC output token is absent from the checkpoint vocabulary")
            confidence = sum(frame_confidences[index:end]) / (end - index)
            result.append({"phone": id_to_phone[token],
                           "start_sec": round(duration_sec * index / frame_count, 4),
                           "end_sec": round(duration_sec * end / frame_count, 4),
                           "confidence": round(float(confidence), 6)})
        index = end
    return result


def canonical_comparison(reference_text, reference, predicted_phones):
    """Compare to canonical prompt phones only, without touching perceived/gold data."""
    if not reference or not normalize_text(reference_text):
        return None
    if normalize_text(reference.get("prompt_text", "")) != normalize_text(reference_text):
        return None
    canonical = reference.get("canonical_phones")
    if (not isinstance(canonical, list) or not canonical or len(canonical) > 1000
            or any(not isinstance(phone, str) or not phone or phone.strip() != phone for phone in canonical)):
        return None
    words = str(reference_text).split()
    mapping = reference.get("canonical_word_indices", [])
    if not isinstance(mapping, list) or len(mapping) != len(canonical):
        mapping = [None] * len(canonical)
    intervals = reference.get("word_intervals")
    if intervals is not None:
        # Some corpus word tiers omit a prompt word or insert a silence token.
        # Do not shift those annotation indices onto unrelated displayed words.
        def word_key(word):
            return "".join(character for character in normalize_text(word) if character.isalnum() or character == "'")

        aligned_words = (isinstance(intervals, list) and len(intervals) == len(words)
                         and all(isinstance(item, dict) and item.get("index") == index
                                 and word_key(item.get("text", "")) == word_key(words[index])
                                 for index, item in enumerate(intervals)))
        if not aligned_words:
            mapping = [None] * len(canonical)
    mapping = [word if isinstance(word, int) and not isinstance(word, bool) and 0 <= word < len(words)
               else None for word in mapping]
    statuses = ["match" if index in mapping else "unavailable" for index in range(len(words))]
    counts = dict.fromkeys(("substitution", "deletion", "insertion", "match"), 0)
    events = []
    for step in align_phones(canonical, predicted_phones):
        kind = "insertion" if step["kind"] == "addition" else step["kind"]
        counts[kind] += 1
        if kind == "match":
            continue
        # The aligner puts additions at a canonical gap. Assign them to the
        # following phone's word, or the final word at the end of the utterance.
        word = mapping[min(step["position"], len(mapping) - 1)]
        if word is not None:
            statuses[word] = "difference"
        events.append({"kind": kind, "expected_phone": step["expected_phone"],
                       "observed_phone": step["observed_phone"], "word_index": word})
    return {"words": [{"text": word, "status": status} for word, status in zip(words, statuses)],
            "counts": counts, "events": events}


class LocalInference:
    """One CPU model cached per server; a failed replacement keeps the old cache."""

    def __init__(self, root: Path, threads: int = 2):
        self.root = Path(root).resolve()
        self.threads = max(1, min(4, int(threads)))
        self._cache_key = None
        self._cached = None

    def _audio(self, audio_bytes):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        if not isinstance(audio_bytes, bytes) or not audio_bytes or len(audio_bytes) > 32 * 1024 * 1024:
            raise ValueError("Chọn tệp âm thanh hợp lệ, tối đa 32 MiB.")
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as stream:
                rate, frames, channels = stream.samplerate, stream.frames, stream.channels
                if not 8000 <= rate <= 192000 or channels not in (1, 2):
                    raise ValueError("Âm thanh cần mono/stereo với sample rate từ 8 đến 192 kHz.")
                # Inspect the header before allocating decoded audio. In
                # particular, a tiny malicious WAV can claim billions of frames.
                if not MIN_AUDIO_SECONDS * rate <= frames <= MAX_AUDIO_SECONDS * rate:
                    raise ValueError("Âm thanh cần có thời lượng từ 0.25 đến 30 giây.")
                samples = stream.read(frames=frames, dtype="float32", always_2d=True)
        except (RuntimeError, OSError) as error:
            raise ValueError("Không đọc được tệp âm thanh; hãy dùng WAV, FLAC hoặc OGG hợp lệ.") from error
        if len(samples) != frames or not np.isfinite(samples).all():
            raise ValueError("Âm thanh thiếu dữ liệu hoặc chứa mẫu không hữu hạn.")
        waveform = samples.mean(axis=1)
        if rate != SAMPLE_RATE:
            divisor = math.gcd(rate, SAMPLE_RATE)
            waveform = resample_poly(waveform, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Âm thanh không có dữ liệu hợp lệ sau khi resample.")
        rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(waveform)))
        if rms < 1e-4 or peak < 5e-4:
            raise ValueError("Âm thanh quá yên lặng; hãy ghi lại giọng nói rõ hơn.")
        # Preserve the signed highest-amplitude sample in each display bucket.
        buckets = np.array_split(waveform, min(160, len(waveform)))
        envelope = [round(float(bucket[np.argmax(np.abs(bucket))]), 6) for bucket in buckets]
        return waveform, {"duration_sec": round(len(waveform) / SAMPLE_RATE, 4),
                          "sample_rate": SAMPLE_RATE, "rms": rms, "peak": peak, "waveform": envelope}

    def _load_bundle(self, path, kind):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Model

        torch.set_num_threads(self.threads)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path), local_files_only=True)
        options = {"local_files_only": True, "attn_implementation": "eager", "output_loading_info": True}
        if kind == "backbone":
            network, loading = Wav2Vec2Model.from_pretrained(str(path), **options)
            vocabulary = None
            experimental = False
        else:
            vocabulary = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
            if (not isinstance(vocabulary, dict) or vocabulary.get("[PAD]") != 0
                    or any(not isinstance(phone, str) or not phone for phone in vocabulary)
                    or any(isinstance(index, bool) or not isinstance(index, int) for index in vocabulary.values())
                    or sorted(vocabulary.values()) != list(range(len(vocabulary)))):
                raise ValueError("Checkpoint không có vocabulary âm vị hợp lệ.")
            network, loading = Wav2Vec2ForCTC.from_pretrained(str(path), **options)
            if network.config.vocab_size != len(vocabulary) or network.config.pad_token_id != 0:
                raise ValueError("Đầu CTC không khớp vocabulary của checkpoint.")
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            experimental = metadata.get("purpose") != "scientific_training"
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError("Checkpoint thiếu trọng số model; không dùng trọng số khởi tạo ngẫu nhiên.")
        network.to("cpu")
        network.eval()
        network.requires_grad_(False)
        return {"network": network, "extractor": extractor, "vocabulary": vocabulary,
                "experimental": experimental}

    def _get_bundle(self, model):
        path = Path(model["path"])
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError("Model local không tồn tại trong workspace.")
        kind = model.get("kind")
        if kind not in {"backbone", "phoneme"}:
            raise ValueError("Loại model không được hỗ trợ.")
        signatures = []
        for filename in ("config.json", "vocab.json", "metadata.json", "model.safetensors", "pytorch_model.bin"):
            candidate = path / filename
            if candidate.is_file():
                status = candidate.stat()
                signatures.append((filename, status.st_mtime_ns, status.st_size))
        key = (str(path), kind, str(model.get("fingerprint", "")), tuple(signatures))
        if key != self._cache_key:
            # Assign only after the entire load/validation succeeds. A damaged
            # newly published checkpoint must not evict the previous model.
            bundle = self._load_bundle(path, kind)
            self._cached, self._cache_key = bundle, key
        return self._cached

    def analyze(self, model: dict[str, Any], audio_bytes: bytes, reference_text: str,
                reference: dict | None = None, *, capture_evidence: bool = False) -> dict[str, Any]:
        started = time.monotonic()
        waveform, audio = self._audio(audio_bytes)
        bundle = self._get_bundle(model)
        import torch

        # The network sees audio only. Reference text and canonical phones are
        # considered solely after inference; perceived labels are never read.
        torch.set_num_threads(self.threads)
        inputs = bundle["extractor"](waveform, sampling_rate=SAMPLE_RATE,
                                     return_attention_mask=True, return_tensors="pt")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        with torch.inference_mode():
            output = bundle["network"](**inputs)
        features, phonemes, comparison = None, [], None
        experimental = bool(model.get("experimental", True)) or bundle["experimental"]
        if model["kind"] == "backbone":
            hidden = output.last_hidden_state[0]
            if not torch.isfinite(hidden).all():
                raise ValueError("Model trả về đặc trưng không hữu hạn.")
            features = {"frames": int(hidden.shape[0]), "dimensions": int(hidden.shape[1]),
                        "norm_mean": float(hidden.norm(dim=-1).mean())}
            analysis_kind = "backbone_features"
            warning = "BASE là backbone chưa có đầu nhận diện âm vị hoặc bản chép lời."
            notice = "Đã trích xuất đặc trưng giọng nói thực trên CPU. Chưa có điểm phát âm."
        else:
            logits = output.logits[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Model trả về logits không hữu hạn.")
            confidence, token_ids = logits.softmax(dim=-1).max(dim=-1)
            phonemes = decode_ctc(token_ids.tolist(), confidence.tolist(),
                                 {index: phone for phone, index in bundle["vocabulary"].items()},
                                 0, audio["duration_sec"])
            analysis_kind = "experimental_phonemes"
            warning = ("Checkpoint thử nghiệm; âm vị có thể chưa chính xác, không dùng để chẩn đoán phát âm."
                       if experimental else "Âm vị do model dự đoán; thời gian và độ tin cậy từng âm chỉ mang tính tương đối.")
            notice = "Chuỗi âm vị được giải mã CTC từ âm thanh; mốc thời gian là xấp xỉ."
            if not experimental:
                comparison = canonical_comparison(reference_text, reference, [item["phone"] for item in phonemes])
                analysis_kind = "phoneme_comparison" if comparison is not None else "phoneme_decoding"
                if comparison is None:
                    notice += " Chưa có chuỗi âm chuẩn khớp chính xác với câu đã nhập để so sánh."
            if not phonemes:
                notice += " Model chỉ phát ra blank; chưa nhận diện được âm vị."
        result = {"model": {"id": model["id"], "label": model["label"], "kind": model["kind"],
                          "experimental": experimental, "warning": warning},
                "audio": audio, "reference_text": str(reference_text), "asr_text": None,
                "phonemes": phonemes, "features": features, "comparison": comparison,
                "notice": notice, "inference_seconds": round(time.monotonic() - started, 3),
                "analysis_kind": analysis_kind}
        if capture_evidence and model['kind'] == 'phoneme':
            result['_acoustic_evidence'] = {
                'log_probs': logits.log_softmax(dim=-1).detach().cpu().numpy(),
                'id_to_phone': {index: phone for phone, index in bundle['vocabulary'].items()},
                'blank_id': 0, 'duration_sec': audio['duration_sec'],
                'phonemes': phonemes, 'audio': audio,
            }
        return result

    def posterior_evidence(self, model: dict[str, Any], audio_bytes: bytes) -> dict[str, Any]:
        """Export audio-only CTC posteriors for an explicitly requested experiment.

        No reference, perceived phones, alignments or human error labels enter
        the network. The original analyze() contract stays unchanged by default.
        """
        if model.get('kind') != 'phoneme':
            raise ValueError('Posterior evidence requires a phoneme checkpoint')
        folder = Path(model['path'])
        folder = (folder if folder.is_absolute() else self.root / folder).resolve()
        if not folder.is_relative_to(self.root) or not folder.is_dir():
            raise ValueError('Posterior checkpoint must be inside the workspace')
        paths, identities, signatures = {}, {}, {}
        for key, filename in (('checkpoint_sha256', 'model.safetensors'), ('vocabulary_sha256', 'vocab.json')):
            path = folder / filename
            if not path.is_file() and filename == 'model.safetensors':
                path = folder / 'pytorch_model.bin'
            if not path.resolve().is_relative_to(self.root):
                raise ValueError('Posterior checkpoint file is outside the workspace')
            info = path.stat()
            paths[key] = path
            signatures[key] = (info.st_mtime_ns, info.st_size)
            identities[key] = _file_digest(str(path), info.st_mtime_ns, info.st_size)
        for name in ('config.json', 'metadata.json'):
            path = folder / name
            if path.is_file():
                paths[name] = path
                signatures[name] = (path.stat().st_mtime_ns, path.stat().st_size)
        result = self.analyze(model, audio_bytes, '', capture_evidence=True)
        if result['model']['experimental']:
            raise ValueError('Unverified experimental weights cannot provide calibrated evidence')
        if any(not path.exists() or (path.stat().st_mtime_ns, path.stat().st_size) != signatures[key]
               for key, path in paths.items()):
            raise ValueError('Checkpoint changed during inference; retry with stable completed weights')
        evidence = result['_acoustic_evidence']
        evidence.update(identities)
        return evidence
