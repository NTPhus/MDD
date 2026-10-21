"""Check local BASE+phone CTC on corpus audio or --audio WAV; no training or benchmark."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path,
                        help="Use an unlabeled local audio file instead of a source-training utterance")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Execution device (default: cpu); auto selects CUDA only when available")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/setup/model_forward.json")
    return parser


def load_audio(path):
    """Read the entire file, downmix channels and resample without inventing labels."""
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf

    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and contain only finite samples")
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio became nonfinite during downmixing or resampling")
    return audio, rate


def source_training_sample():
    processed = ROOT / "data/processed/l2arctic-v1"
    split = json.loads((ROOT / "data/protocol/l2arctic-v1/splits.json").read_text(encoding="utf-8"))
    allowed = set(split["source_train_ids"])
    rows = [json.loads(line) for line in (processed / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = [row for row in rows if row["utterance_id"] in allowed and 0.5 <= row["duration_sec"] <= 3]
    if not candidates:
        raise ValueError("No short source-train sample available for the forward check")
    return min(candidates, key=lambda item: (item["duration_sec"], item["utterance_id"]))


def memory_report():
    """Windows exposes peak working set; other platforms report current RSS explicitly."""
    import psutil

    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", None)
    return {
        "process_rss_mib": round(memory.rss / 1024**2, 2),
        "peak_process_rss_mib": round(peak / 1024**2, 2) if peak is not None else None,
    }


def run_check(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.set_num_threads(4)
    torch.manual_seed(42)
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu explicitly")
    device = torch.device(selected_device)
    vocab = json.loads((ROOT / "data/processed/l2arctic-v1/vocab.json").read_text(encoding="utf-8"))
    row = None if args.audio is not None else source_training_sample()
    audio_path = args.audio.resolve() if args.audio is not None else ROOT / row["audio_path"]
    audio, rate = load_audio(audio_path)
    base_path = ROOT / "models/pretrained/wav2vec2-base"
    started = time.monotonic()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        base_path, local_files_only=True, vocab_size=len(vocab), pad_token_id=vocab["[PAD]"],
        ctc_loss_reduction="mean", ctc_zero_infinity=False, mask_time_prob=0.0, mask_feature_prob=0.0,
        bos_token_id=None, eos_token_id=None, attn_implementation="eager",
        id2label={index: phone for phone, index in vocab.items()}, label2id=vocab,
    ).to(device).eval()
    inputs = {key: value.to(device) for key, value in
              extractor(audio, sampling_rate=16000, return_tensors="pt").items()}
    if not torch.isfinite(inputs["input_values"]).all():
        raise ValueError("Feature extractor returned nonfinite input values")
    frame_count = int(model._get_feat_extract_output_lengths(len(audio)))
    if frame_count < 1:
        raise ValueError("Audio is too short for one model output frame")
    supervision = {}
    if row is not None:
        target_ids = [vocab[phone] for phone in row["perceived_phones"]]
        minimum_frames = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
        if not target_ids or frame_count < minimum_frames:
            raise ValueError("CTC target cannot fit the sample; audio was not truncated")
        inputs["labels"] = torch.tensor([target_ids], dtype=torch.long, device=device)
        supervision.update(ctc_target_length=len(target_ids), minimum_ctc_frames=minimum_frames)
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True)
    features = result.hidden_states[-1]
    if (result.logits.ndim != 3 or result.logits.shape[-1] != len(vocab)
            or result.logits.shape[1] != frame_count or not torch.isfinite(result.logits).all()
            or not torch.isfinite(features).all()):
        raise ValueError("Forward check returned nonfinite features/logits or invalid output dimensions")
    if row is not None:
        if result.loss is None or not torch.isfinite(result.loss):
            raise ValueError("Forward check returned invalid CTC loss")
        supervision["finite_ctc_loss"] = float(result.loss)
    id_to_phone = {index: phone for phone, index in vocab.items()}
    token_ids = result.logits[0].argmax(dim=-1).tolist()
    predicted = [id_to_phone[token] for token, _ in itertools.groupby(token_ids) if token != vocab["[PAD]"]]
    report = {
        "purpose": "technical_forward_only_not_a_benchmark",
        "model": "facebook/wav2vec2-base with newly initialized phoneme CTC head",
        "model_revision": (base_path / "download_revision.txt").read_text(encoding="utf-8").strip(),
        "optimizer_steps": 0, "trained": False, "device": selected_device,
        "requested_device": args.device,
        "input_split": "source_train" if row is not None else "external_unlabeled",
        "utterance_id": row["utterance_id"] if row is not None else None,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "source_sample_rate": rate, "model_sample_rate": 16000,
        "duration_sec": len(audio) / 16000, "vocabulary_size": len(vocab),
        "audio_rms": float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
        "input_values_shape": list(inputs["input_values"].shape), "finite_input_values": True,
        "last_hidden_state_shape": list(features.shape), "finite_hidden_features": True,
        "logits_shape": list(result.logits.shape), "finite_logits": True,
        "logits_min": float(result.logits.min()), "logits_max": float(result.logits.max()),
        "decoder": "greedy_ctc", "predicted_phones_untrained": predicted,
        "decoding_warning": "Randomly initialized phoneme head; these phones are not an MDD diagnosis or transcript.",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_load_and_forward_sec": round(time.monotonic() - started, 3),
        **supervision, **memory_report(),
    }
    return report


def main():
    args = build_parser().parse_args()
    report = run_check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
