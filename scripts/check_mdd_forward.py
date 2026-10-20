"""Check one real source-training utterance through local BASE+phone CTC; no optimizer or training."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/setup/model_forward.json")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.set_num_threads(4)
    torch.manual_seed(42)
    processed = ROOT / "data/processed/l2arctic-v1"
    vocab = json.loads((processed / "vocab.json").read_text())
    split = json.loads((ROOT / "data/protocol/l2arctic-v1/splits.json").read_text())
    allowed = set(split["source_train_ids"])
    rows = [json.loads(line) for line in (processed / "annotations.jsonl").read_text().splitlines()]
    candidates = [row for row in rows if row["utterance_id"] in allowed and 0.5 <= row["duration_sec"] <= 3]
    if not candidates:
        raise ValueError("No short source-train sample available for the forward check")
    row = min(candidates, key=lambda item: (item["duration_sec"], item["utterance_id"]))
    audio_path = ROOT / row["audio_path"]
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    divisor = math.gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    base_path = ROOT / "models/pretrained/wav2vec2-base"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
    started = time.monotonic()
    model = Wav2Vec2ForCTC.from_pretrained(
        base_path, local_files_only=True, vocab_size=len(vocab), pad_token_id=vocab["[PAD]"],
        ctc_loss_reduction="mean", ctc_zero_infinity=False, mask_time_prob=0.0, mask_feature_prob=0.0,
    ).eval()
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    target_ids = [vocab[phone] for phone in row["perceived_phones"]]
    frame_count = int(model._get_feat_extract_output_lengths(len(audio)))
    minimum_frames = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
    if frame_count < minimum_frames:
        raise ValueError("CTC target cannot fit the sample; audio was not truncated")
    with torch.no_grad():
        result = model(**inputs, labels=torch.tensor([target_ids], dtype=torch.long))
    if not torch.isfinite(result.loss) or result.logits.shape[-1] != len(vocab):
        raise ValueError("Forward check returned invalid loss or vocabulary dimension")
    report = {
        "purpose": "technical_forward_only_not_a_benchmark",
        "model": "facebook/wav2vec2-base with newly initialized phoneme CTC head",
        "model_revision": (base_path / "download_revision.txt").read_text().strip(),
        "optimizer_steps": 0, "trained": False, "device": "cpu",
        "input_split": "source_train", "utterance_id": row["utterance_id"],
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "source_sample_rate": rate, "model_sample_rate": 16000,
        "duration_sec": len(audio) / 16000, "vocabulary_size": len(vocab),
        "ctc_target_length": len(target_ids), "minimum_ctc_frames": minimum_frames,
        "logits_shape": list(result.logits.shape), "finite_ctc_loss": float(result.loss),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_load_and_forward_sec": round(time.monotonic() - started, 3),
        "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
