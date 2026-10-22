"""Audio/text pairing on the frozen cascade split; no oracle phones in inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.l2arctic_plus_data import rooted, load as load_native, json_digest
from scripts.prepare_cascade_data import check_common_split, save as atomic_save
from scripts.prepare_feedback_data import digest

CONFIG = ROOT / "configs/projector_l2arctic_plus.json"
PROMPT = ROOT / "prompts/pronunciation_feedback_projector.txt"
PLACEHOLDER = "<audio_latents>"


def save(path, value):
    # Brief Windows reader/AV sharing locks must not kill a long training job.
    for attempt in range(61):
        try:
            return atomic_save(path, value)
        except PermissionError:
            if attempt == 60:
                raise
            time.sleep(min(.05 * (attempt + 1), 1.))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(config):
    cascade = read_json(rooted(config["cascade_config"]))
    protocol, splits, native_manifest = check_common_split(cascade)
    feedback_config = read_json(rooted(cascade["feedback_config"]))
    native, _ = load_native(rooted(feedback_config["dataset_dir"]))
    audio = {row["utterance_id"]: row for row in read_rows(rooted(cascade["protocol_dir"]) / "manifest.jsonl")}
    destination = rooted(config["dataset_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    hashes, counts = {}, {}
    for role, values in native.items():
        rows = []
        for row in values:
            item = audio[row["utterance_id"]]
            rows.append({key: row[key] for key in ("utterance_id", "speaker_id", "reference_text", "feedback", "source_row_sha256", "feedback_provenance")}
                        | {"word_order": list(row["annotation_info"]), "audio_path": item["audio"]["path"],
                           "audio_sha256": item["audio"]["sha256"], "duration_sec": item["duration_sec"]})
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = destination / (role + ".jsonl")
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Frozen projector data changed: " + role)
        path.write_text(content, encoding="utf-8")
        hashes[role], counts[role] = digest(path), len(rows)
    manifest = {"protocol_sha256": digest(rooted(cascade["protocol_dir"]) / "protocol.json"),
                "native_manifest_sha256": digest(rooted(feedback_config["dataset_dir"]) / "manifest.json"),
                "source_commit": native_manifest["source_commit"], "sha256": hashes, "counts": counts,
                "feedback_prompt_sha256": digest(PROMPT),
                "scope": "Same 2249/450/900 speaker holdout as cascade. Stage 1 uses training audio/transcripts only, not the paper's 200k CommonVoice alignment set.",
                "label_limitations": "Training feedback is authors' GPT-4o supervision, not individually human-verified; test references human checked. Repeated word keys collapse in published labels."}
    if (destination / "manifest.json").exists() and read_json(destination / "manifest.json") != manifest:
        raise ValueError("Frozen projector manifest changed")
    save(destination / "manifest.json", manifest)
    return manifest


def load(config):
    folder = rooted(config["dataset_dir"])
    manifest = read_json(folder / "manifest.json")
    cascade = read_json(rooted(config["cascade_config"]))
    check_common_split(cascade)
    if digest(rooted(cascade["protocol_dir"]) / "protocol.json") != manifest["protocol_sha256"] or digest(PROMPT) != manifest["feedback_prompt_sha256"]:
        raise ValueError("Protocol/prompt changed after projector data preparation")
    parts = {}
    for role, expected in manifest["sha256"].items():
        if digest(folder / (role + ".jsonl")) != expected:
            raise ValueError("Projector dataset checksum mismatch: " + role)
        parts[role] = read_rows(folder / (role + ".jsonl"))
    roles = {"train": "joint_train", "validation": "joint_validation", "test": "cascade_test"}
    splits = read_json(rooted(cascade["protocol_dir"]) / "splits.json")
    for role, split in roles.items():
        if {r["utterance_id"] for r in parts[role]} != set(splits[split + "_ids"]):
            raise ValueError("Projector split differs from cascade: " + role)
    return parts, manifest


def prompt_parts(tokenizer, row, stage):
    if stage not in ("alignment", "feedback"):
        raise ValueError("Unknown projector stage")
    if stage == "alignment":
        system = "Transcribe the English speech exactly. Return only the spoken sentence."
        user = "Audio: " + PLACEHOLDER
    else:
        system = PROMPT.read_text(encoding="utf-8").strip()
        # JSON escaping prevents reference text from injecting the audio placeholder.
        reference = json.dumps(row["reference_text"], ensure_ascii=True).replace("<", "\\u003c")
        user = "Learner audio: " + PLACEHOLDER + "\nReference sentence: " + reference
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix, suffix = rendered.split(PLACEHOLDER)
    return (tokenizer.encode(prefix, add_special_tokens=False), tokenizer.encode(suffix, add_special_tokens=False))


def target_ids(tokenizer, row, stage):
    text = row["reference_text"] if stage == "alignment" else json.dumps(row["feedback"], ensure_ascii=False)
    return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]


def encoder_identity(config):
    checkpoint = rooted(config["encoder_checkpoint"])
    completed = read_json(checkpoint.parent / "completed.json")
    metadata = read_json(checkpoint / "metadata.json")
    effective = metadata["effective_config"]
    cascade = read_json(rooted(config["cascade_config"]))
    if (completed.get("status") != "completed" or completed["optimizer_steps"] != effective["max_steps"]
            or metadata.get("purpose") != "scientific_training" or metadata.get("stage") != "B"
            or rooted(effective["protocol_dir"]) != rooted(cascade["protocol_dir"])):
        raise ValueError("Projector requires completed clean scientific encoder B")
    if metadata["provenance"]["files"]["protocol"]["sha256"] != digest(rooted(cascade["protocol_dir"]) / "protocol.json"):
        raise ValueError("Encoder B protocol mismatch")
    files = {name: digest(checkpoint / name) for name in ("model.safetensors", "config.json", "preprocessor_config.json")}
    return {"checkpoint": str(checkpoint), "files": files, "stride": config["feature_stride"],
            "sample_rate": 16000, "layer": "last_hidden_state", "audio_hidden_size": config["audio_hidden_size"]}


def cache_features(config):
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    sys.path.insert(0, str(ROOT / "src"))
    from mdd.runtime import load_audio
    torch.set_num_threads(config["cpu_threads"])
    parts, dataset = load(config)
    identity = {"encoder": encoder_identity(config), "dataset": dataset,
                "cache_code_sha256": digest(Path(__file__)), "dtype": "float16"}
    folder = rooted(config["feature_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    manifest = read_json(path) if path.exists() else {"identity": identity, "items": {}, "status": "caching"}
    if manifest["identity"] != identity:
        raise ValueError("Existing latent feature cache belongs to another encoder/data/code")
    checkpoint = rooted(config["encoder_checkpoint"])
    model = Wav2Vec2ForCTC.from_pretrained(str(checkpoint), local_files_only=True).wav2vec2.cpu().eval().requires_grad_(False)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(checkpoint), local_files_only=True)
    stride = config["feature_stride"]
    if stride != 1:
        raise ValueError("Primary comparison preserves every ~20ms wav2vec frame (stride=1)")
    with torch.inference_mode():
        for index, row in enumerate(r for role in ("train", "validation", "test") for r in parts[role]):
            uid = row["utterance_id"]
            output = folder / (uid + ".npy")
            prior = manifest["items"].get(uid)
            if prior and output.exists() and digest(output) == prior["sha256"]:
                continue
            audio = load_audio(row)
            batch = extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            hidden = model(**batch).last_hidden_state[0].float().numpy().astype(np.float16)
            if hidden.ndim != 2 or hidden.shape[1] != config["audio_hidden_size"] or not np.isfinite(hidden).all():
                raise ValueError("Invalid audio features: " + uid)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, hidden, allow_pickle=False)
            output.with_suffix(".tmp").replace(output)
            manifest["items"][uid] = {"sha256": digest(output), "shape": list(hidden.shape), "audio_sha256": row["audio_sha256"]}
            if index % 20 == 0:
                save(path, manifest)
                print(json.dumps({"cached": len(manifest["items"]), "total": sum(map(len, parts.values()))}), flush=True)
    manifest["status"] = "completed"
    save(path, manifest)
    return manifest


class LatentDataset:
    """Batch size one: keep all frames and all target tokens, never truncate."""
    def __init__(self, rows, tokenizer, config, stage):
        import numpy as np
        self.rows, self.tokenizer, self.config, self.stage = rows, tokenizer, config, stage
        self.folder = rooted(config["feature_dir"])
        self.manifest = read_json(self.folder / "manifest.json")
        if self.manifest["status"] != "completed" or self.manifest["identity"]["encoder"] != encoder_identity(config):
            raise ValueError("Incomplete or stale encoder cache")
        if self.manifest["identity"]["cache_code_sha256"] != digest(Path(__file__)):
            raise ValueError("Feature extraction code changed; use a new cache/run")
        self.lengths = []
        for row in rows:
            prefix, suffix = prompt_parts(tokenizer, row, stage)
            if self.manifest["items"][row["utterance_id"]]["audio_sha256"] != row["audio_sha256"]:
                raise ValueError("Feature audio does not match this dataset row")
            count = len(prefix) + len(suffix) + self.manifest["items"][row["utterance_id"]]["shape"][0] + len(target_ids(tokenizer, row, stage))
            if count > config["max_length"]:
                raise ValueError(f"Context {count} exceeds {config['max_length']}: {row['utterance_id']}; no truncation allowed")
            self.lengths.append(count)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import torch
        row = self.rows[index]
        path = self.folder / (row["utterance_id"] + ".npy")
        item = self.manifest["items"][row["utterance_id"]]
        if digest(path) != item["sha256"]:
            raise ValueError("Feature checksum mismatch: " + row["utterance_id"])
        features = np.load(path, allow_pickle=False)
        if list(features.shape) != item["shape"] or not np.isfinite(features).all():
            raise ValueError("Feature shape/values mismatch")
        prefix, suffix = prompt_parts(self.tokenizer, row, self.stage)
        return {"features": torch.from_numpy(features.astype(np.float32)).unsqueeze(0),
                "prefix_ids": torch.tensor([prefix]), "suffix_ids": torch.tensor([suffix]),
                "target_ids": torch.tensor([target_ids(self.tokenizer, row, self.stage)])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.cache:
        cache_features(config)
        print("Feature cache completed", flush=True)
    else:
        print(json.dumps(prepare(config), ensure_ascii=False, indent=2))
