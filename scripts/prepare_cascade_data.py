"""Freeze one clean split for audio training, published coaching and joint tests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.l2arctic_plus_data import COMMIT, SOURCE_HASHES, identifier, prepare as prepare_feedback, rooted, load
from scripts.prepare_feedback_data import digest
from scripts.prepare_l2arctic import prepare as prepare_audio

CONFIG = ROOT / "configs/cascade_l2arctic_plus.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_common_split(config):
    directory = rooted(config["protocol_dir"])
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    for name, suffix in (("manifest", ".jsonl"), ("splits", ".json")):
        if digest(directory / (name + suffix)) != protocol[name + "_sha256"]:
            raise ValueError("Frozen cascade protocol changed: " + name)
    split = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    roles = {name: set(split[name + "_ids"]) for name in ("joint_train", "joint_validation", "cascade_test")}
    for left, right in (("joint_train", "joint_validation"), ("joint_train", "cascade_test"), ("joint_validation", "cascade_test")):
        if roles[left] & roles[right] or {uid.split("/")[0] for uid in roles[left]} & {uid.split("/")[0] for uid in roles[right]}:
            raise ValueError("Shared utterance or speaker between " + left + " and " + right)
    feedback_config = json.loads(rooted(config["feedback_config"]).read_text(encoding="utf-8"))
    parts, manifest = load(rooted(feedback_config["dataset_dir"]))
    for qwen_role, audio_role in (("train", "joint_train"), ("validation", "joint_validation"), ("test", "cascade_test")):
        if {row["utterance_id"] for row in parts[qwen_role]} != roles[audio_role]:
            raise ValueError("Qwen and audio splits disagree: " + qwen_role)
    if manifest.get("cascade_protocol_sha256") != digest(directory / "protocol.json"):
        raise ValueError("Qwen data are not bound to this audio protocol")
    return protocol, split, manifest


def prepare(config):
    original = rooted(config["original_protocol"])
    source = rooted(config["source_dir"])
    for name, expected in SOURCE_HASHES.items():
        if digest(source / name) != expected:
            raise ValueError("Published source checksum mismatch: " + name)
    old_protocol = json.loads((original / "protocol.json").read_text(encoding="utf-8"))
    if digest(original / "manifest.jsonl") != old_protocol["manifest_sha256"]:
        raise ValueError("Original audio manifest changed")
    records = [json.loads(line) for line in (original / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    native = {name: json.loads((source / "data/L2-Arctic-plus" / f"{name}_data.json").read_text(encoding="utf-8")) for name in ("train", "test")}
    official = {name: {identifier(row) for row in values} for name, values in native.items()}
    if official["train"] & official["test"] or official["train"] | official["test"] != {row["utterance_id"] for row in records}:
        raise ValueError("Published and audio utterance sets differ")
    validation = set(config["validation_speakers"])
    joint_validation = sorted(uid for uid in official["train"] if uid.split("/")[0] in validation)
    joint_train = sorted(official["train"] - set(joint_validation))
    target = set(config["target_train_speakers"])
    source_train = sorted(uid for uid in joint_train if uid.split("/")[0] not in target)
    source_validation = sorted(uid for uid in joint_validation if uid.split("/")[0] not in set(config["target_validation_speakers"]))
    ids_for = lambda speakers: sorted(row["utterance_id"] for row in records if row["speaker_id"] in set(speakers))
    splits = {
        "source_train_ids": source_train, "source_validation_ids": source_validation,
        "source_train_speakers": sorted({uid.split("/")[0] for uid in source_train}),
        "source_validation_speakers": sorted({uid.split("/")[0] for uid in source_validation}),
        "source_test_ids": sorted(uid for uid in official["test"] if uid.split("/")[0] not in set(config["target_test_speakers"])),
        "joint_train_ids": joint_train, "joint_validation_ids": joint_validation,
        "cascade_test_ids": sorted(official["test"]),
        "folds": [{"fold": 1,
            **{role + "_speakers": config["target_" + role + "_speakers"] for role in ("train", "validation", "test")},
            **{role + "_ids": ids_for(config["target_" + role + "_speakers"]) for role in ("train", "validation", "test")}}],
    }
    folder = rooted(config["protocol_dir"])
    if not folder.exists():
        folder.mkdir(parents=True)
        (folder / "manifest.jsonl").write_bytes((original / "manifest.jsonl").read_bytes())
        save(folder / "splits.json", splits)
        protocol = {**old_protocol, "protocol_id": config["protocol_id"], "source_commit": COMMIT,
            "splits_sha256": digest(folder / "splits.json"), "source_train_count": len(source_train),
            "source_validation_count": len(source_validation), "joint_counts": {"train": len(joint_train), "validation": len(joint_validation), "test": len(official["test"])},
            "scope": "Shared speaker/utterance holdout for wav2vec2 and Qwen; shared read prompts follow published setting; old A/B checkpoints excluded"}
        save(folder / "protocol.json", protocol)
        (folder / "SHA256SUMS").write_text("".join(f"{digest(folder / name)}  {name}\n" for name in ("manifest.jsonl", "splits.json", "protocol.json")), encoding="utf-8")
    elif json.loads((folder / "splits.json").read_text(encoding="utf-8")) != splits:
        raise ValueError("Existing frozen split differs; select a new protocol directory")
    audio_audit = prepare_audio(folder / "manifest.jsonl", rooted(config["processed_dir"]), verify_audio=True)
    feedback_config = json.loads(rooted(config["feedback_config"]).read_text(encoding="utf-8"))
    if not (rooted(feedback_config["dataset_dir"]) / "manifest.json").exists():
        prepare_feedback(feedback_config)
    protocol, splits, feedback_manifest = check_common_split(config)
    old_splits = json.loads((original / "splits.json").read_text(encoding="utf-8"))
    report = {"protocol_id": config["protocol_id"], "source_commit": COMMIT, "joint_counts": protocol["joint_counts"],
        "audio_source_train": len(source_train), "audio_source_validation": len(source_validation),
        "audio_target_counts": {role: len(splits["folds"][0][role + "_ids"]) for role in ("train", "validation", "test")},
        "test_train_overlap": 0, "test_validation_overlap": 0, "audio_hashes_verified": audio_audit["audio_sha256_verified"],
        "qwen_token_exclusions": feedback_manifest["excluded"],
        "old_source_train_overlap_with_new_test": len(set(old_splits["source_train_ids"]) & official["test"]),
        "old_checkpoints_eligible": False, "training_started_by_prepare": False}
    save(rooted(config["report_root"]) / "data_readiness.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check:
        protocol, _, _ = check_common_split(config)
        print(json.dumps(protocol["joint_counts"], indent=2))
    else:
        print(json.dumps(prepare(config), indent=2))


if __name__ == "__main__":
    main()
