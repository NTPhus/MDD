"""Published native English labels, immutable speaker splits and token preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_l2arctic_plus import audit, identifier
from scripts.prepare_feedback_data import digest, write_rows

COMMIT = "e8c7684c545d94f2419fc4621e745236c192a353"
SOURCE_HASHES = {
    "README.md": "e4cbc2c6f7c2313231e8f0d722e4544ec0c728b948414f89422fc99a14db7e17",
    "LICENSE": "7402a974fba78d3b2c876c65aec7cd40318bdb87f9206614bde42fd423607a6d",
    "data/L2-Arctic-plus/train_data.json": "35b2176964cfbd0ff29d2b80e2dab9f180e066d8680bbe52f1a4652fd60149c6",
    "data/L2-Arctic-plus/test_data.json": "b922df6605ce16e3ca550122d19ccf99e190af0d671d8a419a53bf730025b041",
}
PROMPT = ROOT / "prompts/pronunciation_feedback_published.txt"
CONFIG = ROOT / "configs/feedback_l2arctic_plus_qlora.json"


def rooted(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def download_source(destination):
    files = {}
    for name, expected in SOURCE_HASHES.items():
        path = destination / name
        url = f"https://raw.githubusercontent.com/zoeyada/ALMs4Learning/{COMMIT}/{name}"
        if not path.is_file():
            with urlopen(Request(url, headers={"User-Agent": "MDD-published-feedback"}), timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Downloaded checksum mismatch: " + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if digest(path) != expected:
            raise ValueError("Source checksum mismatch: " + name)
        files[name] = {"url": url, "sha256": expected, "bytes": path.stat().st_size}
    manifest = destination / "download_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"repository": "https://github.com/zoeyada/ALMs4Learning", "commit": COMMIT,
            "files": files, "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")


def messages(row):
    payload = {"reference_text": row["reference_text"], "annotation_info": row["annotation_info"]}
    return [{"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def encode(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    targets = tokenizer(json.dumps(row["feedback"], ensure_ascii=False, separators=(",", ":")) + tokenizer.eos_token,
                        add_special_tokens=False)["input_ids"]
    if len(inputs) + len(targets) > max_length:
        raise ValueError(f"Pair {row['utterance_id']} exceeds {max_length}; do not truncate labels")
    return {"input_ids": inputs + targets, "attention_mask": [1] * (len(inputs) + len(targets)),
            "labels": [-100] * len(inputs) + targets}


def validate_feedback(value):
    if not isinstance(value, dict):
        raise ValueError("Feedback must be a JSON object")
    for word, pairs in value.items():
        if not isinstance(word, str) or not word.strip() or not isinstance(pairs, list) or not pairs:
            raise ValueError("Each word needs a nonempty list of issue/suggestion pairs")
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"issue", "suggestion"} or not all(
                isinstance(pair[key], str) and pair[key].strip() for key in ("issue", "suggestion")
            ):
                raise ValueError("Each item needs exactly two nonempty strings: issue and suggestion")


def read_native(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_feedback(row["feedback"])
        if row["source_commit"] != COMMIT or row["language"] != "en" or row.get("human_approved") is not None:
            raise ValueError("Incorrect published-label provenance")
        if set(row["feedback"]) - set(row["annotation_info"]):
            raise ValueError("Published feedback references an unknown word")
    return rows


def prepare(config, *, download=False):
    source, destination = rooted(config["source_dir"]), rooted(config["dataset_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Dataset already exists; run --check or choose a new dataset_dir")
    if download:
        download_source(source)
    for name, expected in SOURCE_HASHES.items():
        if digest(source / name) != expected:
            raise ValueError("Pinned source mismatch: " + name)
    source_audit = audit(source)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(rooted(config["model_path"])), local_files_only=True, trust_remote_code=False)
    native = {name: json.loads((source / "data/L2-Arctic-plus" / f"{name}_data.json").read_text(encoding="utf-8"))
              for name in ("train", "test")}
    validation = set(config["validation_speakers"])
    if not validation or not validation < set(source_audit["partitions"]["train"]["speakers"]):
        raise ValueError("Validation speakers must be a proper subset of official train speakers")
    parts = {name: [] for name in ("train", "validation", "test")}
    excluded = []
    for official, values in native.items():
        for row in values:
            uid = identifier(row)
            speaker = uid.split("/")[0]
            split = "test" if official == "test" else "validation" if speaker in validation else "train"
            converted = {"utterance_id": uid, "speaker_id": speaker, "reference_text": row["text"],
                "annotation_info": row["annotation_info"], "feedback": row["mis_exp_sug"],
                "language": "en", "source_commit": COMMIT, "source_split": official,
                "source_row_sha256": json_digest(row), "human_approved": None,
                "feedback_provenance": "authors_human_checked_reference" if official == "test" else "authors_gpt4o_training_label"}
            validate_feedback(converted["feedback"])
            encoded = encode(tokenizer, converted, 1_000_000)
            converted["total_tokens"] = len(encoded["input_ids"])
            converted["target_tokens"] = sum(value != -100 for value in encoded["labels"])
            if split != "test" and converted["total_tokens"] > config["max_length"]:
                excluded.append({"utterance_id": uid, "split": split, "tokens": converted["total_tokens"], "reason": "context_limit_no_truncation"})
                continue
            parts[split].append(converted)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        write_rows(destination / f"{name}.jsonl", rows)
    manifest = {"dataset_kind": "published_l2arctic_plus", "task": config["task"], "language": "en",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": COMMIT, "source_hashes": SOURCE_HASHES,
        "prompt_sha256": digest(PROMPT), "tokenizer_sha256": digest(rooted(config["model_path"]) / "tokenizer.json"),
        "counts": {name: len(rows) for name, rows in parts.items()}, "excluded": excluded,
        "sha256": {name: digest(destination / f"{name}.jsonl") for name in parts},
        "speakers": {name: sorted({row["speaker_id"] for row in rows}) for name, rows in parts.items()},
        "validation_speakers": sorted(validation), "max_length": config["max_length"],
        "official_test_preserved": len(parts["test"]) == len(native["test"]),
        "grouping": "Disjoint speakers and utterances; official test retained; shared read prompts allowed as in publication",
        "source_audit": source_audit,
        "human_review": "No project review invented; train labels are authors' GPT-4o outputs; test verification is authors' procedure",
        "evaluation_scope": "Annotated phones supplied as input; not independent acoustic mispronunciation detection"}
    if config.get("cascade_protocol_dir"):
        manifest["cascade_protocol_dir"] = config["cascade_protocol_dir"]
        manifest["cascade_protocol_sha256"] = digest(rooted(config["cascade_protocol_dir"]) / "protocol.json")
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load(directory=destination)
    return manifest


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_kind") != "published_l2arctic_plus" or manifest.get("source_commit") != COMMIT:
        raise ValueError("Not the pinned published feedback dataset")
    if manifest["prompt_sha256"] != digest(PROMPT) or manifest["source_hashes"] != SOURCE_HASHES:
        raise ValueError("Published prompt or source changed after freezing")
    if manifest.get("cascade_protocol_dir") and manifest["cascade_protocol_sha256"] != digest(rooted(manifest["cascade_protocol_dir"]) / "protocol.json"):
        raise ValueError("Shared cascade protocol changed after freezing")
    parts, seen_ids, seen_speakers = {}, set(), set()
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if digest(path) != manifest["sha256"][name]:
            raise ValueError("Frozen dataset checksum mismatch: " + name)
        rows = read_native(path)
        ids, speakers = {row["utterance_id"] for row in rows}, {row["speaker_id"] for row in rows}
        if len(ids) != len(rows) or ids & seen_ids or speakers & seen_speakers:
            raise ValueError("Published dataset has duplicate IDs or speaker leakage")
        if len(rows) != manifest["counts"][name] or sorted(speakers) != manifest["speakers"][name]:
            raise ValueError("Published manifest count/speaker mismatch")
        if any(row["source_split"] != ("test" if name == "test" else "train") for row in rows):
            raise ValueError("Official test labels entered training or validation")
        seen_ids.update(ids)
        seen_speakers.update(speakers)
        parts[name] = rows
    return parts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check:
        _, manifest = load(rooted(config["dataset_dir"]))
    else:
        manifest = prepare(config, download=args.download)
    print(json.dumps({key: manifest[key] for key in ("counts", "speakers", "excluded", "official_test_preserved")}, indent=2))


if __name__ == "__main__":
    main()
