"""Import coaching drafts and split teacher-reviewed assessment/feedback pairs."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from jsonschema import ValidationError

if __package__:
    from .llm_client import ROOT, validate_local_feedback
    from .validate_contracts import validate_pair
else:
    from llm_client import ROOT, validate_local_feedback
    from validate_contracts import validate_pair


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prompt_key(assessment):
    # Keep the same English exercise out of both train and held-out sets.
    return " ".join(assessment["reference_text"].casefold().split())


def read_rows(path, *, reviewed=True):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        validate_pair(row["assessment"], row["feedback"])
        validate_local_feedback(row["assessment"], row["feedback"])
        if row["feedback"]["language"] != "vi":
            raise ValueError(f"Line {number}: Vietnamese coaching is required")
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError(f"Line {number}: provide a group_id for the exercise/learner family")
        if reviewed:
            review = row.get("review", {})
            if review.get("approved") is not True or not str(review.get("reviewer") or "").strip():
                raise ValueError(f"Line {number}: teacher review is missing; drafts cannot be trained")
            reviewed_at = review.get("reviewed_at")
            if not isinstance(reviewed_at, str):
                raise ValueError(f"Line {number}: review needs an ISO reviewed_at date")
            datetime.fromisoformat(reviewed_at)
        rows.append(row)
    if not rows:
        raise ValueError("No feedback pairs found")
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def split_rows(rows, seed=42):
    # Merge group IDs that share a prompt, including transitive connections.
    parents = {row["group_id"]: row["group_id"] for row in rows}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    prompts, ids = {}, set()
    for row in rows:
        identifier = row["assessment"]["utterance_id"]
        if identifier in ids:
            raise ValueError("Duplicate utterance_id: " + identifier)
        ids.add(identifier)
        key = prompt_key(row["assessment"])
        if key in prompts:
            parents[find(row["group_id"])] = find(prompts[key])
        prompts[key] = row["group_id"]
    groups = sorted({find(group) for group in parents})
    if len(groups) < 3:
        raise ValueError("Need at least three independent reviewed groups for train/validation/test")
    random.Random(seed).shuffle(groups)
    held_out = max(1, len(groups) // 10)
    partitions = {group: "validation" if i < held_out else "test" if i < held_out * 2 else "train"
                  for i, group in enumerate(groups)}
    result = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        result[partitions[find(row["group_id"])]].append(row)
    return result


def prepare(source, destination, seed=42):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty output directory to preserve the frozen split")
    rows = read_rows(source)
    parts = split_rows(rows, seed)
    destination.mkdir(parents=True, exist_ok=True)
    for name, values in parts.items():
        write_rows(destination / f"{name}.jsonl", values)
    manifest = {
        "source_sha256": digest(source), "seed": seed,
        "prompt_sha256": digest(ROOT / "prompts/pronunciation_feedback_local.txt"),
        "counts": {name: len(values) for name, values in parts.items()},
        "sha256": {name: digest(destination / f"{name}.jsonl") for name in parts},
        "data_kinds": dict(Counter(row["assessment"]["data_kind"] for row in rows)),
        "grouping": "group_id with shared reference prompts merged; no group or prompt overlap",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_partitions(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_sha256"] != digest(ROOT / "prompts/pronunciation_feedback_local.txt"):
        raise ValueError("The coaching prompt changed after the dataset split was frozen")
    parts = {}
    seen_ids, seen_groups, seen_prompts = set(), set(), set()
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if digest(path) != manifest["sha256"][name]:
            raise ValueError("Frozen dataset checksum mismatch: " + name)
        rows = read_rows(path)
        ids = {row["assessment"]["utterance_id"] for row in rows}
        groups = {row["group_id"] for row in rows}
        prompts = {prompt_key(row["assessment"]) for row in rows}
        if len(ids) != len(rows) or ids & seen_ids or groups & seen_groups or prompts & seen_prompts:
            raise ValueError("Feedback dataset contains duplicates or held-out leakage")
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_prompts.update(prompts)
        parts[name] = rows
    return parts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="Import JSON downloaded from the demo for teacher review")
    draft.add_argument("demo_json", type=Path, nargs="+")
    draft.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("prepare", help="Validate reviewed pairs and freeze a grouped split")
    split.add_argument("source", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.source, args.output, args.seed), ensure_ascii=False, indent=2))
        else:
            if args.output.exists():
                raise ValueError("Draft output already exists; choose a new file")
            rows = []
            for path in args.demo_json:
                value = json.loads(path.read_text(encoding="utf-8"))
                assessment = value["assessment"]
                feedback = value["guidance"]["feedback"]
                validate_pair(assessment, feedback)
                rows.append({"group_id": "prompt-" + hashlib.sha256(prompt_key(assessment).encode()).hexdigest()[:16],
                             "source": "demo-feedback-draft", "assessment": assessment, "feedback": feedback,
                             "review": {"approved": False, "reviewer": None, "reviewed_at": None}})
            write_rows(args.output, rows)
            print(f"Saved {len(rows)} unreviewed drafts; check teaching accuracy before approving.")
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
