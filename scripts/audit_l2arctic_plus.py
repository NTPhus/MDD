"""Audit the pinned public release; never approve labels, call providers or train."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_feedback_data import digest


def identifier(row):
    path = PurePosixPath(row["audio_path"])
    return path.parts[-3] + "/" + path.stem


def prompt_key(row):
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)*", row["text"].casefold()))


def audit(source):
    downloaded = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
    for name, entry in downloaded["files"].items():
        if digest(source / name) != entry["sha256"]:
            raise ValueError("Published source checksum changed: " + name)
    rows = {name: json.loads((source / "data/L2-Arctic-plus" / f"{name}_data.json").read_text(encoding="utf-8"))
            for name in ("train", "test")}
    summary = {}
    for name, values in rows.items():
        for row in values:
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Missing reference text")
            if not isinstance(row["annotation_info"], dict) or not isinstance(row["mis_exp_sug"], dict):
                raise ValueError("Invalid annotation shape")
            for word, suggestions in row["mis_exp_sug"].items():
                if word not in row["annotation_info"] or not isinstance(suggestions, list):
                    raise ValueError("Feedback references an unknown word")
                for feedback in suggestions:
                    if not all(isinstance(feedback.get(key), str) and feedback[key].strip()
                               for key in ("issue", "suggestion")):
                        raise ValueError("Missing error explanation or corrective suggestion")
        ids = [identifier(row) for row in values]
        summary[name] = {
            "records": len(values), "duplicate_utterance_ids": len(ids) - len(set(ids)),
            "speakers": dict(sorted(Counter(uid.split("/")[0] for uid in ids).items())),
            "unique_prompts": len({prompt_key(row) for row in values}),
            "no_flagged_word_feedback": sum(not row["mis_exp_sug"] for row in values),
            "feedback_pairs": sum(len(pairs) for row in values for pairs in row["mis_exp_sug"].values()),
        }
    overlap = {"utterances": len({identifier(row) for row in rows["train"]} & {identifier(row) for row in rows["test"]}),
               "speakers": sorted(set(summary["train"]["speakers"]) & set(summary["test"]["speakers"])),
               "normalized_reference_prompts": len({prompt_key(row) for row in rows["train"]} & {prompt_key(row) for row in rows["test"]})}
    if overlap["utterances"] or overlap["speakers"] or any(item["duplicate_utterance_ids"] for item in summary.values()):
        raise ValueError("Official utterance/speaker partitions are not disjoint")
    return {
        "dataset": "L2-Arctic-plus", "source_commit": downloaded["commit"],
        "source_directory": source.relative_to(ROOT).as_posix(),
        "paper_url": "https://aclanthology.org/2026.findings-eacl.190.pdf",
        "repository": downloaded["repository"], "source_checksums_verified": True,
        "partitions": summary, "official_train_test_overlap": overlap,
        "annotation_provenance": {
            "phone_labels": "L2-ARCTIC manual annotations",
            "train_feedback": "GPT-4o-generated; paper section 6.1 explicitly states no human verification",
            "test_feedback": "GPT-4o-generated and checked by three annotators; paper section 3.2",
            "original_feedback_language": "en", "vietnamese_translation_included": False,
        },
        "scope": "Source/schema/split audit only; does not establish that every teaching suggestion is correct",
        "split_notes": "Official split holds out speakers, with shared read prompts; it is not an unseen-prompt evaluation",
        "readme_record_count_discrepancy": {"readme_train": 1699, "actual_train": len(rows["train"])},
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/external/l2arctic-plus/e8c7684c545d94f2419fc4621e745236c192a353")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/setup/l2arctic_plus_audit.json")
    args = parser.parse_args()
    report = audit(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
