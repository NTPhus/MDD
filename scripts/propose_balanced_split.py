"""Write an inactive, metadata-balanced alternative to the frozen v1 split.

The proposal uses speaker identity and README M/F metadata, never error labels,
predictions, or test metrics. Creating it does not change any training default.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path

if __package__ in {None, ""}:
    from audit_data_split import assert_group_disjoint, index_rows, read_speaker_metadata, select_rows, sha256
else:
    from .audit_data_split import assert_group_disjoint, index_rows, read_speaker_metadata, select_rows, sha256


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_ID = "l2arctic-v2-balanced"
# Fixed metadata-only design: every target train pair includes one M and one F.
TARGET_FOLDS = (
    (("THV", "TLV"), ("PNV",), ("HQTV",)),
    (("HQTV", "THV"), ("TLV",), ("PNV",)),
    (("PNV", "TLV"), ("HQTV",), ("THV",)),
    (("HQTV", "PNV"), ("THV",), ("TLV",)),
)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def validate_design(manifest, parent_splits, proposed_splits, metadata):
    """Validate group isolation, source identity, balance and exact coverage."""
    source_keys = ("source_train_speakers", "source_validation_speakers",
                   "source_train_ids", "source_validation_ids")
    for key in source_keys:
        if proposed_splits[key] != parent_splits[key]:
            raise ValueError(f"Proposal changed frozen source partition: {key}")
    source_train = select_rows(manifest, proposed_splits["source_train_ids"])
    source_dev = select_rows(manifest, proposed_splits["source_validation_ids"])
    assert_group_disjoint(source_train, source_dev)
    source = source_train + source_dev
    target_ids = {key for key, row in manifest.items() if row["native_language"] == "Vietnamese"}
    if {row["utterance_id"] for row in source} != manifest.keys() - target_ids:
        raise ValueError("Source must cover exactly the non-Vietnamese examples")
    if len(target_ids) != 600:
        raise ValueError("Expected all 600 manually annotated Vietnamese examples")
    role_counts = Counter()
    fold_reports = []
    if len(proposed_splits["folds"]) != 4:
        raise ValueError("Expected exactly four outer folds")
    if [fold["fold"] for fold in proposed_splits["folds"]] != [1, 2, 3, 4]:
        raise ValueError("Expected ordered fold identifiers 1 through 4")
    for fold in proposed_splits["folds"]:
        groups = {}
        for role, count in (("train", 300), ("validation", 150), ("test", 150)):
            subset = select_rows(manifest, fold[role + "_ids"])
            speakers = sorted({row["speaker_id"] for row in subset})
            if speakers != sorted(fold[role + "_speakers"]):
                raise ValueError(f"Fold {fold['fold']} {role} IDs and speakers disagree")
            if len(subset) != count:
                raise ValueError(f"Fold {fold['fold']} {role} expected {count} examples")
            if any(row["native_language"] != "Vietnamese" for row in subset):
                raise ValueError("Target partition contains a non-Vietnamese example")
            assert_group_disjoint(source, subset)
            groups[role] = subset
            for speaker in speakers:
                role_counts[speaker, role] += 1
        if set().union(*({row["utterance_id"] for row in subset} for subset in groups.values())) != target_ids:
            raise ValueError("Target fold must cover every Vietnamese example exactly once")
        for left, right in combinations(groups.values(), 2):
            assert_group_disjoint(left, right)
        train_speakers = fold["train_speakers"]
        if Counter(metadata[speaker]["gender"] for speaker in train_speakers) != {"M": 1, "F": 1}:
            raise ValueError(f"Fold {fold['fold']} train must include one M and one F speaker")
        fold_reports.append({
            "fold": fold["fold"],
            "counts": {role: len(subset) for role, subset in groups.items()},
            "train_gender_by_speaker": {speaker: metadata[speaker]["gender"] for speaker in train_speakers},
            "speaker_and_audio_sha256_disjoint_from_source": True,
            "speaker_and_audio_sha256_disjoint_between_target_partitions": True,
            "exact_target_coverage": True,
        })
    target_speakers = sorted({manifest[key]["speaker_id"] for key in target_ids})
    for speaker in target_speakers:
        if [role_counts[speaker, role] for role in ("train", "validation", "test")] != [2, 1, 1]:
            raise ValueError(f"Unbalanced role assignment for speaker: {speaker}")
    return {
        "all_checks_passed": True,
        "source_partitions_identical_to_parent": True,
        "folds": fold_reports,
        "role_counts": {speaker: {role: role_counts[speaker, role] for role in ("train", "validation", "test")}
                        for speaker in target_speakers},
    }


def build_proposal(root=ROOT):
    root = Path(root).resolve()
    protocol_dir = root / "data/protocol/l2arctic-v1"
    paths = {"manifest": protocol_dir / "manifest.jsonl", "splits": protocol_dir / "splits.json",
             "protocol": protocol_dir / "protocol.json", "protocol_checksums": protocol_dir / "SHA256SUMS",
             "dataset_readme": root / "l2arctic_release_v5.0/README.md"}
    for line in paths["protocol_checksums"].read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if sha256(protocol_dir / filename.lstrip("*")) != digest:
            raise ValueError(f"Frozen parent checksum mismatch: {filename}")
    inputs = {name: {"path": str(path.relative_to(root)), "sha256": sha256(path),
                     "size_bytes": path.stat().st_size} for name, path in paths.items()}
    protocol = json.loads(paths["protocol"].read_text())
    for name in ("manifest", "splits"):
        if inputs[name]["sha256"] != protocol[name + "_sha256"]:
            raise ValueError(f"Frozen parent protocol hash mismatch: {name}")
    manifest = index_rows((json.loads(line) for line in paths["manifest"].read_text().splitlines()), "manifest")
    metadata = read_speaker_metadata(paths["dataset_readme"].read_text())
    if len(manifest) != protocol["utterance_count"]:
        raise ValueError("Parent protocol count mismatch")
    for row in manifest.values():
        if not row["annotation_is_manual"]:
            raise ValueError("Expected only manually annotated examples")
        if row["native_language"] != metadata[row["speaker_id"]]["native_language"]:
            raise ValueError("Manifest and README native-language metadata disagree")
        row["audio_sha256"] = row["audio"]["sha256"]
    actual_counts = Counter(row["speaker_id"] for row in manifest.values())
    if actual_counts != Counter(protocol["counts_by_speaker"]):
        raise ValueError("Frozen parent per-speaker count mismatch")
    for speaker, count in actual_counts.items():
        if count != metadata[speaker]["manual_count"]:
            raise ValueError(f"README annotation count mismatch: {speaker}")
    parent_splits = json.loads(paths["splits"].read_text())
    proposed = {key: deepcopy(value) for key, value in parent_splits.items() if key != "folds"}
    proposed["folds"] = []
    for fold_index, design in enumerate(TARGET_FOLDS, 1):
        fold = {"fold": fold_index}
        for role, speakers in zip(("train", "validation", "test"), design):
            fold[role + "_speakers"] = list(speakers)
            fold[role + "_ids"] = sorted(key for key, row in manifest.items() if row["speaker_id"] in speakers)
        proposed["folds"].append(fold)
    validation = validate_design(manifest, parent_splits, proposed, metadata)
    script = Path(__file__).resolve()
    proposal = {
        "proposal_id": PROPOSAL_ID,
        "status": "proposed_not_active",
        "active_protocol_unchanged": "l2arctic-v1",
        "parent_protocol_id": protocol["protocol_id"],
        "inputs": inputs,
        "implementation": {"path": "scripts/propose_balanced_split.py", "sha256": sha256(script)},
        "proposed_splits_sha256": hashlib.sha256(json_bytes(proposed)).hexdigest(),
        "reason": "Metadata-only balancing: every Vietnamese training pair contains one M and one F speaker while preserving the source partitions and rotating each Vietnamese speaker through train twice, validation once and test once.",
        "selection_inputs": ["speaker identity", "Vietnamese L1 metadata", "README M/F metadata", "manual-example availability"],
        "error_labels_or_test_metrics_used_for_design": False,
        "data_reuse": "References the original frozen manifest and local raw paths; no raw data or model weights copied.",
        "validation": validation,
        "limits": [
            "No accuracy or F1 improvement is guaranteed without training and held-out evaluation.",
            "This balances the dataset's recorded binary M/F metadata; four speakers do not establish demographic representativeness.",
            "Speaker-disjoint evaluation retains shared ARCTIC prompts; this does not establish unseen-sentence or spontaneous-speech generalization.",
            "Audio disjointness uses frozen manifest SHA256 values, detecting byte identity rather than near duplicates; this script does not rehash the raw WAV files.",
            "A proposal file is not an active protocol: configs, runtime, notebook, v1 and Colab bundle remain unchanged.",
        ],
    }
    return proposed, proposal


def export_proposal(root, output, verify_only=False):
    proposed, proposal = build_proposal(root)
    contents = {"splits.json": json_bytes(proposed), "proposal.json": json_bytes(proposal)}
    contents["SHA256SUMS"] = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in contents.items()).encode("utf-8")
    output = Path(output)
    if verify_only:
        if not output.is_dir():
            raise ValueError(f"Proposal directory does not exist: {output}")
        if {path.name for path in output.iterdir()} != set(contents):
            raise ValueError("Proposal directory has missing or unexpected files")
        for name, expected in contents.items():
            if (output / name).read_bytes() != expected:
                raise ValueError(f"Proposal differs from the reproducible metadata-only design: {name}")
    else:
        # mkdir's exclusive creation also rejects an existing empty directory.
        output.mkdir(parents=True, exist_ok=False)
        for name, content in contents.items():
            (output / name).write_bytes(content)
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/proposals/l2arctic-v2-balanced")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    proposal = export_proposal(args.root, args.output, args.verify_only)
    action = "Verified" if args.verify_only else "Wrote"
    print(f"{action} {proposal['proposal_id']}: {proposal['status']}; active protocol remains l2arctic-v1; {args.output}")


if __name__ == "__main__":
    main()
