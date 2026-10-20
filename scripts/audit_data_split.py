"""Audit the frozen data split without fitting, tuning, or changing it.

Only the Python standard library is needed. Prompt overlap is reported both by
CMU ARCTIC sentence ID and by literal normalized transcript because transcript
variants must not be mistaken for genuinely new sentences.
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERROR_KINDS = ("substitution", "deletion", "addition")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt(text):
    """Lowercase and collapse whitespace only; keep punctuation and wording."""
    return " ".join(text.lower().split())


def index_rows(rows, name):
    indexed = {}
    for row in rows:
        key = row["utterance_id"]
        if key in indexed:
            raise ValueError(f"Duplicate utterance ID in {name}: {key}")
        indexed[key] = row
    return indexed


def select_rows(index, ids):
    if len(ids) != len(set(ids)):
        raise ValueError("Split contains a repeated utterance ID")
    missing = set(ids) - index.keys()
    if missing:
        raise ValueError(f"Split references absent utterances: {sorted(missing)}")
    return [index[key] for key in ids]


def group_key(row, group):
    if group == "prompt_id":
        return row["utterance_id"].rsplit("/", 1)[-1]
    if group == "prompt_text":
        return normalize_prompt(row["prompt_text"])
    return row[group]


def assert_group_disjoint(left, right, groups=("speaker_id", "audio_sha256")):
    """Reject leakage for the declared evaluation scope, including hash aliases.

    Prompt disjointness is optional because a fixed-prompt reading task can
    deliberately evaluate new speakers reading familiar prompts.
    """
    for group in groups:
        shared = {group_key(row, group) for row in left} & {
            group_key(row, group) for row in right}
        if shared:
            raise ValueError(f"Overlapping {group}: {sorted(shared)}")


def summarize(rows):
    counts = Counter({kind: 0 for kind in ERROR_KINDS})
    canonical = 0
    canonical_error_positions = set()
    utterances_with_errors = 0
    for row in rows:
        phones = row["canonical_phones"]
        canonical += len(phones)
        events = row["gold_error_events"]
        utterances_with_errors += bool(events)
        for event in events:
            kind = event.get("kind", event.get("type"))
            if kind not in ERROR_KINDS:
                raise ValueError(f"Unexpected gold event kind: {kind}")
            counts[kind] += 1
            if kind == "addition":
                if not 0 <= event["position"] <= len(phones):
                    raise ValueError("Insertion gap outside canonical sequence")
            else:
                position = event["canonical_index"]
                if not isinstance(position, int) or not 0 <= position < len(phones):
                    raise ValueError("Error outside canonical sequence")
                slot = (row["utterance_id"], position)
                if slot in canonical_error_positions:
                    raise ValueError("Repeated error at one canonical phone")
                canonical_error_positions.add(slot)
    duration = sum(row["duration_sec"] for row in rows)
    return {
        "utterances": len(rows),
        "speakers": sorted({row["speaker_id"] for row in rows}),
        "native_languages": dict(sorted(Counter(row["native_language"] for row in rows).items())),
        "genders": dict(sorted(Counter(row["gender"] for row in rows).items())),
        "duration_seconds": round(duration, 9),
        "duration_minutes": round(duration / 60, 6),
        "duration_hours": round(duration / 3600, 9),
        "canonical_phones": canonical,
        "perceived_phones": sum(len(row["perceived_phones"]) for row in rows),
        "gold_error_counts": dict(counts),
        "canonical_error_positions": len(canonical_error_positions),
        "canonical_error_fraction": len(canonical_error_positions) / canonical if canonical else None,
        "correct_canonical_fraction": 1 - len(canonical_error_positions) / canonical if canonical else None,
        "additions_per_canonical_phone": counts["addition"] / canonical if canonical else None,
        "utterances_with_errors": utterances_with_errors,
        "utterance_error_fraction": utterances_with_errors / len(rows) if rows else None,
        "distinct_audio_hashes": len({row["audio_sha256"] for row in rows}),
        "distinct_prompt_ids": len({group_key(row, "prompt_id") for row in rows}),
        "distinct_normalized_prompt_texts": len({group_key(row, "prompt_text") for row in rows}),
    }


def compare_groups(left, right):
    report = {}
    for group in ("utterance_id", "speaker_id", "audio_sha256", "prompt_id", "prompt_text"):
        left_groups = {group_key(row, group) for row in left}
        right_groups = {group_key(row, group) for row in right}
        shared = left_groups & right_groups
        item = {
            "left_distinct": len(left_groups), "right_distinct": len(right_groups),
            "shared_distinct": len(shared),
            "left_utterances_in_shared_groups": sum(group_key(row, group) in shared for row in left),
            "right_utterances_in_shared_groups": sum(group_key(row, group) in shared for row in right),
            "right_distinct_fraction_seen_in_left": len(shared) / len(right_groups) if right_groups else None,
        }
        if group in {"utterance_id", "speaker_id", "audio_sha256"}:
            item["shared_values"] = sorted(shared)
        else:
            item["right_unseen_values"] = sorted(right_groups - left_groups)
        report[group] = item
    return report


def read_speaker_metadata(readme):
    speakers = {}
    for line in readme.splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if (len(fields) == 5 and fields[1] in {"M", "F"}
                and fields[3].isdigit() and fields[4].isdigit()):
            speakers[fields[0]] = {"gender": fields[1], "native_language": fields[2],
                                   "full_wav_count": int(fields[3]), "manual_count": int(fields[4])}
    if not speakers:
        raise ValueError("Could not read speaker table from the raw dataset README")
    return speakers


def duplicate_groups(rows, group):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row, group)].append(row["utterance_id"])
    return {key: sorted(ids) for key, ids in sorted(grouped.items()) if len(ids) > 1}


def build_audit(root=ROOT):
    root = Path(root).resolve()
    paths = {
        "manifest": root / "data/protocol/l2arctic-v1/manifest.jsonl",
        "splits": root / "data/protocol/l2arctic-v1/splits.json",
        "protocol": root / "data/protocol/l2arctic-v1/protocol.json",
        "protocol_checksums": root / "data/protocol/l2arctic-v1/SHA256SUMS",
        "annotations": root / "data/processed/l2arctic-v1/annotations.jsonl",
        "parser_audit": root / "data/processed/l2arctic-v1/audit.json",
        "normalization": root / "data/processed/l2arctic-v1/normalization.json",
        "vocabulary": root / "data/processed/l2arctic-v1/vocab.json",
        "processed_checksums": root / "data/processed/l2arctic-v1/SHA256SUMS",
        "dataset_readme": root / "l2arctic_release_v5.0/README.md",
        "audit_script": Path(__file__).resolve(),
    }
    inputs = {name: {"path": str(path.relative_to(root)), "sha256": sha256(path),
                     "size_bytes": path.stat().st_size} for name, path in paths.items()}
    for name in ("protocol_checksums", "processed_checksums"):
        for line in paths[name].read_text().splitlines():
            checksum, filename = line.split(maxsplit=1)
            if sha256(paths[name].parent / filename.lstrip("*")) != checksum:
                raise ValueError(f"Frozen checksum mismatch in {name}: {filename}")
    protocol = json.loads(paths["protocol"].read_text())
    for name in ("manifest", "splits"):
        if inputs[name]["sha256"] != protocol[name + "_sha256"]:
            raise ValueError(f"Protocol hash mismatch: {name}")
    manifest = index_rows((json.loads(line) for line in paths["manifest"].read_text().splitlines()), "manifest")
    annotations = index_rows((json.loads(line) for line in paths["annotations"].read_text().splitlines()), "annotations")
    if manifest.keys() != annotations.keys():
        raise ValueError("Processed annotations do not cover exactly the frozen manifest")
    if len(manifest) != protocol["utterance_count"]:
        raise ValueError("Protocol utterance count does not match the actual manifest")
    speakers = read_speaker_metadata(paths["dataset_readme"].read_text())
    verified = Counter()
    for key, row in annotations.items():
        source = manifest[key]
        if not source["annotation_is_manual"]:
            raise ValueError(f"Non-manual example: {key}")
        for field in ("speaker_id", "native_language", "sample_rate", "duration_sec"):
            if row[field] != source[field]:
                raise ValueError(f"Metadata mismatch {key}: {field}")
        if row["audio_path"] != source["audio"]["path"] or row["source_annotation_sha256"] != source["annotation"]["sha256"]:
            raise ValueError(f"Processed provenance mismatch: {key}")
        for field in ("audio", "annotation", "transcript"):
            entry = source[field]
            path = root / entry["path"]
            if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Raw source checksum mismatch: {entry['path']}")
            verified[field] += 1
        if row["prompt_text"] != (root / source["transcript"]["path"]).read_text().strip():
            raise ValueError(f"Prompt transcript mismatch: {key}")
        speaker = speakers[row["speaker_id"]]
        if speaker["native_language"] != row["native_language"]:
            raise ValueError(f"README native-language mismatch: {key}")
        row["gender"] = speaker["gender"]
        row["audio_sha256"] = source["audio"]["sha256"]
    rows = list(annotations.values())
    if Counter(row["speaker_id"] for row in rows) != Counter(protocol["counts_by_speaker"]):
        raise ValueError("Protocol speaker counts do not match the actual manifest")
    splits = json.loads(paths["splits"].read_text())
    source_train = select_rows(annotations, splits["source_train_ids"])
    source_validation = select_rows(annotations, splits["source_validation_ids"])
    vn_ids = {row["utterance_id"] for row in rows if row["native_language"] == "Vietnamese"}
    assert_group_disjoint(source_train, source_validation)
    if set(splits["source_train_ids"]) | set(splits["source_validation_ids"]) != annotations.keys() - vn_ids:
        raise ValueError("Source split must cover all and only the non-Vietnamese manual examples")
    for role, subset in (("train", source_train), ("validation", source_validation)):
        if sorted({row["speaker_id"] for row in subset}) != sorted(splits[f"source_{role}_speakers"]):
            raise ValueError(f"Source {role} speaker metadata does not match actual IDs")
    fold_reports = []
    for fold in splits["folds"]:
        groups = {role: select_rows(annotations, fold[role + "_ids"]) for role in ("train", "validation", "test")}
        if set().union(*(set(fold[role + "_ids"]) for role in groups)) != vn_ids:
            raise ValueError(f"Fold {fold['fold']} does not partition all Vietnamese examples")
        for role, subset in groups.items():
            if sorted({row["speaker_id"] for row in subset}) != sorted(fold[role + "_speakers"]):
                raise ValueError(f"Fold {fold['fold']} {role} speaker metadata mismatch")
            assert_group_disjoint(source_train + source_validation, subset)
        for left, right in combinations(groups, 2):
            assert_group_disjoint(groups[left], groups[right])
        comparisons = {f"vn_{left}_vs_vn_{right}": compare_groups(groups[left], groups[right])
                       for left, right in combinations(groups, 2)}
        for name, subset in (("source_train", source_train), ("source_validation", source_validation)):
            for role, target in groups.items():
                comparisons[f"{name}_vs_vn_{role}"] = compare_groups(subset, target)
        # B uses an A checkpoint and source replay: earlier source exposure must
        # be included when assessing whether a test prompt has been unseen.
        for role in ("validation", "test"):
            comparisons[f"all_gradient_data_vs_vn_{role}"] = compare_groups(source_train + groups["train"], groups[role])
        fold_reports.append({"fold": fold["fold"], "splits": {role: summarize(subset) for role, subset in groups.items()},
                             "overlaps": comparisons})
    variants = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variants[group_key(row, "prompt_id")][group_key(row, "prompt_text")].append(row["utterance_id"])
    transcript_variants = {key: {text: sorted(ids) for text, ids in sorted(value.items())}
                           for key, value in sorted(variants.items()) if len(value) > 1}
    return {
        "audit_version": "frozen-split-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "scope": "Descriptive audit only; no training, predictions, split changes, or label-based split optimization",
        "inputs": inputs,
        "raw_files_verified_against_manifest": dict(verified),
        "all_speaker_utterance_audio_hash_disjointness_checks_passed": True,
        "global": summarize(rows),
        "by_speaker": {speaker: summarize([row for row in rows if row["speaker_id"] == speaker]) for speaker in sorted(speakers)},
        "source_splits": {"train": summarize(source_train), "validation": summarize(source_validation)},
        "source_train_vs_source_validation": compare_groups(source_train, source_validation),
        "folds": fold_reports,
        "duplicate_audio_hash_groups": duplicate_groups(rows, "audio_sha256"),
        "prompt_ids_with_multiple_normalized_transcripts": transcript_variants,
        "definitions": {
            "canonical_error_fraction": "(S+D) / N canonical phone positions; manual event labels retained even when normalization makes a substitution look equal",
            "additions_per_canonical_phone": "I / N canonical phones; insertion events are separate from canonical phone classification",
            "prompt_id": "Final component of utterance_id, e.g. arctic_a0003; shared sentence identity across speakers",
            "prompt_text": "Lowercase, strip/collapse whitespace only; no punctuation removal or semantic matching",
            "all_gradient_data": "Union of source_train and current Vietnamese train IDs, including source training before adaptation",
            "gender": "M/F metadata copied from the dataset's speaker table, not inferred from audio",
            "hash_scope": "Byte-identical source WAV files; does not detect re-encoded or near-duplicate recordings",
        },
        "interpretation": {
            "speaker_disjoint": True,
            "prompt_disjoint": False,
            "test_scope": "New Vietnamese speakers reading the fixed ARCTIC prompt inventory; does not establish generalization to unseen sentences or spontaneous conversation",
            "training_improvement": "A split provides evaluation evidence; only training and held-out results can establish a model improvement",
            "uncertainty": "Only four Vietnamese speakers; 600 utterances do not constitute 600 independent speakers",
            "pretraining_overlap": "Not established by this audit; it covers the local fine-tuning corpus only",
        },
    }


def markdown_report(report):
    overall = report["global"]
    lines = ["# Kiểm tra tập dữ liệu đã đóng băng", "",
             "Sinh bằng `python3 scripts/audit_data_split.py`. Không thay đổi split và không train model.", "",
             f"- {overall['utterances']:,} mẫu annotate thủ công; {len(overall['speakers'])} người nói; {overall['duration_hours']:.6f} giờ.",
             f"- {overall['distinct_audio_hashes']:,} SHA256 audio riêng biệt; không trùng người nói, utterance ID hoặc hash WAV giữa train/dev/test trong mỗi thí nghiệm.",
             f"- {overall['distinct_prompt_ids']} mã câu ARCTIC, nhưng {overall['distinct_normalized_prompt_texts']} transcript sau chuẩn hóa; {len(report['prompt_ids_with_multiple_normalized_transcripts'])} mã câu có nhiều biến thể transcript.",
             "- Đã kiểm tra lại SHA256 của 3 loại file gốc cho mọi mẫu; hash file đầu vào đầy đủ trong `split_audit.json`.", "",
             "## Số liệu nguồn", "", "| Tập | Người nói | Mẫu | Phút | S | D | I | (S+D)/N | I/N |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["source_splits"].items():
        errors = item["gold_error_counts"]
        lines.append(f"| source_{name} | {len(item['speakers'])} | {item['utterances']} | {item['duration_minutes']:.3f} | {errors['substitution']} | {errors['deletion']} | {errors['addition']} | {100 * item['canonical_error_fraction']:.3f}% | {100 * item['additions_per_canonical_phone']:.3f}% |")
    lines += ["", "## Bốn người nói Việt Nam", "", "| Người nói | Giới tính metadata | Mẫu | Phút | N canonical | S | D | I | (S+D)/N | I/N |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for speaker, item in report["by_speaker"].items():
        if "Vietnamese" not in item["native_languages"]:
            continue
        errors = item["gold_error_counts"]
        lines.append(f"| {speaker} | {next(iter(item['genders']))} | {item['utterances']} | {item['duration_minutes']:.3f} | {item['canonical_phones']} | {errors['substitution']} | {errors['deletion']} | {errors['addition']} | {100 * item['canonical_error_fraction']:.3f}% | {100 * item['additions_per_canonical_phone']:.3f}% |")
    lines += ["", "N là tổng số vị trí phoneme tham chiếu. S/D là lỗi tại vị trí canonical; I là âm chèn nên được thống kê riêng. Các tỷ lệ trên là đặc điểm nhãn, chưa phải kết quả model.", "",
              "## Kiểm tra từng fold", "", "| Fold | VN train | VN dev | VN test | Mẫu train/dev/test | Phút train/dev/test | Prompt ID test đã thấy trong VN train | Prompt ID test đã thấy trong source train | Transcript test đã thấy trong source train |", "|---|---|---|---|---|---|---:|---:|---:|"]
    for fold in report["folds"]:
        groups = fold["splits"]
        overlap = fold["overlaps"]
        roles = ("train", "validation", "test")
        lines.append(f"| {fold['fold']} | {'/'.join(groups['train']['speakers'])} | {'/'.join(groups['validation']['speakers'])} | {'/'.join(groups['test']['speakers'])} | {'/'.join(str(groups[role]['utterances']) for role in roles)} | {'/'.join(format(groups[role]['duration_minutes'], '.3f') for role in roles)} | {overlap['vn_train_vs_vn_test']['prompt_id']['shared_distinct']}/150 | {overlap['source_train_vs_vn_test']['prompt_id']['shared_distinct']}/150 | {overlap['source_train_vs_vn_test']['prompt_text']['shared_distinct']}/150 |")
    source_overlap = report["source_train_vs_source_validation"]
    lines += ["", f"Source train/dev cũng không tách câu: {source_overlap['prompt_id']['shared_distinct']} mã câu chung, {source_overlap['prompt_text']['shared_distinct']} transcript chung.", "",
              "## Phạm vi kết luận", "",
              "Split hiện tại tách người nói và không có bản ghi WAV trùng byte giữa các tập. Nó phù hợp để đo khả năng chấm người Việt mới trong bài đọc có ngân hàng câu cố định. Nó không đo trực tiếp câu hoàn toàn mới hoặc hội thoại tự do: toàn bộ 150 mã câu VN test ở mỗi fold đã xuất hiện trong source train và VN train.", "",
              "Hai transcript HQTV có wording khác source train nhưng vẫn dùng mã câu ARCTIC đã thấy. Vì vậy phải nhóm theo mã câu khi thiết kế kiểm tra câu mới; chỉ so chuỗi transcript có thể đánh giá sai.", "",
              "Cách chia giúp đánh giá trung thực theo mục tiêu; không tự làm model chính xác hơn. Tỷ lệ nhãn giữa người nói khác nhau được mô tả sau khi split đã đóng băng, không được dùng để chọn lại người nói test cho điểm cao. Chỉ có 4 người nói Việt Nam nên cần báo cáo từng người nói/fold; không coi 600 utterance là 600 người độc lập.", "",
              "JSON lưu đầy đủ thời lượng, lỗi và overlap của train/dev/test, source exposure trước adaptation, giới tính và nguồn hash. Kiểm tra hash chỉ phát hiện audio trùng byte; không kiểm tra near-duplicate/re-encoding hay mức trùng corpus self-supervised pre-training.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/setup/split_audit")
    args = parser.parse_args()
    report = build_audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Audited {report['global']['utterances']} examples: speaker/audio disjoint, prompt overlap present; wrote {args.output}.{{json,md}}")


if __name__ == "__main__":
    main()
