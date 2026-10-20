"""Freeze annotated L2-ARCTIC inputs and speaker splits; does not train a model."""

import argparse
import hashlib
import json
from pathlib import Path
import wave


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {
    "Arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "Chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "Hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "Korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "Spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "Vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}
SOURCE_DEV = {"SKA", "LXC", "RRBI", "HJK", "ERMS"}
TARGET_FOLDS = [
    (["THV", "TLV"], ["PNV"], ["HQTV"]),
    (["HQTV", "TLV"], ["THV"], ["PNV"]),
    (["HQTV", "PNV"], ["TLV"], ["THV"]),
    (["PNV", "THV"], ["HQTV"], ["TLV"]),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def ids_for(records, speakers):
    return [record["utterance_id"] for record in records if record["speaker_id"] in speakers]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "l2arctic_release_v5.0")
    parser.add_argument("--output", type=Path, default=ROOT / "data/protocol/l2arctic-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {args.output}")

    records, counts = [], {}
    for language, speakers in LANGUAGES.items():
        for speaker in speakers:
            speaker_root = args.source / speaker
            annotations = sorted((speaker_root / "annotation").glob("*.TextGrid"))
            expected = 149 if speaker == "YBAA" else 150
            if len(annotations) != expected:
                raise ValueError(f"{speaker}: expected {expected} annotations, got {len(annotations)}")
            counts[speaker] = len(annotations)
            for annotation in annotations:
                stem = annotation.stem
                audio = speaker_root / "wav" / f"{stem}.wav"
                transcript = speaker_root / "transcript" / f"{stem}.txt"
                with wave.open(str(audio), "rb") as source:
                    sample_rate = source.getframerate()
                    frames = source.getnframes()
                    channels = source.getnchannels()
                if sample_rate <= 0 or frames <= 0:
                    raise ValueError(f"Empty or invalid audio: {audio}")
                records.append({
                    "utterance_id": f"{speaker}/{stem}",
                    "speaker_id": speaker,
                    "native_language": language,
                    "annotation_is_manual": True,
                    "audio": file_record(audio),
                    "annotation": file_record(annotation),
                    "transcript": file_record(transcript),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_sec": frames / sample_rate,
                })
            print(f"Indexed and hashed {speaker}: {len(annotations)} annotated utterances", flush=True)

    records.sort(key=lambda item: item["utterance_id"])
    if len(records) != 3599 or len({item["utterance_id"] for item in records}) != len(records):
        raise ValueError("Unexpected total or duplicate utterance IDs")
    target_speakers = set(LANGUAGES["Vietnamese"])
    all_speakers = set(counts)
    source_train = all_speakers - target_speakers - SOURCE_DEV
    splits = {
        "source_train_speakers": sorted(source_train),
        "source_validation_speakers": sorted(SOURCE_DEV),
        "source_train_ids": ids_for(records, source_train),
        "source_validation_ids": ids_for(records, SOURCE_DEV),
        "folds": [],
    }
    for number, (train, dev, test) in enumerate(TARGET_FOLDS, 1):
        roles = [set(train), set(dev), set(test)]
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError("Overlapping target speaker splits")
        if set.union(*roles) != target_speakers:
            raise ValueError("Incomplete target split")
        splits["folds"].append({
            "fold": number,
            "train_speakers": train,
            "validation_speakers": dev,
            "test_speakers": test,
            "train_ids": ids_for(records, train),
            "validation_ids": ids_for(records, dev),
            "test_ids": ids_for(records, test),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    splits_path = args.output / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": "l2arctic-v1",
        "dataset": "L2-ARCTIC v5.0 scripted manual annotations",
        "manifest_sha256": sha256(manifest_path),
        "splits_sha256": sha256(splits_path),
        "utterance_count": len(records),
        "vietnamese_utterance_count": len(ids_for(records, target_speakers)),
        "speaker_count": len(counts),
        "counts_by_speaker": counts,
        "source_train_count": len(splits["source_train_ids"]),
        "source_validation_count": len(splits["source_validation_ids"]),
        "total_duration_hours": sum(record["duration_sec"] for record in records) / 3600,
        "scope": "Source labels and file hashes only; parsed phoneme targets and model predictions are not generated here.",
    }
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (manifest_path, splits_path, protocol_path)),
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(f"Frozen protocol: {args.output}")


if __name__ == "__main__":
    main()
