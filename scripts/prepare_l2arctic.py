"""Parse frozen L2-ARCTIC manual annotations into reproducible phone CTC targets.

Uses only Python's standard library. No split is modified and no vocabulary is
inferred from a speaker or test set. Audio remains untouched at its source rate;
the training loader must resample it to the backbone's 16 kHz input rate.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
# The 40-phone inventory is declared in L2-ARCTIC v5.0/README.md. Deviation
# variants are predefined for every phone, including variants not seen in data.
ARPABET = (
    "AA AE AH AO AW AX AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()
VOCAB = {phone: index for index, phone in enumerate(
    ["[PAD]", "[UNK]"] + ARPABET + [phone + "*" for phone in ARPABET] + ["err"]
)}
SILENCE = {"", "sil", "sp", "spn"}
ERROR_TYPES = {"s": "substitution", "d": "deletion", "a": "addition"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_phone(raw):
    """Discard lexical stress, preserve deviation *, and expose unknown labels.

    Do not guess the intended phone of annotation typos such as ER), D_, or V8.
    They become [UNK] and remain visible in the audit and raw interval labels.
    """
    compact = re.sub(r"\s+", "", raw)
    if compact.lower() in SILENCE:
        return None
    if compact.lower() == "err":
        return "err"
    normalized = re.sub(r"[012]", "", compact.upper())
    return normalized if normalized in VOCAB else "[UNK]"


def _number(text, field):
    match = re.search(rf"(?m)^\s*{re.escape(field)}\s*=\s*([^\s]+)\s*$", text)
    if not match:
        raise ValueError(f"Missing TextGrid field: {field}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite TextGrid {field}")
    return value


def parse_textgrid(text):
    """Read the long text TextGrid format shipped in the frozen corpus.

    The parser deliberately rejects unsupported/malformed files instead of
    producing a plausible but incomplete training target.
    """
    if not re.search(r'Object class\s*=\s*"TextGrid"', text):
        raise ValueError("Expected a long-text Praat TextGrid")
    tiers = {}
    for block in re.split(r"(?m)^\s*item \[\d+\]:\s*$", text)[1:]:
        tier_name = re.search(r'(?m)^\s*name\s*=\s*"((?:[^"]|"")*)"\s*$', block)
        if not tier_name:
            raise ValueError("TextGrid tier has no name")
        name = tier_name.group(1).replace('""', '"')
        if name not in {"phones", "words"}:
            continue
        if name in tiers:
            raise ValueError(f"Duplicate {name} tier")
        if not re.search(r'class\s*=\s*"IntervalTier"', block):
            raise ValueError(f"Expected IntervalTier for {name}")
        count = re.search(r"intervals:\s*size\s*=\s*(\d+)", block)
        if not count:
            raise ValueError(f"Missing interval count for {name}")
        intervals = []
        previous_end = None
        for interval in re.split(r"(?m)^\s*intervals \[\d+\]:\s*$", block)[1:]:
            start, end = _number(interval, "xmin"), _number(interval, "xmax")
            label = re.search(r'(?m)^\s*text\s*=\s*"((?:[^"]|"")*)"\s*$', interval)
            if label is None:
                raise ValueError(f"Missing interval label in {name}")
            if start < 0 or end < start:
                raise ValueError(f"Invalid interval boundaries in {name}: {start}, {end}")
            if previous_end is not None and start < previous_end - 1e-7:
                raise ValueError(f"Overlapping/out-of-order intervals in {name}")
            previous_end = end
            intervals.append({"start_sec": start, "end_sec": end,
                              "text": label.group(1).replace('""', '"')})
        if len(intervals) != int(count.group(1)):
            raise ValueError(f"Interval count mismatch in {name}")
        tiers[name] = intervals
    if set(tiers) != {"phones", "words"}:
        raise ValueError("Expected both manual phones and words tiers")
    return tiers


def _word_index(interval, words):
    """Choose maximum overlap; equal overlaps choose the earlier word.

    An addition in an inter-word pause receives null instead of an invented
    word association. Timestamps are human annotation intervals, not automatic
    timing estimates for model predictions.
    """
    overlaps = [(max(0.0, min(interval["end_sec"], word["end_sec"])
                         - max(interval["start_sec"], word["start_sec"])), word["index"])
                for word in words]
    if overlaps:
        overlap, index = max(overlaps, key=lambda pair: (pair[0], -pair[1]))
        if overlap > 0:
            return index
    return None


def parse_annotation(record, tiers, prompt_text):
    """Construct gold lexical sequences and explicit human error events."""
    words = []
    for interval in tiers["words"]:
        if interval["text"].strip():
            words.append({**interval, "text": interval["text"].strip(), "index": len(words)})
    canonical, perceived, word_indices, phones, errors = [], [], [], [], []
    unknowns = Counter()
    for interval in tiers["phones"]:
        raw_label = interval["text"]
        pieces = re.sub(r"\s+", "", raw_label).split(",")
        if len(pieces) == 1:
            raw_expected = raw_observed = pieces[0]
            expected = observed = normalize_phone(pieces[0])
            kind = "silence" if expected is None else "correct"
        elif len(pieces) == 3 and pieces[2].lower() in ERROR_TYPES:
            raw_expected, raw_observed = pieces[:2]
            expected, observed = normalize_phone(raw_expected), normalize_phone(raw_observed)
            kind = ERROR_TYPES[pieces[2].lower()]
            if ((kind == "addition" and (expected is not None or observed is None))
                    or (kind == "deletion" and (expected is None or observed is not None))
                    or (kind == "substitution" and (expected is None or observed is None))):
                raise ValueError(f"Invalid {kind} semantics: {raw_label!r}")
        else:
            raise ValueError(f"Unsupported phone annotation: {raw_label!r}")
        for raw, normalized in {(raw_expected, expected), (raw_observed, observed)}:
            if normalized == "[UNK]":
                unknowns[raw] += 1
        position = len(canonical)
        canonical_index = position if expected is not None else None
        perceived_index = len(perceived) if observed is not None else None
        word_index = _word_index(interval, words)
        if expected is not None:
            canonical.append(expected)
            word_indices.append(word_index)
        if observed is not None:
            perceived.append(observed)
        phones.append({"raw_label": raw_label, "expected_phone": expected,
                       "observed_phone": observed, "kind": kind,
                       "canonical_index": canonical_index, "perceived_index": perceived_index,
                       "word_index": word_index, "start_sec": interval["start_sec"],
                       "end_sec": interval["end_sec"]})
        if kind in ERROR_TYPES.values():
            errors.append({"kind": kind, "type": kind, "position": position,
                           "expected_phone": expected, "observed_phone": observed,
                           "canonical_index": canonical_index, "perceived_index": perceived_index,
                           "word_index": word_index, "start_sec": interval["start_sec"],
                           "end_sec": interval["end_sec"], "raw_label": raw_label})
    if any(interval["end_sec"] > record["duration_sec"] + 0.025 for interval in phones):
        raise ValueError(f"Annotation extends beyond audio: {record['utterance_id']}")
    return {
        "utterance_id": record["utterance_id"], "speaker_id": record["speaker_id"],
        "native_language": record["native_language"], "audio_path": record["audio"]["path"],
        "sample_rate": record["sample_rate"], "duration_sec": record["duration_sec"],
        "prompt_text": prompt_text.strip(), "canonical_phones": canonical,
        "perceived_phones": perceived, "target_phones": perceived,
        "canonical_word_indices": word_indices, "gold_error_events": errors,
        "phone_intervals": phones, "word_intervals": words,
        "usable_for_training": bool(perceived),
        "unknown_phone_labels": dict(sorted(unknowns.items())),
        "source_annotation_sha256": record["annotation"]["sha256"],
    }


def normalization_policy():
    return {
        "version": "l2arctic-phones-v1",
        "inventory_source": "l2arctic_release_v5.0/README.md, Phoneme set (40 phones)",
        "vocabulary_policy": "Predeclared inventory; never fitted on any training/validation/test labels",
        "phones": ARPABET, "deviation_variants": [phone + "*" for phone in ARPABET],
        "ctc_blank": "[PAD]", "unknown_token": "[UNK]", "ambiguous_perception": "err",
        "silence_and_nonlexical_labels_removed": sorted(SILENCE),
        "whitespace": "Remove whitespace within phone labels and error codes",
        "case": "Uppercase ARPAbet; lowercase error codes; case-insensitive err/silence",
        "stress": "Remove digits 0, 1, 2; keep deviation marker * (including AA*1 -> AA*)",
        "other_characters": "Do not infer corrections: unrecognized label maps to [UNK], audited with raw label",
        "gold_errors": "Keep explicit human s/d/a events even when stress normalization makes phones equal",
        "canonical_sequence": "Include correct/substituted/deleted expected phones; exclude additions and silence",
        "perceived_sequence": "Include correct/substituted/added perceived phones; exclude deletions and silence",
        "addition_position": "Gap index in canonical sequence, from 0 to canonical length",
        "word_mapping": "Maximum positive overlap with nonempty human words tier; earlier index breaks ties; otherwise null",
        "annotation_timing": "Retained annotation intervals; deletion duration is not measured production of a deleted phone",
    }


def prepare(manifest, output, root=ROOT, verify_audio=False, overwrite=False):
    manifest, output, root = Path(manifest), Path(output), Path(root)
    protocol_dir = manifest.parent
    checksum_path = protocol_dir / "SHA256SUMS"
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            if sha256(protocol_dir / name) != digest:
                raise ValueError(f"Frozen protocol checksum mismatch: {name}")
    source = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [record["utterance_id"] for record in source]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate utterance IDs in manifest")
    records = []
    by_speaker = defaultdict(lambda: {"utterances": 0, "canonical_phones": 0, "perceived_phones": 0,
                                      "error_counts": Counter(), "unknown_phone_labels": Counter()})
    raw_label_counts, unknown_counts, error_counts = Counter(), Counter(), Counter()
    normalized_equal_errors = 0
    for entry in source:
        for key in ("annotation", "transcript") + (("audio",) if verify_audio else ()):
            if sha256(root / entry[key]["path"]) != entry[key]["sha256"]:
                raise ValueError(f"Source checksum mismatch: {entry['utterance_id']} {key}")
        if not (root / entry["audio"]["path"]).is_file():
            raise FileNotFoundError(entry["audio"]["path"])
        tiers = parse_textgrid((root / entry["annotation"]["path"]).read_text(encoding="utf-8"))
        record = parse_annotation(entry, tiers, (root / entry["transcript"]["path"]).read_text(encoding="utf-8"))
        records.append(record)
        speaker = by_speaker[record["speaker_id"]]
        speaker["utterances"] += 1
        speaker["canonical_phones"] += len(record["canonical_phones"])
        speaker["perceived_phones"] += len(record["perceived_phones"])
        unknown_counts.update(record["unknown_phone_labels"])
        speaker["unknown_phone_labels"].update(record["unknown_phone_labels"])
        for interval in record["phone_intervals"]:
            raw_label_counts[interval["raw_label"]] += 1
        for error in record["gold_error_events"]:
            error_counts[error["kind"]] += 1
            speaker["error_counts"][error["kind"]] += 1
            normalized_equal_errors += error["kind"] == "substitution" and error["expected_phone"] == error["observed_phone"]
    audit = {
        "manifest_sha256": sha256(manifest), "utterance_count": len(records),
        "speaker_count": len(by_speaker), "audio_sha256_verified": verify_audio,
        "annotation_and_transcript_sha256_verified": True,
        "vocabulary_size": len(VOCAB), "vocabulary_fitted_on_data": False,
        "total_canonical_phones": sum(len(record["canonical_phones"]) for record in records),
        "total_perceived_phones": sum(len(record["perceived_phones"]) for record in records),
        "error_counts": dict(sorted(error_counts.items())),
        "substitution_events_with_equal_normalized_phones": normalized_equal_errors,
        "unknown_phone_labels": dict(sorted(unknown_counts.items())),
        "utterances_with_unknown_labels": [record["utterance_id"] for record in records if record["unknown_phone_labels"]],
        "unusable_utterance_ids": [record["utterance_id"] for record in records if not record["usable_for_training"]],
        "by_speaker": dict(sorted(by_speaker.items())),
        "raw_phone_label_counts": dict(sorted(raw_label_counts.items())),
        "scope": "Manual scripted annotation only; no forced-alignment-only data, no suitcase speech; no training or predictions",
    }
    outputs = {
        "annotations.jsonl": "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        "vocab.json": json.dumps(VOCAB, ensure_ascii=False, indent=2) + "\n",
        "normalization.json": json.dumps(normalization_policy(), ensure_ascii=False, indent=2) + "\n",
        "audit.json": json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    }
    outputs["SHA256SUMS"] = "".join(f"{hashlib.sha256(content.encode()).hexdigest()}  {name}\n" for name, content in outputs.items())
    # Idempotent for identical inputs; different existing output is not replaced.
    for name, content in outputs.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content and not overwrite:
            raise FileExistsError(f"Refusing to replace different prepared data: {path}; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = output / name
        if not path.exists() or overwrite:
            path.write_text(content, encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/protocol/l2arctic-v1/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/l2arctic-v1")
    parser.add_argument("--verify-audio", action="store_true", help="Also rehash every raw audio file (slower)")
    parser.add_argument("--overwrite", action="store_true", help="Refresh only the five generated files in --output; raw inputs remain untouched")
    args = parser.parse_args()
    audit = prepare(args.manifest, args.output, verify_audio=args.verify_audio, overwrite=args.overwrite)
    print(json.dumps({key: audit[key] for key in ["utterance_count", "speaker_count", "vocabulary_size", "error_counts",
                                                  "unknown_phone_labels", "substitution_events_with_equal_normalized_phones"]},
                     indent=2, ensure_ascii=False))
    print(f"Prepared data: {args.output}")


if __name__ == "__main__":
    main()
