"""Evaluate a frozen local MDD checkpoint on human-rated SpeechOcean762.

This is an external canonical-position benchmark, not the L2-ARCTIC scoring
protocol. No perceived-phone sequence, PER, training, HTTP, LLM or TTS is used.
Example (run from the project):
  .venv/Scripts/python.exe scripts/evaluate_external_speechocean.py \
    --selection full-test --output data/generated/external/B-full
Use --resume with the same arguments to reuse completed, content-verified rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdd.metrics import align_phones

COMMIT = "613968e3b0b789fc33936fb5eba1973176ba7d11"
PROTOCOL_ID = "speechocean762-canonical-majority-v1"
BASE_PHONES = set("AA AE AH AO AW AX AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())
SUPPORTED = BASE_PHONES | {phone + "*" for phone in BASE_PHONES} | {"err"}
ANNOTATION_TOKEN = re.compile(r"\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|[^\s]+")
COUNT_KEYS = (
    "utterances reference_phones reference_words position_tp position_fp position_fn position_tn "
    "unanimous_correct_positions unanimous_correct_rejections unanimous_acceptable_positions "
    "unanimous_acceptable_rejections predicted_insertions word_tp word_fp word_fn word_tn "
    "strict_perfect_utterances strict_perfect_sentence_flags strict_perfect_reference_phones "
    "strict_perfect_phone_rejections strict_perfect_reference_words strict_perfect_word_flags"
).split()


def normalize_phone(raw: str) -> str:
    """Remove terminal stress only; never merge AX/AH or remove a deviation *."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Invalid phone: {raw!r}")
    phone = re.sub(r"[012]+$", "", raw.strip().upper())
    if phone == "ERR":
        phone = "err"
    if phone not in SUPPORTED:
        raise ValueError(f"Unsupported canonical/annotation phone: {raw!r}")
    return phone


def _phones(value) -> list[str]:
    values = value.split() if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty phone string/list")
    return [normalize_phone(phone) for phone in values]


def parse_expert_word(annotation: str, reference_phones: list[str]) -> dict:
    """Parse one expert's notation; insertion phones do not consume positions."""
    if not isinstance(annotation, str):
        raise ValueError("Expert annotation must be a string")
    reference = _phones(reference_phones)
    observed, ratings, insertions = [], [], []
    for token in ANNOTATION_TOKEN.findall(annotation):
        grouped = token[0] in "[{("
        if grouped:
            closing = {"[": "]", "{": "}", "(": ")"}[token[0]]
            if not token.endswith(closing) or any(c in token[1:-1] for c in "[]{}()"):
                raise ValueError(f"Malformed expert annotation group: {token!r}")
            contents = _phones(token[1:-1])
        else:
            if any(c in token for c in "[]{}()"):
                raise ValueError(f"Malformed expert annotation: {token!r}")
            contents = _phones([token])
        if token.startswith("["):
            insertions.extend({"position": len(observed), "phone": phone} for phone in contents)
        else:
            rating = 1 if token.startswith("{") else 0 if token.startswith("(") else 2
            observed.extend(contents)
            ratings.extend([rating] * len(contents))
    if observed != reference:
        raise ValueError(f"Expert canonical positions do not match reference: {observed!r} != {reference!r}")
    return {"ratings": ratings, "insertions": insertions}


def _text(value) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _word_key(value) -> str:
    return "".join(character for character in _text(value) if character.isalnum() or character in "'’")


def parse_human_labels(score: dict, detail: dict, reference_text: str) -> dict:
    """Use the five original annotations, never manufacture perceived phones."""
    words = reference_text.split()
    # Official text removes punctuation. Three original strings also omit the
    # space after a period; the nested word arrays below enforce boundaries.
    if _word_key(score.get("text")) != _word_key(reference_text) or _word_key(detail.get("text")) != _word_key(reference_text):
        raise ValueError("Human annotation transcript differs from official test text")
    if len(score.get("words", [])) != len(words) or len(detail.get("words", [])) != len(words):
        raise ValueError("Human words do not match the transcript word count")
    canonical, mapping, phone_ratings, word_votes, insertion_votes = [], [], [], [], []
    for index, (word, aggregate, original) in enumerate(zip(words, score["words"], detail["words"])):
        if _word_key(aggregate.get("text")) != _word_key(word) or _word_key(original.get("text")) != _word_key(word):
            raise ValueError("Human word order/text differs from transcript")
        reference = _phones(original.get("ref-phones"))
        if _phones(aggregate.get("phones")) != reference:
            raise ValueError("Aggregate and original canonical phones differ")
        if len(aggregate.get("phones-accuracy", [])) != len(reference):
            raise ValueError("Aggregate phone accuracy length differs from canonical phones")
        annotations = original.get("phones")
        if not isinstance(annotations, list) or len(annotations) != 5:
            raise ValueError("Exactly five original expert annotations are required")
        parsed = [parse_expert_word(annotation, reference) for annotation in annotations]
        canonical.extend(reference)
        mapping.extend([index] * len(reference))
        phone_ratings.extend([entry["ratings"][position] for entry in parsed] for position in range(len(reference)))
        word_votes.append(sum(any(rating == 0 for rating in entry["ratings"]) or bool(entry["insertions"]) for entry in parsed))
        insertion_votes.append([len(entry["insertions"]) for entry in parsed])
    majority_phone = [sum(rating == 0 for rating in ratings) >= 3 for ratings in phone_ratings]
    return {
        "reference_text": reference_text, "words": words, "canonical_phones": canonical,
        "canonical_word_indices": mapping, "phone_ratings": phone_ratings,
        "phone_rating_counts": [{str(r): ratings.count(r) for r in (0, 1, 2)} for ratings in phone_ratings],
        "word_error_votes": word_votes, "word_insertion_counts_by_expert": insertion_votes,
        "majority_phone_error": majority_phone, "majority_word_error": [votes >= 3 for votes in word_votes],
        "strict_perfect": all(all(rating == 2 for rating in ratings) for ratings in phone_ratings)
                          and not any(any(counts) for counts in insertion_votes),
    }


def score_prediction(human: dict, predicted_phones: list[str]) -> dict:
    canonical, mapping = human["canonical_phones"], human["canonical_word_indices"]
    if not canonical or len(mapping) != len(canonical) or len(human["phone_ratings"]) != len(canonical):
        raise ValueError("Canonical phone/mapping/rating lengths differ")
    if len(human["majority_phone_error"]) != len(canonical):
        raise ValueError("Phone gold length differs")
    counts, per_phone = Counter({key: 0 for key in COUNT_KEYS}), defaultdict(Counter)
    alignment = align_phones(canonical, predicted_phones)
    events = []
    for step in alignment:
        if step["kind"] != "match":
            events.append({**step, "word_index": mapping[min(step["position"], len(mapping) - 1)]})
    rejected = {event["position"] for event in events if event["kind"] != "addition"}
    flagged_words = {event["word_index"] for event in events}
    counts.update(utterances=1, reference_phones=len(canonical), reference_words=len(human["words"]))
    for index, (phone, ratings, gold) in enumerate(zip(canonical, human["phone_ratings"], human["majority_phone_error"])):
        predicted = index in rejected
        key = "position_" + ("tp" if predicted and gold else "fp" if predicted else "fn" if gold else "tn")
        local = Counter(reference_phones=1)
        local[key] += 1
        if all(rating == 2 for rating in ratings):
            local["unanimous_correct_positions"] += 1
            local["unanimous_correct_rejections"] += int(predicted)
        if all(rating >= 1 for rating in ratings):
            local["unanimous_acceptable_positions"] += 1
            local["unanimous_acceptable_rejections"] += int(predicted)
        counts.update(local)
        per_phone[phone].update(local)
    # reference_phones was set once above; local phone totals only belong in the table.
    counts["reference_phones"] = len(canonical)
    for event in events:
        if event["kind"] == "addition":
            counts["predicted_insertions"] += 1
            per_phone[event["observed_phone"]]["predicted_insertions"] += 1
    for index, gold in enumerate(human["majority_word_error"]):
        predicted = index in flagged_words
        counts["word_" + ("tp" if predicted and gold else "fp" if predicted else "fn" if gold else "tn")] += 1
    if human["strict_perfect"]:
        counts.update(strict_perfect_utterances=1, strict_perfect_sentence_flags=int(bool(events)),
                      strict_perfect_reference_phones=len(canonical), strict_perfect_phone_rejections=len(rejected),
                      strict_perfect_reference_words=len(human["words"]), strict_perfect_word_flags=len(flagged_words))
    return {"counts": dict(counts), "events": events, "per_phone": {phone: dict(c) for phone, c in per_phone.items()}}


def _ratio(a, b):
    return a / b if b else None


def _rate(numerator: int, denominator: int) -> dict:
    interval = None
    if denominator:
        p, z = numerator / denominator, 1.959963984540054
        center = (p + z * z / (2 * denominator)) / (1 + z * z / denominator)
        half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator * denominator)) / (1 + z * z / denominator)
        interval = [max(0., center - half), min(1., center + half)]
    return {"rate": _ratio(numerator, denominator), "numerator": numerator, "denominator": denominator, "wilson95": interval}


def derive_metrics(counts: dict) -> dict:
    c = Counter(counts)
    return {
        "canonical_position_detection": {
            "precision": _ratio(c["position_tp"], c["position_tp"] + c["position_fp"]),
            "recall": _ratio(c["position_tp"], c["position_tp"] + c["position_fn"]),
            "f1": _ratio(2 * c["position_tp"], 2 * c["position_tp"] + c["position_fp"] + c["position_fn"]),
        },
        "word_detection": {
            "precision": _ratio(c["word_tp"], c["word_tp"] + c["word_fp"]),
            "recall": _ratio(c["word_tp"], c["word_tp"] + c["word_fn"]),
            "f1": _ratio(2 * c["word_tp"], 2 * c["word_tp"] + c["word_fp"] + c["word_fn"]),
        },
        "unanimous_correct_position_rejection_rate": _rate(c["unanimous_correct_rejections"], c["unanimous_correct_positions"]),
        "unanimous_acceptable_position_rejection_rate": _rate(c["unanimous_acceptable_rejections"], c["unanimous_acceptable_positions"]),
        "majority_accepted_word_false_positive_rate": _rate(c["word_fp"], c["word_fp"] + c["word_tn"]),
        "strict_perfect_sentence_any_flag_rate": _rate(c["strict_perfect_sentence_flags"], c["strict_perfect_utterances"]),
        "strict_perfect_phone_false_rejection_rate": _rate(c["strict_perfect_phone_rejections"], c["strict_perfect_reference_phones"]),
        "strict_perfect_word_false_positive_rate": _rate(c["strict_perfect_word_flags"], c["strict_perfect_reference_words"]),
        "predicted_insertions_per_100_reference_phones": 100 * c["predicted_insertions"] / c["reference_phones"] if c["reference_phones"] else None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pairs(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        if key in result:
            raise ValueError(f"Duplicate ID {key} in {path}")
        result[key] = value.strip()
    return result


def load_dataset(dataset_root: Path, selection: str) -> tuple[list[dict], dict]:
    source = _json(dataset_root / "source-manifest.json")
    if source.get("commit") != COMMIT:
        raise ValueError("Dataset author commit differs from frozen protocol")
    metadata_hashes = {}
    for relative, item in source["files"].items():
        path = (dataset_root / relative).resolve()
        if not path.is_relative_to(dataset_root.resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe metadata file: {relative}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Frozen metadata hash mismatch: {relative}")
        metadata_hashes[relative] = digest
    auxiliary = {}
    for key in ("audio_manifest", "frozen_protocol"):
        item = source.get(key, {})
        path = (dataset_root / item.get("path", "")).resolve()
        if not path.is_relative_to(dataset_root.resolve()) or not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError(f"Missing/changed pinned {key}")
        auxiliary[key] = path
    frozen = _json(auxiliary["frozen_protocol"])
    if frozen.get("commit") != COMMIT or frozen.get("protocol_id") != PROTOCOL_ID or frozen.get("metadata_sha256") != metadata_hashes:
        raise ValueError("Frozen protocol/metadata identity differs")
    audio_manifest = {}
    for line in auxiliary["audio_manifest"].read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["utterance_id"] in audio_manifest:
            raise ValueError("Duplicate ID in pinned audio manifest")
        audio_manifest[item["utterance_id"]] = item
    wavs, speakers, texts = (_pairs(dataset_root / "test" / name) for name in ("wav.scp", "utt2spk", "text"))
    ages, genders = (_pairs(dataset_root / "test" / name) for name in ("spk2age", "spk2gender"))
    if len(wavs) != 2500 or set(wavs) != set(speakers) or set(wavs) != set(texts):
        raise ValueError("Expected the complete matching official test split of 2500 IDs")
    if set(audio_manifest) != set(wavs) or sorted(wavs) != frozen.get("official_test_ids"):
        raise ValueError("Audio/official-test IDs differ from the frozen protocol")
    scores, details = (_json(dataset_root / "resource" / name) for name in ("scores.json", "scores-detail.json"))
    records = []
    for identifier in wavs:
        try:
            human = parse_human_labels(scores[identifier], details[identifier], texts[identifier])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid/unsupported human annotation {identifier}: {error}; no records were silently excluded") from error
        relative = Path(wavs[identifier])
        audio = (dataset_root / relative).resolve()
        if relative.is_absolute() or "|" in wavs[identifier] or not audio.is_relative_to(dataset_root.resolve()):
            raise ValueError(f"Unsafe audio path for {identifier}")
        speaker = speakers[identifier]
        pinned = audio_manifest[identifier]
        if pinned["path"] != wavs[identifier] or pinned["speaker_id"] != speaker:
            raise ValueError(f"Pinned audio path/speaker differs for {identifier}")
        age = int(ages[speaker])
        records.append({"utterance_id": identifier, "speaker_id": speaker, "age": age,
                        "age_group": "adult" if age >= 18 else "child", "gender": genders[speaker],
                        "audio_path": str(audio), "expected_audio_sha256": pinned["sha256"],
                        "expected_audio_bytes": pinned["bytes"], "human": human})
    perfect = sum(row["human"]["strict_perfect"] for row in records)
    if perfect != 179:
        raise ValueError(f"Frozen strict-perfect control expected 179 utterances, found {perfect}")
    if sorted(row["utterance_id"] for row in records if row["human"]["strict_perfect"]) != frozen.get("strict_perfect_control_ids"):
        raise ValueError("Strict-perfect IDs differ from the frozen pre-prediction selection")
    chosen = records if selection == "full-test" else [row for row in records if row["human"]["strict_perfect"]]
    return chosen, {"official_test_count": len(records), "strict_perfect_count": perfect,
                    "selected_count": len(chosen), "metadata_sha256": metadata_hashes,
                    "source_manifest_sha256": sha256(dataset_root / "source-manifest.json"),
                    "audio_manifest_sha256": sha256(auxiliary["audio_manifest"]),
                    "frozen_protocol_sha256": sha256(auxiliary["frozen_protocol"]),
                    "study_config_sha256": frozen["study_config_sha256"],
                    "unsupported_canonical_phone_count": 0, "excluded_record_count": 0}


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def read_verified_audio(row: dict) -> tuple[bytes, str]:
    """Verify the exact bytes subsequently passed to the model, not a second read."""
    content = Path(row["audio_path"]).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != row["expected_audio_bytes"] or digest != row["expected_audio_sha256"]:
        raise ValueError("Audio bytes/SHA256 differ from the pinned author-commit manifest")
    return content, digest


def load_prediction_cache(path: Path, identity_sha: str, selected_ids: list[str]) -> dict:
    """Recover only an interrupted final write; malformed earlier rows fail."""
    cached, valid_bytes, allowed = {}, 0, set(selected_ids)
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
                break
            raise ValueError("Malformed completed prediction cache row") from error
        if record.get("identity_sha256") != identity_sha or record.get("utterance_id") not in allowed:
            raise ValueError("Prediction cache identity/ID differs")
        cached[record["utterance_id"]] = record
        valid_bytes += len(line)
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return cached


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(_flatten(item, name + "."))
        elif not isinstance(item, list):
            result[name] = item
    return result


def _csv(path, groups):
    rows = [{"group": group, **dict(counts), **_flatten(derive_metrics(counts))} for group, counts in sorted(groups.items())]
    fields = ["group"] + sorted({key for row in rows for key in row} - {"group"})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
        if not path.exists():
            path = ROOT / "runs" / value
    if (path / "best_checkpoint").is_dir():
        path /= "best_checkpoint"
    if not all((path / name).is_file() for name in ("model.safetensors", "vocab.json", "config.json", "metadata.json")):
        raise ValueError("Checkpoint must be a local scientific run or checkpoint directory")
    if _json(path / "metadata.json").get("purpose") != "scientific_training":
        raise ValueError("Expected a completed scientific checkpoint, not BASE/smoke")
    return path.resolve()


def _overlap_audit(records):
    path = ROOT / "data/processed/l2arctic-v1/annotations.jsonl"
    if not path.is_file():
        return {"available": False}
    source = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = _json(ROOT / "data/protocol/l2arctic-v1/splits.json")
    training_ids = set(splits["source_train_ids"])
    folds = splits["folds"]
    fold = next(item for item in (folds if isinstance(folds, list) else folds.values()) if item["fold"] == 1)
    training_ids.update(fold["train_ids"])
    training = [row for row in source if row["utterance_id"] in training_ids]
    train_texts = {_text(row["prompt_text"]) for row in training}
    train_audio = {row.get("audio_sha256") for row in training} - {None}
    return {"available": True, "scope": "source train plus Vietnamese fold-1 train; no weight changes",
            "training_utterance_count": len(training), "normalized_text_overlap_ids": [row["utterance_id"] for row in records if _text(row["human"]["reference_text"]) in train_texts],
            "audio_hash_audit_available": bool(train_audio), "training_audio_hash_count": len(train_audio),
            "audio_hash_audit_note": "Audio duplicate checking unavailable when frozen training rows contain no hashes; no zero-overlap claim is made." if not train_audio else "Only training records with recorded audio hashes can be compared.",
            "training_audio_sha256": sorted(train_audio)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/speechocean762" / COMMIT)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/external_speechocean762.json")
    parser.add_argument("--checkpoint", default="windows-notebook-seed42-B-fold1")
    parser.add_argument("--selection", choices=("full-test", "perfect-control"), default="full-test")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing results require --resume")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    protocol = _json(args.protocol)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("commit") != COMMIT or protocol.get("no_training") is not True:
        raise ValueError("Protocol does not match this frozen evaluator")
    records, dataset_audit = load_dataset(args.dataset_root.resolve(), args.selection)
    if sha256(args.protocol) != dataset_audit["study_config_sha256"]:
        raise ValueError("Study configuration differs from the frozen pre-prediction protocol")
    checkpoint = _checkpoint_path(args.checkpoint)
    checkpoint_hashes = {name: sha256(checkpoint / name) for name in ("model.safetensors", "vocab.json", "config.json", "metadata.json")}
    pipeline_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (Path(__file__), ROOT / "src/mdd/demo_inference.py", ROOT / "src/mdd/metrics.py")}
    identity = {"protocol": protocol, "protocol_sha256": sha256(args.protocol), "selection": args.selection,
                "selected_ids": [row["utterance_id"] for row in records], "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hashes, "pipeline_sha256": pipeline_hashes,
                "dataset": dataset_audit, "threads": args.threads, "device": "cpu", "precision": "float32"}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"Output already exists: {output}; use a new directory or --resume")
    if args.resume and (not output.is_dir() or not (output / "manifest.json").is_file()):
        raise ValueError("--resume requires an existing evaluator manifest")
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    previous = _json(output / "manifest.json") if args.resume else None
    if previous and previous.get("identity_sha256") != identity_sha:
        raise ValueError("Resume inputs/checkpoint/protocol/code differ from saved manifest")
    overlap = _overlap_audit(records)
    manifest = {"identity_sha256": identity_sha, "identity": identity, "started_at_utc": previous["started_at_utc"] if previous else _utc(),
                "resumed_at_utc": _utc() if previous else None, "status": "running", "overlap_audit": overlap}
    _atomic_json(output / "manifest.json", manifest)
    prediction_path, cached = output / "predictions.jsonl", {}
    if args.resume and prediction_path.exists():
        cached = load_prediction_cache(prediction_path, identity_sha, identity["selected_ids"])
    from mdd.demo_inference import LocalInference
    engine = LocalInference(ROOT, threads=args.threads)
    model = {"id": checkpoint.parent.name, "label": checkpoint.parent.name, "path": str(checkpoint), "kind": "phoneme", "experimental": False}
    completed, failures, interrupted = [], [], False
    try:
        with prediction_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(records, 1):
                started = time.monotonic()
                identifier, human = row["utterance_id"], row["human"]
                item = {key: value for key, value in row.items() if key != "human"}
                item.update(identity_sha256=identity_sha, human=human)
                try:
                    audio_bytes, digest = read_verified_audio(row)
                    old = cached.get(identifier)
                    if old and old.get("status") == "ok" and old.get("audio_sha256") != digest:
                        raise ValueError("Audio differs from cached content")
                    if old and old.get("status") == "ok":
                        item = old
                    else:
                        result = engine.analyze(model, audio_bytes, human["reference_text"], reference={
                            "prompt_text": human["reference_text"], "canonical_phones": human["canonical_phones"],
                            "canonical_word_indices": human["canonical_word_indices"]})
                        predicted = [phone["phone"] for phone in result["phonemes"]]
                        scored = score_prediction(human, predicted)
                        item.update(status="ok", audio_sha256=digest, audio=result["audio"], predicted_phones=predicted,
                                    events=scored["events"], counts=scored["counts"], per_phone=scored["per_phone"],
                                    inference_seconds=time.monotonic() - started)
                        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                    completed.append(item)
                except (OSError, ValueError, RuntimeError) as error:
                    item.update(status="error", message=str(error), elapsed_seconds=time.monotonic() - started)
                    failures.append(item)
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                print(f"[{index}/{len(records)}] {identifier} {item['status']} ({len(failures)} failures)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    counts, groups = Counter({key: 0 for key in COUNT_KEYS}), {key: defaultdict(Counter) for key in ("phone", "speaker", "age", "gender")}
    for item in completed:
        counts.update(item["counts"])
        for phone, local in item["per_phone"].items():
            groups["phone"][phone].update(local)
        for kind, field in (("speaker", "speaker_id"), ("age", "age_group"), ("gender", "gender")):
            groups[kind][str(item[field])].update(item["counts"])
    after = {name: sha256(checkpoint / name) for name in checkpoint_hashes}
    unchanged = checkpoint_hashes == after
    hashes = set(overlap.pop("training_audio_sha256", []))
    overlap["audio_sha256_overlap_ids"] = [item["utterance_id"] for item in completed if item.get("audio_sha256") in hashes]
    status = "invalid_checkpoint_changed" if not unchanged else "interrupted" if interrupted else "partial" if failures else "completed"
    metrics = {"protocol_id": PROTOCOL_ID, "label": "canonical-position majority-expert detection", "status": status,
               "selection": args.selection, "selected_count": len(records), "evaluated_count": len(completed),
               "failed_count": len(failures), "unprocessed_count": len(records) - len(completed) - len(failures),
               "counts": dict(counts), "metrics": derive_metrics(counts), "dataset_audit": dataset_audit,
               "groups": {kind: {name: {"counts": dict(c), "metrics": derive_metrics(c)} for name, c in sorted(items.items())} for kind, items in groups.items() if kind != "phone"},
               "failures": [{"utterance_id": item["utterance_id"], "message": item["message"]} for item in failures],
               "overlap_audit": overlap, "checkpoint_sha256_before": checkpoint_hashes, "checkpoint_sha256_after": after,
               "checkpoint_unchanged": unchanged, "identity_sha256": identity_sha,
               "limitations": ["No perceived-phone PER or L2-ARCTIC insertion-slot F1 is claimed.",
                               "Predicted insertions are excluded from canonical-position phone detection; word detection includes them.",
                               "Wilson intervals are descriptive binomial intervals; phones/words within speakers are not independent.",
                               "SpeechOcean762 speakers are Mandarin learners; these results do not establish native/TTS performance."]}
    _atomic_json(output / "metrics.json", metrics)
    for kind, items in groups.items():
        _csv(output / f"per_{kind}.csv", items)
    manifest.update(status=status, finished_at_utc=_utc(), evaluated_count=len(completed), failed_count=len(failures),
                    checkpoint_sha256_after=after, checkpoint_unchanged=unchanged, overlap_audit=overlap,
                    predictions_sha256=sha256(prediction_path), metrics_sha256=sha256(output / "metrics.json"))
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": status, "evaluated": len(completed), "failed": len(failures), "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"External evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
