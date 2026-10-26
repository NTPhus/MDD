"""Frozen 24-case stratified phoneme pilot: official CROTTC-IF versus local B.

No ASR, Qwen, TTS, training or threshold tuning. Models receive full audio only;
manual perceived phones and error events are accessed only after decoding.
This project's scoring protocol is not a reproduction of a paper's F1 script.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.compare_asr_backends import canonical_hash, digest, load_audio, write_json
from mdd.metrics import aggregate_scores, score_utterance

MODELS = ("cascade-B", "crottc-if")
SHARED = frozenset("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())
NONPHONES = frozenset({"sil", "<bos>", "<eos>", "<blank>"})
SCHEMA = "crottc-paired-stratified-pilot-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_cases(rows, splits):
    """Two deterministic lower-half and two upper-half error-density cases/speaker."""
    tests = set(splits["cascade_test_ids"])
    forbidden = set(splits["joint_train_ids"]) | set(splits["joint_validation_ids"])
    if tests & forbidden:
        raise ValueError("Protocol test IDs overlap model train/validation IDs")
    candidates = [row for row in rows if row["utterance_id"] in tests]
    speakers = sorted({row["speaker_id"] for row in candidates})
    if len(speakers) != 6:
        raise ValueError("Expected the fixed six-speaker cascade test split")
    output = []
    for speaker in speakers:
        ordered = sorted((row for row in candidates if row["speaker_id"] == speaker),
                         key=lambda row: (len(row["gold_error_events"]) / max(1, len(row["canonical_phones"])), row["utterance_id"]))
        if len(ordered) < 4:
            raise ValueError("Need four pilot cases per speaker")
        halfway = len(ordered) // 2
        for label, pool in (("lower_error_density_half", ordered[:halfway]), ("upper_error_density_half", ordered[halfway:])):
            chosen = sorted(pool, key=lambda row: hashlib.sha256(("crottc-pilot-v1\n" + row["utterance_id"]).encode()).hexdigest())[:2]
            for original in chosen:
                row = deepcopy(original)
                row.update(group="expert_test_pilot", stratum=label,
                           manual_error_density=len(row["gold_error_events"]) / max(1, len(row["canonical_phones"])))
                output.append(row)
    return sorted(output, key=lambda row: row["utterance_id"])


def shared_phone(phone):
    if phone is None:
        return None
    value = re.sub(r"[012]$", "", str(phone).upper().rstrip("*"))
    value = "AH" if value == "AX" else value
    return value if value in SHARED else "[UNK]"


def scored(row, predicted, *, normalized=False):
    canonical, perceived, hypothesis = list(row["canonical_phones"]), list(row["perceived_phones"]), list(predicted)
    events = deepcopy(row["gold_error_events"])
    if normalized:
        canonical, perceived, hypothesis = ([shared_phone(value) for value in values] for values in (canonical, perceived, hypothesis))
        for event in events:
            for key in ("expected_phone", "observed_phone"):
                event[key] = shared_phone(event.get(key))
    return score_utterance(canonical, perceived, hypothesis, gold_error_events=events,
                           canonical_word_indices=row["canonical_word_indices"])


def raw_decoded(evidence, model):
    if model == "crottc-if":
        # The UI filters unresolved labels. Research scoring retains them as
        # [UNK], so dropping unknown output cannot artificially improve PER.
        return [label.upper() if label.upper() in SHARED else "[UNK]"
                for label in evidence["decoded_labels"] if label.lower() not in NONPHONES]
    return [item["phone"] for item in evidence["phonemes"]]


def review_counts(row, fields):
    """Tentative review-list behaviour, separate from research F1 and decisions."""
    word_count = len(row["word_intervals"])
    gold = {event["word_index"] for event in row["gold_error_events"]
            if type(event.get("word_index")) is int and 0 <= event["word_index"] < word_count}
    diagnostic = fields.get("sound_candidates") or {}
    available = diagnostic.get("status") == "available"
    hints = {word["index"] for word in diagnostic.get("words", [])
             if type(word.get("index")) is int and 0 <= word["index"] < word_count} if available else set()
    decisions = fields.get("assessment", {}).get("words", [])
    return {"reference_words": word_count, "gold_error_words": len(gold),
            "review_available_words": word_count if available else 0,
            "unavailable_words": 0 if available else word_count,
            "hint_tp": len(hints & gold), "hint_fp": len(hints - gold),
            "hint_fn": len(gold - hints) if available else 0,
            "not_flagged_gold_correct": len(set(range(word_count)) - gold - hints) if available else 0,
            "hint_word_indices": sorted(hints), "gold_word_indices": sorted(gold),
            "calibrated_decided_words": sum(word.get("status") in {"match", "difference"} for word in decisions),
            "calibration_applied": fields.get("assessment", {}).get("calibration", {}).get("applied") is True}


def identity(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}


def prepare(root=ROOT):
    from scripts.crottc_pronunciation import FILE_PINS, RELATIVE, MODEL_ID, REVISION
    from mdd.demo_reference import COMMIT, DICTIONARY_SHA256
    annotations = root / "data/processed/l2arctic-plus-cascade-v1/annotations.jsonl"
    protocol = root / "data/protocol/l2arctic-plus-cascade-v1"
    rows, splits = read_jsonl(annotations), read(protocol / "splits.json")
    selected = selected_cases(rows, splits)
    source_rows = {row["utterance_id"]: row for row in read_jsonl(protocol / "manifest.jsonl")}
    cases = []
    for row in selected:
        provenance = source_rows[row["utterance_id"]]
        for key in ("audio", "annotation"):
            if digest(root / provenance[key]["path"]) != provenance[key]["sha256"]:
                raise ValueError("Corpus source checksum mismatch: " + row["utterance_id"])
        if row["source_annotation_sha256"] != provenance["annotation"]["sha256"]:
            raise ValueError("Parsed expert annotation source differs")
        cases.append({"id": row["utterance_id"].replace("/", "-"), "group": row["group"],
                      "speaker_id": row["speaker_id"], "audio": provenance["audio"]["path"],
                      "audio_sha256": provenance["audio"]["sha256"], "reference_text": row["prompt_text"],
                      "duration_sec": row["duration_sec"], "stratum": row["stratum"],
                      "manual_error_density": row["manual_error_density"], "annotation": row,
                      "source_annotation": provenance["annotation"]})
    user_manifest = root / "data/generated/asr-backends-20261023/manifest.json"
    user_cases = [case for case in read(user_manifest)["cases"] if case["group"] == "user_development"]
    if len(user_cases) != 3:
        raise ValueError("Expected three frozen user development recordings")
    for case in user_cases:
        if digest(root / case["audio"]) != case["audio_sha256"]:
            raise ValueError("User audio changed: " + case["id"])
        cases.append(deepcopy(case))
    b_path = root / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"
    files = [annotations, protocol / "protocol.json", protocol / "splits.json", protocol / "manifest.jsonl", user_manifest,
             root / "scripts/evaluate_crottc.py", root / "scripts/crottc_pronunciation.py", root / "scripts/crottc_worker.py",
             root / "src/mdd/metrics.py", root / "src/mdd/phone_sequence_assessment.py", root / "src/mdd/chat_assessment.py",
             root / "src/mdd/pronunciation_assessment.py", root / "src/mdd/chat_diagnostics.py", root / "src/mdd/demo_inference.py",
             root / "src/mdd/demo_reference.py", root / "src/mdd/chat_pronunciation.py", root / "src/mdd/acoustic_evidence.py",
             root / "configs/pronunciation_calibration.json"]
    dictionary = root / "data/external/cmudict" / COMMIT / "cmudict.dict"
    if digest(dictionary) != DICTIONARY_SHA256:
        raise ValueError("Pronunciation dictionary checksum mismatch")
    files.append(dictionary)
    files.extend(path for path in (root / "data/generated/pronunciation-calibration-v1/full-validation/calibration.json",
                                   root / "data/generated/pronunciation-calibration-v1/full-validation/selection.json") if path.is_file())
    b_files = [path for path in b_path.iterdir() if path.name in {"model.safetensors", "pytorch_model.bin", "vocab.json", "config.json", "metadata.json", "preprocessor_config.json"}]
    c_files = [root / RELATIVE / "bundle" / name for name in FILE_PINS]
    c_files.append(root / RELATIVE / "setup_manifest.json")
    source_hashes = {path.relative_to(root).as_posix(): digest(path) for path in files}
    return {"schema": SCHEMA, "cases": cases, "source_sha256": source_hashes,
            "models": {"cascade-B": {"path": b_path.relative_to(root).as_posix(), "decoder": "CTC greedy", "device": "cpu", "threads": 2,
                                      "files": [identity(path, root) for path in b_files]},
                       "crottc-if": {"model_id": MODEL_ID, "revision": REVISION,
                                     "decoder": "Official SpeechBrain AR/CTC beam10 weight0.99", "device": "cpu", "threads": 2,
                                     "files": [identity(path, root) for path in c_files]}},
            "selection": {"test_cases": 24, "speakers": sorted({row["speaker_id"] for row in selected}),
                          "stratification": "Within each of six test speakers, sort by expert error-event density; split in halves; select two per half using SHA256(crottc-pilot-v1\\nID). No outputs or thresholds used.",
                          "sampling_limitation": "Small label-stratified exploratory pilot, not an unbiased full-test or speaker-population estimate.",
                          "training_validation_ids_overlap": [], "test_used_for_tuning": False},
            "protocol": {"model_inputs": "Full mono 16kHz waveform only; no reference/gold/ASR/Qwen/TTS.",
                         "raw_scoring": "mdd.metrics project canonical-position/insertion-slot protocol, exact original inventory; unknown CROTTC labels retained as [UNK].",
                         "shared_scoring": "Apply same normalization to canonical, perceived, prediction and explicit gold event labels: strip stress/distortion suffix; AX->AH; outside shared39-> [UNK]. Gold error events retained even when labels collapse.",
                         "shared_inventory": sorted(SHARED), "diagnosis_unknowns": "mdd.metrics excludes err/[UNK] from diagnosis only, not detection; PER retains tokens.",
                         "ui_review": "Actual dictionary-variant helpers with confirmed reference; tentative hint lists versus manual word_index labels, not research F1 and not calibrated pass/fail.",
                         "user_cases": "Previously inspected intended references; no expert labels, so no accuracy/precision/recall/F1 is computed.",
                         "timing": "CPU2 sequential resident models; load separate; CROTTC lacks phone timestamps/posterior probabilities.",
                         "no_training": True, "no_threshold_tuning": True,
                         "pretraining_overlap": "CROTTC released L2-ARCTIC model training overlap with local test cases must be considered unknown unless release provenance proves otherwise."}}


def verify(manifest, root=ROOT):
    for name, expected in manifest["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for metadata in manifest["models"].values():
        for item in metadata["files"]:
            if digest(root / item["path"]) != item["sha256"]:
                raise ValueError("Frozen model changed: " + item["path"])
    for case in manifest["cases"]:
        if digest(root / case["audio"]) != case["audio_sha256"]:
            raise ValueError("Frozen audio changed: " + case["id"])
        if "source_annotation" in case and digest(root / case["source_annotation"]["path"]) != case["source_annotation"]["sha256"]:
            raise ValueError("Frozen expert source changed: " + case["id"])


def wav_bytes(waveform):
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, waveform, 16000, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def load_backend(model, root):
    if model == "crottc-if":
        from scripts.crottc_pronunciation import CROTTCPronunciation
        engine = CROTTCPronunciation(root)
        engine.load()
        return engine, engine.infer
    from mdd.demo_inference import LocalInference
    engine = LocalInference(root, threads=2)
    selected = {"id": "cascade-B", "label": "wav2vec2 B", "kind": "phoneme", "experimental": False,
                "path": root / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"}
    engine._get_bundle(selected)
    return engine, lambda waveform: engine.posterior_evidence(selected, wav_bytes(waveform))


def summary(manifest, predictions):
    paired = [case for case in manifest["cases"] if case["group"] == "expert_test_pilot"
              and all(predictions.get(model, {}).get(case["id"], {}).get("status") == "completed" for model in MODELS)]
    result = {"paired_completed_expert_cases": len(paired), "planned_expert_cases": 24, "models": {}}
    for model in MODELS:
        rows = [predictions[model][case["id"]] for case in paired]
        fields = {"raw": aggregate_scores(row["research_raw"] for row in rows),
                  "shared_inventory": aggregate_scores(row["research_shared"] for row in rows)}
        counts = {key: sum(row["ui_review"][key] for row in rows)
                  for key in ("reference_words", "gold_error_words", "review_available_words", "unavailable_words",
                              "hint_tp", "hint_fp", "hint_fn", "not_flagged_gold_correct", "calibrated_decided_words")}
        ratio = lambda n, d: n / d if d else None
        counts.update(hint_precision=ratio(counts["hint_tp"], counts["hint_tp"] + counts["hint_fp"]),
                      hint_recall=ratio(counts["hint_tp"], counts["hint_tp"] + counts["hint_fn"]),
                      hint_f1=ratio(2 * counts["hint_tp"], 2 * counts["hint_tp"] + counts["hint_fp"] + counts["hint_fn"]),
                      hint_false_positive_rate=ratio(counts["hint_fp"], counts["hint_fp"] + counts["not_flagged_gold_correct"]),
                      review_coverage=ratio(counts["review_available_words"], counts["reference_words"]))
        fields["ui_review"] = counts
        fields["inference_sec"] = sum(row["inference_sec"] for row in rows)
        result["models"][model] = fields
    return result


def run(manifest, output, *, resume=False, root=ROOT):
    verify(manifest, root)
    import torch
    import psutil
    from mdd.chat_assessment import assessment_fields
    torch.set_num_threads(2)
    path, manifest_hash = output / "results.json", canonical_hash(manifest)
    if path.exists():
        if not resume:
            raise ValueError("Existing results require --resume")
        results = read(path)
        if results["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume identity differs")
    else:
        results = {"schema": SCHEMA, "manifest_sha256": manifest_hash, "status": "running", "predictions": {},
                   "runtime": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
    process = psutil.Process()
    for model in MODELS:
        pending = [case for case in manifest["cases"] if results["predictions"].get(model, {}).get(case["id"], {}).get("status") != "completed"]
        if not pending:
            continue
        print(json.dumps({"stage": "load", "model": model, "pending": len(pending)}), flush=True)
        started = time.monotonic()
        engine, infer = load_backend(model, root)
        runtime = {"load_sec": time.monotonic() - started, "rss_after_load_mib": process.memory_info().rss / 2**20,
                   "worker_runtime": getattr(engine, "runtime", None)}
        results["runtime"].setdefault(model, []).append(runtime)
        model_predictions = results["predictions"].setdefault(model, {})
        try:
            for index, case in enumerate(pending, 1):
                try:
                    waveform = load_audio(root / case["audio"])
                    started = time.monotonic()
                    evidence = infer(waveform)
                    elapsed = time.monotonic() - started
                    phones = raw_decoded(evidence, model)
                    # All labels and reference text are used after completed decoding.
                    fields = assessment_fields(root, case["reference_text"], evidence, confirmed=True)
                    row = {"status": "completed", "phonemes": phones, "inference_sec": elapsed,
                           "audio_sha256": case["audio_sha256"], "model_identity": manifest["models"][model],
                           "decoded_labels": evidence.get("decoded_labels"), "unresolved_labels": evidence.get("unresolved_labels", []),
                           "assessment_sec": time.monotonic() - started - elapsed,
                           "ui_assessment": fields, "reference_supplied_to_model": False,
                           "phone_timing": evidence.get("phone_timing", "CTC_frame_approximate"),
                           "shared_normalization_changes": dict(Counter(phone + " -> " + shared_phone(phone) for phone in phones if shared_phone(phone) != phone))}
                    if case["group"] == "expert_test_pilot":
                        row.update(research_raw=scored(case["annotation"], phones), research_shared=scored(case["annotation"], phones, normalized=True),
                                   ui_review=review_counts(case["annotation"], fields))
                    else:
                        row["expert_metrics"] = None
                    model_predictions[case["id"]] = row
                    del evidence
                except Exception as error:
                    model_predictions[case["id"]] = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
                results["summary"] = summary(manifest, results["predictions"])
                write_json(path, results)
                print(json.dumps({"stage": "infer", "model": model, "done": index, "total": len(pending),
                                  "case": case["id"], "status": model_predictions[case["id"]]["status"]}), flush=True)
        finally:
            close = getattr(engine, "close", None)
            if close:
                close()
            del infer, engine
            gc.collect()
    verify(manifest, root)
    results["status"] = ("completed" if all(results["predictions"].get(model, {}).get(case["id"], {}).get("status") == "completed"
                                           for model in MODELS for case in manifest["cases"]) else "partial_errors")
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must be inside the workspace")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = read(path)
        verify(manifest)
    else:
        manifest = prepare()
        write_json(path, manifest)
    print(json.dumps({"manifest_sha256": canonical_hash(manifest), "cases": len(manifest["cases"]),
                      "groups": dict(Counter(case["group"] for case in manifest["cases"]))}), flush=True)
    if args.run:
        result = run(manifest, output, resume=args.resume)
        print(json.dumps({"status": result["status"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
