"""Frozen audio-only 120-case comparison of B, TIMIT-XLSR and CROTTC-IF.

No training, ASR text, Qwen, TTS, threshold fitting or forced reference decoding.
The new model's IPA inventory lacks AO/ZH distinctions. Full39 and reduced37
tasks are separately reported; the reduced task applies the same map to every
model, canonical/perceived reference and expert event, and explicitly counts
the expert substitutions rendered indistinguishable by that map.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.compare_asr_backends import canonical_hash, digest, load_audio, write_json
from scripts.evaluate_crottc import read, read_jsonl, identity, shared_phone, scored, raw_decoded, wav_bytes
from scripts.setup_phoneme_baseline import MODEL_ID, REVISION, RELATIVE, PINS, verify as verify_baseline
from mdd.metrics import aggregate_scores, score_utterance

MODELS = ("cascade-B", "timit-xlsr-300m", "crottc-if")
SCHEMA = "phoneme-three-models-fixed-120-v1"
SALT = "pronunciation-comparison-20261024-v1\n"
IPA_TO_ARPA = {
    "ɑ": "AA", "æ": "AE", "ə": "AH", "aʊ": "AW", "aɪ": "AY", "b": "B",
    "ʧ": "CH", "d": "D", "ð": "DH", "ɾ": "T", "ɛ": "EH", "ɝ": "ER",
    "eɪ": "EY", "f": "F", "g": "G", "h": "HH", "ɪ": "IH", "i": "IY",
    "ʤ": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "ŋ": "NG",
    "oʊ": "OW", "ɔɪ": "OY", "p": "P", "ɹ": "R", "s": "S", "ʃ": "SH",
    "t": "T", "θ": "TH", "ʊ": "UH", "u": "UW", "v": "V", "w": "W",
    "j": "Y", "z": "Z",
}
NONPHONE = frozenset({" ", "|", "[PAD]", "<s>", "</s>"})
MERGES = {"AO": "AA", "ZH": "SH", "DX": "T", "AX": "AH"}


def common_phone(phone):
    if phone is None:
        return None
    return MERGES.get(str(phone).upper().rstrip("*012"), shared_phone(phone))


def common_reference(row):
    """Keep expert locations; remove only substitutions lost in reduced inventory."""
    result = deepcopy(row)
    for key in ("canonical_phones", "perceived_phones"):
        result[key] = [common_phone(phone) for phone in row[key]]
    events, removed = [], []
    for original in row["gold_error_events"]:
        event = deepcopy(original)
        for key in ("expected_phone", "observed_phone"):
            event[key] = common_phone(event.get(key))
        if (event["kind"] == "substitution" and event["expected_phone"] not in {None, "[UNK]"}
                and event["expected_phone"] == event["observed_phone"]):
            removed.append(original)
        else:
            events.append(event)
    result["gold_error_events"] = events
    return result, removed


def common_scored(row, phones):
    normalized, removed = common_reference(row)
    score = score_utterance(normalized["canonical_phones"], normalized["perceived_phones"],
                            [common_phone(phone) for phone in phones],
                            gold_error_events=normalized["gold_error_events"],
                            canonical_word_indices=normalized["canonical_word_indices"])
    score["removed_indistinguishable_expert_events"] = removed
    return score


def selected_cases(rows, splits, *, per_speaker=20):
    tests = set(splits["cascade_test_ids"])
    forbidden = set(splits["joint_train_ids"]) | set(splits["joint_validation_ids"])
    if tests & forbidden:
        raise ValueError("Test overlaps local train/validation")
    eligible = [row for row in rows if row["utterance_id"] in tests]
    speakers = sorted({row["speaker_id"] for row in eligible})
    if len(speakers) != 6:
        raise ValueError("Expected six held-out speakers")
    output = []
    for speaker in speakers:
        candidates = [row for row in eligible if row["speaker_id"] == speaker]
        if len(candidates) < per_speaker:
            raise ValueError("Not enough test audio for speaker " + speaker)
        output.extend(sorted(candidates, key=lambda row: hashlib.sha256(
            (SALT + row["utterance_id"]).encode()).hexdigest())[:per_speaker])
    return sorted(output, key=lambda row: row["utterance_id"])


def decode_ipa_ids(frame_ids, id_to_token, blank_id):
    """Collapse consecutive CTC IDs before deleting blank; retain unknown tokens."""
    previous = None
    tokens = []
    for value in frame_ids:
        value = int(value)
        if value != previous and value != blank_id:
            token = id_to_token.get(value, "[UNK]")
            if token not in NONPHONE:
                tokens.append(token)
        previous = value
    return tokens, [IPA_TO_ARPA.get(token, "[UNK]") for token in tokens]


def review_counts(annotation, hinted, *, available=True):
    count = len(annotation["word_intervals"])
    gold = {event["word_index"] for event in annotation["gold_error_events"]
            if type(event.get("word_index")) is int and 0 <= event["word_index"] < count}
    hints = {index for index in hinted if type(index) is int and 0 <= index < count} if available else set()
    return {"reference_words": count, "gold_error_words": len(gold), "available_words": count if available else 0,
            "unavailable_error_words": 0 if available else len(gold),
            "unavailable_correct_words": 0 if available else count - len(gold),
            "hint_tp": len(gold & hints), "hint_fp": len(hints - gold),
            "hint_fn": len(gold - hints) if available else 0,
            "hint_tn": len(set(range(count)) - gold - hints) if available else 0,
            "hint_word_indices": sorted(hints), "gold_word_indices": sorted(gold),
            "calibrated_decided_words": 0}


def variant_review(root, text, phones, annotation=None, *, reduced=False):
    """Identical CMUdict-variant review algorithm for every model; no calibrated grade."""
    from mdd.pronunciation_assessment import accepted_variants
    from mdd.phone_sequence_assessment import align_variants
    result = {"status": "unavailable", "method": "shared_dictionary_variant_edit_review",
              "calibrated": False, "inventory": "common37" if reduced else "full39"}
    try:
        variants = accepted_variants(root, text)
        if not variants or any(not item["available"] for item in variants):
            raise ValueError("reference_contains_unknown_words")
        if annotation and len(variants) != len(annotation["word_intervals"]):
            raise ValueError("reference_word_count_differs_from_expert_annotation")
        observed = [common_phone(p) if reduced else shared_phone(p) for p in phones]
        if "[UNK]" in observed:
            raise ValueError("unsupported_perceived_phone")
        choices = [[[common_phone(p) if reduced else shared_phone(p) for p in entry["phones"]]
                    for entry in word["variants"]] for word in variants]
        path, distance = align_variants(choices, observed)
        errors = [step for step in path if step["kind"] != "match"]
        hints = sorted({step["word_index"] for step in errors})
        result.update(status="available", hint_word_indices=hints, edit_distance=distance,
                      events=[{key: event.get(key) for key in ("kind", "word_index", "expected_phone", "observed_phone")}
                              for event in errors])
        if annotation:
            target = common_reference(annotation)[0] if reduced else annotation
            result["counts"] = review_counts(target, hints)
    except ValueError as error:
        result["reason"] = str(error)
        if annotation:
            target = common_reference(annotation)[0] if reduced else annotation
            result["counts"] = review_counts(target, [], available=False)
    return result


def prepare(root=ROOT, *, device="cuda", per_speaker=20):
    from scripts.crottc_pronunciation import FILE_PINS, RELATIVE as CRO_RELATIVE, MODEL_ID as CRO_ID, REVISION as CRO_REV
    from mdd.demo_reference import COMMIT, DICTIONARY_SHA256
    verify_baseline(root)
    annotations = root / "data/processed/l2arctic-plus-cascade-v1/annotations.jsonl"
    protocol = root / "data/protocol/l2arctic-plus-cascade-v1"
    split = read(protocol / "splits.json")
    selected = selected_cases(read_jsonl(annotations), split, per_speaker=per_speaker)
    sources = {row["utterance_id"]: row for row in read_jsonl(protocol / "manifest.jsonl")}
    cases = []
    for row in selected:
        source = sources[row["utterance_id"]]
        if row["source_annotation_sha256"] != source["annotation"]["sha256"]:
            raise ValueError("Expert source identity differs")
        for field in ("audio", "annotation"):
            if digest(root / source[field]["path"]) != source[field]["sha256"]:
                raise ValueError("Source changed for " + row["utterance_id"])
        cases.append({"id": row["utterance_id"].replace("/", "-"), "group": "expert_test",
                      "speaker_id": row["speaker_id"], "audio": source["audio"]["path"],
                      "audio_sha256": source["audio"]["sha256"], "duration_sec": row["duration_sec"],
                      "reference_text": row["prompt_text"], "annotation": row, "source_annotation": source["annotation"]})
    user_manifest = root / "data/generated/asr-backends-20261023/manifest.json"
    users = [deepcopy(case) for case in read(user_manifest)["cases"] if case["group"] == "user_development"]
    if len(users) != 3:
        raise ValueError("Expected three fixed user recordings")
    cases.extend(users)
    files = [annotations, protocol / "protocol.json", protocol / "splits.json", protocol / "manifest.jsonl", user_manifest,
             root / "scripts/evaluate_pronunciation_models.py", root / "scripts/setup_phoneme_baseline.py",
             root / "scripts/evaluate_crottc.py", root / "scripts/compare_asr_backends.py",
             root / "scripts/crottc_pronunciation.py", root / "scripts/crottc_worker.py",
             root / "src/mdd/demo_inference.py", root / "src/mdd/metrics.py",
             root / "src/mdd/demo_reference.py", root / "src/mdd/phone_sequence_assessment.py",
             root / "src/mdd/pronunciation_assessment.py"]
    dictionary = root / "data/external/cmudict" / COMMIT / "cmudict.dict"
    if digest(dictionary) != DICTIONARY_SHA256:
        raise ValueError("Dictionary checksum differs")
    files.append(dictionary)
    b_path = root / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"
    b_files = [p for p in b_path.iterdir() if p.name in {"model.safetensors", "pytorch_model.bin", "vocab.json", "config.json", "metadata.json", "preprocessor_config.json"}]
    return {"schema": SCHEMA, "cases": cases, "source_sha256": {p.relative_to(root).as_posix(): digest(p) for p in files},
            "environment_packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy", "soundfile")},
            "models": {
                "cascade-B": {"path": b_path.relative_to(root).as_posix(), "device": device, "compute_type": "float32", "decoder": "CTC greedy", "files": [identity(p, root) for p in b_files], "train_overlap": "Speaker/utterance disjoint local training; reading prompts can overlap."},
                "timit-xlsr-300m": {"model_id": MODEL_ID, "revision": REVISION, "path": RELATIVE.as_posix(), "device": device, "compute_type": "float32", "decoder": "CTC greedy; fixed IPA-to-ARPAbet map", "files": [identity(root / RELATIVE / name, root) for name in PINS], "train_overlap": "Author reports TIMIT fine-tuning only; no documented L2-ARCTIC task fine-tuning; upstream self-supervised pretraining overlap cannot be proven absent."},
                "crottc-if": {"model_id": CRO_ID, "revision": CRO_REV, "device": device, "compute_type": "float32", "decoder": "Released SpeechBrain AR/CTC beam10 weight0.99; no extra CTC collapse", "files": [identity(root / CRO_RELATIVE / "bundle" / name, root) for name in FILE_PINS] + [identity(root / CRO_RELATIVE / "setup_manifest.json", root)], "train_overlap": "Released model fine-tuned on L2-ARCTIC; local test overlap with released model training unknown."}},
            "selection": {"expert_cases": len(selected), "per_speaker": per_speaker, "speakers": sorted({r["speaker_id"] for r in selected}), "salt": SALT,
                          "algorithm": "Within each of six fixed test speakers choose smallest SHA256(salt+utterance_id), without reading errors, model outputs or quality labels; no threshold tuning.",
                          "scope": "Previously observed corpus; exploratory fixed subset, not a new blind benchmark or unbiased cross-speaker population estimate.", "local_train_validation_id_overlap": []},
            "protocol": {"model_inputs": "Full waveform only, mono16kHz, no reference text/expert phones/ASR/Qwen/TTS; references accessed after decoding.",
                         "full39": "Same shared39 normalization across all models, canonical/perceived reference and explicit expert events; stress/distortion suffix stripped, AX->AH, unknown retained. Expert events retained even if normalized labels match.",
                         "common37": "AO->AA, ZH->SH, DX->T, AX->AH for every model and reference. Only expert substitution events with identical known normalized expected/observed are removed and counted. Locations retained; insertions/deletions/unknown events retained. This is a reduced task, not full39 MDD.",
                         "ipa_to_arpa": IPA_TO_ARPA, "common_merges": MERGES,
                         "variant_review": "Same dictionary-variant sequence aligner for every model after decoding, on confirmed reference; full39 and reduced37 reported separately. Tentative review lists, not calibrated diagnoses or the exact deployed posterior-B policy.",
                         "user_cases": "Three intended sentences previously supplied by user; no expert phone labels, no MDD accuracy assigned.",
                         "timing": "Requested sequential CUDA FP32. Effective device and fallback recorded per model/audio; mixed-device times are not a fair speed ranking.", "no_training": True, "no_tuning": True}}


def verify(manifest, root=ROOT):
    for name, expected in manifest["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for model in manifest["models"].values():
        for item in model["files"]:
            if digest(root / item["path"]) != item["sha256"]:
                raise ValueError("Frozen model changed: " + item["path"])
    for case in manifest["cases"]:
        if digest(root / case["audio"]) != case["audio_sha256"]:
            raise ValueError("Frozen audio changed: " + case["id"])
        if "source_annotation" in case and digest(root / case["source_annotation"]["path"]) != case["source_annotation"]["sha256"]:
            raise ValueError("Expert source changed: " + case["id"])


class TimitBaseline:
    def __init__(self, root, device):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForCTC
        folder = root / RELATIVE
        self.torch = torch
        self.feature = AutoFeatureExtractor.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCTC.from_pretrained(str(folder), local_files_only=True,
                                                    trust_remote_code=False, use_safetensors=True,
                                                    torch_dtype=torch.float32).eval()
        self.device = "cuda:0" if device == "cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.model.to(self.device)
        vocab = read(folder / "vocab.json") | read(folder / "added_tokens.json")
        self.id_to_token = {index: token for token, index in vocab.items()}
        self.runtime = {"device": self.device, "compute_type": "float32", "fallback_reason": None,
                        "parameter_count": sum(p.numel() for p in self.model.parameters())}

    def infer(self, waveform):
        torch = self.torch
        inputs = self.feature(waveform, sampling_rate=16000, return_tensors="pt", padding=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        with torch.inference_mode():
            ids = self.model(**inputs).logits.argmax(dim=-1)[0].cpu().tolist()
        tokens, phones = decode_ipa_ids(ids, self.id_to_token, self.model.config.pad_token_id)
        return {"phonemes": phones, "native_decoded_labels": tokens, "runtime": dict(self.runtime),
                "unknown_tokens": [token for token in tokens if token not in IPA_TO_ARPA]}

    def close(self):
        self.model.to("cpu")


def load_backend(model, root, requested_device):
    if model == "timit-xlsr-300m":
        engine = TimitBaseline(root, requested_device)
        return engine, engine.infer
    if model == "crottc-if":
        from scripts.crottc_pronunciation import CROTTCPronunciation
        engine = CROTTCPronunciation(root, device=requested_device, timeout=300.)
        engine.load()
        return engine, engine.infer
    from mdd.demo_inference import LocalInference
    engine = LocalInference(root, threads=2, device=requested_device, compute_type="float32")
    selected = {"id": "cascade-B", "kind": "phoneme", "label": "wav2vec2 B", "experimental": False,
                "path": root / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"}
    engine._get_bundle(selected)
    return engine, lambda waveform: engine.posterior_evidence(selected, wav_bytes(waveform))


def summarize(manifest, predictions):
    output = {"planned_expert_cases": manifest["selection"]["expert_cases"], "models": {}, "paired": {}}
    for model in MODELS:
        cases = [case for case in manifest["cases"] if case["group"] == "expert_test"
                 and predictions.get(model, {}).get(case["id"], {}).get("status") == "completed"]
        rows = [predictions[model][case["id"]] for case in cases]
        values = summarize_rows(rows)
        values["completed_expert_cases"] = len(rows)
        values["effective_devices"] = dict(Counter(str(row.get("runtime", {}).get("device")) for row in rows))
        output["models"][model] = values
    paired = [case for case in manifest["cases"] if case["group"] == "expert_test" and all(
        predictions.get(model, {}).get(case["id"], {}).get("status") == "completed" for model in MODELS)]
    output["paired"] = {"completed_expert_cases": len(paired), "models": {
        model: summarize_rows([predictions[model][case["id"]] for case in paired]) for model in MODELS}}
    return output


def summarize_rows(rows):
    result = {key: aggregate_scores(row[key] for row in rows)
              for key in ("research_original_inventory", "research_full39", "research_common37")}
    result["removed_indistinguishable_expert_events"] = sum(len(row["research_common37"]["removed_indistinguishable_expert_events"]) for row in rows)
    for key in ("variant_review_full39", "variant_review_common37"):
        counts = {name: sum(row[key]["counts"][name] for row in rows) for name in (
            "reference_words", "gold_error_words", "available_words", "unavailable_error_words", "unavailable_correct_words",
            "hint_tp", "hint_fp", "hint_fn", "hint_tn", "calibrated_decided_words")}
        divide = lambda n, d: n / d if d else None
        counts.update(precision=divide(counts["hint_tp"], counts["hint_tp"] + counts["hint_fp"]),
                      recall=divide(counts["hint_tp"], counts["hint_tp"] + counts["hint_fn"]),
                      all_reference_recall=divide(counts["hint_tp"], counts["gold_error_words"]),
                      false_positive_rate=divide(counts["hint_fp"], counts["hint_fp"] + counts["hint_tn"]),
                      f1=divide(2*counts["hint_tp"], 2*counts["hint_tp"] + counts["hint_fp"] + counts["hint_fn"]),
                      coverage=divide(counts["available_words"], counts["reference_words"]))
        result[key] = counts
    result["inference_sec"] = sum(row["inference_sec"] for row in rows)
    result["audio_sec"] = sum(row["audio_sec"] for row in rows)
    result["unknown_prediction_phones"] = sum(row["phonemes"].count("[UNK]") for row in rows)
    return result


def run(manifest, output, *, resume=False, only_models=None, root=ROOT):
    verify(manifest, root)
    import torch
    import psutil
    torch.set_num_threads(2)
    result_path = output / "results.json"
    manifest_hash = canonical_hash(manifest)
    if result_path.exists():
        if not resume:
            raise ValueError("Results already exist; use --resume")
        results = read(result_path)
        if results["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume identity differs")
    else:
        results = {"schema": SCHEMA, "manifest_sha256": manifest_hash, "status": "running", "predictions": {},
                   "runtime": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
    if torch.cuda.is_available():
        results["hardware"] = {"gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                               "torch": torch.__version__, "cuda": torch.version.cuda}
    for model in only_models or MODELS:
        if model not in MODELS:
            raise ValueError("Unknown model")
        pending = [case for case in manifest["cases"] if results["predictions"].get(model, {}).get(case["id"], {}).get("status") != "completed"]
        if not pending:
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print(json.dumps({"stage": "load", "model": model, "pending": len(pending)}), flush=True)
        started = time.monotonic()
        engine, infer = load_backend(model, root, manifest["models"][model]["device"])
        results["runtime"].setdefault(model, []).append({"load_sec": time.monotonic() - started,
            "initial_runtime": getattr(engine, "runtime", None), "parent_rss_mib": psutil.Process().memory_info().rss / 2**20})
        model_rows = results["predictions"].setdefault(model, {})
        try:
            for index, case in enumerate(pending, 1):
                try:
                    waveform = load_audio(root / case["audio"])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.monotonic()
                    evidence = infer(waveform)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    phones = evidence["phonemes"] if model == "timit-xlsr-300m" else raw_decoded(evidence, model)
                    annotation = case.get("annotation")
                    row = {"status": "completed", "phonemes": phones, "native_decoded_labels": evidence.get("native_decoded_labels", evidence.get("decoded_labels")),
                           "unknown_tokens": evidence.get("unknown_tokens", evidence.get("unresolved_labels", [])),
                           "runtime": evidence.get("runtime", getattr(engine, "runtime", None)) or {},
                           "inference_sec": elapsed, "audio_sec": len(waveform)/16000,
                           "audio_sha256": case["audio_sha256"], "reference_supplied_to_model": False}
                    # Ground truth is first accessed after completed audio-only inference.
                    row["variant_review_full39"] = variant_review(root, case["reference_text"], phones, annotation)
                    row["variant_review_common37"] = variant_review(root, case["reference_text"], phones, annotation, reduced=True)
                    if annotation:
                        row.update(research_original_inventory=scored(annotation, phones),
                                   research_full39=scored(annotation, phones, normalized=True),
                                   research_common37=common_scored(annotation, phones))
                    else:
                        row["expert_metrics"] = None
                    model_rows[case["id"]] = row
                    del evidence, waveform
                except Exception as error:
                    model_rows[case["id"]] = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
                results["summary"] = summarize(manifest, results["predictions"])
                write_json(result_path, results)
                print(json.dumps({"stage": "infer", "model": model, "case": case["id"], "done": index,
                                  "total": len(pending), "status": model_rows[case["id"]]["status"]}), flush=True)
        finally:
            if hasattr(engine, "close"):
                engine.close()
            results["runtime"][model][-1]["effective_runtime_end"] = getattr(engine, "runtime", None)
            if torch.cuda.is_available():
                results["runtime"][model][-1]["parent_peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            del infer, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_json(result_path, results)
    verify(manifest, root)
    results["status"] = "completed" if all(results["predictions"].get(model, {}).get(case["id"], {}).get("status") == "completed"
        for model in MODELS for case in manifest["cases"]) else "incomplete"
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    results["summary"] = summarize(manifest, results["predictions"])
    write_json(result_path, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must stay in workspace")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = read(path)
        verify(manifest)
        if args.device != manifest["models"]["cascade-B"]["device"] or args.per_speaker != manifest["selection"]["per_speaker"]:
            parser.error("Requested settings differ from frozen manifest")
    else:
        manifest = prepare(device=args.device, per_speaker=args.per_speaker)
        write_json(path, manifest)
    print(json.dumps({"manifest_sha256": canonical_hash(manifest), "cases": len(manifest["cases"]),
                      "models": list(MODELS)}), flush=True)
    if args.run:
        result = run(manifest, output, resume=args.resume, only_models=args.models)
        print(json.dumps({"status": result["status"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
