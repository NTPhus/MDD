"""Frozen offline raw-ASR comparison; no training, text recovery, or reference prompts.

The fixed TLV test split has been inspected previously. External SAA recordings
share a reading prompt; user recordings are development cases. These groups are
reported separately, and their references describe intended orthographic speech,
not independently annotated words actually realized in the recordings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SCHEMA = "raw-asr-three-backends-v1"
MODEL_IDS = ("wav2vec2-base-960h", "wav2vec2-vi-en-v1", "whisper-small.en")
STELLA = ("Please call Stella. Ask her to bring these things with her from the store: "
          "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack "
          "for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. "
          "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def words(text):
    """Uniform strict WER tokens: case/punctuation ignored; number words retained."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def alignment(reference, hypothesis):
    """Deterministic minimum edit alignment with separately counted S/D/I."""
    ref, hyp = words(reference), words(hypothesis)
    table = [[(0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (0, 0, j)
    for i, expected in enumerate(ref, 1):
        for j, actual in enumerate(hyp, 1):
            if expected == actual:
                table[i][j] = table[i - 1][j - 1]
            else:
                s, d, ins = table[i - 1][j - 1]
                candidates = [(s + 1, d, ins)]
                s, d, ins = table[i - 1][j]
                candidates.append((s, d + 1, ins))
                s, d, ins = table[i][j - 1]
                candidates.append((s, d, ins + 1))
                table[i][j] = min(candidates, key=sum)
    s, d, ins = table[-1][-1]
    return {"substitutions": s, "deletions": d, "insertions": ins,
            "word_errors": s + d + ins, "reference_words": len(ref),
            "wer": (s + d + ins) / len(ref) if ref else None,
            "exact_match": ref == hyp}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_splits(parts):
    """Recompute speaker, audio and normalized-prompt disjointness."""
    from scripts.asr_vi_data import normalized, prompt_split
    checks = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("speaker_id", "audio_sha256", "utterance_id"):
            overlap = sorted({row[key] for row in parts[left]} & {row[key] for row in parts[right]})
            checks[f"{left}_{right}_{key}_overlap"] = overlap
        overlap = sorted({normalized(row["text"]) for row in parts[left]}
                         & {normalized(row["text"]) for row in parts[right]})
        checks[f"{left}_{right}_normalized_prompt_overlap"] = overlap
    if any(checks.values()):
        raise ValueError("ASR split leakage: " + repr(checks))
    if any(row["split"] != name or prompt_split(row["text"]) != name
           for name, rows in parts.items() for row in rows):
        raise ValueError("Prepared split label/prompt bucket mismatch")
    return {"passed": True, **checks,
            "counts": {name: len(rows) for name, rows in parts.items()},
            "speakers": {name: sorted({row["speaker_id"] for row in rows}) for name, rows in parts.items()}}


def file_identity(path, root):
    path = path.resolve()
    return {"path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def freeze_models(root):
    paths = {"wav2vec2-base-960h": root / "models/pretrained/wav2vec2-base-960h",
             "wav2vec2-vi-en-v1": root / "runs/asr-vietnamese-english-v1/best_checkpoint",
             "whisper-small.en": root / "models/pretrained/faster-whisper-small.en"}
    result = {}
    for identifier, path in paths.items():
        weight = "model.bin" if identifier == "whisper-small.en" else "model.safetensors"
        names = [weight, "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "vocabulary.txt", "preprocessor_config.json", "feature_extractor_config.json",
                 "special_tokens_map.json", "download_manifest.json", "setup_manifest.json"]
        if not (path / weight).is_file():
            raise ValueError("Local model is not installed: " + str(path / weight))
        result[identifier] = {"path": path.relative_to(root).as_posix(),
                              "files": {name: file_identity(path / name, root) for name in names
                                        if (path / name).is_file()},
                              "decoder": ("autoregressive beam search, beam_size=5, native word timestamps"
                                          if identifier == "whisper-small.en" else "CTC greedy argmax"),
                              "device": "cpu", "threads": 2,
                              "compute_type": "int8" if identifier == "whisper-small.en" else "float32"}
    return result


def prepare_manifest(root=ROOT, *, limit=None):
    root = Path(root).resolve()
    directory = root / "data/processed/asr-vietnamese-english-v1"
    parts = {name: jsonl(directory / (name + ".jsonl")) for name in ("train", "validation", "test")}
    dataset = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sources = {str(path.relative_to(root).as_posix()): digest(path)
               for path in [directory / "manifest.json", *(directory / (name + ".jsonl") for name in parts)]}
    for name in parts:
        if digest(directory / (name + ".jsonl")) != dataset["parts"][name]["sha256"]:
            raise ValueError("Prepared ASR dataset hash mismatch: " + name)
    audit = audit_splits(parts)
    cases = [{"id": row["utterance_id"].replace("/", "-"), "group": "tlv_previously_observed_heldout",
              "speaker_id": row["speaker_id"], "audio": row["audio"], "audio_sha256": row["audio_sha256"],
              "reference_text": row["raw_transcript"], "reference_kind": "corpus_orthographic_reading_prompt",
              "source_url": "https://psi.engr.tamu.edu/l2-arctic-corpus/", "split": "test"}
             for row in parts["test"]]
    external_path = root / "data/generated/vi-voice-pipeline-v1/manifest.json"
    external = json.loads(external_path.read_text(encoding="utf-8"))
    sources[external_path.relative_to(root).as_posix()] = digest(external_path)
    for row in external["cases"]:
        if row["id"] in {"saa-vietnamese12", "saa-vietnamese13", "saa-vietnamese22", "saa-vietnamese32"}:
            cases.append({key: row[key] for key in ("id", "speaker_id", "audio", "audio_sha256",
                                                    "reference_text", "reference_kind", "source_url")}
                         | {"group": "saa_external_same_prompt", "split": "external"})
    if sum(row["group"] == "saa_external_same_prompt" for row in cases) != 4:
        raise ValueError("Expected four fixed SAA external recordings")
    downloads = Path.home() / "Downloads"
    for identifier, filename, reference in (
            ("user-stella", "audio-file.mp3", STELLA),
            ("user-road", "End of the road for some players.mp3", "End of the road for some players"),
            ("user-digits", "Zero five eight three.mp3", "Zero five eight three")):
        path = downloads / filename
        cases.append({"id": identifier, "group": "user_development", "speaker_id": "unverified_user_recording",
                      "audio": str(path), "audio_sha256": digest(path), "reference_text": reference,
                      "reference_kind": "user_confirmed_intended_sentence", "source_url": "user_supplied_local_file",
                      "split": "development"})
    all_training_hashes = {row["audio_sha256"] for row in parts["train"]}
    all_validation_hashes = {row["audio_sha256"] for row in parts["validation"]}
    import soundfile as sf
    for case in cases:
        path = root / case["audio"]
        if digest(path) != case["audio_sha256"]:
            raise ValueError("Audio changed: " + case["id"])
        info = sf.info(path)
        case.update(duration_sec=info.frames / info.samplerate, source_sample_rate=info.samplerate,
                    source_channels=info.channels)
        if case["audio_sha256"] in all_training_hashes | all_validation_hashes:
            raise ValueError("Evaluation audio overlaps training/validation: " + case["id"])
    duplicates = [sorted(case["id"] for case in cases if case["audio_sha256"] == sha)
                  for sha, count in Counter(case["audio_sha256"] for case in cases).items() if count > 1]
    audit["duplicate_evaluation_audio_groups"] = duplicates
    audit["evaluation_vs_training_validation_audio_overlap"] = []
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    implementation_files = [root / "scripts/compare_asr_backends.py", root / "scripts/whisper_asr.py"]
    sources.update({path.relative_to(root).as_posix(): digest(path) for path in implementation_files})
    return {"schema": SCHEMA, "cases": cases, "models": freeze_models(root), "split_audit": audit,
            "source_sha256": sources, "limit": limit,
            "protocol": {"sample_rate": 16000, "mono": "channel arithmetic mean", "resample": "scipy.signal.resample_poly",
                         "audio_cropping": False, "silence_removal": False, "references_supplied_to_asr": False,
                         "history": [], "hotwords": None, "training": False, "bert_or_qwen": False,
                         "all_audio_local": True, "whisper_word_timestamps": True,
                         "strict_wer": "lowercase, punctuation ignored, apostrophes retained, number words retained; Whisper raw_text scored before production-only number spelling",
                         "secondary_wer": "transformers Whisper EnglishNumberNormalizer, then identical strict tokens",
                         "group_interpretation": "Groups are separate; TLV is one previously observed held-out speaker; SAA shares one prompt; user clips are development cases.",
                         "reference_limitation": "Intended orthographic prompts, not independently verified realized-word transcripts.",
                         "decoder_limitation": "Production configurations compared: CTC greedy FP32 vs Whisper beam5 INT8; this does not isolate fine-tuning/model architecture alone."}}


def load_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Invalid audio samples")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_backend(identifier, metadata, root=ROOT):
    """Return one resident backend and its audio-only callable, with no references."""
    if identifier == "whisper-small.en":
        from scripts.whisper_asr import WhisperASR
        backend = WhisperASR(root, cpu_threads=2)
        backend.load()
        return backend, lambda waveform: backend.transcribe(waveform)
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
    path = root / metadata["path"]
    processor = Wav2Vec2Processor.from_pretrained(str(path), local_files_only=True)
    config = Wav2Vec2Config.from_pretrained(str(path), local_files_only=True)
    config.apply_spec_augment = False
    config.mask_time_prob = config.mask_feature_prob = 0.
    model = Wav2Vec2ForCTC.from_pretrained(str(path), config=config, local_files_only=True,
                                        attn_implementation="eager").cpu().eval().requires_grad_(False)

    def predict(waveform):
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = model(**inputs).logits.argmax(-1)
        return {"text": processor.batch_decode(ids)[0].strip()}

    return (model, processor), predict


def aggregate(cases, predictions):
    output = {}
    for group in sorted({case["group"] for case in cases}):
        group_cases = [case for case in cases if case["group"] == group]
        complete = [case for case in group_cases if all(
            predictions.get(model, {}).get(case["id"], {}).get("status") == "completed" for model in MODEL_IDS)]
        models = {}
        for model in MODEL_IDS:
            rows = [predictions[model][case["id"]] for case in complete]
            model_result = {"cases": len(rows), "exact_matches": sum(row["strict"]["exact_match"] for row in rows),
                            "audio_seconds": sum(case["duration_sec"] for case in complete),
                            "inference_seconds": sum(row["inference_sec"] for row in rows)}
            for metric in ("strict", "number_normalized"):
                totals = {key: sum(row[metric][key] for row in rows)
                          for key in ("substitutions", "deletions", "insertions", "word_errors", "reference_words")}
                totals["wer"] = totals["word_errors"] / totals["reference_words"] if totals["reference_words"] else None
                model_result[metric] = totals
            model_result["real_time_factor"] = (model_result["inference_seconds"] / model_result["audio_seconds"]
                                                 if model_result["audio_seconds"] else None)
            models[model] = model_result
        paired = {}
        for before, after in ((MODEL_IDS[0], MODEL_IDS[1]), (MODEL_IDS[1], MODEL_IDS[2]), (MODEL_IDS[0], MODEL_IDS[2])):
            changes = [predictions[after][case["id"]]["strict"]["word_errors"]
                       - predictions[before][case["id"]]["strict"]["word_errors"] for case in complete]
            paired[before + " -> " + after] = {"improved_cases": sum(delta < 0 for delta in changes),
                                                "worse_cases": sum(delta > 0 for delta in changes),
                                                "same_cases": sum(delta == 0 for delta in changes),
                                                "word_error_delta": sum(changes)}
        output[group] = {"planned_cases": len(group_cases), "paired_complete_cases": len(complete),
                         "excluded_incomplete_ids": [case["id"] for case in group_cases if case not in complete],
                         "models": models, "paired_comparisons": paired}
    return output


def verify_frozen(manifest, root=ROOT):
    for relative, expected in manifest["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    for case in manifest["cases"]:
        if digest(root / case["audio"]) != case["audio_sha256"]:
            raise ValueError("Frozen audio changed: " + case["id"])
    for model in manifest["models"].values():
        for identity in model["files"].values():
            if digest(root / identity["path"]) != identity["sha256"]:
                raise ValueError("Frozen model changed: " + identity["path"])


def run(manifest, output, *, resume=False, root=ROOT):
    verify_frozen(manifest, root)
    import torch
    import psutil
    from transformers.models.whisper.english_normalizer import EnglishNumberNormalizer
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    normalizer = EnglishNumberNormalizer()
    identity = canonical_hash(manifest)
    result_path = output / "results.json"
    if result_path.exists():
        if not resume:
            raise ValueError("Results exist; use --resume with the same frozen manifest")
        results = json.loads(result_path.read_text(encoding="utf-8"))
        if results["manifest_sha256"] != identity:
            raise ValueError("Resume manifest identity differs")
    else:
        results = {"schema": SCHEMA, "manifest_sha256": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                   "predictions": {}, "runtime": {}, "status": "running"}
    packages = {}
    for package in ("torch", "transformers", "numpy", "scipy", "soundfile", "faster-whisper", "ctranslate2"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_in_standard_distribution_path"
    results["package_versions"] = packages
    process = psutil.Process()
    for identifier in MODEL_IDS:
        pending = [case for case in manifest["cases"]
                   if results["predictions"].get(identifier, {}).get(case["id"], {}).get("status") != "completed"]
        if not pending:
            continue
        print(json.dumps({"stage": "loading", "model": identifier, "pending": len(pending)}), flush=True)
        started = time.monotonic()
        backend, predict = load_backend(identifier, manifest["models"][identifier], root)
        runtime = results["runtime"].setdefault(identifier, {"load_attempts": []})
        runtime["load_attempts"].append({"load_sec": time.monotonic() - started,
                                         "rss_after_load_mib": process.memory_info().rss / 2**20})
        predictions = results["predictions"].setdefault(identifier, {})
        for index, case in enumerate(pending, 1):
            try:
                # Verify immediately before decoding, then strip all reference metadata.
                source = root / case["audio"]
                if digest(source) != case["audio_sha256"]:
                    raise ValueError("Source changed before inference")
                decode_started = time.monotonic()
                waveform = load_audio(source)
                decode_sec = time.monotonic() - decode_started
                then = time.monotonic()
                prediction = predict(waveform)
                elapsed = time.monotonic() - then
                text = prediction.get("raw_text", prediction["text"])
                if not isinstance(text, str):
                    raise ValueError("ASR must return text")
                # Reference is accessed only after hypothesis generation has completed.
                ref = case["reference_text"]
                predictions[case["id"]] = {"status": "completed", "text": text,
                                          "production_text": prediction["text"],
                                          "inference_sec": elapsed, "audio_decode_resample_sec": decode_sec,
                                          "sample_count_16khz": len(waveform), "audio_sha256": case["audio_sha256"],
                                          "strict": alignment(ref, text),
                                          "number_normalized": alignment(normalizer(ref.lower()), normalizer(text.lower())),
                                          "rss_after_inference_mib": process.memory_info().rss / 2**20,
                                          "whisper_word_intervals": prediction.get("word_intervals")}
            except Exception as error:
                predictions[case["id"]] = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            results["groups"] = aggregate(manifest["cases"], results["predictions"])
            write_json(result_path, results)
            if index == len(pending) or index % 10 == 0:
                print(json.dumps({"stage": "inference", "model": identifier, "done": index,
                                  "total": len(pending), "last_case": case["id"],
                                  "last_status": predictions[case["id"]]["status"]}), flush=True)
        runtime["rss_before_unload_mib"] = process.memory_info().rss / 2**20
        del predict, backend
        gc.collect()
        runtime["rss_after_unload_mib"] = process.memory_info().rss / 2**20
    verify_frozen(manifest, root)
    results["status"] = ("completed" if all(row["paired_complete_cases"] == row["planned_cases"]
                                           for row in results["groups"].values()) else "partial_errors")
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    memory = process.memory_info()
    results["process_peak_working_set_mib"] = getattr(memory, "peak_wset", memory.rss) / 2**20
    results["memory_note"] = "Process RSS is sampled, and Windows peak is cumulative across sequential backends; model weights are unloaded between backends."
    write_json(result_path, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke subset only; omit for the full fixed comparison")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must remain in the workspace")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if args.limit is not None and manifest["limit"] != args.limit:
            raise ValueError("Existing manifest uses a different limit")
        verify_frozen(manifest)
    else:
        manifest = prepare_manifest(limit=args.limit)
        write_json(path, manifest)
    print(json.dumps({"manifest_sha256": canonical_hash(manifest), "cases": len(manifest["cases"]),
                      "groups": dict(Counter(case["group"] for case in manifest["cases"])),
                      "duration_sec": sum(case["duration_sec"] for case in manifest["cases"]),
                      "split_audit_passed": manifest["split_audit"]["passed"]}), flush=True)
    if args.run:
        result = run(manifest, output, resume=args.resume)
        print(json.dumps({"status": result["status"], "groups": result["groups"]}), flush=True)


if __name__ == "__main__":
    main()
