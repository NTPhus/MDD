"""Register two Charsiu heads before inference; reuse the immutable three-model test.

Both heads receive waveform only. References enter scoring after decoding. CTC
and frame-CE models differ in objective, resolution and training, so this is an
observational checkpoint comparison, not a controlled loss-function ablation.
The separate B posterior cache contains raw test evidence for an independently
locked validation-fitted policy; this script never fits policy on test data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.compare_asr_backends import canonical_hash, digest, load_audio, write_json
from scripts.evaluate_crottc import read, identity, scored, wav_bytes
from scripts.evaluate_pronunciation_models import (
    MODELS as OLD_MODELS, common_scored, variant_review, summarize_rows, verify as verify_old,
)
from scripts.setup_charsiu_phoneme import RELATIVE, MODELS as NEW_MODELS, PREPROCESSING, verify as verify_setup
from scripts.experiment_pronunciation_calibration import _cache_path, save_evidence, load_evidence
from scripts.crottc_worker import physical_cuda_memory

OLD_FOLDER = Path("data/generated/model-comparison-20261024/pronunciation")
OLD_MANIFEST_SHA256 = "80d28fee3cd0ef21d01552b9ed39dbbb201ec4a894846da22297800a71f13eac"
OLD_RESULTS_SHA256 = "124d1a6a5244461e919710384f51b5f6dfba99eaa6b1b6d773dda58223a5f6e4"
DEFAULT_CACHE = Path("data/generated/pronunciation-improvement-20261024/b-test-posteriors")
SCHEMA = "charsiu-ctc-fc-and-three-frozen-baselines-v1"
ALL_MODELS = (*OLD_MODELS, *NEW_MODELS)
ARPABET = frozenset("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())


def save(path, data):
    # Keep Windows readers from turning a completed neural inference into loss.
    for attempt in range(8):
        try:
            write_json(path, data)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def decode_frames(frame_ids, vocabulary, *, ctc):
    """RLE IDs first; CTC removes blank, CE preserves unexpected PAD as unknown."""
    labels, phones, previous = [], [], None
    for raw in frame_ids:
        index = int(raw)
        if index != previous:
            token = vocabulary.get(index, "[UNK]")
            if not (ctc and index == 41) and token != "[SIL]":
                labels.append(token)
                phones.append(token if token in ARPABET else "[UNK]")
        previous = index
    return labels, phones


def baseline_bundle(root=ROOT):
    old = read(root / OLD_FOLDER / "manifest.json")
    if canonical_hash(old) != OLD_MANIFEST_SHA256:
        raise ValueError("Previous fixed test manifest identity differs")
    if digest(root / OLD_FOLDER / "results.json") != OLD_RESULTS_SHA256:
        raise ValueError("Previous three-model baseline results changed")
    results = read(root / OLD_FOLDER / "results.json")
    if results.get("status") != "completed" or results.get("manifest_sha256") != OLD_MANIFEST_SHA256:
        raise ValueError("Previous baseline run is incomplete or not paired")
    for key in OLD_MODELS:
        if any(results["predictions"][key][case["id"]].get("status") != "completed"
               or results["predictions"][key][case["id"]].get("audio_sha256") != case["audio_sha256"] for case in old["cases"]):
            raise ValueError("Baseline predictions and audio identities differ")
    return old, results


def prepare(root=ROOT):
    verify_setup(root)
    old, results = baseline_bundle(root)
    verify_old(old, root)
    folder = root / RELATIVE
    setup = read(folder / "setup_manifest.json")
    models = deepcopy(old["models"])
    tokenizer = read(folder / "tokenizer/vocab.json")
    if set(tokenizer) - {"[SIL]", "[UNK]", "[PAD]"} != ARPABET or tokenizer["[PAD]"] != 41:
        raise ValueError("Charsiu vocabulary is not the fixed full CMU39 inventory")
    for key, metadata in NEW_MODELS.items():
        models[key] = {**metadata, "path": (RELATIVE / key).as_posix(), "device": "cuda", "compute_type": "float32",
                       "files": [identity(folder / key / filename, root) for filename in ("config.json", "pytorch_model.bin")]
                       + [identity(folder / "tokenizer" / filename, root) for filename in ("vocab.json", "tokenizer_config.json", "special_tokens_map.json")],
                       "decoder": "CTC ID-collapse before blank/SIL removal" if key == "charsiu-ctc" else "Frame argmax RLE; remove SIL only; unexpected PAD/UNK retained as [UNK]",
                       "train_overlap": "Paper phone recognizer trained on LibriSpeech/Common Voice; family frame classifiers use aligned native speech. Exact released-checkpoint train ID manifest is unavailable; no documented L2-ARCTIC fine-tuning, upstream overlap not proven absent."}
    sources = dict(old["source_sha256"])
    for path in (ROOT / "scripts/setup_charsiu_phoneme.py", ROOT / "scripts/evaluate_charsiu_pronunciation.py",
                 ROOT / "scripts/experiment_pronunciation_calibration.py", ROOT / "scripts/setup_phoneme_baseline.py"):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    return {"schema": SCHEMA, "cases": deepcopy(old["cases"]), "selection": deepcopy(old["selection"]),
            "models": models, "source_sha256": sources, "environment_packages": old["environment_packages"],
            "baseline_bundle": {"manifest_path": (OLD_FOLDER / "manifest.json").as_posix(),
                                "manifest_sha256": OLD_MANIFEST_SHA256, "results_path": (OLD_FOLDER / "results.json").as_posix(),
                                "results_sha256": OLD_RESULTS_SHA256, "models_reused": list(OLD_MODELS),
                                "cases_sha256": canonical_hash(old["cases"]), "new_inference_models": list(NEW_MODELS)},
            "charsiu_setup_manifest": identity(folder / "setup_manifest.json", root),
            "charsiu_preprocessing": PREPROCESSING, "charsiu_vocabulary": tokenizer,
            "checkpoint_loading": {"weights_only": True, "unsafe_globals_allowed": [], "remote_code": False,
                                   "state_inspection": setup["state_inspection"],
                                   "fc_forward": "Author frame classifier subclasses standard Wav2Vec2ForCTC; audio-to-logits inference identical, only training loss differs. No label/loss is supplied."},
            "protocol": {**deepcopy(old["protocol"]),
                         "new_models": "Pre-register BOTH author CTC20ms and FC10ms before new predictions, no GT/ASR/reference at neural input. Both full CMU39, no IPA translation required.",
                         "ablation_limit": "Different objective, resolution and training data/procedure. This is not a causal CTC-vs-CE loss ablation.",
                         "baseline_reuse": "All 369 prior predictions reused byte-identical by frozen source results checksum, not rerun. New timings and old timings come from separate sessions; no controlled speedup claim.",
                         "b_cache": "Collect original B raw FP32 posteriors for the same120 expert test audio only, no test threshold selection; policy must be frozen from validation independently before test opening."},
            "references": [{"title": "Charsiu ICASSP2022 paper", "url": "https://arxiv.org/abs/2110.03876"},
                           {"title": "Author implementation", "url": "https://github.com/lingjzhu/charsiu/tree/13a69f2a22ca0c0962b75cc693399b0ae23a12c9"}]}


def verify(manifest, root=ROOT):
    verify_old(manifest, root)
    old, results = baseline_bundle(root)
    if canonical_hash(manifest["cases"]) != canonical_hash(old["cases"]):
        raise ValueError("Cases/order/references differ from fixed prior test")
    item = manifest["charsiu_setup_manifest"]
    if digest(root / item["path"]) != item["sha256"]:
        raise ValueError("Charsiu setup proof changed")


def require_free_gpu(minimum_mib=1900):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently change the registered test")
    free, _ = torch.cuda.mem_get_info(0)
    physical = physical_cuda_memory()
    if physical:
        free = min(free, physical["free_mib"] * 2**20)
    if free < minimum_mib * 2**20:
        raise RuntimeError("GPU busy; pause evaluation instead of evicting demo or risking OOM")
    return {"free_mib_before_load": free / 2**20, "physical": physical}


class Charsiu:
    def __init__(self, key, root=ROOT):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        self.key = key
        self.torch = torch
        self.gate = require_free_gpu()
        folder = root / RELATIVE
        path = folder / key
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path / "pytorch_model.bin")
        if unsafe:
            raise ValueError("Unsafe checkpoint globals")
        self.model, info = Wav2Vec2ForCTC.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, use_safetensors=False, weights_only=True,
            torch_dtype=torch.float32, output_loading_info=True)
        if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Checkpoint does not strictly match the standard inference network: " + str(info))
        self.feature = Wav2Vec2FeatureExtractor(**PREPROCESSING)
        self.vocab = {index: token for token, index in read(folder / "tokenizer/vocab.json").items()}
        self.model.eval().to("cuda:0", dtype=torch.float32)
        self.runtime = {"device": "cuda:0", "compute_type": "float32", "fallback_reason": None,
                        "parameter_count": sum(p.numel() for p in self.model.parameters()),
                        "loading_info": info, "unsafe_globals": unsafe, "weights_only": True,
                        "nominal_frame_stride_sec": NEW_MODELS[key]["stride_samples"] / 16000,
                        "training_objective": NEW_MODELS[key]["objective"], **self.gate}

    def infer(self, waveform):
        inputs = self.feature(waveform, sampling_rate=16000, return_tensors="pt", padding=False)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            ids = logits.argmax(dim=-1)[0].cpu().tolist()
        labels, phones = decode_frames(ids, self.vocab, ctc=self.key == "charsiu-ctc")
        return {"phonemes": phones, "native_decoded_labels": labels,
                "unknown_tokens": [token for token in labels if token not in ARPABET],
                "frame_count": len(ids), "runtime": dict(self.runtime)}

    def close(self):
        del self.model


def cache_b(manifest, destination, root=ROOT):
    """Create legacy-compatible raw posterior NPZs, independent of any policy."""
    import numpy as np
    import torch
    from mdd.demo_inference import LocalInference
    destination.mkdir(parents=True, exist_ok=True)
    expert = [case for case in manifest["cases"] if case["group"] == "expert_test"]
    b_info = manifest["models"]["cascade-B"]
    weight = next(item for item in b_info["files"] if item["path"].endswith("model.safetensors"))
    vocab = next(item for item in b_info["files"] if item["path"].endswith("vocab.json"))
    plan = {"schema": "fixed120-b-test-ctc-posterior-plan-v1", "comparison_manifest_sha256": canonical_hash(manifest),
            "source_comparison_manifest_sha256": OLD_MANIFEST_SHA256, "checkpoint_sha256": weight["sha256"],
            "vocabulary_sha256": vocab["sha256"], "device": "cuda", "compute_type": "float32",
            "cases": [{key: case[key] for key in ("id", "speaker_id", "audio", "audio_sha256")}
                      | {"utterance_id": case["annotation"]["utterance_id"]} for case in expert]}
    plan_path = destination / "plan.json"
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError("B cache collection plan changed")
    else:
        save(plan_path, plan)
    index_path = destination / "manifest.json"
    index = read(index_path) if index_path.exists() else {
        "schema": "legacy-ctc-posterior-cache-fixed120-v1", "status": "running", "cases": [],
        "checkpoint_sha256": weight["sha256"], "vocabulary_sha256": vocab["sha256"],
        "source_comparison_manifest_sha256": OLD_MANIFEST_SHA256, "collection_plan_sha256": canonical_hash(plan)}
    if index.get("collection_plan_sha256") != canonical_hash(plan):
        raise ValueError("B cache index identity changed")
    existing = {entry["case_id"]: entry for entry in index["cases"]}
    pending = []
    for case in expert:
        entry = existing.get(case["id"])
        if entry:
            if entry["audio_sha256"] != case["audio_sha256"] or digest(destination / entry["path"]) != entry["sha256"]:
                raise ValueError("Existing B posterior evidence identity mismatch")
        else:
            pending.append(case)
    if not pending:
        return index
    verify(manifest, root)
    require_free_gpu()
    old_predictions = baseline_bundle(root)[1]["predictions"]["cascade-B"]
    engine = LocalInference(root, threads=2, device="cuda", compute_type="float32")
    selected = {"id": "cascade-B", "kind": "phoneme", "label": "Frozen B posterior cache", "experimental": False,
                "path": root / b_info["path"]}
    engine._get_bundle(selected)
    try:
        for number, case in enumerate(pending, 1):
            waveform = load_audio(root / case["audio"])
            started = time.monotonic()
            evidence = engine.posterior_evidence(selected, wav_bytes(waveform))
            elapsed = time.monotonic() - started
            if not evidence["runtime"]["device"].startswith("cuda") or evidence["runtime"]["compute_type"] != "float32":
                raise RuntimeError("Registered B posterior GPU run fell back; retain partial cache and report")
            matrix = np.asarray(evidence["log_probs"], dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError("Invalid raw posterior matrix")
            evidence.update(case_id=case["id"], utterance_id=case["annotation"]["utterance_id"],
                source_audio_sha256=case["audio_sha256"], sample_rate=16000, sample_count=len(waveform),
                frame_count=matrix.shape[0], nominal_frame_stride_sec=.02,
                effective_frame_stride_sec=len(waveform)/16000/matrix.shape[0], reference_supplied_to_model=False)
            path = _cache_path(destination, case["annotation"]["utterance_id"])
            save_evidence(path, evidence)
            loaded = load_evidence(path)
            if not np.array_equal(loaded["log_probs"], matrix) or loaded["checkpoint_sha256"] != weight["sha256"]:
                raise ValueError("Saved posterior differs from actual neural evidence")
            phones = [item["phone"] for item in evidence["phonemes"]]
            entry = {"case_id": case["id"], "utterance_id": case["annotation"]["utterance_id"], "speaker_id": case["speaker_id"],
                     "audio_sha256": case["audio_sha256"], "path": path.relative_to(destination).as_posix(), "sha256": digest(path),
                     "checkpoint_sha256": evidence["checkpoint_sha256"], "vocabulary_sha256": evidence["vocabulary_sha256"],
                     "runtime": evidence["runtime"], "inference_sec": elapsed, "frame_count": matrix.shape[0],
                     "old_b_prediction_equal": phones == old_predictions[case["id"]]["phonemes"]}
            existing[case["id"]] = entry
            index["cases"] = [existing[c["id"]] for c in expert if c["id"] in existing]
            save(index_path, index)
            print(json.dumps({"stage": "B_posteriors", "done": number, "pending": len(pending), "case": case["id"]}), flush=True)
            del waveform, evidence, loaded, matrix
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    verify(manifest, root)
    index["status"] = "completed" if len(index["cases"]) == len(expert) else "incomplete"
    index["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(index_path, index)
    return index


def summary(manifest, predictions):
    expert = [case for case in manifest["cases"] if case["group"] == "expert_test"]
    all_paired = [c for c in expert if all(predictions.get(model, {}).get(c["id"], {}).get("status") == "completed" for model in ALL_MODELS)]
    result = {"planned_expert_cases": len(expert), "models": {}, "paired": {"completed_expert_cases": len(all_paired), "models": {}}}
    for model in ALL_MODELS:
        rows = [predictions[model][c["id"]] for c in expert if predictions.get(model, {}).get(c["id"], {}).get("status") == "completed"]
        values = summarize_rows(rows)
        values["completed_expert_cases"] = len(rows)
        values["effective_devices"] = dict(Counter(str(row.get("runtime", {}).get("device")) for row in rows))
        result["models"][model] = values
        result["paired"]["models"][model] = summarize_rows([predictions[model][c["id"]] for c in all_paired])
    return result


def run(manifest, output, *, resume=False, root=ROOT):
    import torch
    torch.set_num_threads(2)
    verify(manifest, root)
    path = output / "results.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing results require --resume")
        result = read(path)
        if result["manifest_sha256"] != canonical_hash(manifest):
            raise ValueError("Resume identity differs")
    else:
        old = baseline_bundle(root)[1]
        result = {"schema": SCHEMA, "manifest_sha256": canonical_hash(manifest), "status": "running",
                  "predictions": deepcopy(old["predictions"]), "runtime": deepcopy(old["runtime"]),
                  "baseline_reuse": manifest["baseline_bundle"], "started_utc": datetime.now(timezone.utc).isoformat(),
                  "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}}
    for key in NEW_MODELS:
        pending = [case for case in manifest["cases"] if result["predictions"].get(key, {}).get(case["id"], {}).get("status") != "completed"]
        if not pending:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        engine = Charsiu(key, root)
        result["runtime"].setdefault(key, []).append({"load_sec": time.monotonic()-started, "runtime": engine.runtime})
        rows = result["predictions"].setdefault(key, {})
        try:
            for number, case in enumerate(pending, 1):
                try:
                    waveform = load_audio(root / case["audio"])
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    evidence = engine.infer(waveform)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                    phones = evidence["phonemes"]
                    annotation = case.get("annotation")
                    row = {"status": "completed", **evidence, "inference_sec": elapsed,
                           "audio_sec": len(waveform)/16000, "audio_sha256": case["audio_sha256"],
                           "reference_supplied_to_model": False,
                           "variant_review_full39": variant_review(root, case["reference_text"], phones, annotation),
                           "variant_review_common37": variant_review(root, case["reference_text"], phones, annotation, reduced=True)}
                    if annotation:
                        row.update(research_original_inventory=scored(annotation, phones),
                                   research_full39=scored(annotation, phones, normalized=True),
                                   research_common37=common_scored(annotation, phones))
                    else:
                        row["expert_metrics"] = None
                    rows[case["id"]] = row
                    del evidence, waveform
                except Exception as error:
                    rows[case["id"]] = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
                result["summary"] = summary(manifest, result["predictions"])
                save(path, result)
                print(json.dumps({"stage": "Charsiu", "model": key, "case": case["id"], "done": number,
                                  "pending": len(pending), "status": rows[case["id"]]["status"]}), flush=True)
        finally:
            result["runtime"][key][-1]["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
            engine.close()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    verify(manifest, root)
    result["status"] = "completed" if all(result["predictions"].get(model, {}).get(case["id"], {}).get("status") == "completed"
        for model in ALL_MODELS for case in manifest["cases"]) else "incomplete"
    result["summary"] = summary(manifest, result["predictions"])
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-output", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect-b-posteriors", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare_only, args.run, args.collect_b_posteriors)):
        parser.error("Select a preparation or inference action")
    output = args.output.resolve()
    cache_output = args.cache_output.resolve()
    if not output.is_relative_to(ROOT) or not cache_output.is_relative_to(ROOT):
        parser.error("Outputs must stay in workspace")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = read(path)
        verify(manifest)
    else:
        manifest = prepare()
        save(path, manifest)
    print(json.dumps({"manifest_sha256": canonical_hash(manifest), "cases": len(manifest["cases"]), "registered_new_models": list(NEW_MODELS)}), flush=True)
    if args.collect_b_posteriors:
        cache = cache_b(manifest, cache_output)
        print(json.dumps({"B_cache_status": cache["status"], "completed_cases": len(cache["cases"])}), flush=True)
    if args.run:
        result = run(manifest, output, resume=args.resume)
        print(json.dumps({"status": result["status"], "paired_expert_cases": result["summary"]["paired"]["completed_expert_cases"]}), flush=True)


if __name__ == "__main__":
    main()
