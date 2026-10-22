"""Evaluate available weights honestly: full transcript NLL and balanced generation pilots."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import LatentDataset, load, read_json, read_rows, save
from scripts.l2arctic_plus_data import rooted, json_digest
from scripts.prepare_feedback_data import digest
from scripts.projector_inference import ProjectorPipeline
from scripts.validation_records import read_json_lines, word_counts


def select_balanced(rows, per_speaker, seed, task):
    if per_speaker < 1:
        raise ValueError("per_speaker must be positive")
    selected = set()
    for speaker in sorted({row["speaker_id"] for row in rows}):
        values = [row for row in rows if row["speaker_id"] == speaker]
        if per_speaker > len(values):
            raise ValueError("Requested more examples than the speaker has")
        ranked = sorted(values, key=lambda r: hashlib.sha256(f"{seed}:{task}:{r['utterance_id']}".encode()).hexdigest())
        selected.update(row["utterance_id"] for row in ranked[:per_speaker])
    return [row for row in rows if row["utterance_id"] in selected]


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_case(output, case, identity, rows, predict):
    folder = output / case
    folder.mkdir(parents=True, exist_ok=True)
    path, meta_path = folder / "predictions.jsonl", folder / "metadata.json"
    identity = {**identity, "case": case, "ordered_rows_sha256": json_digest([(r["utterance_id"], r["source_row_sha256"]) for r in rows]),
                "rows": len(rows), "speaker_counts": dict(Counter(r["speaker_id"] for r in rows))}
    previous = read_json_lines(path) if path.exists() else []
    if meta_path.exists():
        metadata = read_json(meta_path)
        if metadata["identity"] != identity:
            raise ValueError("Evaluation identity changed; choose a new output directory: " + case)
        if metadata["status"] == "completed":
            if digest(path) != metadata["predictions_sha256"] or len(previous) != len(rows):
                raise ValueError("Completed evaluation changed")
            return previous
    elif previous:
        raise ValueError("Predictions have no provenance")
    if len(previous) > len(rows):
        raise ValueError("Too many cached rows")
    for old, row in zip(previous, rows):
        if (old["utterance_id"], old["source_row_sha256"]) != (row["utterance_id"], row["source_row_sha256"]):
            raise ValueError("Resume is not a frozen prefix")
    metadata = {"identity": identity, "status": "running", "completed_rows": len(previous)}
    save(meta_path, metadata)
    try:
        for index in range(len(previous), len(rows)):
            row = rows[index]
            observation = {"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"],
                           "source_row_sha256": row["source_row_sha256"], **predict(index, row)}
            append(path, observation)
            previous.append(observation)
            metadata["completed_rows"] = len(previous)
            if len(rows) < 100 or (index + 1) % 25 == 0:
                save(meta_path, metadata)
                print(f"{case}: {index + 1}/{len(rows)}", flush=True)
        metadata.update(status="completed", predictions_sha256=digest(path))
        save(meta_path, metadata)
        return previous
    except BaseException as failure:
        metadata.update(status="failed", error=str(failure), completed_rows=len(previous))
        save(meta_path, metadata)
        raise


def evaluate(settings):
    import numpy as np
    import torch
    from transformers import set_seed
    from scripts.projector_model import AudioProjector
    from scripts.published_feedback_metrics import parse_prediction
    config = read_json(rooted(settings["projector_config"]))
    parts, manifest = load(config)
    rows = parts["test"]
    output = rooted(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ProjectorPipeline(config, "alignment")
    tokenizer, model, dtype = pipeline.tokenizer, pipeline.model, pipeline.dtype
    torch.cuda.reset_peak_memory_stats()
    trained = deepcopy(model.projector.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        untrained = AudioProjector(config["audio_hidden_size"], config["text_hidden_size"], config["projector_hidden_size"]).state_dict()
    random_sha = hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in untrained.values())).hexdigest()
    shared = {"settings": settings, "split": "test", "full_split_rows": len(rows), "dataset": manifest,
              "projector_training": pipeline.training, "encoder": pipeline.encoder_identity,
              "random_projector_sha256": random_sha, "random_baseline_scope": "Seeded random projector, not a saved exact step-0 training checkpoint",
              "feature_manifest_sha256": digest(rooted(config["feature_dir"]) / "manifest.json"),
              "base_weight_sha256": pipeline.training["identity"]["base_weight_sha256"],
              "evaluation_code_sha256": digest(Path(__file__)), "inference_code_sha256": digest(ROOT / "scripts/projector_inference.py"),
              "metrics_code_sha256": digest(ROOT / "scripts/published_feedback_metrics.py"),
              "gpu": torch.cuda.get_device_name(), "precision": str(dtype),
              "packages": {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "sacrebleu", "rouge-score")},
              "no_training": True, "no_external_api": True}
    save(output / "benchmark_manifest.json", shared)
    dataset = LatentDataset(rows, tokenizer, config, "alignment")
    def compute_nll(index, row, *, mute=False):
        batch = {key: value.cuda() for key, value in dataset[index].items()}
        if mute:
            batch["features"].zero_()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            loss = float(model(**batch))
        if not math.isfinite(loss):
            raise ValueError("Nonfinite held-out loss")
        tokens = batch["target_ids"].numel()
        return {"reference_text": row["reference_text"], "nll_sum": loss * tokens, "target_tokens": tokens}
    # Full held-out likelihood: the target is supplied only to the loss, never to generation prompts.
    for case, state, mute in (("transcript_nll_random", untrained, False),
                              ("transcript_nll_aligned", trained, False), ("transcript_nll_muted", trained, True)):
        model.projector.load_state_dict(state)
        run_case(output, case, {**shared, "purpose": "full_held_out_teacher_forced_nll", "task": "transcription", "muted": mute},
                 rows, lambda i, r: compute_nll(i, r, mute=mute))
    samples = select_balanced(rows, settings["transcription_per_speaker"], settings["seed"], "transcription")
    pilot = LatentDataset(samples, tokenizer, config, "alignment")
    for case, state, mute in (("transcript_generation_random", untrained, False),
                              ("transcript_generation_aligned", trained, False), ("transcript_generation_muted", trained, True)):
        model.projector.load_state_dict(state)
        def transcribe(index, row):
            features = pilot[index]["features"]
            if mute:
                features.zero_()
            result = pipeline.generate(features, row, "alignment", max_new_tokens=settings["transcription_max_new_tokens"])
            return {"reference_text": row["reference_text"], **result, **word_counts(row["reference_text"], result["raw_text"])}
        run_case(output, case, {**shared, "purpose": "balanced_generation_pilot", "task": "transcription", "muted": mute,
                               "generation": {"do_sample": False, "max_new_tokens": settings["transcription_max_new_tokens"]}}, samples, transcribe)
    model.projector.load_state_dict(trained)
    # One fixed raw-audio check per held-out speaker; compare the end-to-end encoder to the hashed cache.
    smoke = select_balanced(rows, 1, settings["seed"], "raw_audio_parity")
    frozen_features = LatentDataset(smoke, tokenizer, config, "alignment")
    def raw_audio_check(index, row):
        if digest(rooted(row["audio_path"])) != row["audio_sha256"]:
            raise ValueError("Raw audio checksum changed")
        features = pipeline.encode_audio(rooted(row["audio_path"]))
        expected = frozen_features[index]["features"]
        if not torch.equal(features, expected):
            raise ValueError("Raw audio encoder differs from frozen feature cache")
        result = pipeline.generate(features, row, "alignment", max_new_tokens=settings["transcription_max_new_tokens"])
        return {"raw_cache_equal": True, "audio_frames": features.shape[1], "reference_text": row["reference_text"], **result,
                **word_counts(row["reference_text"], result["raw_text"])}
    run_case(output, "raw_audio_parity", {**shared, "purpose": "end_to_end_integrity_check", "task": "transcription"}, smoke, raw_audio_check)
    pipeline.encoder = pipeline.extractor = None
    samples = select_balanced(rows, settings["feedback_per_speaker"], settings["seed"], "feedback")
    feedback_dataset = LatentDataset(samples, tokenizer, config, "feedback")
    def feedback(index, row):
        batch = {key: value.cuda() for key, value in feedback_dataset[index].items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            nll = float(model(**batch))
        tokens = batch["target_ids"].numel()
        result = pipeline.generate(batch["features"], row, "feedback", max_new_tokens=settings["feedback_max_new_tokens"])
        return {"reference_feedback": row["feedback"], "word_order": row["word_order"],
                "nll_sum": nll * tokens, "target_tokens": tokens, **result}
    feedback_identity = {**shared, "purpose": "balanced_generation_pilot", "task": "english_feedback",
                         "generation": {"do_sample": False, "max_new_tokens": settings["feedback_max_new_tokens"]}}
    run_case(output, "feedback_projector_P0", {**feedback_identity, "feedback_task_trained": False}, samples, feedback)
    # Control: the SAME sampled utterances/reference labels, conditioned only on actual B predictions.
    from peft import PeftModel
    from scripts.cascade_bridge import load_condition
    from scripts.l2arctic_plus_data import load as load_native, messages, encode
    from scripts.train_feedback_model import collate_pairs, chunked_causal_loss
    cascade = read_json(rooted(config["cascade_config"]))
    qconfig = read_json(rooted(cascade["feedback_config"]))
    native, native_manifest = load_native(rooted(qconfig["dataset_dir"]))
    lookup = {row["utterance_id"]: row for row in native["test"]}
    native_samples = [lookup[row["utterance_id"]] for row in samples]
    conditioned, condition = load_condition(native_samples, native_manifest, rooted(cascade["report_root"]) / "audio/B")
    adapter = rooted(cascade["run_root"]) / "Q/adapter"
    provenance = read_json(adapter.parent / "training_provenance.json")
    if (provenance["status"] != "completed" or provenance["purpose"] != "scientific_training"
            or provenance["dataset"] != native_manifest
            or provenance["base_weight_sha256"] != shared["base_weight_sha256"]
            or provenance["adapter_sha256"] != digest(adapter / "adapter_model.safetensors")):
        raise ValueError("Invalid cascade Qwen training provenance")
    if [(r["utterance_id"], r["feedback"]) for r in native_samples] != [(r["utterance_id"], r["feedback"]) for r in samples]:
        raise ValueError("Cascade/Projector references differ")
    base = pipeline.model.llm
    def coaching(index, row, llm):
        if row.get("input_alignment_status", "ok") != "ok":
            return {"reference_feedback": row["feedback"], "word_order": [], "prediction": None, "valid_json": False,
                    "raw_text": "", "validation_error": "unmappable_input_abstained", "finished_with_eos": False,
                    "generation_sec": 0., "generated_tokens": 0, "nll_sum": None, "target_tokens": 0}
        prompt = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        count = inputs["input_ids"].shape[1]
        if count + settings["feedback_max_new_tokens"] > 4096:
            raise ValueError("Cascade prompt exceeds the shared context budget")
        encoded = encode(tokenizer, row, 4096)
        batch = {key: val.cuda() for key, val in collate_pairs([encoded], tokenizer.eos_token_id, assistant_only_logits=True).items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            nll = float(chunked_causal_loss(llm, batch, 128))
        tokens = sum(label != -100 for label in encoded["labels"])
        set_seed(settings["seed"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            generated = llm.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=settings["feedback_max_new_tokens"],
                                     use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)[0, count:]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        finished = bool(len(generated) and int(generated[-1]) == tokenizer.eos_token_id)
        parsed, error = None, None
        try:
            if not finished:
                raise ValueError("Token budget reached without EOS")
            parsed = parse_prediction(text)
        except (ValueError, TypeError) as failure:
            error = str(failure)
        return {"reference_feedback": row["feedback"], "word_order": list(row["annotation_info"]),
                "prediction": parsed, "valid_json": error is None, "validation_error": error, "raw_text": text,
                "finished_with_eos": finished, "prompt_tokens": count, "generated_tokens": len(generated),
                "generation_sec": elapsed, "nll_sum": nll * tokens, "target_tokens": tokens}
    run_case(output, "feedback_cascade_B_Q0", {**feedback_identity, "input_condition": condition, "adapter_sha256": None},
             conditioned, lambda i, r: coaching(i, r, base))
    adapted = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True).eval().requires_grad_(False)
    run_case(output, "feedback_cascade_B_Q1", {**feedback_identity, "input_condition": condition, "adapter_sha256": provenance["adapter_sha256"]},
             conditioned, lambda i, r: coaching(i, r, adapted))
    run_case(output, "feedback_oracle_Q1", {**feedback_identity, "input_condition": "oracle_published_phones_diagnostic_only",
             "adapter_sha256": provenance["adapter_sha256"]}, native_samples, lambda i, r: coaching(i, r, adapted))
    save(output / "execution.json", {"status": "completed", "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "full_transcript_nll_rows": len(rows), "transcript_generation_rows": len(pilot), "feedback_generation_rows": len(samples),
        "raw_audio_checks": len(smoke), "feedback_projector_stage": "alignment_only", "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/projector_validation.json")
    args = parser.parse_args()
    evaluate(read_json(args.config))
