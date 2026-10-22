"""Paired held-out feedback benchmark: aligned P0 versus task-tuned P1."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import CONFIG, PROMPT, LatentDataset, load, read_json, read_rows, save
from scripts.l2arctic_plus_data import rooted, json_digest
from scripts.prepare_feedback_data import digest
from scripts.train_projector import scientific_stage, model_files
from scripts.published_feedback_metrics import parse_prediction, summarize


def evaluate(config, case, *, limit=None):
    import torch
    from transformers import AutoTokenizer, set_seed
    from scripts.projector_model import build_model
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    stage = "alignment" if case == "P0" else "feedback"
    weights, training = scientific_stage(config, stage)
    parts, dataset = load(config)
    feature_path = rooted(config["feature_dir"]) / "manifest.json"
    base_hash = digest(rooted(config["model_path"]) / "model.safetensors")
    for key, actual in (("config", config), ("dataset", dataset), ("base_weight_sha256", base_hash),
                        ("feature_manifest_sha256", digest(feature_path))):
        if training["identity"][key] != actual:
            raise ValueError("Training/evaluation mismatch: " + key)
    rows = parts["test"][:limit] if limit is not None else parts["test"]
    if not rows:
        raise ValueError("Empty projector evaluation")
    tokenizer = AutoTokenizer.from_pretrained(str(rooted(config["model_path"])), local_files_only=True)
    examples = LatentDataset(rows, tokenizer, config, "feedback")
    feature = read_json(feature_path)
    identity = {"case": case, "task": "audio_reference_to_english_feedback", "split": "test",
                "purpose": "technical_smoke" if limit else "held_out_benchmark", "rows": len(rows), "full_split_rows": len(parts["test"]),
                "dataset_split_sha256": dataset["sha256"]["test"], "protocol_sha256": dataset["protocol_sha256"],
                "ordered_rows_sha256": json_digest([(r["utterance_id"], r["source_row_sha256"]) for r in rows]),
                "base_weight_sha256": base_hash, "tokenizer_sha256": digest(rooted(config["model_path"]) / "tokenizer.json"),
                "weights": model_files(weights), "training_identity_sha256": json_digest(training["identity"]),
                "input_condition": {"encoder": feature["identity"]["encoder"], "features_sha256": digest(feature_path),
                                    "gold_phone_annotations": False},
                "prompt_sha256": digest(PROMPT), "generation": config["generation"], "context_length": 4096,
                "precision_profile": "none-bf16-or-fp16", "metrics_code_sha256": digest(ROOT / "scripts/published_feedback_metrics.py"),
                "seed": config["seed"], "gpu": torch.cuda.get_device_name(),
                "packages": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "sacrebleu", "rouge-score")},
                "evaluation_code_sha256": digest(Path(__file__)), "model_code_sha256": digest(ROOT / "scripts/projector_model.py")}
    output = rooted(config["report_root"]) / (("technical-" + case) if limit else case)
    output.mkdir(parents=True, exist_ok=True)
    meta_path, prediction_path = output / "metadata.json", output / "predictions.jsonl"
    if meta_path.exists():
        metadata = read_json(meta_path)
        if metadata["identity"] != identity:
            raise ValueError("Evaluation resume identity differs")
        if metadata["status"] == "completed":
            if digest(prediction_path) != metadata["predictions_sha256"]:
                raise ValueError("Completed predictions changed")
            return read_json(output / "summary.json")
    observations = read_rows(prediction_path) if prediction_path.exists() else []
    if len(observations) > len(rows):
        raise ValueError("Too many resumed predictions")
    for previous, row in zip(observations, rows):
        if (previous["utterance_id"] != row["utterance_id"] or previous["reference_feedback"] != row["feedback"]
                or previous["source_row_sha256"] != row["source_row_sha256"]):
            raise ValueError("Resumed rows are not the expected held-out prefix")
    metadata = {"identity": identity, "status": "running", "completed_rows": len(observations)}
    save(meta_path, metadata)
    model = build_model(config, stage, training=False, initial=weights).eval()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(len(observations), len(rows)):
            row = rows[i]
            batch = {key: value.cuda() for key, value in examples[i].items()}
            prompt_length = examples.lengths[i] - batch["target_ids"].numel()
            if prompt_length + config["generation"]["max_new_tokens"] > 4096:
                raise ValueError("Generation context exceeds the locked 4096-token budget")
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                nll = float(model(**batch))
            if not math.isfinite(nll):
                raise ValueError("Nonfinite target NLL")
            set_seed(config["seed"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                generated = model.generate_response(**{key: value for key, value in batch.items() if key != "target_ids"},
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, **config["generation"])[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            finished = bool(len(generated) and generated[-1].item() == tokenizer.eos_token_id)
            prediction, error = None, None
            try:
                if not finished:
                    raise ValueError("Token budget reached without EOS")
                prediction = parse_prediction(text)
            except (ValueError, TypeError) as failure:
                error = str(failure)[:500]
            observation = {"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"],
                           "source_row_sha256": row["source_row_sha256"], "word_order": row["word_order"],
                           "reference_feedback": row["feedback"], "prediction": prediction, "raw_text": text,
                           "valid_json": error is None, "validation_error": error, "finished_with_eos": finished,
                           "nll_sum": nll * batch["target_ids"].numel(), "target_tokens": batch["target_ids"].numel(),
                           "generated_tokens": len(generated), "prompt_tokens": prompt_length, "generation_sec": elapsed}
            with prediction_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
            observations.append(observation)
            metadata["completed_rows"] = len(observations)
            save(meta_path, metadata)
            print(f"{case} {i + 1}/{len(rows)} {row['utterance_id']} JSON={error is None} seconds={elapsed:.2f}", flush=True)
        summary, _ = summarize(observations)
        save(output / "summary.json", summary)
        metadata.update(status="completed", predictions_sha256=digest(prediction_path),
                        summary_sha256=digest(output / "summary.json"), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(meta_path, metadata)
        return summary
    except BaseException as error:
        metadata.update(status="failed", error_type=type(error).__name__, completed_rows=len(observations))
        save(meta_path, metadata)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--case", choices=("P0", "P1"), required=True)
    parser.add_argument("--limit", type=int, help="Technical output only; excluded from scientific reports")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    evaluate(read_json(args.config), args.case, limit=args.limit)
