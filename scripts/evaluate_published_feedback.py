"""Offline deterministic Qwen evaluation, raw outputs, no fallback and safe resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from scripts.l2arctic_plus_data import CONFIG, PROMPT, encode, json_digest, load, messages, rooted
from scripts.prepare_feedback_data import digest
from scripts.published_feedback_metrics import parse_prediction, summarize
from scripts.train_feedback_model import collate_pairs, chunked_causal_loss

EVAL_CONFIG = ROOT / "configs/feedback_l2arctic_plus_eval.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_observation(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(args):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.eval_config.read_text(encoding="utf-8"))
    if config.get("dataset_kind") != "published_l2arctic_plus" or protocol["task"] != config["task"]:
        raise ValueError("Training and evaluation tasks differ")
    if protocol["generation"].get("do_sample") is not False:
        raise ValueError("The locked comparison uses greedy generation")
    torch.set_num_threads(protocol.get("cpu_threads", 4))
    parts, manifest = load(rooted(config["dataset_dir"]))
    rows = parts[args.split]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    condition = {"condition": "oracle_published_phone_annotations"}
    if args.mdd_evaluation:
        from scripts.cascade_bridge import load_condition
        rows, condition = load_condition(rows, manifest, args.mdd_evaluation.resolve(), allow_technical=args.limit is not None)
    purpose = "technical_smoke" if args.limit is not None or args.allow_technical_adapter else "held_out_benchmark"
    model_path = rooted(config["model_path"])
    if not torch.cuda.is_available():
        raise ValueError("This comparison profile requires CUDA/NF4 for both models")
    adapter = args.adapter.resolve() if args.adapter else None
    training_provenance = None
    if adapter:
        training_provenance = json.loads((adapter.parent / "training_provenance.json").read_text(encoding="utf-8"))
        if training_provenance["status"] != "completed" or training_provenance["dataset"] != manifest:
            raise ValueError("Adapter is incomplete or trained on a different frozen dataset")
        if training_provenance["purpose"] != "scientific_training" and not args.allow_technical_adapter:
            raise ValueError("A technical adapter cannot be reported as a trained scientific model")
        if digest(adapter / "adapter_model.safetensors") != training_provenance["adapter_sha256"]:
            raise ValueError("Adapter checksum differs from its training record")
    packages = {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "sacrebleu", "rouge-score")}
    identity = {
        "task": config["task"], "purpose": purpose, "split": args.split, "rows": len(rows),
        "full_split_rows": len(parts[args.split]), "dataset_split_sha256": manifest["sha256"][args.split],
        "source_commit": manifest["source_commit"], "prompt_sha256": digest(PROMPT),
        "tokenizer_sha256": digest(model_path / "tokenizer.json"),
        "base_weight_sha256": digest(model_path / "model.safetensors"),
        "base_revision": (model_path / "download_revision.txt").read_text().strip(),
        "adapter_sha256": digest(adapter / "adapter_model.safetensors") if adapter else None,
        "adapter_path": str(adapter) if adapter else None,
        "protocol": protocol, "protocol_sha256": json_digest(protocol),
        "ordered_rows_sha256": json_digest([(row["utterance_id"], row["source_row_sha256"]) for row in rows]),
        "gpu": torch.cuda.get_device_name(), "packages": packages, "platform": platform.platform(),
        "precision_profile": protocol["quantization"],
        "input_condition": condition,
    }
    if training_provenance and identity["base_weight_sha256"] != training_provenance["base_weight_sha256"]:
        raise ValueError("Adapter and evaluation base weights differ")
    if identity["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from frozen preflight")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".evaluation.lock"), timeout=1):
        metadata_path = output / "metadata.json"
        prediction_path = output / "predictions.jsonl"
        if metadata_path.exists():
            if not args.resume:
                raise ValueError("Output exists; use --resume or select a new directory")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                raise ValueError("Resume fingerprint differs: dataset, prompt, weights or evaluation settings changed")
        else:
            if prediction_path.exists():
                raise ValueError("Predictions exist without matching metadata")
            metadata = {"identity": identity, "started_at_utc": utc(), "status": "running"}
            write_json(metadata_path, metadata)
        observations = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()] if prediction_path.exists() else []
        if len(observations) > len(rows) or [row["utterance_id"] for row in observations] != [row["utterance_id"] for row in rows[:len(observations)]]:
            raise ValueError("Cached predictions are not a prefix of this frozen evaluation")
        for observed, original in zip(observations, rows):
            if observed["source_row_sha256"] != original["source_row_sha256"] or observed["reference_feedback"] != original["feedback"]:
                raise ValueError("Cached references differ from the frozen dataset")
        if len(observations) < len(rows):
            set_seed(protocol["seed"])
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            identity_dtype = str(dtype)
            quantization = None
            if protocol["quantization"] == "nf4-double-bf16-or-fp16":
                quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            elif protocol["quantization"] != "none-bf16-or-fp16":
                raise ValueError("Unknown evaluation precision profile")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                quantization_config=quantization, device_map={"": torch.cuda.current_device()}, dtype=dtype,
                attn_implementation=protocol.get("attention_implementation", "sdpa"))
            if quantization:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            if adapter:
                model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True, is_trainable=False)
            model.eval().requires_grad_(False)
            model.config.use_cache = True
            torch.cuda.reset_peak_memory_stats()
            metadata.update(status="running", compute_dtype=identity_dtype)
            write_json(metadata_path, metadata)
            try:
                for original in rows[len(observations):]:
                    if original.get("input_alignment_status", "ok") != "ok":
                        observed = {"utterance_id": original["utterance_id"], "speaker_id": original["speaker_id"],
                            "source_row_sha256": original["source_row_sha256"], "word_order": [],
                            "reference_feedback": original["feedback"], "prediction": None, "valid_json": False,
                            "raw_text": "", "validation_error": "unmappable_input_abstained", "finished_with_eos": False,
                            "prompt_tokens": 0, "generated_tokens": 0, "generation_sec": 0., "nll_sum": None, "target_tokens": 0}
                        append_observation(prediction_path, observed)
                        observations.append(observed)
                        print(f"{len(observations)}/{len(rows)} {observed['utterance_id']} input uncertain: abstained", flush=True)
                        continue
                    # Re-seeding each utterance also makes interrupted/resumed runs comparable.
                    set_seed(protocol["seed"])
                    prompt = tokenizer.apply_chat_template(messages(original), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                    prompt_tokens = inputs["input_ids"].shape[1]
                    budget = protocol["generation"]["max_new_tokens"]
                    if prompt_tokens + budget > protocol["context_length"]:
                        raise ValueError("Prompt exceeds the locked generation context budget")
                    encoded = encode(tokenizer, original, protocol["context_length"])
                    batch = {key: value.to("cuda") for key, value in collate_pairs([encoded], tokenizer.pad_token_id, assistant_only_logits=True).items()}
                    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                        # Teacher-forced NLL on reference tokens is measured even when generation fails its schema.
                        loss = float(chunked_causal_loss(model, batch, config.get("loss_projection_chunk_size", 128)))
                    if not math.isfinite(loss):
                        raise ValueError("Nonfinite reference loss")
                    target_tokens = sum(label != -100 for label in encoded["labels"])
                    del batch
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                        generated = model.generate(**inputs, **protocol["generation"],
                            use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    new_ids = generated[0, prompt_tokens:]
                    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    finished = bool(len(new_ids) and int(new_ids[-1]) == tokenizer.eos_token_id)
                    prediction, validation_error = None, None
                    try:
                        if not finished:
                            raise ValueError("Token budget reached without EOS")
                        prediction = parse_prediction(text)
                    except (ValueError, TypeError) as error:
                        validation_error = str(error)[:500]
                    observed = {"utterance_id": original["utterance_id"], "speaker_id": original["speaker_id"],
                        "source_row_sha256": original["source_row_sha256"], "word_order": list(original["annotation_info"]),
                        "reference_feedback": original["feedback"], "prediction": prediction, "valid_json": validation_error is None,
                        "raw_text": text, "validation_error": validation_error, "finished_with_eos": finished,
                        "prompt_tokens": prompt_tokens, "generated_tokens": len(new_ids), "generation_sec": elapsed,
                        "nll_sum": loss * target_tokens, "target_tokens": target_tokens}
                    append_observation(prediction_path, observed)
                    observations.append(observed)
                    print(f"{len(observations)}/{len(rows)} {observed['utterance_id']} JSON={observed['valid_json']} tokens={len(new_ids)} seconds={elapsed:.2f}", flush=True)
                metadata["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            except BaseException as error:
                metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error_type=type(error).__name__, completed_rows=len(observations))
                write_json(metadata_path, metadata)
                raise
        summary, _ = summarize(observations)
        write_json(output / "summary.json", summary)
        metadata.update(status="completed", completed_rows=len(observations), finished_at_utc=utc(),
                        predictions_sha256=digest(prediction_path), summary_sha256=digest(output / "summary.json"))
        write_json(metadata_path, metadata)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--mdd-evaluation", type=Path, help="Audio evaluation directory for real cascade input; omit for oracle-only coaching")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Technical preview only; never a full scientific benchmark")
    parser.add_argument("--allow-technical-adapter", action="store_true", help="Mark output as technical_smoke")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
