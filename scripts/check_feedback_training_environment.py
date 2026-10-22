"""One explicitly technical QLoRA optimizer update; never a reviewed-data run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_feedback_data import digest
from scripts.train_feedback_model import collate_pairs, encode_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical-smoke-step", action="store_true", required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad-to-limit", action="store_true", help="Exercise the configured 3072-token memory limit; padding labels remain ignored")
    parser.add_argument("--full-logits", action="store_true", help="Compare the original full-sequence projection; may exceed physical VRAM")
    parser.add_argument("--trainer-step", action="store_true", help="Also exercise Trainer, eight-way accumulation, validation and checkpoint save")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Choose a new smoke directory inside this workspace")
    output.mkdir(parents=True)
    report = {"purpose": "technical_smoke_only", "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "optimizer_updates": 0, "approved_training_dataset": False, "production_adapter_enabled": False,
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}}
    destination = output / "report.json"
    started = time.monotonic()
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        if not torch.cuda.is_available():
            raise ValueError("CUDA is required")
        torch.set_num_threads(4)
        config = json.loads((ROOT / "configs/feedback_qlora_rtx3050.json").read_text(encoding="utf-8"))
        set_seed(config["seed"])
        path = ROOT / config["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        rows = [json.loads(line) for line in args.drafts.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = max(rows, key=lambda row: row["provenance"]["training_tokens"])
        encoded = encode_pair(tokenizer, row, config["max_length"])
        actual_tokens = len(encoded["input_ids"])
        if args.pad_to_limit:
            padding = config["max_length"] - actual_tokens
            encoded["input_ids"].extend([tokenizer.pad_token_id] * padding)
            encoded["attention_mask"].extend([0] * padding)
            encoded["labels"].extend([-100] * padding)
        report.update(gpu=torch.cuda.get_device_name(0), total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      sample_id=row["assessment"]["utterance_id"], sample_review_approved=row["review"]["approved"],
                      sample_actual_tokens=actual_tokens, tested_sequence_length=len(encoded["input_ids"]),
                      target_tokens=sum(value != -100 for value in encoded["labels"]),
                      source_drafts_sha256=digest(args.drafts), base_weight_sha256_before=digest(path / "model.safetensors"))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        report["compute_dtype"] = str(dtype)
        torch.cuda.reset_peak_memory_stats()
        print("Loading pinned Qwen3-0.6B in NF4; this is a technical check, not the reviewed-data training run.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                  bnb_4bit_compute_dtype=dtype), device_map={"": 0}, dtype=dtype)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none"))
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
        report["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
        batch = {key: value.to("cuda") for key, value in collate_pairs([encoded], tokenizer.pad_token_id,
                    assistant_only_logits=not args.full_logits).items()}
        report["logit_projection"] = "all_positions" if args.full_logits else "supervised_next_token_positions_only"
        model.train()
        print(f"Forward/backward at {len(encoded['input_ids'])} tokens, supervised assistant tokens only.", flush=True)
        if args.trainer_step:
            from transformers import Trainer, TrainingArguments
            training_args = TrainingArguments(output_dir=str(output / "trainer-checkpoint"), max_steps=1,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                learning_rate=config["learning_rate"], warmup_ratio=0., bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1, save_total_limit=1,
                load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                logging_steps=1, report_to="none", seed=config["seed"], dataloader_num_workers=0,
                optim="adamw_torch", remove_unused_columns=False, prediction_loss_only=True, disable_tqdm=True)
            trainer = Trainer(model=model, args=training_args,
                train_dataset=[encoded] * config["gradient_accumulation_steps"], eval_dataset=[encoded],
                data_collator=lambda rows: collate_pairs(rows, tokenizer.pad_token_id, assistant_only_logits=not args.full_logits))
            outcome = trainer.train()
            loss_value = outcome.training_loss
            gradient_value = next(float(entry["grad_norm"]) for entry in trainer.state.log_history if "grad_norm" in entry)
            report.update(trainer_global_step=trainer.state.global_step,
                          trainer_gradient_accumulation_steps=config["gradient_accumulation_steps"],
                          trainer_best_validation_loss=trainer.state.best_metric,
                          trainer_checkpoint_saved=bool(trainer.state.best_model_checkpoint))
        else:
            optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=config["learning_rate"])
            with torch.autocast("cuda", dtype=dtype):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
            if not torch.isfinite(gradient_norm):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            loss_value, gradient_value = float(loss.detach()), float(gradient_norm)
        torch.cuda.synchronize()
        changed = [name for name, parameter in trainable if not torch.equal(before[name], parameter.detach().cpu())]
        if not changed:
            raise ValueError("No LoRA weight changed")
        model.save_pretrained(output / "technical_adapter")
        report.update(status="passed", optimizer_updates=1, loss=loss_value, gradient_norm_before_clip=gradient_value,
                      changed_lora_tensors=len(changed), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), base_weight_sha256_after=digest(path / "model.safetensors"))
        report["base_weight_file_unchanged"] = report["base_weight_sha256_before"] == report["base_weight_sha256_after"]
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, message=str(error)[:1200])
        if "torch" in locals() and torch.cuda.is_available():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, finished_at_utc=datetime.now(timezone.utc).isoformat())
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
