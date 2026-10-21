"""Train a Qwen3-0.6B QLoRA coaching adapter from reviewed feedback pairs."""

import argparse
import json
from pathlib import Path

from jsonschema import ValidationError

if __package__:
    from .llm_client import ROOT, feedback_messages
    from .prepare_feedback_data import digest, load_partitions
else:
    from llm_client import ROOT, feedback_messages
    from prepare_feedback_data import digest, load_partitions


def from_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def encode_pair(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(feedback_messages(row["assessment"], local=True),
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target = json.dumps(row["feedback"], ensure_ascii=False, separators=(",", ":"))
    target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError(f"Pair {row['assessment']['utterance_id']} has {len(ids)} tokens, limit {max_length}; shorten it, do not truncate labels")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + target_ids}


def collate_pairs(rows, pad_token_id):
    import torch
    length = max(len(row["input_ids"]) for row in rows)
    return {key: torch.tensor([row[key] + [(-100 if key == "labels" else 0 if key == "attention_mask" else pad_token_id)]
                              * (length - len(row[key])) for row in rows], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")}


def data_check(config):
    directory = from_root(config["dataset_dir"])
    if not (directory / "manifest.json").is_file():
        raise ValueError("No frozen reviewed feedback dataset. Review pairs and run prepare_feedback_data.py prepare first.")
    parts, manifest = load_partitions(directory)
    for name, minimum in (("train", config["min_train_examples"]), ("validation", config["min_validation_examples"])):
        if len(parts[name]) < minimum:
            raise ValueError(f"Need at least {minimum} reviewed {name} pairs; found {len(parts[name])}")
    return parts, manifest


def train(config, parts, manifest):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise ValueError("QLoRA requires the CUDA training machine; the current Linux workspace is CPU-only.")
    output = from_root(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output already exists; select a new output_dir to preserve checkpoints")
    path = from_root(config["model_path"])
    if not (path / "model.safetensors").is_file():
        raise ValueError("Download the feedback model before training")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    # Preflight every example; never truncate away a diagnosis or target JSON.
    encoded = {name: [encode_pair(tokenizer, row, config["max_length"]) for row in parts[name]]
               for name in ("train", "validation")}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, quantization_config=quantization,
        device_map={"": torch.cuda.current_device()}, dtype=dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none",
    ))
    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=config["epochs"], per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], warmup_ratio=0.05, weight_decay=0.01,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, report_to="none", seed=config["seed"], dataloader_num_workers=0,
        optim="adamw_torch", remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      data_collator=lambda rows: collate_pairs(rows, tokenizer.pad_token_id))
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"effective_config": config, "dataset": manifest,
                  "base_model_revision": (path / "download_revision.txt").read_text().strip(),
                  "base_weight_sha256": digest(path / "model.safetensors"),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "supervision": "assistant feedback tokens only; prompt labels masked with -100",
                  "test_used_for_training": False, "status": "started"}
    provenance_path = output / "training_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    provenance.update(status="completed", metrics=result.metrics, global_step=trainer.state.global_step,
                      best_validation_loss=trainer.state.best_metric)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Adapter saved. Evaluate against the original Qwen checkpoint before enabling it in the demo.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/feedback_qlora_rtx3050.json")
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        parts, manifest = data_check(config)
        print(json.dumps({"reviewed_pairs": {name: len(rows) for name, rows in parts.items()}}, indent=2))
        if not args.check_data_only:
            train(config, parts, manifest)
    except (OSError, ValueError, ImportError, ValidationError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
