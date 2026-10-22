"""Train a Qwen3-0.6B QLoRA coaching adapter from reviewed feedback pairs."""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

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


def collate_pairs(rows, pad_token_id, *, assistant_only_logits=False):
    import torch
    length = max(len(row["input_ids"]) for row in rows)
    batch = {key: torch.tensor([row[key] + [(-100 if key == "labels" else 0 if key == "attention_mask" else pad_token_id)]
                              * (length - len(row[key])) for row in rows], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")}
    if assistant_only_logits:
        # The prompt still traverses the entire Transformer. Only the large
        # vocabulary projection at ignored next-token positions is skipped.
        # Explicit shift_labels preserves the exact causal assistant-only loss.
        shifted = torch.nn.functional.pad(batch["labels"][:, 1:], (0, 1), value=-100)
        positions = shifted.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        if not len(positions):
            raise ValueError("A batch needs supervised next-token labels")
        batch["logits_to_keep"] = positions
        batch["shift_labels"] = shifted[:, positions].contiguous()
    return batch


def chunked_causal_loss(model, batch, chunk_size=128, num_items_in_batch=None):
    """Exact assistant-only causal loss with bounded vocabulary-projection memory.

    The entire prompt traverses Qwen. Checkpointing recomputes only the small
    projection chunks, avoiding retention of all target-token vocabulary logits.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    if chunk_size < 1:
        raise ValueError("Projection chunk size must be positive")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    features = hidden[:, :-1, :][mask]
    targets = labels[mask]
    if not len(targets):
        raise ValueError("A batch needs supervised next-token labels")
    def projection_loss(values, truth):
        return torch.nn.functional.cross_entropy(base.lm_head(values).float(), truth, reduction="sum")
    losses = []
    for start in range(0, len(targets), chunk_size):
        values, truth = features[start:start + chunk_size], targets[start:start + chunk_size]
        losses.append(checkpoint(projection_loss, values, truth, use_reentrant=False) if torch.is_grad_enabled()
                      else projection_loss(values, truth))
    denominator = len(targets) if num_items_in_batch is None else num_items_in_batch
    return torch.stack(losses).sum() / denominator


def data_check(config):
    directory = from_root(config["dataset_dir"])
    if not (directory / "manifest.json").is_file():
        raise ValueError("No frozen reviewed feedback dataset. Review pairs and run prepare_feedback_data.py prepare first.")
    kind = config.get("dataset_kind", "reviewed_feedback")
    if kind == "published_l2arctic_plus":
        if __package__:
            from .l2arctic_plus_data import load
        else:
            from l2arctic_plus_data import load
        parts, manifest = load(directory)
        if config["task"] != manifest["task"] or config["max_length"] != manifest["max_length"]:
            raise ValueError("Training task/context differs from frozen published dataset")
    elif kind == "published_grammar_fluency":
        if __package__:
            from .grammar_data import load
        else:
            from grammar_data import load
        parts, manifest = load(directory)
        if config["task"] != manifest["task"] or config["max_length"] != manifest["max_length"]:
            raise ValueError("Training task/context differs from frozen grammar dataset")
    elif kind == "reviewed_feedback":
        parts, manifest = load_partitions(directory)
    else:
        raise ValueError("Unknown dataset_kind: " + str(kind))
    for name, minimum in (("train", config["min_train_examples"]), ("validation", config["min_validation_examples"])):
        if len(parts[name]) < minimum:
            raise ValueError(f"Need at least {minimum} eligible {name} pairs; found {len(parts[name])}")
    return parts, manifest


def train(config, parts, manifest, *, resume=False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise ValueError("QLoRA requires a CUDA-capable training machine.")
    output = from_root(config["output_dir"])
    provenance_path = output / "training_provenance.json"
    checkpoint = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError("Training output exists; use --resume or choose a new output_dir")
        prior = json.loads(provenance_path.read_text(encoding="utf-8"))
        if prior["effective_config"] != config or prior["dataset"] != manifest:
            raise ValueError("Cannot resume with changed config or frozen dataset")
        if prior["status"] == "completed":
            print("Training already completed; use its saved adapter for comparison.")
            return
        checkpoint = get_last_checkpoint(str(output))
        if not checkpoint:
            raise ValueError("No resumable checkpoint; keep logs and choose a new output_dir")
    path = from_root(config["model_path"])
    if not (path / "model.safetensors").is_file():
        raise ValueError("Download the feedback model before training")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    # Preflight every example; never truncate away a diagnosis or target JSON.
    encoder = encode_pair
    if config.get("dataset_kind") in {"published_l2arctic_plus", "published_grammar_fluency"}:
        if config["dataset_kind"] == "published_grammar_fluency":
            if __package__:
                from .grammar_data import encode
            else:
                from grammar_data import encode
        elif __package__:
            from .l2arctic_plus_data import encode
        else:
            from l2arctic_plus_data import encode
        encoder = encode
        if manifest["tokenizer_sha256"] != digest(path / "tokenizer.json"):
            raise ValueError("Tokenizer differs from dataset preflight")
    encoded = {name: [encoder(tokenizer, row, config["max_length"]) for row in parts[name]]
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
        output_dir=str(output), num_train_epochs=config["epochs"], max_steps=config.get("max_steps", -1), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], warmup_ratio=config.get("warmup_ratio", 0.05), weight_decay=0.01,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if config.get("eval_steps") else "epoch",
        save_strategy="steps" if config.get("eval_steps") else "epoch",
        eval_steps=config.get("eval_steps"), save_steps=config.get("eval_steps", 500), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=config.get("logging_steps", 10), report_to="none", seed=config["seed"], dataloader_num_workers=0, prediction_loss_only=True,
        optim="adamw_torch", remove_unused_columns=False,
    )
    class History(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            output.mkdir(parents=True, exist_ok=True)
            with (output / "training_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, "at_utc": datetime.now(timezone.utc).isoformat(),
                                         **(logs or {})}, default=str) + "\n")

    class BoundedProjectionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = chunked_causal_loss(model, inputs, config["loss_projection_chunk_size"], num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss

    trainer_class = BoundedProjectionTrainer if config.get("loss_projection_chunk_size") else Trainer
    trainer = trainer_class(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      callbacks=[History()],
                      data_collator=lambda rows: collate_pairs(rows, tokenizer.pad_token_id, assistant_only_logits=True))
    if config.get("loss_projection_chunk_size"):
        trainer.model_accepts_loss_kwargs = True
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"effective_config": config, "dataset": manifest,
                  "base_model_revision": (path / "download_revision.txt").read_text().strip(),
                  "base_weight_sha256": digest(path / "model.safetensors"),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "supervision": "assistant feedback tokens only; prompt labels masked with -100",
                  "logit_projection": "supervised next-token positions only; full prompt and unchanged causal loss",
                  "test_used_for_training": False, "status": "started",
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "resume_from_checkpoint": checkpoint,
                  "purpose": config.get("purpose", "scientific_training")}
    provenance["samples_used"] = {name: len(encoded[name]) for name in encoded}
    provenance["loss_projection_chunk_size"] = config.get("loss_projection_chunk_size")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
    except BaseException as error:
        provenance.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error_type=type(error).__name__, global_step=trainer.state.global_step)
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    provenance.update(status="completed", metrics=result.metrics, global_step=trainer.state.global_step,
                      best_validation_loss=trainer.state.best_metric,
                      best_checkpoint=trainer.state.best_model_checkpoint,
                      adapter_sha256=digest(output / "adapter/adapter_model.safetensors"),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Adapter saved. Evaluate against the original Qwen checkpoint before enabling it in the demo.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/feedback_qlora_rtx3050.json")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume matching checkpoints; completed runs are kept")
    parser.add_argument("--technical-smoke", action="store_true", help="One update on a small subset, saved separately; not a scientific run")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        parts, manifest = data_check(config)
        if args.technical_smoke:
            config.update(purpose="technical_smoke", max_steps=1, eval_steps=1, warmup_ratio=0., logging_steps=1,
                          output_dir="runs/local-checks/l2plus-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            ordered = sorted(parts["train"], key=lambda row: row.get("total_tokens", 0), reverse=True)
            parts = {**parts, "train": ordered[:config["gradient_accumulation_steps"]], "validation": parts["validation"][:1]}
            print("Technical output:", config["output_dir"], flush=True)
        print(json.dumps({"eligible_pairs": {name: len(rows) for name, rows in parts.items()},
                          "dataset_kind": config.get("dataset_kind", "reviewed_feedback")}, indent=2))
        if not args.check_data_only:
            train(config, parts, manifest, resume=args.resume)
    except (OSError, ValueError, ImportError, ValidationError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
