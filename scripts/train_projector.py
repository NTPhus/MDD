"""Two-stage local QLoRA/projector trainer with resumable optimizer checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import CONFIG, LatentDataset, load, read_json, save
from scripts.l2arctic_plus_data import rooted, json_digest
from scripts.prepare_feedback_data import digest


def model_files(folder):
    return {str(path.relative_to(folder)).replace("\\", "/"): digest(path)
            for path in folder.rglob("*.safetensors")}


def scientific_stage(config, stage):
    folder = rooted(config["run_root"]) / stage
    record = read_json(folder / "training_provenance.json")
    if record.get("status") != "completed" or record.get("purpose") != "scientific_training" or record.get("stage") != stage:
        raise ValueError("Incomplete/non-scientific stage: " + stage)
    if record["identity"]["config"] != config:
        raise ValueError("Completed stage recipe differs from current configuration")
    for name, expected in record["identity"]["code"].items():
        if digest(ROOT / "scripts" / name) != expected:
            raise ValueError("Training implementation changed after stage completion: " + name)
    if record["identity"]["feature_manifest_sha256"] != digest(rooted(config["feature_dir"]) / "manifest.json"):
        raise ValueError("Completed stage feature cache changed")
    if model_files(folder / "best") != record["best_weights"]:
        raise ValueError("Stage weights changed: " + stage)
    return folder / "best", record


def train(config, stage, technical=False):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
    from scripts.projector_model import build_model
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    parts, manifest = load(config)
    tokenizer = AutoTokenizer.from_pretrained(str(rooted(config["model_path"])), local_files_only=True)
    train_set = LatentDataset(parts["train"], tokenizer, config, stage)
    val_set = LatentDataset(parts["validation"], tokenizer, config, stage)
    if technical:
        # Stress the longest combined sequence rather than claiming a fast short-input fit.
        order = sorted(range(len(train_set)), key=lambda i: train_set.lengths[i], reverse=True)[:config["gradient_accumulation_steps"]]
        val_order = [max(range(len(val_set)), key=lambda i: val_set.lengths[i])]
    else:
        order, val_order = list(range(len(train_set))), list(range(len(val_set)))
    initial_record, initial = None, None
    if stage == "feedback":
        initial, initial_record = scientific_stage(config, "alignment")
    folder = rooted(config["run_root"]) / ("technical-" + stage if technical else stage)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "stage": stage, "purpose": "technical_smoke" if technical else "scientific_training",
                "dataset": manifest, "feature_manifest_sha256": digest(rooted(config["feature_dir"]) / "manifest.json"),
                "base_weight_sha256": digest(rooted(config["model_path"]) / "model.safetensors"),
                "tokenizer_sha256": digest(rooted(config["model_path"]) / "tokenizer.json"),
                "code": {name: digest(ROOT / "scripts" / name) for name in ("projector_model.py", "projector_data.py", "train_projector.py")},
                "alignment_weights": initial_record["best_weights"] if initial_record else None}
    record_path = folder / "training_provenance.json"
    if record_path.exists():
        prior = read_json(record_path)
        if prior["identity"] != identity:
            raise ValueError("Training resume fingerprint differs; use a new run directory")
        if prior["status"] == "completed":
            if model_files(folder / "best") != prior["best_weights"]:
                raise ValueError("Completed weights changed")
            return prior
    record = {"identity": identity, "purpose": identity["purpose"], "stage": stage, "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    save(record_path, record)
    model = build_model(config, stage, training=True, initial=initial)
    if model.llm.config.hidden_size != config["text_hidden_size"]:
        raise ValueError("Qwen embedding dimension differs from configured projector")
    trainable = [p for p in model.parameters() if p.requires_grad]
    original = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad} if technical else None
    counts = {"projector_trainable": sum(p.numel() for p in model.projector.parameters() if p.requires_grad),
              "llm_trainable": sum(p.numel() for p in model.llm.parameters() if p.requires_grad),
              "encoder_trainable": 0}
    save(folder / "parameter_counts.json", counts)
    optimizer = torch.optim.AdamW(trainable, lr=config[stage]["learning_rate"], weight_decay=0.01)
    accumulation = config["gradient_accumulation_steps"]
    epochs = 1 if technical else config[stage]["epochs"]
    total_steps = 1 if technical else math.ceil(len(order) / accumulation) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0 if technical else int(.05 * total_steps), total_steps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    cursor, step, best, epoch = 0, 0, float("inf"), 0
    latest = folder / "checkpoint"
    if not latest.exists() and (folder / "checkpoint.previous/state.pt").exists():
        (folder / "checkpoint.previous").rename(latest)
    if (latest / "state.pt").exists():
        state = torch.load(latest / "state.pt", map_location="cpu", weights_only=False)
        if state["identity_sha256"] != json_digest(identity):
            raise ValueError("Optimizer checkpoint fingerprint mismatch")
        model.load_weights(latest)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, cursor, step, best = (state[key] for key in ("epoch", "cursor", "step", "best"))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def batch(dataset, index):
        return {key: value.cuda() for key, value in dataset[index].items()}

    def validate():
        model.eval()
        weighted, tokens = 0., 0
        with torch.no_grad():
            for i in val_order:
                inputs = batch(val_set, i)
                count = inputs["target_ids"].numel()
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**inputs)
                weighted += float(loss) * count
                tokens += count
        model.train()
        return weighted / tokens

    def checkpoint_state():
        writing = folder / "checkpoint.writing"
        # Recoverable swap; a prior complete checkpoint stays available until serialization succeeds.
        if writing.exists():
            import shutil
            if not writing.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Checkpoint cleanup escaped run folder")
            shutil.rmtree(writing)
        model.save_weights(writing)
        torch.save({"identity_sha256": json_digest(identity), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "cursor": cursor, "step": step, "best": best, "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, writing / "state.pt")
        previous = folder / "checkpoint.previous"
        if previous.exists():
            import shutil
            if not previous.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Checkpoint cleanup escaped run folder")
            shutil.rmtree(previous)
        if latest.exists():
            latest.rename(previous)
        writing.rename(latest)

    while epoch < epochs and step < total_steps:
        permutation = np.random.default_rng(config["seed"] + epoch).permutation(order).tolist()
        while cursor < len(permutation) and step < total_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            indices = permutation[cursor:cursor + accumulation]
            losses = []
            for i in indices:
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**batch(train_set, i))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite projector loss; scientific run stopped")
                scaler.scale(loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite projector gradient; checkpoint retained for diagnosis")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            cursor += len(indices)
            step += 1
            observation = {"step": step, "total_steps": total_steps, "epoch": epoch + cursor / len(permutation),
                           "loss": sum(losses) / len(losses), "gradient_norm": float(norm),
                           "learning_rate": scheduler.get_last_lr()[0], "elapsed_sec": time.monotonic() - started,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if step % config["eval_steps"] == 0 or cursor == len(permutation) or technical:
                observation["validation_loss"] = validate()
                if observation["validation_loss"] < best:
                    best = observation["validation_loss"]
                    model.save_weights(folder / "best")
            if step % 25 == 0 or "validation_loss" in observation:
                checkpoint_state()
            with (folder / "training_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(observation, allow_nan=False) + "\n")
            save(folder / "progress.json", observation)
            print(json.dumps(observation), flush=True)
        epoch, cursor = epoch + 1, 0
    if technical:
        changed = [name for name, p in model.named_parameters() if p.requires_grad and not torch.equal(original[name], p.detach().cpu())]
        if not any(name.startswith("projector.") for name in changed):
            raise RuntimeError("Technical update did not change projector weights")
        if stage == "feedback" and not any("lora_" in name for name in changed):
            raise RuntimeError("Technical feedback update did not change LoRA weights")
        record["changed_trainable_tensors"] = changed
    record.update(status="completed", optimizer_steps=step, best_validation_loss=best,
                  best_weights=model_files(folder / "best"), parameter_counts=counts,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_sec=time.monotonic() - started)
    save(record_path, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=("alignment", "feedback"), required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    train(read_json(args.config), args.stage, args.technical_smoke)
