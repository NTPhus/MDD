"""Local-files-only wav2vec2 phone-CTC training and explicit held-out scoring.

The model consumes audio only. Canonical and gold perceived phones are used by
the scorer after greedy decoding; they never condition model inference.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from mdd.metrics import SCORING_PROTOCOL, aggregate_scores, score_utterance


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "stage": "A", "purpose": "scientific_training", "seed": 42,
    "model_path": "models/pretrained/wav2vec2-base",
    "annotations": "data/processed/l2arctic-v1/annotations.jsonl",
    "vocab": "data/processed/l2arctic-v1/vocab.json",
    "normalization": "data/processed/l2arctic-v1/normalization.json",
    "protocol_dir": "data/protocol/l2arctic-v1", "device": "cuda",
    "mixed_precision": "fp16", "cpu_threads": 4, "batch_size": 1,
    "gradient_accumulation_steps": 4, "max_steps": 2000,
    "eval_every": 250, "checkpoint_every": 250,
    "backbone_learning_rate": 3e-5, "head_learning_rate": 3e-4,
    "weight_decay": 0.01, "max_grad_norm": 1.0,
    "freeze_feature_encoder": True, "freeze_backbone": False,
    "gradient_checkpointing": True, "source_replay_fraction": 0.5,
    "fold": 1, "max_train_samples": None, "max_eval_samples": None,
    "max_audio_seconds": None, "selection_metric": "f1",
    "save_optimizer": True,
}


def path_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config(config: dict[str, Any]) -> None:
    if config["stage"] not in {"A", "B", "C"}:
        raise ValueError("stage must be A, B, or C")
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "checkpoint_every", "cpu_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda; no implicit device fallback")
    if not 0 <= config["source_replay_fraction"] <= 1:
        raise ValueError("source_replay_fraction must be between zero and one")
    if config["selection_metric"] not in {"f1", "per"}:
        raise ValueError("selection_metric must be f1 or per")
    if config["fold"] not in {1, 2, 3, 4}:
        raise ValueError("fold must be 1, 2, 3, or 4")
    for key in ("max_train_samples", "max_eval_samples", "max_audio_seconds"):
        if config[key] is not None and config[key] <= 0:
            raise ValueError(f"{key} must be positive when specified")
    if config["head_learning_rate"] <= 0 or config["backbone_learning_rate"] <= 0:
        raise ValueError("learning rates must be positive")


def configure_device(config: dict[str, Any]) -> torch.device:
    torch.set_num_threads(config["cpu_threads"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Train on a Colab GPU runtime; --device cpu is an explicit override.")
    if config["device"] == "cuda" and config["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; select fp16 instead.")
    return torch.device(config["device"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def environment_report(config: dict[str, Any]) -> dict[str, Any]:
    names = ("torch", "transformers", "numpy", "scipy", "soundfile", "safetensors")
    return {
        "created_at": utc_now(), "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "device": config["device"], "cpu_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
        "mixed_precision_effective": config["mixed_precision"] if config["device"] == "cuda" else "no",
        "determinism": "Fixed seeds and deterministic cuDNN; bitwise equality across hardware is not guaranteed.",
    }


def load_records(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, Any]]:
    annotations = path_from_root(config["annotations"])
    records = {}
    with annotations.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["utterance_id"] in records:
                raise ValueError(f"Duplicate annotation ID: {record['utterance_id']}")
            records[record["utterance_id"]] = record
    vocab = read_json(path_from_root(config["vocab"]))
    if sorted(vocab.values()) != list(range(len(vocab))) or vocab.get("[PAD]") != 0 or "[UNK]" not in vocab:
        raise ValueError("Vocabulary needs contiguous IDs and [PAD]=0 (CTC blank), plus [UNK].")
    protocol_dir = path_from_root(config["protocol_dir"])
    protocol = read_json(protocol_dir / "protocol.json")
    for name in ("manifest", "splits"):
        suffix = ".jsonl" if name == "manifest" else ".json"
        if sha256(protocol_dir / (name + suffix)) != protocol[name + "_sha256"]:
            raise ValueError(f"Frozen {name} checksum does not match protocol.json")
    manifest = {item["utterance_id"]: item for item in
                (json.loads(line) for line in (protocol_dir / "manifest.jsonl").read_text().splitlines())}
    if records.keys() != manifest.keys():
        raise ValueError("Prepared and frozen manifest utterance IDs differ")
    for utterance_id, record in records.items():
        original = manifest[utterance_id]
        if record["audio_path"] != original["audio"]["path"] or record["speaker_id"] != original["speaker_id"]:
            raise ValueError(f"Prepared audio/speaker differs from manifest: {utterance_id}")
        record["audio_sha256"] = original["audio"]["sha256"]
    splits = read_json(protocol_dir / "splits.json")
    return records, vocab, splits


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    protocol_dir = path_from_root(config["protocol_dir"])
    paths = {
        "annotations": path_from_root(config["annotations"]),
        "vocab": path_from_root(config["vocab"]),
        "normalization": path_from_root(config["normalization"]),
        "manifest": protocol_dir / "manifest.jsonl", "splits": protocol_dir / "splits.json",
        "protocol": protocol_dir / "protocol.json", "scorer": Path(__file__).with_name("metrics.py"),
        "runtime": Path(__file__), "training_cli": ROOT / "scripts/train_mdd.py",
        "evaluation_cli": ROOT / "scripts/evaluate_mdd.py",
    }
    base_dir = path_from_root(config["model_path"])
    for name in ("download_manifest.json", "download_revision.txt", "config.json"):
        if (base_dir / name).is_file():
            paths["base_model_" + name.replace(".", "_")] = base_dir / name
    return {"scoring_protocol": SCORING_PROTOCOL,
            "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}}


def select_records(records: dict[str, dict[str, Any]], ids: list[str], *, limit: int | None = None,
                   max_seconds: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing = sorted(set(ids) - records.keys())
    if missing:
        raise ValueError(f"Prepared annotations missing frozen IDs: {missing[:5]}")
    chosen, excluded = [], []
    for utterance_id in ids:
        row = records[utterance_id]
        reason = None
        if not row.get("usable_for_training", True) or not row["perceived_phones"]:
            reason = "empty_or_unusable_target"
        elif max_seconds is not None and row["duration_sec"] > max_seconds:
            reason = "duration_filter_no_audio_truncation"
        if reason:
            excluded.append({"utterance_id": utterance_id, "reason": reason})
        else:
            chosen.append(row)
    if limit is not None:
        # Technical subsets are explicitly shortest-first; no scientific claims from these subsets.
        chosen.sort(key=lambda item: (item["duration_sec"], item["utterance_id"]))
        excluded.extend({"utterance_id": row["utterance_id"], "reason": "sample_limit"} for row in chosen[limit:])
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("Selection produced no usable utterances")
    return chosen, {"requested_count": len(ids), "selected_count": len(chosen),
                    "selected_ids": [row["utterance_id"] for row in chosen], "excluded": excluded}


def training_selections(config: dict[str, Any], records: dict, splits: dict) -> tuple[list, list, list, dict]:
    selection_args = {"limit": config["max_train_samples"], "max_seconds": config["max_audio_seconds"]}
    source, source_audit = select_records(records, splits["source_train_ids"], **selection_args)
    fold = next(item for item in splits["folds"] if item["fold"] == config["fold"])
    target, target_audit = ([], None)
    if config["stage"] == "B":
        target, target_audit = select_records(records, fold["train_ids"], **selection_args)
    validation_ids = splits["source_validation_ids"] if config["stage"] == "A" else fold["validation_ids"]
    validation, val_audit = select_records(records, validation_ids, limit=config["max_eval_samples"],
                                         max_seconds=config["max_audio_seconds"])
    train_ids = {row["utterance_id"] for row in source + target}
    if train_ids & {row["utterance_id"] for row in validation}:
        raise ValueError("Train/validation overlap")
    if {row["speaker_id"] for row in source + target} & {row["speaker_id"] for row in validation}:
        raise ValueError("Train/validation speaker overlap")
    return source, target, validation, {"source_train": source_audit, "target_train": target_audit,
                                      "validation": val_audit, "test_used_for_training": False}


def load_audio(row: dict[str, Any]) -> np.ndarray:
    audio_path = path_from_root(row["audio_path"])
    if row.get("audio_sha256") and sha256(audio_path) != row["audio_sha256"]:
        raise ValueError(f"Audio no longer matches frozen manifest: {row['utterance_id']}")
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Empty/nonfinite audio: {row['utterance_id']}")
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    return waveform


def ctc_required_frames(labels: list[int]) -> int:
    return len(labels) + sum(left == right for left, right in zip(labels, labels[1:]))


class PhoneCollator:
    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, vocab: dict[str, int], model: Any):
        self.feature_extractor, self.vocab, self.model = feature_extractor, vocab, model

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [load_audio(row) for row in rows]
        # Never crop audio while retaining a full-utterance phone target.
        encoded = self.feature_extractor(audio, sampling_rate=16000, padding=True,
                                         return_attention_mask=True, return_tensors="pt")
        labels = [[self.vocab.get(phone, self.vocab["[UNK]"]) for phone in row["perceived_phones"]] for row in rows]
        for row, waveform, target in zip(rows, audio, labels):
            frames = int(self.model._get_feat_extract_output_lengths(len(waveform)))
            if not target or frames < ctc_required_frames(target):
                raise ValueError(f"CTC-infeasible sample {row['utterance_id']}: frames={frames}, required={ctc_required_frames(target)}")
        encoded["labels"] = torch.full((len(labels), max(map(len, labels))), -100, dtype=torch.long)
        for index, target in enumerate(labels):
            encoded["labels"][index, :len(target)] = torch.tensor(target, dtype=torch.long)
        return dict(encoded)


def load_model(config: dict[str, Any], vocab: dict[str, int], checkpoint: Path | None = None) -> tuple[Any, Any]:
    location = checkpoint or path_from_root(config["model_path"])
    if checkpoint:
        stored_vocab = read_json(checkpoint / "vocab.json")
        if stored_vocab != vocab:
            raise ValueError("Checkpoint phone vocabulary differs from the frozen vocabulary")
    model = Wav2Vec2ForCTC.from_pretrained(
        str(location), local_files_only=True, vocab_size=len(vocab), pad_token_id=vocab["[PAD]"],
        ctc_loss_reduction="mean", ctc_zero_infinity=False, bos_token_id=None, eos_token_id=None,
        id2label={index: phone for phone, index in vocab.items()}, label2id=vocab,
        attn_implementation="eager",
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(location), local_files_only=True)
    if config["freeze_backbone"]:
        model.wav2vec2.requires_grad_(False)
    elif config["freeze_feature_encoder"]:
        model.freeze_feature_encoder()
    if config["gradient_checkpointing"] and not config["freeze_backbone"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    return model, extractor


def autocast_context(config: dict[str, Any]):
    if config["device"] != "cuda" or config["mixed_precision"] == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if config["mixed_precision"] == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def greedy_decode(token_ids: list[int], id_to_phone: dict[int, str], blank: int = 0) -> list[str]:
    return [id_to_phone[token] for token, _ in itertools.groupby(token_ids) if token != blank]


def evaluate_model(model: Any, collator: PhoneCollator, rows: list[dict[str, Any]], vocab: dict[str, int],
                   config: dict[str, Any], device: torch.device, output_path: Path) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    scores, losses = [], []
    per_speaker: dict[str, list] = {}
    id_to_phone = {index: phone for phone, index in vocab.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for row in rows:
            sample_started = time.monotonic()
            batch = {key: tensor.to(device) for key, tensor in collator([row]).items()}
            with autocast_context(config):
                result = model(**batch)
            if not torch.isfinite(result.loss):
                raise RuntimeError(f"Nonfinite evaluation loss on {row['utterance_id']}")
            predicted = greedy_decode(result.logits[0].argmax(dim=-1).tolist(), id_to_phone, vocab["[PAD]"])
            score = score_utterance(row["canonical_phones"], row["perceived_phones"], predicted,
                                    gold_error_events=row.get("gold_error_events"),
                                    canonical_word_indices=row.get("canonical_word_indices"))
            scores.append(score)
            per_speaker.setdefault(row["speaker_id"], []).append(score)
            losses.append(float(result.loss))
            audio_hash = row.get("audio_sha256")
            if not audio_hash:
                audio_hash = sha256(path_from_root(row["audio_path"]))
            prediction = {"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"],
                          "audio_path": row["audio_path"], "audio_sha256": audio_hash,
                          "status": "ok", "duration_sec": row["duration_sec"],
                          "inference_seconds": time.monotonic() - sample_started,
                          "canonical_phones": row["canonical_phones"], "perceived_phones": row["perceived_phones"],
                          "canonical_word_indices": row.get("canonical_word_indices"),
                          "predicted_phones": predicted, "ctc_loss": losses[-1],
                          "counts": score["counts"], "metrics": score["metrics"],
                          "gold_events": score["gold_events"], "predicted_events": score["predicted_events"]}
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
    aggregate = aggregate_scores(scores)
    aggregate["by_speaker"] = {speaker: aggregate_scores(items) for speaker, items in sorted(per_speaker.items())}
    speaker_f1 = [item["metrics"]["f1"] for item in aggregate["by_speaker"].values() if item["metrics"]["f1"] is not None]
    aggregate["speaker_macro_f1"] = sum(speaker_f1) / len(speaker_f1) if speaker_f1 else None
    aggregate["mean_ctc_loss"] = sum(losses) / len(losses)
    aggregate["elapsed_seconds"] = time.monotonic() - started
    aggregate["utterance_count"] = len(rows)
    aggregate["audio_duration_seconds"] = sum(row["duration_sec"] for row in rows)
    aggregate["prediction_sha256"] = sha256(output_path)
    aggregate["scoring_protocol"] = SCORING_PROTOCOL
    model.train(was_training)
    return aggregate


def optimizer_for(model: Any, config: dict[str, Any]) -> torch.optim.Optimizer:
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (head if name.startswith("lm_head.") else backbone).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": config["backbone_learning_rate"], "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": config["head_learning_rate"], "name": "phone_head"})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def save_checkpoint(path: Path, model: Any, extractor: Any, vocab: dict, metadata: dict,
                    training_state: dict | None = None) -> None:
    temporary = path.with_name(path.name + ".writing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    extractor.save_pretrained(temporary)
    write_json(temporary / "vocab.json", vocab)
    write_json(temporary / "metadata.json", metadata)
    normalization = path_from_root(metadata["effective_config"]["normalization"])
    if normalization.is_file():
        shutil.copy2(normalization, temporary / "normalization.json")
    if training_state is not None:
        torch.save(training_state, temporary / "training_state.pt")
    # Keep the previous checkpoint until the next complete copy is on disk.
    old = path.with_name(path.name + ".previous")
    if old.exists():
        shutil.rmtree(old)
    if path.exists():
        path.rename(old)
    temporary.rename(path)
    if old.exists():
        shutil.rmtree(old)


def checkpoint_metadata(config: dict, run_id: str, step: int, best_value: float | None,
                        hashes: dict, initialized_from: str | None,
                        initialization: dict | None = None) -> dict:
    return {"run_id": run_id, "stage": config["stage"], "purpose": config["purpose"],
            "step": step, "best_validation_value": best_value,
            "selection_metric": config["selection_metric"], "effective_config": config,
            "initialized_from": initialized_from, "provenance": hashes,
            "initialization": initialization,
            "created_at": utc_now(), "decoder": "greedy_ctc", "sample_rate": 16000}


def resolve_checkpoint(path: Path) -> Path:
    """Recover the previous complete copy after a disconnect during directory renames."""
    if path.is_dir():
        return path
    previous = path.with_name(path.name + ".previous")
    if previous.is_dir() and (previous / "metadata.json").is_file():
        print(f"Recovering previous complete checkpoint: {previous}", flush=True)
        return previous
    raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")


def archive_uncheckpointed_logs(run_dir: Path, step: int) -> None:
    """Retain stale evidence separately so the resumed main log has unique steps."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    for name in ("training_log.jsonl", "validation_log.jsonl"):
        path = run_dir / name
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        stale = [row for row in rows if row["step"] > step]
        if stale:
            archived = path.with_name(path.stem + ".uncheckpointed-" + stamp + ".jsonl")
            archived.write_text("".join(json.dumps(row) + "\n" for row in stale))
            path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step))
    for suffix in ("", "_best"):
        metrics = run_dir / f"metrics_validation{suffix}.json"
        if metrics.is_file() and read_json(metrics)["step"] > step:
            for path in (metrics, run_dir / f"predictions_validation{suffix}.jsonl"):
                if path.is_file():
                    path.rename(path.with_name(path.name + ".uncheckpointed-" + stamp))


def run_train(args: argparse.Namespace) -> Path:
    supplied = read_json(path_from_root(args.config))
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = dict(DEFAULTS, **supplied)
    for key in ("device", "max_steps", "fold", "max_train_samples", "max_eval_samples", "max_audio_seconds", "purpose", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if any(config[key] is not None for key in ("max_train_samples", "max_eval_samples", "max_audio_seconds")):
        config["purpose"] = "technical_smoke_only"
    validate_config(config)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use --resume for continuation OR --init-checkpoint for new adaptation, not both")
    if config["stage"] in {"B", "C"} and not (args.init_checkpoint or args.resume):
        raise ValueError("Stages B/C require --init-checkpoint from the same trained stage-A checkpoint")
    run_dir = path_from_root(args.runs_root) / args.run_id
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    resume = resolve_checkpoint(path_from_root(args.resume)) if args.resume else None
    if resume and resume.parent.resolve() != run_dir.resolve():
        raise ValueError("--resume checkpoint must belong to the same --runs-root/--run-id directory")
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists; choose a new run ID or --resume: {run_dir}")
    device = configure_device(config)
    seed_everything(config["seed"])
    records, vocab, splits = load_records(config)
    source, target, validation, selection = training_selections(config, records, splits)
    hashes = provenance(config)
    initialization = resume or (resolve_checkpoint(path_from_root(args.init_checkpoint)) if args.init_checkpoint else None)
    if initialization:
        initial_metadata = read_json(initialization / "metadata.json")
        if not resume and initial_metadata["stage"] != "A":
            raise ValueError("B/C must initialize from a stage-A checkpoint")
        if not resume and config["stage"] in {"B", "C"} and config["purpose"] == "scientific_training":
            completed = initial_metadata.get("completed_training", {})
            if initial_metadata["purpose"] != "scientific_training" or completed.get("optimizer_steps") != initial_metadata["effective_config"]["max_steps"]:
                raise ValueError("Scientific B/C require a completed scientific stage-A run, not a smoke/incomplete checkpoint")
        for name in hashes["files"]:
            if initial_metadata["provenance"]["files"][name]["sha256"] != hashes["files"][name]["sha256"]:
                raise ValueError(f"Checkpoint provenance mismatch: {name}")
    model, extractor = load_model(config, vocab, initialization)
    model.to(device)
    collator = PhoneCollator(extractor, vocab, model)
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config["device"] == "cuda" and config["mixed_precision"] == "fp16")
    sampler_rng = random.Random(config["seed"] + 1009)
    step, best_value = 0, None
    seen = {"source": 0, "target": 0}
    seen_audio_seconds = {"source": 0.0, "target": 0.0}
    if resume:
        # Only load state files created locally by this project. Never load an untrusted pickle.
        state_path = resume / "training_state.pt"
        if not state_path.is_file():
            raise ValueError("Resume requires checkpoint/training_state.pt; best_checkpoint contains weights only")
        previous = initial_metadata["effective_config"]
        mutable = {"max_steps", "cpu_threads"}
        if any(previous[key] != config[key] for key in DEFAULTS if key not in mutable):
            raise ValueError("Resume configuration differs from saved training configuration")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        scaler.load_state_dict(state["scaler"])
        sampler_rng.setstate(state["sampler_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, best_value, seen = state["step"], state["best_value"], state["seen_samples"]
        seen_audio_seconds = state["seen_audio_seconds"]
        if config["max_steps"] <= step:
            raise ValueError(f"Checkpoint is already at step {step}; increase --max-steps to continue")
        best_metadata_path = run_dir / "best_checkpoint/metadata.json"
        if best_metadata_path.is_file() and read_json(best_metadata_path)["step"] > step:
            raise RuntimeError("Best checkpoint is newer than the recovered training state. Preserve the run and recover its matching checkpoint before resuming.")
        archive_uncheckpointed_logs(run_dir, step)
        if (run_dir / "completed.json").is_file():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            (run_dir / "completed.json").rename(run_dir / f"completed.before-resume-{stamp}.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_report(config))
    write_json(run_dir / "protocol_hashes.json", hashes)
    write_json(run_dir / "data_selection.json", selection)
    write_json(run_dir / "parameter_counts.json", {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    if args.init_checkpoint:
        weight_path = initialization / "model.safetensors"
        write_json(run_dir / "initialization.json", {
            "checkpoint": str(initialization), "weight_sha256": sha256(weight_path),
            "mode": "weights_only_new_optimizer", "source_metadata": initial_metadata,
        })
    elif not resume:
        base_model = path_from_root(config["model_path"])
        weights = next((base_model / name for name in ("model.safetensors", "pytorch_model.bin")
                        if (base_model / name).is_file()), None)
        write_json(run_dir / "initialization.json", {
            "checkpoint": str(base_model), "weight_sha256": sha256(weights) if weights else None,
            "mode": "pretrained_backbone_new_random_phone_head", "random_seed": config["seed"],
            "download_revision": ((base_model / "download_revision.txt").read_text().strip()
                                  if (base_model / "download_revision.txt").is_file() else None),
        })
    initialization_info = read_json(run_dir / "initialization.json")
    origin_checkpoint = initial_metadata.get("initialized_from") if resume else args.init_checkpoint
    started = time.monotonic()
    model.train()
    consecutive_overflows = 0
    while step < config["max_steps"]:
        # A skipped FP16 attempt is retried on the same samples and RNG draws,
        # so counts and A/B/C budgets measure successful optimizer updates.
        attempt_sampler_state = sampler_rng.getstate()
        attempt_python_state = random.getstate()
        attempt_numpy_state = np.random.get_state()
        attempt_torch_state = torch.get_rng_state()
        attempt_cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        previous_seen, previous_audio = dict(seen), dict(seen_audio_seconds)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(config["gradient_accumulation_steps"]):
            batch_rows = []
            for _ in range(config["batch_size"]):
                choose_source = not target or sampler_rng.random() < config["source_replay_fraction"]
                pool, name = (source, "source") if choose_source else (target, "target")
                batch_rows.append(sampler_rng.choice(pool))
                seen[name] += 1
                seen_audio_seconds[name] += batch_rows[-1]["duration_sec"]
            batch = {key: tensor.to(device) for key, tensor in collator(batch_rows).items()}
            with autocast_context(config):
                output = model(**batch)
                loss = output.loss / config["gradient_accumulation_steps"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite train loss at step {step + 1}")
            running_loss += float(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"],
                                                       error_if_nonfinite=not scaler.is_enabled())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < old_scale:
            consecutive_overflows += 1
            overflow = {"optimizer_steps_completed": step, "attempted_loss": running_loss,
                        "old_scale": old_scale, "new_scale": scaler.get_scale(),
                        "consecutive_overflows": consecutive_overflows,
                        "attempted_samples": {name: seen[name] - previous_seen[name] for name in seen},
                        "created_at": utc_now()}
            with (run_dir / "amp_overflows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(overflow, allow_nan=False) + "\n")
            print(json.dumps({"amp_retry": overflow}), flush=True)
            sampler_rng.setstate(attempt_sampler_state)
            random.setstate(attempt_python_state)
            np.random.set_state(attempt_numpy_state)
            torch.set_rng_state(attempt_torch_state)
            if attempt_cuda_state is not None:
                torch.cuda.set_rng_state_all(attempt_cuda_state)
            seen, seen_audio_seconds = previous_seen, previous_audio
            if consecutive_overflows >= 20:
                raise RuntimeError("20 consecutive AMP overflows; select bf16 if supported or mixed_precision=no")
            continue
        consecutive_overflows = 0
        step += 1
        record = {"step": step, "loss": running_loss, "gradient_norm": float(gradient_norm),
                  "elapsed_seconds": time.monotonic() - started, "seen_samples": dict(seen),
                  "seen_audio_seconds": dict(seen_audio_seconds),
                  "lr": {group["name"]: group["lr"] for group in optimizer.param_groups}, "created_at": utc_now()}
        with (run_dir / "training_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        should_eval = step % config["eval_every"] == 0 or step == config["max_steps"]
        if should_eval:
            prediction_path = run_dir / "predictions_validation.jsonl"
            metrics = evaluate_model(model, collator, validation, vocab, config, device, prediction_path)
            metrics.update(step=step, purpose=config["purpose"], split="validation", fold=config["fold"])
            write_json(run_dir / "metrics_validation.json", metrics)
            value = metrics["metrics"][config["selection_metric"]]
            improved = value is not None and (best_value is None or
                       (value > best_value if config["selection_metric"] == "f1" else value < best_value))
            if improved or not (run_dir / "best_checkpoint").exists():
                best_value = value
                metadata = checkpoint_metadata(config, args.run_id, step, best_value, hashes, origin_checkpoint,
                                               initialization_info)
                save_checkpoint(run_dir / "best_checkpoint", model, extractor, vocab, metadata)
                shutil.copy2(prediction_path, run_dir / "predictions_validation_best.jsonl")
                write_json(run_dir / "metrics_validation_best.json", metrics)
            with (run_dir / "validation_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "metrics": metrics["metrics"], "best_value": best_value}, allow_nan=False) + "\n")
        if step % config["checkpoint_every"] == 0 or step == config["max_steps"]:
            metadata = checkpoint_metadata(config, args.run_id, step, best_value, hashes, origin_checkpoint,
                                           initialization_info)
            state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step,
                     "best_value": best_value, "seen_samples": seen, "sampler_rng": sampler_rng.getstate(),
                     "seen_audio_seconds": seen_audio_seconds,
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            save_checkpoint(run_dir / "checkpoint", model, extractor, vocab, metadata,
                            state if config["save_optimizer"] else None)
    completed = {"status": "completed", "optimizer_steps": step,
               "purpose": config["purpose"], "best_validation_value": best_value,
               "seen_samples": seen, "seen_audio_seconds": seen_audio_seconds,
               "elapsed_seconds_this_session": time.monotonic() - started,
               "test_evaluated": False, "created_at": utc_now()}
    write_json(run_dir / "completed.json", completed)
    for name in ("checkpoint", "best_checkpoint"):
        metadata_path = run_dir / name / "metadata.json"
        final_metadata = read_json(metadata_path)
        final_metadata["completed_training"] = completed
        final_metadata["effective_config"] = config
        write_json(metadata_path, final_metadata)
    print(f"Completed: {run_dir}", flush=True)
    return run_dir


def run_evaluate(args: argparse.Namespace) -> Path:
    checkpoint = resolve_checkpoint(path_from_root(args.checkpoint))
    metadata = read_json(checkpoint / "metadata.json")
    config = dict(DEFAULTS, **metadata["effective_config"])
    config["device"] = args.device or config["device"]
    config["gradient_checkpointing"] = False
    config["freeze_backbone"] = True
    fold_number = args.fold or config["fold"]
    if metadata["stage"] in {"B", "C"} and fold_number != config["fold"]:
        raise ValueError("B/C must be evaluated on their own fold; another fold can contain training/validation speakers")
    config["fold"] = fold_number
    validate_config(config)
    device = configure_device(config)
    records, vocab, splits = load_records(config)
    hashes = provenance(config)
    for name, value in hashes["files"].items():
        if value["sha256"] != metadata["provenance"]["files"][name]["sha256"]:
            raise ValueError(f"Checkpoint/evaluation provenance mismatch: {name}")
    fold = next(item for item in splits["folds"] if item["fold"] == fold_number)
    if args.split == "source_validation":
        ids = splits["source_validation_ids"]
    elif args.split in {"cascade_test", "source_test"}:
        key = args.split + "_ids"
        if key not in splits:
            raise ValueError("This protocol has no " + key)
        ids = splits[key]
    elif args.split == "test":
        ids = fold["test_ids"]
    else:
        ids = fold["validation_ids"]
    rows, selection = select_records(records, ids, limit=args.max_samples)
    destination = path_from_root(args.output_dir)
    if destination.exists():
        raise FileExistsError(f"Evaluation destination already exists: {destination}")
    destination.mkdir(parents=True)
    model, extractor = load_model(config, vocab, checkpoint)
    model.to(device)
    collator = PhoneCollator(extractor, vocab, model)
    metrics = evaluate_model(model, collator, rows, vocab, config, device, destination / "predictions.jsonl")
    metrics.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint / "model.safetensors"),
                   split=args.split, fold=fold_number, purpose=metadata["purpose"], model_step=metadata["step"],
                   stage=metadata["stage"], effective_config=metadata["effective_config"],
                   training_metadata=metadata,
                   selection=selection, protocol=hashes,
                   limited_technical_subset=args.max_samples is not None)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "environment.json", environment_report(config))
    print(json.dumps(metrics["metrics"], indent=2), flush=True)
    return destination


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train audio-only phone-CTC from local wav2vec2-base; no test evaluation")
    parser.add_argument("--config", default="configs/source.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--init-checkpoint", help="Stage-A weights for B/C; starts a new optimizer")
    parser.add_argument("--resume", help="Same-run checkpoint with trusted local optimizer/RNG state")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-audio-seconds", type=float, help="Filter whole utterances; never truncate audio")
    parser.add_argument("--purpose", choices=("scientific_training", "technical_smoke_only"))
    return parser


def eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit phone-CTC evaluation, including held-out test only when requested")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test", "source_validation", "cascade_test", "source_test"), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-samples", type=int, help="Technical subset only, never a full benchmark")
    return parser
