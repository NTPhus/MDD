"""Evaluation only: frozen checkpoints -> audio test -> projector/cascade -> report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from scripts.projector_data import read_json, save
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest

CONFIG = ROOT / "configs/projector_validation.json"


def commands(settings, phase):
    config = read_json(rooted(settings["projector_config"]))
    cascade = read_json(rooted(config["cascade_config"]))
    queue = []
    if phase in ("all", "audio"):
        for stage in ("A", "B"):
            folder = rooted(cascade["report_root"]) / "audio" / stage
            weights = rooted(cascade["run_root"]) / stage / "best_checkpoint"
            if (folder / "metrics.json").exists():
                prior = read_json(folder / "metrics.json")
                if (prior["utterance_count"] != 900 or prior["split"] != "cascade_test"
                        or digest(folder / "predictions.jsonl") != prior["prediction_sha256"]
                        or digest(weights / "model.safetensors") != prior["checkpoint_sha256"]):
                    raise ValueError("Invalid cached full audio evaluation: " + stage)
                continue
            if folder.exists():
                # Keep every partial result, then let the non-resumable audio CLI restart.
                archive = folder.with_name(stage + ".incomplete-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                if not folder.resolve().is_relative_to(rooted(cascade["report_root"]).resolve()):
                    raise ValueError("Archive escapes experiment root")
                folder.rename(archive)
            queue.append(("audio_" + stage, ["scripts/evaluate_mdd.py", "--checkpoint", str(weights),
                         "--split", "cascade_test", "--output-dir", str(folder)]))
    if phase in ("all", "benchmark"):
        completed = rooted(settings["output_dir"]) / "execution.json"
        if not completed.exists() or read_json(completed).get("status") != "completed":
            queue.append(("checkpoint_benchmark", ["scripts/evaluate_projector_checkpoint.py", "--config", str(CONFIG)]))
    if phase in ("all", "benchmark", "transfer"):
        completed = rooted(settings["output_dir"]) / "transfer_execution.json"
        if not completed.exists() or read_json(completed).get("status") != "completed":
            queue.append(("adapter_transfer", ["scripts/evaluate_projector_adapter_transfer.py", "--config", str(CONFIG)]))
    if phase in ("all", "benchmark", "transfer", "report"):
        queue.append(("report", ["scripts/report_projector_validation.py", "--config", str(CONFIG)]))
    return queue


def run(settings, phase):
    output = rooted(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ROOT / "data/generated/cascade_gpu.lock"), timeout=1), FileLock(str(output / ".pipeline.lock"), timeout=1):
        state = {"status": "running", "phase": phase, "pid": os.getpid(), "training_started": False}
        save(output / "pipeline_status.json", state)
        try:
            for label, arguments in commands(settings, phase):
                # This runner deliberately has no training command or optimizer step.
                if any("train_" in Path(arg).name for arg in arguments[:1]):
                    raise ValueError("Training command forbidden")
                command = [sys.executable, "-X", "utf8", "-u", *arguments]
                state.update(current_command=label, argv=command)
                save(output / "pipeline_status.json", state)
                print(label, flush=True)
                with (output / (label + ".log")).open("a", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
                             "TOKENIZERS_PARALLELISM": "false"},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if result.returncode:
                    raise RuntimeError(f"{label} failed; see {output / (label + '.log')}")
            state.update(status="completed", current_command=None)
            save(output / "pipeline_status.json", state)
        except BaseException as error:
            state.update(status="failed", error=str(error))
            save(output / "pipeline_status.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--phase", choices=("all", "audio", "benchmark", "transfer", "report", "status"), default="status")
    args = parser.parse_args()
    settings = read_json(args.config)
    if args.config.resolve() != CONFIG.resolve():
        CONFIG = args.config.resolve()
    if args.phase == "status":
        path = rooted(settings["output_dir"]) / "pipeline_status.json"
        print(json.dumps(read_json(path) if path.exists() else {"status": "not_started"}, indent=2))
    else:
        run(settings, args.phase)
