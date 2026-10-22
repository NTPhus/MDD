"""Queue the separate projector branch without competing with cascade for VRAM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filelock import FileLock, Timeout
from scripts.projector_data import CONFIG, prepare, read_json, encoder_identity, save
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def pipeline(config, phase, config_path):
    output = rooted(config["report_root"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    if phase == "status":
        print(status_path.read_text(encoding="utf-8") if status_path.exists() else "Not started")
        return
    if phase == "prepare":
        print(json.dumps(prepare(config), indent=2))
        return
    with FileLock(str(output / "pipeline.lock"), timeout=1):
        dataset = prepare(config)
        files = [CONFIG if config_path is None else config_path, ROOT / "prompts/pronunciation_feedback_projector.txt"]
        files += [ROOT / "scripts" / name for name in ("projector_data.py", "projector_model.py", "train_projector.py", "evaluate_projector.py", "report_projector.py", "run_projector_training.py")]
        files += [ROOT / "scripts" / name for name in ("l2arctic_plus_data.py", "prepare_cascade_data.py", "prepare_feedback_data.py", "published_feedback_metrics.py")]
        files += [ROOT / "src/mdd/runtime.py", ROOT / "src/mdd/metrics.py"]
        frozen = {"config": config, "dataset": dataset, "files": {str(p.relative_to(ROOT)): digest(p) for p in files}}
        manifest_path = output / "pipeline_manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != frozen:
            raise ValueError("Projector recipe changed; select new output/run roots")
        save(manifest_path, frozen)
        state = {"status": "starting", "pid": os.getpid(), "started_at_utc": utc(), "current_command": None}

        def status(value, **extra):
            state.update(status=value, updated_at_utc=utc(), **extra)
            save(status_path, state)

        def verify():
            for name, expected in frozen["files"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError("Frozen projector code/config changed: " + name)

        def command(label, script, *arguments):
            verify()
            argv = [sys.executable, "-X", "utf8", "-u", "scripts/" + script, "--config", str(config_path or CONFIG), *arguments]
            path = output / (label + ".log")
            status("running", current_command=label, argv=argv, log=str(path))
            with path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                status("running", child_pid=process.pid)
                try:
                    while process.poll() is None:
                        time.sleep(2)
                        status("running")
                except BaseException:
                    process.terminate()
                    process.wait(timeout=20)
                    raise
                if process.returncode:
                    raise RuntimeError(f"{label} failed (exit {process.returncode}); see {path}")

        try:
            if phase == "report":
                command("report", "report_projector.py")
                status("report_ready")
                return
            completed = rooted(config["encoder_checkpoint"]).parent / "completed.json"
            cascade = read_json(rooted(config["cascade_config"]))
            cascade_status = rooted(cascade["report_root"]) / "pipeline_status.json"
            while not completed.exists():
                status("waiting_for_encoder_B", current_command="wait_B")
                if cascade_status.exists() and read_json(cascade_status).get("status") in ("failed", "completed"):
                    raise RuntimeError("Encoder B not completed; inspect cascade status before resuming")
                time.sleep(15)
            encoder_identity(config)
            command("cache_cpu", "projector_data.py", "--cache")
            gpu_lock = FileLock(str(ROOT / "data/generated/cascade_gpu.lock"))
            while True:
                status("waiting_for_gpu", current_command="wait_cascade_gpu")
                try:
                    gpu_lock.acquire(timeout=10)
                    break
                except Timeout:
                    continue
            try:
                command("technical_alignment", "train_projector.py", "--stage", "alignment", "--technical-smoke")
                command("alignment", "train_projector.py", "--stage", "alignment")
                command("technical_feedback", "train_projector.py", "--stage", "feedback", "--technical-smoke")
                command("feedback", "train_projector.py", "--stage", "feedback")
                command("evaluate_P0", "evaluate_projector.py", "--case", "P0")
                command("evaluate_P1", "evaluate_projector.py", "--case", "P1")
                command("report", "report_projector.py")
            finally:
                gpu_lock.release()
            status("completed", current_command=None, finished_at_utc=utc())
        except BaseException as error:
            status("failed", error_type=type(error).__name__, error=str(error)[:1000])
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--phase", choices=("prepare", "all", "status", "report"), default="all")
    args = parser.parse_args()
    pipeline(read_json(args.config), args.phase, args.config.resolve())
