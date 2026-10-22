"""Keep the current Qwen child; remove automatic follow-up jobs at user request.

This changes orchestration only. It neither starts training nor edits model state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from filelock import FileLock
from scripts.projector_data import read_json, save
from scripts.prepare_feedback_data import digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked_process(pid, expected_scripts):
    process = psutil.Process(pid)
    if Path(process.cwd()).resolve() != ROOT.resolve():
        raise RuntimeError("Refusing to control a process outside this project")
    if not any(Path(argument).name in expected_scripts for argument in process.cmdline()):
        raise RuntimeError("Process command differs from the recorded training job")
    return process


def main():
    cascade_folder = ROOT / "data/generated/cascade-l2arctic-plus-seed42"
    projector_folder = ROOT / "data/generated/projector-l2arctic-plus-seed42"
    cascade_path = cascade_folder / "pipeline_status.json"
    projector_path = projector_folder / "pipeline_status.json"
    state = read_json(cascade_path)
    if state.get("status") != "running" or state.get("current_command") != "train_Q":
        raise RuntimeError("Expected the current cascade Qwen training job; nothing changed")
    launcher = checked_process(state["pid"], {"run_cascade_training.py", "resume_cascade_with_io_retry.py"})
    qwen = checked_process(state["child_pid"], {"train_feedback_model.py"})
    qwen_workers = [p for p in qwen.children(recursive=True) if "train_feedback_model.py" in " ".join(p.cmdline())]
    worker = qwen_workers[-1] if qwen_workers else qwen
    worker_created = worker.create_time()
    snapshot = {"at_utc": utc(), "reason": "User keeps current Qwen training; future train/evaluation starts are manual",
                "cascade_before": state, "monitor_script_sha256": digest(Path(__file__)),
                "qwen_compute_pid": worker.pid, "qwen_compute_created": worker_created}
    if projector_path.exists():
        projector_state = read_json(projector_path)
        snapshot["projector_before"] = projector_state
        if psutil.pid_exists(projector_state.get("pid", -1)):
            projector = checked_process(projector_state["pid"], {"run_projector_training.py"})
            if projector_state.get("current_command") not in ("cache_cpu", "wait_B", "wait_cascade_gpu"):
                raise RuntimeError("Projector already moved beyond preparation; refusing to interrupt model training")
            children = projector.children(recursive=True)
            for child in children:
                checked_process(child.pid, {"projector_data.py"})
            projector.terminate()  # Stop scheduling first, then stop CPU cache workers.
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs([projector, *children], timeout=10)
        projector_state.update(status="stopped_by_user", control_mode="manual", child_pid=None,
                               updated_at_utc=utc(), note="Cache files retained. No projector training has started.")
        save(projector_path, projector_state)
    save(cascade_folder / "manual_control.json", snapshot)
    launcher.terminate()  # Intentionally leave Qwen and its launcher untouched.
    launcher.wait(timeout=10)
    with FileLock(str(ROOT / "data/generated/cascade_gpu.lock"), timeout=15):
        state.update(pid=os.getpid(), previous_launcher_pid=launcher.pid, phase="current_training_only", total_commands=1,
                     command_index=1, control_mode="manual_no_followup", qwen_compute_pid=worker.pid,
                     note="Only the already running Qwen job continues. No automatic benchmark or projector job follows.")
        state.pop("error", None)
        while psutil.pid_exists(worker.pid):
            try:
                if psutil.Process(worker.pid).create_time() != worker_created:
                    break
            except psutil.NoSuchProcess:
                break
            state.update(updated_at_utc=utc())
            save(cascade_path, state)
            time.sleep(3)
        record_path = ROOT / "runs/cascade-l2arctic-plus-seed42/Q/training_provenance.json"
        record = read_json(record_path) if record_path.exists() else {}
        state.update(status="completed" if record.get("status") == "completed" else "failed", child_pid=None,
                     updated_at_utc=utc(), finished_at_utc=utc(), qwen_training_status=record.get("status"),
                     note="Current Qwen job ended. Further training and evaluation require user commands.")
        save(cascade_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
