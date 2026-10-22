"""Run clean A/B/Q training and paired evaluation sequentially on one GPU."""
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
from filelock import FileLock
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_cascade_data import CONFIG, check_common_split, save, prepare
from scripts.prepare_feedback_data import digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def commands(config, phase, config_path=CONFIG):
    run_root, reports = Path(config["run_root"]), Path(config["report_root"])
    result = []
    if phase in {"train", "all"}:
        for stage in ("A", "B"):
            folder = rooted(run_root / stage)
            wanted = json.loads(rooted(config["audio_configs"][stage]).read_text(encoding="utf-8"))
            if (folder / "completed.json").is_file():
                metadata = json.loads((folder / "best_checkpoint/metadata.json").read_text(encoding="utf-8"))
                if any(metadata["effective_config"].get(key) != value for key, value in wanted.items()):
                    raise ValueError("Existing completed audio run has changed settings: " + stage)
                if metadata["provenance"]["files"]["protocol"]["sha256"] != digest(rooted(config["protocol_dir"]) / "protocol.json"):
                    raise ValueError("Existing audio checkpoint has a different protocol")
                continue
            args = ["scripts/train_mdd.py", "--config", config["audio_configs"][stage], "--runs-root", str(run_root), "--run-id", stage]
            if folder.exists():
                if not (folder / "checkpoint/metadata.json").is_file():
                    raise ValueError("Incomplete run without a resumable checkpoint: " + str(folder))
                args += ["--resume", str(run_root / stage / "checkpoint")]
            elif stage == "B":
                args += ["--init-checkpoint", str(run_root / "A/best_checkpoint")]
            result.append(("train_" + stage, args))
        result.append(("train_Q", ["scripts/train_feedback_model.py", "--config", config["feedback_config"], "--resume"]))
    if phase in {"evaluate", "all"}:
        for stage in ("A", "B"):
            destination = rooted(reports / "audio" / stage)
            if (destination / "metrics.json").is_file():
                prior = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                if digest(destination / "predictions.jsonl") != prior["prediction_sha256"]:
                    raise ValueError("Cached audio predictions changed")
                if prior["checkpoint_sha256"] != digest(rooted(run_root / stage / "best_checkpoint/model.safetensors")):
                    raise ValueError("Cached audio evaluation belongs to different weights")
                continue
            if destination.exists():
                archive = destination.with_name(destination.name + ".incomplete-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                if not destination.resolve().is_relative_to(rooted(reports).resolve()) or not archive.resolve().is_relative_to(rooted(reports).resolve()):
                    raise ValueError("Incomplete-result archive must stay inside this experiment")
                destination.rename(archive)
            result.append(("evaluate_audio_" + stage, ["scripts/evaluate_mdd.py", "--checkpoint", str(run_root / stage / "best_checkpoint"),
                "--split", "cascade_test", "--output-dir", str(destination)]))
        for condition in ("oracle", "A", "B"):
            for qwen in ("Q0", "Q1"):
                case = condition + "_" + qwen
                args = ["scripts/evaluate_published_feedback.py", "--config", config["feedback_config"],
                    "--eval-config", config["feedback_eval_config"], "--split", "test",
                    "--output", str(reports / "feedback" / case), "--resume"]
                if qwen == "Q1":
                    args += ["--adapter", str(run_root / "Q/adapter")]
                if condition != "oracle":
                    args += ["--mdd-evaluation", str(reports / "audio" / condition)]
                result.append(("evaluate_" + case, args))
        result.append(("report", ["scripts/report_cascade.py", "--config", str(config_path)]))
    return result


def run(config, phase, config_path=CONFIG):
    check_common_split(config)
    output = rooted(config["report_root"])
    output.mkdir(parents=True, exist_ok=True)
    tracked = [config_path.resolve(), *[rooted(value) for value in config["audio_configs"].values()], rooted(config["feedback_config"]), rooted(config["feedback_eval_config"]),
               ROOT / "src/mdd/runtime.py", ROOT / "src/mdd/metrics.py", ROOT / "scripts/train_feedback_model.py", ROOT / "scripts/l2arctic_plus_data.py",
               ROOT / "scripts/cascade_bridge.py", ROOT / "scripts/evaluate_published_feedback.py", ROOT / "scripts/published_feedback_metrics.py",
               ROOT / "scripts/report_cascade.py", ROOT / "scripts/run_cascade_training.py", ROOT / "scripts/prepare_cascade_data.py",
               ROOT / "prompts/pronunciation_feedback_published.txt"]
    feedback_settings = json.loads(rooted(config["feedback_config"]).read_text(encoding="utf-8"))
    tracked += [rooted(config["protocol_dir"]) / name for name in ("protocol.json", "splits.json", "manifest.jsonl")]
    tracked += [rooted(feedback_settings["dataset_dir"]) / name for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")]
    fingerprint = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    manifest_path = output / "pipeline_manifest.json"
    with FileLock(str(output / "pipeline.lock"), timeout=1), FileLock(str(ROOT / "data/generated/cascade_gpu.lock"), timeout=1):
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior["files"] != fingerprint:
                raise ValueError("Experiment code/config changed; keep this run and choose a new run/report root")
        else:
            save(manifest_path, {"files": fingerprint, "created_at_utc": utc(), "test_used_for_selection": False})
        queue = commands(config, phase, config_path)
        status_path = output / "pipeline_status.json"
        state = {"status": "running", "phase": phase, "started_at_utc": utc(), "pid": os.getpid(), "total_commands": len(queue)}
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONIOENCODING": "utf-8"}
        try:
            for number, (label, arguments) in enumerate(queue, 1):
                command = [sys.executable, "-X", "utf8", "-u", *arguments]
                log = output / (label + ".log")
                state.update(command_index=number, current_command=label, argv=command, log=str(log), updated_at_utc=utc())
                save(status_path, state)
                print(f"[{number}/{len(queue)}] {label} -> {log}", flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    state["child_pid"] = process.pid
                    try:
                        while process.poll() is None:
                            state["updated_at_utc"] = utc()
                            save(status_path, state)
                            time.sleep(2)
                    except BaseException:
                        process.terminate()
                        process.wait(timeout=20)
                        raise
                if process.returncode:
                    raise RuntimeError(f"{label} failed with exit {process.returncode}; see {log}")
            state.update(status="completed", finished_at_utc=utc(), child_pid=None)
            save(status_path, state)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), updated_at_utc=utc())
            save(status_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "all", "status", "report"), default="status")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.phase == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.phase == "status":
        path = rooted(config["report_root"]) / "pipeline_status.json"
        print(path.read_text(encoding="utf-8") if path.is_file() else "No full training run has started.")
    elif args.phase == "report":
        from scripts.report_cascade import report
        print(json.dumps(report(config), indent=2))
    else:
        run(config, args.phase, args.config)


if __name__ == "__main__":
    main()
