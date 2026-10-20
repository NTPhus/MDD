"""Inspect the local CPU training environment without network access or API keys."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "safetensors", "numpy", "scipy", "soundfile", "jsonschema", "psutil")


def report():
    missing = [name for name in PACKAGES if importlib.util.find_spec(name) is None]
    memory = {}
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[name + "_bytes"] = int(value.split()[0]) * 1024
    paths = {
        "model_config": ROOT / "models/pretrained/wav2vec2-base/config.json",
        "model_weights": ROOT / "models/pretrained/wav2vec2-base/pytorch_model.bin",
        "data_manifest": ROOT / "data/protocol/l2arctic-v1/manifest.jsonl",
        "speaker_splits": ROOT / "data/protocol/l2arctic-v1/splits.json",
        "prepared_annotations": ROOT / "data/processed/l2arctic-v1/annotations.jsonl",
        "phone_vocabulary": ROOT / "data/processed/l2arctic-v1/vocab.json",
        "normalization": ROOT / "data/processed/l2arctic-v1/normalization.json",
        "prepared_audit": ROOT / "data/processed/l2arctic-v1/audit.json",
    }
    result = {
        "purpose": "environment_probe_only",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "platform": platform.platform(), "cpu_logical_count": os.cpu_count(),
        "memory": memory, "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES if name not in missing},
        "missing_packages": missing,
        "model_optimizer_steps": 0,
        "training_executed": False,
        "required_inputs": {key: {"path": path.relative_to(ROOT).as_posix(), "exists": path.is_file()}
                            for key, path in paths.items()},
    }
    for key in ("data_manifest", "speaker_splits", "model_config"):
        path = paths[key]
        if path.is_file():
            result["required_inputs"][key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not missing:
        import numpy as np
        import scipy.signal
        import soundfile as sf
        import torch
        from transformers import Wav2Vec2ForCTC  # Verify the actual model import, including optional deps.

        torch.set_num_threads(4)
        probe = torch.tensor([1.0, 2.0], requires_grad=True)
        probe.square().sum().backward()
        result["torch"] = {
            "version": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda, "threads": torch.get_num_threads(),
            "cpu_backward_ok": probe.grad.tolist() == [2.0, 4.0],
        }
        result["audio"] = {"libsndfile": sf.__libsndfile_version__,
                           "resample_44100_to_16000_ok": len(scipy.signal.resample_poly(np.zeros(441), 160, 441)) == 160}
        result["wav2vec2_ctc_import_ok"] = Wav2Vec2ForCTC is not None
    result["ready_for_cpu_pipeline_check"] = (
        not missing and all(item["exists"] for item in result["required_inputs"].values())
        and result.get("torch", {}).get("cpu_backward_ok", False)
        and result.get("wav2vec2_ctc_import_ok", False)
    )
    local_configs = {}
    for name in ("local_smoke", "local_source", "local_adapt_vi", "local_control"):
        config_path = ROOT / "configs" / f"{name}.json"
        local_configs[name] = {"path": config_path.relative_to(ROOT).as_posix(),
                               "exists": config_path.is_file()}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            local_configs[name].update(device=config.get("device"),
                                       mixed_precision=config.get("mixed_precision"),
                                       sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    result["local_cpu_configs"] = local_configs
    audit_path = paths["prepared_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    result["prepared_data"] = {key: audit.get(key) for key in
                               ("utterance_count", "speaker_count", "vocabulary_size",
                                "audio_sha256_verified", "annotation_and_transcript_sha256_verified")}
    result["ready_for_local_cpu_training"] = (
        result["ready_for_cpu_pipeline_check"]
        and result.get("torch", {}).get("cuda_build") is None
        and all(config.get("device") == "cpu" and config.get("mixed_precision") == "no"
                for config in local_configs.values())
        and audit.get("audio_sha256_verified", False)
        and audit.get("annotation_and_transcript_sha256_verified", False)
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-local-cpu", action="store_true",
                        help="Fail unless prepared data and all local CPU presets are ready")
    args = parser.parse_args()
    result = report()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    readiness = "ready_for_local_cpu_training" if args.require_local_cpu else "ready_for_cpu_pipeline_check"
    return 0 if result[readiness] else 1


if __name__ == "__main__":
    raise SystemExit(main())
