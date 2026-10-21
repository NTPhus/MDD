"""Inspect the local CPU/CUDA training environment without network access or API keys."""

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


def model_weights_path(directory):
    """Honor the downloaded format while keeping legacy bin-only installs usable."""
    manifest = directory / "download_manifest.json"
    if manifest.is_file():
        weights = json.loads(manifest.read_text(encoding="utf-8")).get("weights_file")
        if weights in {"model.safetensors", "pytorch_model.bin"}:
            return directory / weights
    safe_weights = directory / "model.safetensors"
    return safe_weights if safe_weights.is_file() else directory / "pytorch_model.bin"


def config_report(prefix):
    configs = {}
    for suffix in ("smoke", "source", "adapt_vi", "control"):
        name = f"{prefix}_{suffix}"
        config_path = ROOT / "configs" / f"{name}.json"
        configs[name] = {"path": config_path.relative_to(ROOT).as_posix(),
                         "exists": config_path.is_file()}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configs[name].update(device=config.get("device"),
                                 mixed_precision=config.get("mixed_precision"),
                                 sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    return configs


def cuda_probe(torch):
    """A CUDA wheel or a visible GPU alone does not prove backward execution works."""
    result = {"cuda_backward_ok": False}
    if not torch.cuda.is_available():
        return result
    try:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        result["gpu"] = {
            "index": device_index, "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        probe = torch.ones(16, device=f"cuda:{device_index}", requires_grad=True)
        probe.square().sum().backward()
        torch.cuda.synchronize(device_index)
        result["cuda_backward_ok"] = bool((probe.grad == 2).all().item())
        free_memory, _ = torch.cuda.mem_get_info(device_index)
        result["gpu"]["free_memory_bytes"] = free_memory
    except Exception as error:
        result["cuda_backward_ok"] = False
        result["cuda_probe_error"] = f"{type(error).__name__}: {error}"
    return result


def local_training_ready(result, configs, audit, *, device):
    torch_info = result.get("torch", {})
    if device == "cpu":
        device_ready = torch_info.get("cuda_build") is None
        configs_ready = all(config.get("device") == "cpu" and config.get("mixed_precision") == "no"
                            for config in configs.values())
    else:
        device_ready = (bool(torch_info.get("cuda_build"))
                        and torch_info.get("cuda_available", False)
                        and torch_info.get("cuda_backward_ok", False))
        configs_ready = all(
            config.get("device") == "cuda"
            and config.get("mixed_precision") in {"no", "fp16", "bf16"}
            and (config.get("mixed_precision") != "bf16" or torch_info.get("bf16_supported", False))
            for config in configs.values())
    return bool(result["ready_for_cpu_pipeline_check"] and device_ready and configs_ready
                and audit.get("audio_sha256_verified", False)
                and audit.get("annotation_and_transcript_sha256_verified", False))


def report():
    missing = [name for name in PACKAGES if importlib.util.find_spec(name) is None]
    memory = {}
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[name + "_bytes"] = int(value.split()[0]) * 1024
    elif "psutil" not in missing:
        import psutil

        physical = psutil.virtual_memory()
        swap = psutil.swap_memory()
        memory = {"MemTotal_bytes": physical.total, "MemAvailable_bytes": physical.available,
                  "SwapTotal_bytes": swap.total, "SwapFree_bytes": swap.free}
    paths = {
        "model_config": ROOT / "models/pretrained/wav2vec2-base/config.json",
        "model_weights": model_weights_path(ROOT / "models/pretrained/wav2vec2-base"),
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
        result["torch"].update(cuda_probe(torch))
        result["audio"] = {"libsndfile": sf.__libsndfile_version__,
                           "resample_44100_to_16000_ok": len(scipy.signal.resample_poly(np.zeros(441), 160, 441)) == 160}
        result["wav2vec2_ctc_import_ok"] = Wav2Vec2ForCTC is not None
    result["ready_for_cpu_pipeline_check"] = (
        not missing and all(item["exists"] for item in result["required_inputs"].values())
        and result.get("torch", {}).get("cpu_backward_ok", False)
        and result.get("wav2vec2_ctc_import_ok", False)
    )
    local_configs = config_report("local")
    windows_configs = config_report("windows")
    result["local_cpu_configs"] = local_configs
    result["windows_cuda_configs"] = windows_configs
    audit_path = paths["prepared_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    result["prepared_data"] = {key: audit.get(key) for key in
                               ("utterance_count", "speaker_count", "vocabulary_size",
                                "audio_sha256_verified", "annotation_and_transcript_sha256_verified")}
    result["ready_for_local_cpu_training"] = local_training_ready(result, local_configs, audit, device="cpu")
    result["ready_for_local_cuda_training"] = local_training_ready(result, windows_configs, audit, device="cuda")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    required_device = parser.add_mutually_exclusive_group()
    required_device.add_argument("--require-local-cpu", action="store_true",
                                 help="Fail unless prepared data and all local CPU presets are ready")
    required_device.add_argument("--require-local-cuda", action="store_true",
                                 help="Fail unless CUDA backward, prepared data and all Windows CUDA presets are ready")
    args = parser.parse_args()
    result = report()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    readiness = ("ready_for_local_cuda_training" if args.require_local_cuda else
                 "ready_for_local_cpu_training" if args.require_local_cpu else "ready_for_cpu_pipeline_check")
    return 0 if result[readiness] else 1


if __name__ == "__main__":
    raise SystemExit(main())
