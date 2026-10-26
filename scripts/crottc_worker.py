"""JSONL worker for verified CROTTC; third-party imports stay out of the demo."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def physical_cuda_memory():
    """Read GPU 0's physical memory; WDDM CUDA budgets can overstate free VRAM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        fields = result.stdout.strip().split(",")
        if len(fields) != 2:
            return None
        free_mib, total_mib = (int(value.strip()) for value in fields)
        if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
            return None
        return {"free_mib": free_mib, "total_mib": total_mib}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def select_runtime_device(torch, requested_device="cpu", min_free_cuda_mib=2560):
    """Keep FP32 decoding, using CUDA only with sufficient observed free VRAM.

    The memory gate is an operational safety margin, not a claim that every
    audio duration fits. Allocation failure has a separate, explicit fallback.
    """
    if requested_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("CROTTC device must be cpu, cuda or auto")
    if isinstance(min_free_cuda_mib, bool) or not isinstance(min_free_cuda_mib, int) or min_free_cuda_mib < 0:
        raise ValueError("CROTTC minimum free CUDA memory must be a nonnegative integer MiB")
    runtime = {"requested_device": requested_device, "device": "cpu", "effective_device": "cpu",
               "compute_type": "float32", "device_fallback_reason": None,
               "min_free_cuda_mib": min_free_cuda_mib}
    if requested_device == "cpu":
        return "cpu", runtime
    if not torch.cuda.is_available():
        runtime["device_fallback_reason"] = "cuda_unavailable"
        return "cpu", runtime
    free, total = torch.cuda.mem_get_info(0)
    runtime.update(cuda_free_mib=round(free / (1024 ** 2), 1), cuda_total_mib=round(total / (1024 ** 2), 1))
    physical = physical_cuda_memory()
    runtime["cuda_memory_gate_source"] = "torch"
    runtime["cuda_physical_free_mib"] = None
    runtime["cuda_physical_total_mib"] = None
    if physical is not None:
        runtime.update(cuda_physical_free_mib=physical["free_mib"],
                       cuda_physical_total_mib=physical["total_mib"],
                       cuda_memory_gate_source="min_torch_and_nvidia_smi")
        free = min(free, physical["free_mib"] * (1024 ** 2))
    runtime["cuda_memory_gate_free_mib"] = round(free / (1024 ** 2), 1)
    if free < min_free_cuda_mib * (1024 ** 2):
        runtime["device_fallback_reason"] = "insufficient_free_cuda_memory"
        return "cpu", runtime
    runtime.update(device="cuda:0", effective_device="cuda:0")
    return "cuda:0", runtime


def clear_cuda_cache(torch):
    # This worker owns this CUDA context; it never changes other processes.
    gc.collect()
    torch.cuda.empty_cache()


def load_with_device_fallback(torch, loader, requested_device="cpu", min_free_cuda_mib=2560):
    device, runtime = select_runtime_device(torch, requested_device, min_free_cuda_mib)
    try:
        model = loader(device)
    except torch.cuda.OutOfMemoryError:
        if device == "cpu":
            raise
        runtime.update(device="cpu", effective_device="cpu", device_fallback_reason="cuda_out_of_memory_during_load")
    else:
        return model, runtime
    # Leave the exception scope before collecting partially constructed tensors.
    clear_cuda_cache(torch)
    return loader("cpu"), runtime


def inference_with_device_fallback(torch, model, runtime, batch_factory, loader):
    """Retry only CUDA allocation failures; invalid/model errors remain visible."""
    try:
        prediction = model.brain.inference_batch(batch_factory())
    except torch.cuda.OutOfMemoryError:
        if runtime["device"] == "cpu":
            raise
        runtime.update(device="cpu", effective_device="cpu", device_fallback_reason="cuda_out_of_memory_during_inference")
    else:
        return model, prediction
    # Move the old modules off GPU before reloading to release this model's VRAM.
    # CPU tensors remain FP32, matching the default evaluation path.
    model.brain.modules.to("cpu")
    model = None
    clear_cuda_cache(torch)
    model = loader("cpu")
    return model, model.brain.inference_batch(batch_factory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-downloaded", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--min-free-cuda-mib", type=int, default=2560)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    folder = root / ".tools/crottc"
    bundle = folder / "bundle"
    sys.path[:0] = [str(root), str(folder / "python"), str(bundle)]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
                      WANDB_DISABLED="true", HF_HUB_DISABLE_TELEMETRY="1")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from scripts.crottc_pronunciation import MODEL_ID, REVISION, MODEL_SHA256, SSL_SHA256, verify, phone_evidence
    model, record, runtime = None, None, None
    protocol = sys.stdout
    for line in sys.stdin:
        identifier = None
        try:
            request = json.loads(line)
            identifier = request["id"]
            with redirect_stdout(sys.stderr):
                if model is None:
                    started = time.monotonic()
                    record = verify(root, require_ready=not args.allow_downloaded)
                    import torch
                    import torchaudio
                    import speechbrain
                    import transformers
                    if torch.__version__ != "2.7.1+cu126" or torchaudio.__version__ != "2.7.1+cu126":
                        raise ValueError("CROTTC requires matching torch/torchaudio 2.7.1+cu126")
                    if speechbrain.__version__ != "1.0.3" or transformers.__version__ != "4.57.0":
                        raise ValueError("CROTTC runtime package versions changed")
                    torch.set_num_threads(2)
                    torch.set_num_interop_threads(1)
                    original_load = torch.load
                    allowed = {(bundle / name).resolve() for name in ("checkpoint/model.ckpt", "checkpoint/perceived_ssl.ckpt")}
                    for path in allowed:
                        if torch.serialization.get_unsafe_globals_in_checkpoint(path):
                            raise ValueError("CROTTC checkpoint contains unsafe pickle globals")
                    def weights_only_load(path, *positional, **kwargs):
                        if Path(path).resolve() not in allowed:
                            raise ValueError("CROTTC attempted to load an unverified checkpoint")
                        kwargs.update(weights_only=True, map_location="cpu")
                        return original_load(path, *positional, **kwargs)
                    torch.load = weights_only_load
                    spec = importlib.util.spec_from_file_location("crottc_official_interface", bundle / "custom_interface.py")
                    interface = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(interface)
                    def loader(device):
                        # No reference text, phone labels or teacher forcing enter inference.
                        loaded = interface.from_pretrained(bundle, device=device, overrides=[
                            "output_folder=" + json.dumps(str(folder / "runtime-output")),
                            "pretrained_models_path=" + json.dumps(str(folder / "cache")),
                            "pretrained_model_path=" + json.dumps(str(folder / "unused-pretrainer")),
                            "precision=fp32", "auto_mix_prec=false",
                        ])
                        loaded.brain.modules.eval()
                        for parameter in loaded.brain.modules.parameters():
                            parameter.requires_grad_(False)
                        return loaded
                    model, device_runtime = load_with_device_fallback(torch, loader, args.device, args.min_free_cuda_mib)
                    runtime = {**device_runtime, "cpu_threads": 2,
                               "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                               "speechbrain": speechbrain.__version__, "transformers": transformers.__version__,
                               "load_sec": round(time.monotonic() - started, 4), "load_only": True,
                               "weights_only": True, "training": False, "offline": True,
                               "decoder": "official_speechbrain_ar_ctc_beam10_weight0.99"}
                if request["type"] == "load":
                    result = runtime
                elif request["type"] == "infer":
                    import numpy as np
                    audio = np.asarray(request["waveform"], dtype=np.float32)
                    if audio.ndim != 1 or not 400 <= audio.size <= 480000 or not np.isfinite(audio).all():
                        raise ValueError("Invalid CROTTC waveform")
                    started = time.monotonic()
                    def batch_factory():
                        return interface._AudioBatch(torch.from_numpy(audio).unsqueeze(0), torch.ones(1), ["local-recording"])
                    with torch.inference_mode():
                        model, prediction = inference_with_device_fallback(torch, model, runtime, batch_factory, loader)
                    hyps = prediction["hyps"]
                    if torch.is_tensor(hyps):
                        hyps = hyps.detach().cpu().tolist()
                    labels = model.label_encoder.decode_ndim(hyps[0])
                    phones, unresolved, excluded = phone_evidence(labels)
                    result = {"provider": "crottc-if", "model_id": MODEL_ID, "revision": REVISION,
                              "checkpoint_sha256": MODEL_SHA256, "ssl_checkpoint_sha256": SSL_SHA256,
                              "vocabulary_sha256": record["files"]["label_encoder.txt"]["sha256"],
                              "duration_sec": len(audio) / 16000., "sample_rate": 16000,
                              "phonemes": phones, "decoded_labels": labels, "sequence_tokens": hyps[0],
                              "unresolved_labels": [row["label"] for row in unresolved],
                              "unresolved_tokens": unresolved, "excluded_labels": excluded,
                              "phone_timing": "unavailable", "log_probs": None,
                              "reference_supplied_to_model": False, "runtime": runtime,
                              "inference_sec": round(time.monotonic() - started, 4)}
                else:
                    raise ValueError("Unknown CROTTC request type")
            protocol.write(json.dumps({"id": identifier, "result": result}, allow_nan=False) + "\n")
            protocol.flush()
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({"id": identifier, "error": f"{type(error).__name__}: {error}"}) + "\n")
            protocol.flush()


if __name__ == "__main__":
    main()
