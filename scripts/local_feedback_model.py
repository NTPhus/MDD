"""Offline Qwen feedback inference. No HTTP or automatic model downloads."""

import json
import os
from pathlib import Path
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_CACHE = None
_CACHE_KEY = None


def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def settings():
    path = local_path(os.environ.get("LLM_LOCAL_MODEL_PATH", "models/pretrained/Qwen3-0.6B"))
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise ValueError("Local Qwen weights are missing. Run scripts/download_feedback_model.py first.")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("The local feedback backend requires a Qwen3 text model.")
    device = os.environ.get("LLM_LOCAL_DEVICE", "cpu").strip()
    if device not in {"cpu", "cuda"}:
        raise ValueError("LLM_LOCAL_DEVICE must be cpu or cuda.")
    threads = int(os.environ.get("LLM_LOCAL_THREADS", "2"))
    context = int(os.environ.get("LLM_LOCAL_CONTEXT", "4096"))
    if not 1 <= threads <= 8 or not 256 <= context <= 32768:
        raise ValueError("Local threads must be 1..8 and context 256..32768.")
    adapter_value = os.environ.get("LLM_LOCAL_ADAPTER_PATH", "").strip()
    adapter = local_path(adapter_value) if adapter_value else None
    if adapter and not all((adapter / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")):
        raise ValueError("The configured local LoRA adapter is incomplete.")
    return {"path": path, "device": device, "threads": threads, "context": context, "adapter": adapter}


def _load(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for feedback but is unavailable.")
    torch.set_num_threads(config["threads"])
    tokenizer = AutoTokenizer.from_pretrained(str(config["path"]), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(config["path"]), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32 if config["device"] == "cpu" else torch.float16,
    )
    if config["adapter"]:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(config["adapter"]), local_files_only=True, is_trainable=False)
    model.to(config["device"]).eval().requires_grad_(False)
    return model, tokenizer


def complete(messages, config, max_tokens, timeout):
    import torch

    global _CACHE, _CACHE_KEY
    started = time.monotonic()
    files = [config["path"] / "model.safetensors"]
    if config["adapter"]:
        files.append(config["adapter"] / "adapter_model.safetensors")
    key = (str(config["path"]), str(config["adapter"]), config["device"],
           tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files))
    try:
        with _LOCK:
            if key != _CACHE_KEY:
                bundle = _load(config)
                _CACHE, _CACHE_KEY = bundle, key
            model, tokenizer = _CACHE
            torch.set_num_threads(config["threads"])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])
            length = inputs["input_ids"].shape[-1]
            limit = min(config["context"], model.config.max_position_embeddings)
            if length + max_tokens > limit:
                raise ValueError("Feedback exceeds the local context budget; reduce the assessment or token limit.")
            # The model card recommends sampling in non-thinking mode.
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_tokens, max_time=timeout,
                                        do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                        pad_token_id=tokenizer.eos_token_id)
            new_ids = output[0, length:]
            eos = model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            if not len(new_ids) or int(new_ids[-1]) not in eos_ids:
                raise ValueError("Local feedback generation reached its token/time limit before completion.")
            content = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if not content:
                raise ValueError("Local feedback model returned no text.")
            return {"content": content, "requested_model": "Qwen/Qwen3-0.6B",
                    "response_model": "Qwen/Qwen3-0.6B" + ("+LoRA" if config["adapter"] else ""),
                    "request_id": None, "usage": {"prompt_tokens": length, "completion_tokens": len(new_ids)},
                    "finish_reason": "stop", "latency_sec": round(time.monotonic() - started, 4)}
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        if isinstance(error, ValueError):
            raise
        raise RuntimeError("Local feedback model failed (" + type(error).__name__ + "). Check weights and optional dependencies.") from None
