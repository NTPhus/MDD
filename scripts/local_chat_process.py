"""Keep native PyTorch inference outside the web server and its MDD features."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from queue import Queue, Empty
import secrets
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_PROCESS = None
_EVENTS = None


def _start():
    global _PROCESS, _EVENTS
    folder = ROOT / "data/generated/demo"
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "TOKENIZERS_PARALLELISM":"false",
           "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "CUDA_VISIBLE_DEVICES":"", "PYTHONIOENCODING":"utf-8"}
    with (folder / "chat-worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts/local_chat_worker.py")],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            text=True, encoding="utf-8", bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    events = Queue()
    def receive():
        try:
            for line in process.stdout:
                try: events.put(json.loads(line))
                except ValueError: continue
        finally:
            events.put(None)
    threading.Thread(target=receive, daemon=True, name="chat-worker-output").start()
    _PROCESS, _EVENTS = process, events


def _send(value):
    _PROCESS.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
    _PROCESS.stdin.flush()


def shutdown():
    global _PROCESS, _EVENTS
    process = _PROCESS
    if process is None: return
    try:
        if process.poll() is None:
            try:
                process.stdin.write('{"type":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                # Only descendants of this exact owned worker are terminated.
                import psutil
                try:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for child in reversed(descendants):
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                process.terminate()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream: stream.close()
    finally:
        _PROCESS = _EVENTS = None


def complete(messages, config, max_tokens, timeout, *, on_text, cancelled, on_ready=None):
    if config["device"] != "cpu":
        raise ValueError("Chat inference is CPU only")
    profile = config.get("generation_profile")
    if profile not in (None, "conversation"):
        raise ValueError("Unsupported generation profile")
    return _request({"type": "generate", "model_id": "qwen-trained" if config["adapter"] else "qwen-base",
                     "messages": messages, "max_tokens": min(max_tokens,384), "timeout": min(timeout,120.),
                     "generation_profile": profile}, timeout,
                    on_text=on_text, cancelled=cancelled, on_ready=on_ready)


def speech(request, *, on_stage, on_speech, on_text, cancelled):
    return _request({"type": "speech", **request}, 240., on_stage=on_stage, on_speech=on_speech,
                    on_text=on_text, cancelled=cancelled)


def regenerate_reference(request, *, on_stage, on_speech, on_text, cancelled):
    return _request({**request, 'type': 'regenerate_reference'}, 240., on_stage=on_stage, on_speech=on_speech,
                    on_text=on_text, cancelled=cancelled)


def _request(request, timeout, *, on_text, cancelled, on_ready=None, on_stage=None, on_speech=None):
    with _LOCK:
        if cancelled.is_set(): return {"finish_reason":"cancelled","usage":{}}
        if _PROCESS is None or _PROCESS.poll() is not None:
            shutdown(); _start()
        identifier = secrets.token_hex(16)
        _send({**request, "id":identifier})
        sent_cancel = False
        started = time.monotonic()
        while True:
            if cancelled.is_set() and not sent_cancel:
                _send({"type":"cancel","id":identifier}); sent_cancel = True
            if time.monotonic() - started > timeout + 90:
                shutdown()
                raise RuntimeError("Local chat worker timed out")
            try:
                value = _EVENTS.get(timeout=.2)
            except Empty:
                if _PROCESS.poll() is not None:
                    shutdown()
                    raise RuntimeError("Local chat worker stopped; web server remains available") from None
                continue
            if value is None:
                shutdown()
                raise RuntimeError("Local chat worker stopped; web server remains available") from None
            if value.get("id") != identifier: continue
            if value["type"] == "ready" and on_ready: on_ready()
            elif value["type"] == "stage" and on_stage: on_stage(value["stage"])
            elif value["type"] == "speech" and on_speech: on_speech(value["speech"])
            elif value["type"] == "chunk" and not cancelled.is_set(): on_text(value["text"])
            elif value["type"] == "done":
                metadata = value["metadata"]
                metadata["latency_sec"] = round(time.monotonic() - started, 3)
                return metadata
            elif value["type"] == "error": raise ValueError(value.get("code", "generation_failed"))


atexit.register(shutdown)
