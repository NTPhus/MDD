#!/usr/bin/env python3
"""All-in-one manager for MDD demo backend and Cloudflare tunnel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_demo_cloudflare as cloudflare_deployment
import deploy_demo_nginx as nginx_deployment

PYTHON = ROOT / ".venv/Scripts/python.exe"
SERVE_SCRIPT = ROOT / "scripts/serve_demo.py"
LOG_DIR = ROOT / "data/generated/demo/logs"
SERVER_STATE = ROOT / "data/generated/demo/server.json"
BACKEND_PORT = 8765
HTTP = build_opener(ProxyHandler({}))

CREATE_BREAKAWAY_FROM_JOB = 0x01000000
FLAGS_DETACHED = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NEW_PROCESS_GROUP
    | subprocess.CREATE_NO_WINDOW
    | CREATE_BREAKAWAY_FROM_JOB
    if sys.platform == "win32"
    else 0
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def backend_health(timeout: float = 3.0) -> bool:
    try:
        req = Request(f"http://127.0.0.1:{BACKEND_PORT}/api/status", headers={"User-Agent": "MDD-health/1"})
        with HTTP.open(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_backend_process() -> psutil.Process | None:
    # 1. Check server.json
    state = read_json(SERVER_STATE)
    pid = state.get("pid")
    if pid:
        try:
            p = psutil.Process(int(pid))
            cmd = " ".join(p.cmdline()).lower()
            if "serve_demo.py" in cmd:
                return p
        except (psutil.Error, ValueError):
            pass

    # 2. Check listeners on port 8765
    listeners = nginx_deployment.listeners(BACKEND_PORT)
    for conn in listeners:
        try:
            p = psutil.Process(conn.pid)
            return p
        except psutil.Error:
            pass

    # 3. Check processes by command line
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if "serve_demo.py" in cmd:
                return p
        except (psutil.Error, TypeError):
            pass
    return None


def stop_backend() -> bool:
    stopped_any = False
    p = get_backend_process()
    if p:
        try:
            children = p.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            p.terminate()
            gone, alive = psutil.wait_procs(children + [p], timeout=5)
            for item in alive:
                try:
                    item.kill()
                except psutil.Error:
                    pass
            stopped_any = True
        except psutil.Error:
            pass

    # Clean up any orphan workers or leftovers
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if any(target in cmd for target in ("serve_demo.py", "local_chat_worker.py", "kokoro_worker.py")):
                p.kill()
                stopped_any = True
        except (psutil.Error, TypeError):
            pass

    state = read_json(SERVER_STATE)
    state.update(status="stopped", stopped_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(SERVER_STATE, state)
    return stopped_any


def start_backend(threads: int = 2) -> dict:
    existing = get_backend_process()
    if existing and backend_health():
        state = read_json(SERVER_STATE)
        state.update(status="ready", pid=existing.pid, port=BACKEND_PORT)
        return state

    # Stop any half-dead process first
    stop_backend()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_file = LOG_DIR / "server.stdout.log"
    stderr_file = LOG_DIR / "server.stderr.log"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    cmd = [
        str(PYTHON),
        "-X",
        "utf8",
        "-u",
        str(SERVE_SCRIPT),
        "--port",
        str(BACKEND_PORT),
        "--threads",
        str(threads),
    ]

    with stdout_file.open("w", encoding="utf-8") as out, stderr_file.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            creationflags=FLAGS_DETACHED,
            env=env,
        )

    # Wait for backend to become healthy
    deadline = time.monotonic() + 35
    ready = False
    while time.monotonic() < deadline:
        if backend_health(timeout=1.0):
            ready = True
            break
        if proc.poll() is not None:
            err_text = ""
            try:
                err_text = stderr_file.read_text(encoding="utf-8", errors="replace")[-500:]
            except OSError:
                pass
            raise RuntimeError(f"Backend process exited prematurely. Error log: {err_text}")
        time.sleep(0.5)

    if not ready:
        raise TimeoutError("Backend server did not respond with HTTP 200 within 35 seconds.")

    state = {
        "status": "ready",
        "pid": proc.pid,
        "port": BACKEND_PORT,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": f"http://127.0.0.1:{BACKEND_PORT}/",
        "stdout": str(stdout_file),
        "stderr": str(stderr_file),
    }
    save_json(SERVER_STATE, state)
    return state


def start_all(local_only: bool = False, threads: int = 2) -> dict:
    backend_info = start_backend(threads=threads)
    cf_info = {}
    if not local_only:
        try:
            cf_info = cloudflare_deployment.start()
        except Exception as e:
            cf_info = {"status": "error", "message": str(e)}

    return {
        "system_status": "ready" if backend_info.get("status") == "ready" else "error",
        "backend": backend_info,
        "cloudflare": cf_info,
    }


def stop_all() -> dict:
    cf_info = {}
    try:
        cf_info = cloudflare_deployment.stop()
    except Exception as e:
        cf_info = {"status": "error", "message": str(e)}

    be_stopped = stop_backend()
    return {
        "system_status": "stopped",
        "backend_stopped": be_stopped,
        "cloudflare": cf_info,
    }


def status_all() -> dict:
    proc = get_backend_process()
    healthy = backend_health() if proc else False
    cf_info = {}
    try:
        cf_info = cloudflare_deployment.status()
    except Exception as e:
        cf_info = {"status": "error", "message": str(e)}

    return {
        "system_status": "ready" if healthy else "stopped_or_unhealthy",
        "backend": {
            "running": bool(proc),
            "pid": proc.pid if proc else None,
            "port": BACKEND_PORT,
            "healthy": healthy,
            "url": f"http://127.0.0.1:{BACKEND_PORT}/",
        },
        "cloudflare": cf_info,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["start", "stop", "restart", "status"], default="start")
    parser.add_argument("--local-only", action="store_true", help="Do not start Cloudflare tunnel")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run in background and exit immediately")
    parser.add_argument("--threads", type=int, default=2, help="Number of inference threads")
    parser.add_argument("--json", action="store_true", help="Output pure JSON")
    args = parser.parse_args()

    action = args.action.lower()
    if action == "stop":
        res = stop_all()
    elif action == "restart":
        stop_all()
        time.sleep(1)
        res = start_all(local_only=args.local_only, threads=args.threads)
    elif action == "status":
        res = status_all()
    else:
        res = start_all(local_only=args.local_only, threads=args.threads)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    print("=" * 60)
    print("           MDD SYSTEM STATUS & ACCESS URLs           ")
    print("=" * 60)
    be = res.get("backend", {})
    be_status = "READY (Running)" if be.get("status") == "ready" or be.get("healthy") else "STOPPED / UNHEALTHY"
    print(f"[*] Backend Local  : {be_status}")
    print(f"    -> Local URL   : http://127.0.0.1:{BACKEND_PORT}/")

    cf = res.get("cloudflare", {})
    if cf:
        cf_url = cf.get("public_url")
        cf_status = cf.get("status")
        if cf_status == "ready" and cf_url:
            print(f"[*] Cloudflare     : READY (Online)")
            print(f"    -> Public URL  : {cf_url}")
        elif cf_status == "stopped":
            print(f"[*] Cloudflare     : STOPPED")
        else:
            print(f"[*] Cloudflare     : {cf_status} ({cf.get('message', 'N/A')})")
    print("=" * 60)

    if not args.daemon and action in ("start", "restart") and be_status.startswith("READY"):
        print("\n[i] He thong dang hoat dong. Nhan Ctrl+C de dung toan bo dich vu...")
        try:
            while True:
                time.sleep(2)
                if not backend_health():
                    print("\n[!] Backend server da dung bat thuong.")
                    break
        except KeyboardInterrupt:
            print("\nDang dung tat ca dich vu...")
            stop_all()
            print("Da dung xong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
