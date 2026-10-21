#!/usr/bin/env python3
"""Install and manage this workspace's Windows Nginx + loopback MDD demo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener, urlopen
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.31.7"
ARCHIVE_SHA256 = "7c717ccf861f8925be64431d06d57634512a5014bcc8d149f279928f5e8151bf"
TOOL_ROOT = ROOT / ".tools/nginx"
NGINX = TOOL_ROOT / f"nginx-{VERSION}/nginx.exe"
PYTHON = ROOT / ".venv/Scripts/python.exe"
SERVE = ROOT / "scripts/serve_demo.py"
RUNTIME = ROOT / "data/generated/demo/nginx"
CONFIG = RUNTIME / "conf/nginx.conf"
STATE = RUNTIME / "deployment.json"
LEGACY_STATE = ROOT / "data/generated/demo/server.json"
HTTP = build_opener(ProxyHandler({}))
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def command(arguments, *, timeout=30):
    result = subprocess.run([str(value) for value in arguments], cwd=ROOT,
                            capture_output=True, text=True, errors="replace",
                            creationflags=HIDDEN, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"Command exited {result.returncode}")
    return result


def nginx_arguments(*extra):
    return [NGINX, "-p", RUNTIME.as_posix() + "/", "-c", CONFIG.as_posix(), *extra]


def install():
    TOOL_ROOT.mkdir(parents=True, exist_ok=True)
    archive = TOOL_ROOT / f"nginx-{VERSION}.zip"
    source = f"https://nginx.org/download/nginx-{VERSION}.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        with urlopen(source, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError("Nginx archive SHA256 differs from the pinned official download; no extraction performed.")
    if not NGINX.is_file():
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = (TOOL_ROOT / member.filename).resolve()
                if not target.is_relative_to(TOOL_ROOT.resolve() / f"nginx-{VERSION}"):
                    raise RuntimeError("Unexpected path in Nginx archive.")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Symbolic links are not accepted in the Nginx archive.")
            package.extractall(TOOL_ROOT)
    save_json(TOOL_ROOT / "download.json", {"version": VERSION, "url": source,
              "archive_sha256": digest, "verification": "HTTPS download and pinned SHA256; no PGP verification"})
    return command([NGINX, "-v"]).stderr.strip()


def configure(address, port, backend_port):
    parsed = ipaddress.IPv4Address(address)
    private_ranges = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    if str(parsed) != address or not any(parsed in ipaddress.IPv4Network(net) for net in private_ranges):
        raise ValueError("Use a canonical RFC1918 LAN IPv4 address.")
    assigned = {entry.address for entries in psutil.net_if_addrs().values() for entry in entries}
    if address not in assigned:
        raise ValueError("The LAN address is not assigned to this machine.")
    runtime_path = RUNTIME.as_posix()
    if any(character in runtime_path for character in ('"', ";", "\r", "\n", "$")):
        raise ValueError("The workspace path contains an unsupported Nginx configuration character.")
    replacements = {
        "@@LAN_ADDRESS@@": address,
        "@@LAN_ADDRESS_REGEX@@": address.replace(".", "[.]"),
        "@@PUBLIC_PORT@@": str(port),
        "@@AUTHORITY_PORT_REGEX@@": "(?::80)?" if port == 80 else f":{port}",
        "@@BACKEND_PORT@@": str(backend_port),
        "@@RUNTIME_DIRECTORY@@": runtime_path,
    }
    content = (ROOT / "deploy/nginx/nginx.conf.template").read_text(encoding="utf-8")
    for marker, value in replacements.items():
        content = content.replace(marker, value)
    if "@@" in content:
        raise ValueError("Unresolved Nginx configuration placeholders.")
    directories = [CONFIG.parent, RUNTIME / "logs", RUNTIME / "client_body_temp", RUNTIME / "proxy_temp"]
    directories.extend(RUNTIME / "temp" / name for name in
                       ("client_body_temp", "proxy_temp", "fastcgi_temp", "uwsgi_temp", "scgi_temp"))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(content, encoding="utf-8")
    command(nginx_arguments("-t"))


def snapshot(process):
    return {"pid": process.pid, "created": process.create_time(),
            "executable": process.exe(), "command": process.cmdline()}


def verified(record, kind):
    """Never act on a reused PID or an executable outside this deployment."""
    if not record:
        return None
    try:
        process = psutil.Process(record["pid"])
        expected = NGINX if kind == "nginx" else PYTHON
        if abs(process.create_time() - record["created"]) > .01:
            return None
        if Path(process.exe()).resolve() != expected.resolve():
            return None
        actual = process.cmdline()
        marker = RUNTIME.as_posix() + "/" if kind == "nginx" else str(SERVE)
        if marker not in actual:
            return None
        return process
    except (psutil.Error, KeyError, ValueError):
        return None


def listeners(port):
    return [connection for connection in psutil.net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port]


def project_backend(connection, port):
    """Recognize this project's venv launcher, including its base-Python child."""
    process = psutil.Process(connection.pid)
    candidates = [process, process.parent()]
    legacy_pid = read_json(LEGACY_STATE).get("pid")
    for candidate in candidates:
        if candidate is None or Path(candidate.exe()).resolve() != PYTHON.resolve():
            continue
        arguments = candidate.cmdline()
        absolute_script = str(SERVE) in arguments
        known_legacy = candidate.pid == legacy_pid and "scripts/serve_demo.py" in arguments
        if not (absolute_script or known_legacy):
            continue
        if known_legacy and Path(candidate.cwd()).resolve() != ROOT:
            continue
        if "--port" in arguments and arguments[arguments.index("--port") + 1] != str(port):
            continue
        if "--port" not in arguments and port != 8765:
            continue
        return candidate
    raise RuntimeError(f"Port {port} belongs to an unrecognized process; no process was stopped.")


def terminate_backend(process):
    # Capture identities before stopping; a PID reused during shutdown must be untouched.
    processes = process.children(recursive=True) + [process]
    records = [(item, item.create_time()) for item in processes]
    for item, created in records:
        try:
            if item.create_time() == created:
                item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    if alive:
        raise RuntimeError("The verified demo process did not stop; inspect backend logs before retrying.")


def http_status(url):
    try:
        with HTTP.open(Request(url), timeout=3) as response:
            return response.status
    except OSError:
        return None


def stop_nginx(record):
    process = verified(record, "nginx")
    if process is None:
        return
    try:
        saved_pid = int((RUNTIME / "nginx.pid").read_text().strip())
    except (OSError, ValueError):
        raise RuntimeError("Nginx PID file is missing or invalid; no shutdown signal sent.") from None
    if saved_pid != process.pid:
        raise RuntimeError("Nginx PID file differs from the verified master; no shutdown signal sent.")
    command(nginx_arguments("-s", "quit"))
    try:
        process.wait(timeout=8)
    except psutil.TimeoutExpired:
        raise RuntimeError("Nginx is still finishing requests; retry Status later.") from None


def stop():
    state = read_json(STATE)
    stop_nginx(state.get("nginx"))
    if state.get("backend_owned"):
        process = verified(state.get("backend"), "backend")
        if process is not None:
            terminate_backend(process)
    state["status"] = "stopped"
    save_json(STATE, state)
    return state


def status():
    state = read_json(STATE)
    nginx = verified(state.get("nginx"), "nginx")
    state["nginx_running"] = nginx is not None
    state["http_status"] = http_status(state["local_url"] + "api/status") if nginx else None
    state["status"] = "ready" if nginx and state["http_status"] == 200 else "stopped_or_unhealthy"
    return state


def start(address, port, backend_port, threads):
    old_state = read_json(STATE)
    if verified(old_state.get("nginx"), "nginx"):
        if (address, port, backend_port) != (old_state.get("lan_address"), old_state.get("port"), old_state.get("backend_port")):
            raise RuntimeError("This workspace already runs Nginx with different parameters; stop it first.")
        current = status()
        if current["status"] != "ready":
            raise RuntimeError("Existing deployment is unhealthy; inspect logs before restarting.")
        return current
    if listeners(port):
        raise RuntimeError(f"Public port {port} is already in use; no process was stopped.")
    existing = listeners(backend_port)
    owners = {project_backend(connection, backend_port).pid for connection in existing}
    if len(owners) > 1:
        raise RuntimeError("More than one demo launcher owns the backend port.")
    backend = psutil.Process(next(iter(owners))) if owners else None
    replace_backend = backend is not None and any(connection.laddr.ip != "127.0.0.1" for connection in existing)
    previous_owned_backend = verified(old_state.get("backend"), "backend")
    reuse_owned_backend = bool(old_state.get("backend_owned") and backend is not None and
                              previous_owned_backend is not None and previous_owned_backend.pid == backend.pid)
    legacy_record = snapshot(backend) if replace_backend else None
    legacy_metadata = read_json(LEGACY_STATE)
    legacy_stopped = False
    version = install()
    configure(address, port, backend_port)
    local_url = f"http://127.0.0.1{'' if port == 80 else ':' + str(port)}/"
    lan_url = f"http://{address}{'' if port == 80 else ':' + str(port)}/"
    state = {"status": "starting", "version": VERSION, "version_output": version,
             "lan_address": address, "port": port, "backend_port": backend_port,
             "local_url": local_url, "lan_url": lan_url,
             "backend_url": f"http://127.0.0.1:{backend_port}/",
             "config": str(CONFIG), "backend_owned": reuse_owned_backend,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    new_backend = None
    try:
        with (RUNTIME / "logs/launcher.log").open("ab") as output:
            launcher = subprocess.Popen([str(value) for value in nginx_arguments()], cwd=ROOT,
                                        stdout=output, stderr=output, creationflags=HIDDEN)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                master_pid = int((RUNTIME / "nginx.pid").read_text().strip())
                master = psutil.Process(master_pid)
                record = snapshot(master)
                if verified(record, "nginx") and any(item.pid in {master.pid, *(child.pid for child in master.children())} for item in listeners(port)):
                    state["nginx"] = record
                    break
            except (OSError, ValueError, psutil.Error):
                pass
            if launcher.poll() is not None and launcher.returncode:
                raise RuntimeError("Nginx failed to bind; inspect logs/error.log. Existing backend was kept.")
            time.sleep(.15)
        else:
            raise RuntimeError("Nginx listener was not ready; existing backend was kept.")
        save_json(STATE, state)
        if replace_backend:
            # Nginx has successfully bound before changing the existing LAN backend.
            if not (RUNTIME / "server-before-nginx.json").exists():
                save_json(RUNTIME / "server-before-nginx.json", read_json(LEGACY_STATE))
            terminate_backend(backend)
            legacy_stopped = True
            backend = None
        if backend is None:
            environment = os.environ.copy()
            environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            with (RUNTIME / "logs/backend.stdout.log").open("ab") as output, (RUNTIME / "logs/backend.stderr.log").open("ab") as errors:
                new_backend = subprocess.Popen([str(PYTHON), "-X", "utf8", "-u", str(SERVE), "--port", str(backend_port), "--threads", str(threads)],
                                               cwd=ROOT, env=environment, stdout=output, stderr=errors, creationflags=HIDDEN)
            backend = psutil.Process(new_backend.pid)
            state["backend_owned"] = True
        state["backend"] = snapshot(backend)
        save_json(STATE, state)
        deadline = time.monotonic() + 45
        while http_status(local_url + "api/status") != 200:
            if time.monotonic() >= deadline or not backend.is_running():
                raise RuntimeError("Backend readiness failed; inspect logs/backend.stderr.log.")
            time.sleep(.25)
        if any(item.laddr.ip != "127.0.0.1" for item in listeners(backend_port)):
            raise RuntimeError("The backend must listen only on 127.0.0.1.")
        state["status"] = "ready"
        state["http_status"] = 200
        save_json(STATE, state)
        save_json(LEGACY_STATE, {"pid": backend.pid, "port": backend_port, "bind_address": "127.0.0.1",
                  "url": local_url, "lan_url": lan_url, "backend_url": state["backend_url"],
                  "reverse_proxy": "nginx", "deployment_state": str(STATE),
                  "stdout": str(RUNTIME / "logs/backend.stdout.log"), "stderr": str(RUNTIME / "logs/backend.stderr.log")})
        return state
    except Exception as error:
        cleanup_errors = []
        if new_backend is not None:
            try:
                candidate = verified(state.get("backend"), "backend")
                if candidate is not None:
                    terminate_backend(candidate)
            except (OSError, RuntimeError, psutil.Error) as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        try:
            stop_nginx(state.get("nginx"))
        except (OSError, RuntimeError, psutil.Error) as cleanup_error:
            cleanup_errors.append(str(cleanup_error))
        state.update(status="error", message=str(error))
        if legacy_stopped and legacy_record is not None:
            try:
                environment = os.environ.copy()
                environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                                   OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
                with (RUNTIME / "logs/rollback.stdout.log").open("ab") as output, (RUNTIME / "logs/rollback.stderr.log").open("ab") as errors:
                    restored = subprocess.Popen(legacy_record["command"], cwd=ROOT, env=environment,
                                                stdout=output, stderr=errors, creationflags=HIDDEN)
                deadline = time.monotonic() + 30
                while http_status(f"http://127.0.0.1:{backend_port}/api/status") != 200:
                    if time.monotonic() >= deadline or restored.poll() is not None:
                        raise RuntimeError("Previous demo did not become ready after rollback.")
                    time.sleep(.25)
                legacy_metadata.update(pid=restored.pid, stdout=str(RUNTIME / "logs/rollback.stdout.log"),
                                       stderr=str(RUNTIME / "logs/rollback.stderr.log"))
                save_json(LEGACY_STATE, legacy_metadata)
                state["legacy_backend_restored"] = True
            except (OSError, RuntimeError) as rollback_error:
                cleanup_errors.append(str(rollback_error))
        if cleanup_errors:
            state["cleanup_errors"] = cleanup_errors
        save_json(STATE, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("setup", "start", "stop", "status"), default="start")
    parser.add_argument("--lan-address")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This launcher installs native Windows Nginx; use the deployment template separately on Linux.")
    if not 1 <= args.port <= 65535 or not 1024 <= args.backend_port <= 65535 or args.port == args.backend_port:
        parser.error("Use distinct public port1..65535 and backend port1024..65535.")
    if any(not path.resolve().is_relative_to(ROOT) for path in (TOOL_ROOT, NGINX, RUNTIME)):
        parser.error("Deployment paths must stay inside this workspace.")
    if args.action in {"setup", "start"} and not args.lan_address:
        parser.error("--lan-address is required for setup/start.")
    if args.action == "setup":
        version = install()
        configure(args.lan_address, args.port, args.backend_port)
        result = {"status": "configured", "version": version, "config": str(CONFIG), "program": str(NGINX)}
    elif args.action == "start":
        result = start(args.lan_address, args.port, args.backend_port, args.threads)
    else:
        result = stop() if args.action == "stop" else status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, psutil.Error) as error:
        print(f"Nginx deployment failed: {error}", file=sys.stderr)
        sys.exit(1)
