#!/usr/bin/env python3
"""Manage an isolated HTTPS ngrok ingress without restarting the MDD backend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

import psutil

import deploy_demo_nginx as nginx_deployment

ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.39.11"
ARCHIVE_SHA256 = "699bbf1932ec43a573b764bd03e6568efa2c4e45955eb3cc2089c19bb4be4464"
DOWNLOAD_URL = "https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
TOOL_ROOT = ROOT / ".tools/ngrok"
NGROK = TOOL_ROOT / "ngrok.exe"
ARCHIVE = TOOL_ROOT / "ngrok-v3-stable-windows-amd64.zip"
RUNTIME = ROOT / "data/generated/demo/ngrok"
NGINX_RUNTIME = RUNTIME / "nginx"
CONFIG = NGINX_RUNTIME / "conf/nginx.conf"
AGENT_CONFIG = RUNTIME / "agent.json"
STATE = RUNTIME / "deployment.json"
HTTP = nginx_deployment.HTTP
HIDDEN = nginx_deployment.HIDDEN
save_json = nginx_deployment.save_json
read_json = nginx_deployment.read_json


class DeploymentError(RuntimeError):
    """A fixed, secret-free diagnostic that may be shown to the user."""


def settings():
    """Read only the two ngrok settings; never evaluate dotenv content."""
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in {"NGROK_AUTHTOKEN", "NGROK_DOMAIN"}:
                continue
            if value[:1] in {"'", '"'}:
                match = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", value)
                if not match:
                    raise DeploymentError("An ngrok setting has unmatched quotes in .env.")
                value = match.group(2)
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            values[key] = value
    for key in ("NGROK_AUTHTOKEN", "NGROK_DOMAIN"):
        if key in os.environ:
            values[key] = os.environ[key].strip()
    token = values.get("NGROK_AUTHTOKEN", "")
    if len(token) > 4096 or any(character.isspace() or character == "\0" for character in token):
        raise DeploymentError("NGROK_AUTHTOKEN must be a single token without whitespace.")
    domain = values.get("NGROK_DOMAIN", "")
    if domain:
        domain = public_host(domain if domain.startswith("https://") else "https://" + domain)
    return token, domain


def public_host(url):
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        valid_labels = all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                           for label in hostname.split("."))
        if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or not 1 <= len(hostname) <= 253 or "." not in hostname or not valid_labels
                or parsed.netloc not in {hostname, hostname + ":443"}):
            raise ValueError
        return hostname
    except ValueError:
        raise DeploymentError("Use one exact HTTPS public hostname without credentials, path or query.") from None


def run(arguments, *, timeout=30, environment=None):
    result = subprocess.run([str(item) for item in arguments], cwd=ROOT, env=environment,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=HIDDEN, timeout=timeout)
    if result.returncode:
        # Do not relay arbitrary child stderr: ngrok can echo a rejected token.
        raise DeploymentError("A deployment validation command failed; no provider request was made.")
    return result


def install():
    TOOL_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        partial = ARCHIVE.with_suffix(".zip.part")
        with urlopen(DOWNLOAD_URL, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise DeploymentError("The ngrok download differs from the pinned SHA256; installation stopped.")
        partial.replace(ARCHIVE)
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise DeploymentError("The ngrok archive differs from the pinned SHA256; installation stopped.")
    with zipfile.ZipFile(ARCHIVE) as package:
        executable = package.read("ngrok.exe")
    if not NGROK.is_file():
        temporary = NGROK.with_suffix(".exe.tmp")
        temporary.write_bytes(executable)
        temporary.replace(NGROK)
    executable_hash = hashlib.sha256(executable).hexdigest()
    if hashlib.sha256(NGROK.read_bytes()).hexdigest() != executable_hash:
        raise DeploymentError("The installed ngrok executable differs from the pinned archive.")
    environment = os.environ.copy()
    environment["MDD_NGROK_EXECUTABLE"] = str(NGROK)
    script = ("$s=Get-AuthenticodeSignature -LiteralPath $env:MDD_NGROK_EXECUTABLE; "
              "[pscustomobject]@{status=[string]$s.Status;subject=[string]$s.SignerCertificate.Subject}"
              " | ConvertTo-Json -Compress")
    signature = json.loads(run([shutil.which("pwsh") or "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                               environment=environment).stdout)
    if signature.get("status") != "Valid" or not re.search(r"(?i)\bngrok\b", signature.get("subject", "")):
        raise DeploymentError("ngrok must have a valid Authenticode signature from ngrok.")
    version = run([NGROK, "version"]).stdout.strip()
    if not version.startswith("ngrok version " + VERSION):
        raise DeploymentError("The installed ngrok version differs from the pinned version.")
    nginx_deployment.install()
    save_json(TOOL_ROOT / "download.json", {"version": VERSION, "url": DOWNLOAD_URL,
              "archive_sha256": ARCHIVE_SHA256, "executable_sha256": executable_hash,
              "authenticode_status": "Valid", "signer": signature["subject"]})
    return version


def write_agent_config(control_port):
    save_json(AGENT_CONFIG, {"version": "3", "agent": {
        "console_ui": False, "inspect_db_size": -1, "web_addr": f"127.0.0.1:{control_port}",
        "remote_management": False, "update_check": False, "log_level": "error",
        "log_format": "json", "log": "false"}})
    run([NGROK, "config", "check", "--config", AGENT_CONFIG])


def nginx_arguments(*extra):
    return [nginx_deployment.NGINX, "-p", NGINX_RUNTIME.as_posix() + "/", "-c", CONFIG.as_posix(), *extra]


def configure(hostname, ingress_port, backend_port):
    hostname = public_host("https://" + hostname)
    prefix = NGINX_RUNTIME.as_posix()
    if any(character in prefix for character in ('"', "\r", "\n", "$")):
        raise DeploymentError("The workspace path contains an unsupported Nginx character.")
    replacements = {"@@PUBLIC_HOST@@": hostname, "@@PUBLIC_HOST_REGEX@@": hostname.replace(".", "[.]"),
                    "@@INGRESS_PORT@@": str(ingress_port), "@@BACKEND_PORT@@": str(backend_port),
                    "@@RUNTIME_DIRECTORY@@": prefix}
    content = (ROOT / "deploy/nginx/ngrok.conf.template").read_text(encoding="utf-8")
    for marker, value in replacements.items():
        content = content.replace(marker, value)
    if "@@" in content:
        raise DeploymentError("The HTTPS ingress template contains unresolved placeholders.")
    for path in (CONFIG.parent, NGINX_RUNTIME / "logs", NGINX_RUNTIME / "client_body_temp",
                 NGINX_RUNTIME / "proxy_temp", NGINX_RUNTIME / "temp/client_body",
                 NGINX_RUNTIME / "temp/proxy"):
        path.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(content, encoding="utf-8")
    run(nginx_arguments("-t"))


def snapshot(process):
    return {"pid": process.pid, "created": process.create_time(), "executable": process.exe(),
            "command": process.cmdline()}


def verified(record, kind):
    if not record:
        return None
    try:
        process = psutil.Process(record["pid"])
        expected = NGROK if kind == "ngrok" else nginx_deployment.NGINX
        marker = str(AGENT_CONFIG) if kind == "ngrok" else NGINX_RUNTIME.as_posix() + "/"
        if (abs(process.create_time() - record["created"]) > .01
                or Path(process.exe()).resolve() != expected.resolve()
                or marker not in process.cmdline()
                or process.cmdline()[1:] != record["command"][1:]):
            return None
        return process
    except (psutil.Error, KeyError, TypeError, ValueError):
        return None


def listeners(port):
    return nginx_deployment.listeners(port)


def owns_port(process, port):
    candidates = {process.pid, *(child.pid for child in process.children())}
    connections = listeners(port)
    return bool(connections) and all(item.pid in candidates and item.laddr.ip == "127.0.0.1"
                                     for item in connections)


def fetch_json(url, headers=None):
    with HTTP.open(Request(url, headers=headers or {}), timeout=5) as response:
        if response.status != 200:
            raise ValueError("Unexpected status")
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Response exceeds limit")
        return json.loads(raw)


def backend_ready(port):
    try:
        data = fetch_json(f"http://127.0.0.1:{port}/api/status")
        return isinstance(data, dict) and isinstance(data.get("models"), list)
    except (OSError, ValueError):
        return False


def discover(control_port, ingress_port, requested_domain):
    # Caller verifies that our own ngrok process owns the control socket first.
    for resource, field, url_field in (("endpoints", "endpoints", "url"), ("tunnels", "tunnels", "public_url")):
        try:
            data = fetch_json(f"http://127.0.0.1:{control_port}/api/{resource}")
            for endpoint in data.get(field, []):
                upstream = (endpoint.get("upstream") or {}).get("url") if resource == "endpoints" else (endpoint.get("config") or {}).get("addr")
                if upstream not in {f"http://127.0.0.1:{ingress_port}", f"127.0.0.1:{ingress_port}"}:
                    continue
                try:
                    hostname = public_host(endpoint.get(url_field, ""))
                except DeploymentError:
                    continue
                if requested_domain and hostname != requested_domain:
                    continue
                return hostname
        except (OSError, ValueError, TypeError, AttributeError, DeploymentError):
            pass
    return None


def public_ready(hostname):
    try:
        data = fetch_json(f"https://{hostname}/api/status", {"ngrok-skip-browser-warning": "true"})
        return isinstance(data, dict) and isinstance(data.get("models"), list)
    except (OSError, ValueError):
        return False


def stop_owned(state):
    errors = []
    try:
        agent = verified(state.get("ngrok"), "ngrok")
        if agent is not None:
            agent.terminate()
            agent.wait(timeout=8)
    except Exception as error:
        errors.append("ngrok shutdown failed: " + type(error).__name__)
    try:
        master = verified(state.get("nginx"), "nginx")
        if master is not None:
            saved_pid = int((NGINX_RUNTIME / "nginx.pid").read_text(encoding="utf-8").strip())
            if saved_pid != master.pid:
                raise DeploymentError("Nginx PID file differs from the verified ingress master.")
            run(nginx_arguments("-s", "quit"))
            master.wait(timeout=8)
    except Exception as error:
        errors.append("ingress shutdown failed: " + type(error).__name__)
    return errors


def stop():
    state = read_json(STATE)
    errors = stop_owned(state)
    state.update(status="stop_incomplete" if errors else "stopped", cleanup_errors=errors)
    save_json(STATE, state)
    return state


def status():
    state = read_json(STATE)
    agent = verified(state.get("ngrok"), "ngrok")
    ingress = verified(state.get("nginx"), "nginx")
    state.update(ngrok_running=agent is not None, ingress_running=ingress is not None)
    healthy = bool(agent and ingress and public_ready(state.get("public_host", "")))
    state.update(status="ready" if healthy else "stopped_or_unhealthy", https_status=200 if healthy else None)
    return state


def start(ingress_port, control_port, backend_port):
    previous = read_json(STATE)
    if verified(previous.get("ngrok"), "ngrok") or verified(previous.get("nginx"), "nginx"):
        if (ingress_port, control_port, backend_port) != tuple(previous.get(key) for key in ("ingress_port", "control_port", "backend_port")):
            raise DeploymentError("This workspace already runs an ingress with different ports; stop it first.")
        current = status()
        if current["status"] != "ready":
            raise DeploymentError("The existing owned ingress is unhealthy; inspect Status or Stop before retrying.")
        return current
    token, domain = settings()
    if not token:
        raise DeploymentError("Set NGROK_AUTHTOKEN in the local .env file or process environment first.")
    if listeners(control_port) or listeners(ingress_port):
        raise DeploymentError("The ingress or control port is occupied; no existing process was changed.")
    backend_connections = listeners(backend_port)
    if (not backend_connections or any(item.laddr.ip != "127.0.0.1" for item in backend_connections)
            or not backend_ready(backend_port)):
        raise DeploymentError("The existing MDD backend must already be healthy and bound only to 127.0.0.1.")
    install()
    write_agent_config(control_port)
    state = {"status": "starting", "version": VERSION, "ingress_port": ingress_port,
             "control_port": control_port, "backend_port": backend_port,
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "inspection_enabled": False, "agent_logging": "disabled",
             "agent_config": str(AGENT_CONFIG), "ingress_config": str(CONFIG), "backend_owned": False}
    try:
        environment = os.environ.copy()
        environment["NGROK_AUTHTOKEN"] = token
        arguments = [str(NGROK), "http", f"http://127.0.0.1:{ingress_port}", "--inspect=false",
                     "--config", str(AGENT_CONFIG), "--name", "mdd-workspace-demo"]
        if domain:
            arguments += ["--url", "https://" + domain]
        # The agent can echo a rejected token even on fatal stderr. Neither
        # stdout nor stderr is persisted; the parent consumes only its local API.
        launched = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=HIDDEN)
        state["ngrok"] = snapshot(psutil.Process(launched.pid))
        save_json(STATE, state)
        deadline = time.monotonic() + 45
        hostname = None
        while time.monotonic() < deadline:
            if launched.poll() is not None:
                raise DeploymentError("ngrok exited before its HTTPS endpoint was ready; verify the token and account locally.")
            agent = verified(state["ngrok"], "ngrok")
            if agent and owns_port(agent, control_port):
                hostname = discover(control_port, ingress_port, domain)
                if hostname:
                    break
            time.sleep(.2)
        if not hostname:
            raise DeploymentError("ngrok did not advertise its own HTTPS endpoint within 45 seconds.")
        state.update(public_host=hostname, public_url="https://" + hostname + "/")
        # Remains closed to all public traffic until this exact hostname is known.
        configure(hostname, ingress_port, backend_port)
        launched_ingress = subprocess.Popen([str(item) for item in nginx_arguments()], cwd=ROOT,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, creationflags=HIDDEN)
        state["nginx"] = snapshot(psutil.Process(launched_ingress.pid))
        save_json(STATE, state)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            ingress = verified(state["nginx"], "nginx")
            if ingress and owns_port(ingress, ingress_port):
                break
            if launched_ingress.poll() is not None:
                raise DeploymentError("The isolated Nginx ingress failed to start; inspect its error.log.")
            time.sleep(.15)
        else:
            raise DeploymentError("The isolated Nginx ingress did not bind its loopback port.")
        deadline = time.monotonic() + 30
        while not public_ready(hostname):
            if time.monotonic() >= deadline:
                raise DeploymentError("The HTTPS endpoint did not return the MDD status response; no paid API was called.")
            time.sleep(.3)
        state.update(status="ready", https_status=200)
        save_json(STATE, state)
        return state
    except Exception as error:
        errors = stop_owned(state)
        message = str(error) if isinstance(error, DeploymentError) else "Startup failed: " + type(error).__name__
        state.update(status="error", message=message, cleanup_errors=errors)
        save_json(STATE, state)
        raise DeploymentError(message) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("setup", "start", "status", "stop"), default="start")
    parser.add_argument("--ingress-port", type=int, default=8081)
    parser.add_argument("--control-port", type=int, default=4040)
    parser.add_argument("--backend-port", type=int, default=8765)
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("This helper manages native Windows ngrok and Nginx.")
    ports = (args.ingress_port, args.control_port, args.backend_port)
    if len(set(ports)) != 3 or any(not 1024 <= port <= 65535 for port in ports):
        parser.error("Use three distinct ports between 1024 and 65535.")
    if args.action == "setup":
        version = install()
        write_agent_config(args.control_port)
        result = {"status": "configured", "version": version, "program": str(NGROK),
                  "agent_config": str(AGENT_CONFIG), "inspection_enabled": False,
                  "ingress_config_pending": "The exact public hostname is discovered at Start."}
    elif args.action == "start":
        result = start(*ports)
    else:
        result = stop() if args.action == "stop" else status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except DeploymentError as error:
        print(f"ngrok deployment failed: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"ngrok deployment failed ({type(error).__name__}); no child output or credentials were logged.", file=sys.stderr)
        sys.exit(1)
