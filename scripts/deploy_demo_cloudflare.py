#!/usr/bin/env python3
"""Manage a temporary Cloudflare Quick Tunnel for the local MDD demo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import psutil

import deploy_demo_ngrok as ngrok_deployment
import deploy_demo_nginx as nginx_deployment

ROOT = Path(__file__).resolve().parents[1]
CLOUDFLARED = ROOT / ".tools/cloudflared/cloudflared.exe"
EXPECTED_SHA256 = "76b2805253d0d6ffacb092a7acfbee99e7af6917835c0c5410e23f3ef5052da2"
VERSION = "2026.10.3"
RUNTIME = ROOT / "data/generated/demo/cloudflare"
NGINX_RUNTIME = RUNTIME / "nginx"
CONFIG = NGINX_RUNTIME / "conf/nginx.conf"
STATE = RUNTIME / "deployment.json"
LOG = RUNTIME / "cloudflared.log"
INGRESS_PORT = 8082
BACKEND_PORT = 8765
HIDDEN = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
DETACHED = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NEW_PROCESS_GROUP
    | subprocess.CREATE_NO_WINDOW
    | CREATE_BREAKAWAY_FROM_JOB
    if sys.platform == "win32"
    else 0
)
HTTP = build_opener(ProxyHandler({}))


class DeploymentError(RuntimeError):
    pass


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(arguments, timeout=20):
    result = subprocess.run([str(item) for item in arguments], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout,
                            creationflags=HIDDEN)
    if result.returncode:
        raise DeploymentError("Cloudflare/Nginx validation command failed.")
    return result


def nginx_arguments(*extra):
    return [nginx_deployment.NGINX, "-p", NGINX_RUNTIME.as_posix() + "/", "-c", CONFIG.as_posix(), *extra]


def snapshot(process):
    return {"pid": process.pid, "created": process.create_time(), "executable": process.exe(),
            "command": process.cmdline()}


def verified(record, kind):
    if not isinstance(record, dict):
        return None
    try:
        process = psutil.Process(int(record["pid"]))
        expected = CLOUDFLARED if kind == "cloudflared" else nginx_deployment.NGINX
        marker = "tunnel" if kind == "cloudflared" else NGINX_RUNTIME.as_posix() + "/"
        if (abs(process.create_time() - float(record["created"])) > .01
                or Path(process.exe()).resolve() != expected.resolve()
                or marker not in process.cmdline()
                or process.cmdline() != record.get("command")):
            return None
        return process
    except (psutil.Error, KeyError, TypeError, ValueError, OSError):
        return None


def nginx_master():
    path = NGINX_RUNTIME / "nginx.pid"
    try:
        return psutil.Process(int(path.read_text(encoding="ascii").strip()))
    except (OSError, ValueError, psutil.Error):
        return None


def configure(hostname=None):
    NGINX_RUNTIME.mkdir(parents=True, exist_ok=True)
    for name in ("conf", "logs", "client_body_temp", "proxy_temp", "temp/client_body", "temp/proxy"):
        (NGINX_RUNTIME / name).mkdir(parents=True, exist_ok=True)
    prefix = NGINX_RUNTIME.as_posix()
    if any(c in prefix for c in ('"', "$", "\r", "\n", ";")):
        raise DeploymentError("Workspace path contains a character unsupported by the Nginx config.")
    common = f'''worker_processes 1;
pid "{prefix}/nginx.pid";
error_log "{prefix}/error.log" warn;
events {{ worker_connections 256; }}
http {{
  server_tokens off;
  access_log off;
  client_body_temp_path "{prefix}/client_body_temp";
  proxy_temp_path "{prefix}/proxy_temp";
'''
    if hostname is None:
        body = f'''  server {{
    listen 127.0.0.1:{INGRESS_PORT} default_server;
    server_name _;
    return 404;
  }}
}}
'''
    else:
        if not re.fullmatch(r"[a-z0-9-]+\.trycloudflare\.com", hostname):
            raise DeploymentError("Cloudflare Quick Tunnel returned an unexpected hostname.")
        exact = re.escape(hostname)
        body = f'''  map $http_host $mdd_host_allowed {{
    default 0;
    ~^{exact}(?::443)?$ 1;
  }}
  map "$http_host|$http_origin" $mdd_origin_allowed {{
    default 0;
    ~^{exact}(?::443)?[|]$ 1;
    ~^{exact}(?::443)?[|]https://{exact}(?::443)?$ 1;
  }}
  map $http_origin $mdd_backend_origin {{
    "" "";
    default "http://127.0.0.1:{BACKEND_PORT}";
  }}
  server {{
    listen 127.0.0.1:{INGRESS_PORT} default_server;
    server_name {hostname};
    if ($mdd_host_allowed = 0) {{ return 403; }}
    if ($mdd_origin_allowed = 0) {{ return 403; }}
    if ($http_transfer_encoding != "") {{ return 400; }}
    client_max_body_size 16m;
    client_header_timeout 15s;
    client_body_timeout 60s;
    keepalive_timeout 30s;
    send_timeout 60s;
    location / {{
      proxy_pass http://127.0.0.1:{BACKEND_PORT};
      proxy_http_version 1.1;
      proxy_set_header Host "127.0.0.1:{BACKEND_PORT}";
      proxy_set_header Origin $mdd_backend_origin;
      proxy_set_header Sec-Fetch-Site $http_sec_fetch_site;
      proxy_set_header Connection "";
      proxy_set_header Forwarded "";
      proxy_set_header X-Forwarded-Host "";
      proxy_set_header X-Forwarded-Proto "";
      proxy_set_header X-Forwarded-For "";
      proxy_connect_timeout 5s;
      proxy_send_timeout 60s;
      proxy_read_timeout 1500s;
      proxy_request_buffering off;
      proxy_buffering off;
      proxy_max_temp_file_size 0;
      proxy_cache off;
      proxy_store off;
      proxy_intercept_errors off;
      proxy_redirect off;
    }}
  }}
}}
'''
    CONFIG.write_text(common + body, encoding="utf-8")
    run(nginx_arguments("-t"))


def http_status(url):
    try:
        request = Request(url, headers={"User-Agent": "MDD-Cloudflare-health-check/1"})
        with HTTP.open(request, timeout=5) as response:
            return response.status
    except Exception:
        return None


def parse_url(text):
    matches = re.findall(r"https://([a-z0-9-]+\.trycloudflare\.com)", text)
    return matches[-1] if matches else None


def stop_owned(state):
    errors = []
    tunnel = verified(state.get("cloudflared"), "cloudflared")
    if tunnel:
        try:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        except (psutil.Error, psutil.TimeoutExpired):
            errors.append("cloudflared_shutdown_failed")
    master = verified(state.get("nginx"), "nginx")
    if master:
        try:
            pid = int((NGINX_RUNTIME / "nginx.pid").read_text(encoding="ascii").strip())
            if pid != master.pid:
                raise DeploymentError("Nginx PID file changed; refusing to stop a different process.")
            run(nginx_arguments("-s", "quit"))
            master.wait(timeout=10)
        except Exception:
            errors.append("nginx_shutdown_failed")
    return errors


def status():
    state = read_json(STATE)
    tunnel = verified(state.get("cloudflared"), "cloudflared")
    master = verified(state.get("nginx"), "nginx")
    code = http_status(state.get("public_url", "") + "api/status") if tunnel and master and state.get("public_url") else None
    state.update(cloudflared_running=bool(tunnel), ingress_running=bool(master), https_status=code,
                 status="ready" if tunnel and master and code == 200 else "stopped_or_unhealthy")
    return state


def stop():
    state = read_json(STATE)
    errors = stop_owned(state)
    state.update(status="stop_incomplete" if errors else "stopped", cleanup_errors=errors)
    save_json(STATE, state)
    return state


def retire_ngrok(state):
    old = ngrok_deployment.read_json(ngrok_deployment.STATE)
    if ngrok_deployment.verified(old.get("ngrok"), "ngrok"):
        subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8",
                        str(ROOT / "scripts/deploy_demo_ngrok.py"), "--action", "stop"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=20, creationflags=HIDDEN)
        old = ngrok_deployment.read_json(ngrok_deployment.STATE)
    state["old_ngrok_stopped"] = ngrok_deployment.verified(old.get("ngrok"), "ngrok") is None
    state["old_nginx_ingress_stopped"] = ngrok_deployment.verified(old.get("nginx"), "nginx") is None
    save_json(STATE, state)
    return state


def start():
    previous = read_json(STATE)
    if verified(previous.get("cloudflared"), "cloudflared") or verified(previous.get("nginx"), "nginx"):
        current = status()
        if current["status"] == "ready":
            return retire_ngrok(current)
        raise DeploymentError("An owned Cloudflare deployment is already present but unhealthy; inspect Status or Stop first.")
    if not CLOUDFLARED.is_file() or hashlib.sha256(CLOUDFLARED.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise DeploymentError("cloudflared is missing or does not match the pinned official SHA256.")
    version = run([CLOUDFLARED, "--version"]).stdout.strip()
    if VERSION not in version:
        raise DeploymentError("cloudflared version differs from the pinned release.")
    backend = nginx_deployment.listeners(BACKEND_PORT)
    if not backend or any(item.laddr.ip != "127.0.0.1" for item in backend):
        raise DeploymentError("The MDD backend must already be running on 127.0.0.1:8765.")
    if http_status(f"http://127.0.0.1:{BACKEND_PORT}/api/status") != 200:
        raise DeploymentError("The MDD backend health check failed.")
    if nginx_deployment.listeners(INGRESS_PORT):
        raise DeploymentError("Cloudflare ingress port 8082 is occupied; no process was changed.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    configure()
    launched_nginx = subprocess.Popen(nginx_arguments(), cwd=ROOT, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      creationflags=DETACHED)
    deadline = time.monotonic() + 8
    master = None
    while time.monotonic() < deadline:
        master = nginx_master()
        if master and nginx_deployment.listeners(INGRESS_PORT):
            break
        if launched_nginx.poll() is not None:
            raise DeploymentError("Cloudflare Nginx bootstrap failed; inspect its error.log.")
        time.sleep(.1)
    if not master:
        raise DeploymentError("Cloudflare Nginx bootstrap did not start.")
    state = {"status": "starting", "version": VERSION, "ingress_port": INGRESS_PORT,
             "backend_port": BACKEND_PORT, "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "public_access": "Anyone with the temporary URL can reach this demo.",
             "nginx": snapshot(master), "backend_owned": False}
    save_json(STATE, state)
    try:
        with LOG.open("w", encoding="utf-8") as log:
            tunnel_proc = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate",
                                            "--loglevel", "info", "--logfile", str(LOG),
                                            "--url", f"http://127.0.0.1:{INGRESS_PORT}"],
                                           cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, creationflags=DETACHED)
        state["cloudflared"] = snapshot(psutil.Process(tunnel_proc.pid))
        save_json(STATE, state)
        deadline = time.monotonic() + 60
        hostname = None
        while time.monotonic() < deadline:
            if tunnel_proc.poll() is not None:
                raise DeploymentError("cloudflared exited before it created a Quick Tunnel.")
            try:
                hostname = parse_url(LOG.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
            if hostname:
                break
            time.sleep(.25)
        if not hostname:
            raise DeploymentError("Timed out waiting for the Cloudflare trycloudflare.com URL.")
        configure(hostname)
        run(nginx_arguments("-s", "reload"))
        public_url = f"https://{hostname}/"
        deadline = time.monotonic() + 45
        code = None
        while time.monotonic() < deadline:
            code = http_status(public_url + "api/status")
            if code == 200:
                break
            time.sleep(.5)
        if code != 200:
            raise DeploymentError("Cloudflare URL did not return the MDD health response; inspect the tunnel log.")
        state.update(status="ready", public_host=hostname, public_url=public_url, https_status=200)
        save_json(STATE, state)
        # Retire the prior public tunnel only after Cloudflare has passed the health check.
        return retire_ngrok(state)
    except Exception as error:
        cleanup = stop_owned(state)
        state.update(status="error", message=(str(error) if isinstance(error, DeploymentError)
                                               else "Cloudflare startup failed: " + type(error).__name__),
                     cleanup_errors=cleanup)
        save_json(STATE, state)
        raise DeploymentError(state["message"]) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("start", "status", "stop"), default="start")
    args = parser.parse_args()
    try:
        result = start() if args.action == "start" else status() if args.action == "status" else stop()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except DeploymentError as error:
        print("Cloudflare deployment failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
