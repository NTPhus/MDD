"""Exercise the rendered Nginx configuration against a guarded loopback backend.

Run: python -m unittest discover -s tests -p test_nginx_deployment.py -v
Set MDD_NGINX_EXECUTABLE to test another installed Nginx binary. No installer,
application server, provider, administrator port, or external network is used.
"""

from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy/nginx/nginx.conf.template"
MAX_BODY_BYTES = 16 * 1024 * 1024
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_nginx():
    configured = os.environ.get("MDD_NGINX_EXECUTABLE")
    if configured:
        executable = shutil.which(configured) or configured
        if not Path(executable).is_file():
            raise RuntimeError("MDD_NGINX_EXECUTABLE does not identify an installed binary")
        return str(Path(executable).resolve())
    installed = sorted((ROOT / ".tools/nginx").glob("nginx-*/nginx.exe"), reverse=True) if os.name == "nt" else []
    if installed:
        return str(installed[0].resolve())
    return shutil.which("nginx")


def render_template(template, public_port, backend_port, runtime, authority_port_regex=None):
    replacements = {
        "@@LAN_ADDRESS@@": "127.0.0.2",
        "@@LAN_ADDRESS_REGEX@@": "127[.]0[.]0[.]2",
        "@@PUBLIC_PORT@@": str(public_port),
        "@@AUTHORITY_PORT_REGEX@@": authority_port_regex or f":{public_port}",
        "@@BACKEND_PORT@@": str(backend_port),
        "@@RUNTIME_DIRECTORY@@": Path(runtime).resolve().as_posix().rstrip("/"),
    }
    result = template
    for token, value in replacements.items():
        result = result.replace(token, value)
    if re.search(r"@@[A-Z_]+@@", result):
        raise AssertionError("Rendered Nginx configuration contains unresolved placeholders")
    return result


class GuardedBackend(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        self.paid_calls = 0
        self.response_status = 200
        super().__init__(("127.0.0.1", 0), GuardedHandler)

    @property
    def authority(self):
        return f"127.0.0.1:{self.server_port}"

    def reset(self):
        with self.lock:
            self.requests.clear()
            self.paid_calls = 0
            self.response_status = 200

    def snapshot(self):
        with self.lock:
            return list(self.requests), self.paid_calls


class GuardedHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_arguments):
        pass

    def dispatch(self):
        self.connection.settimeout(3)
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        record = {"method": self.command, "path": self.path, "hosts": hosts,
                  "origins": origins, "headers": list(self.headers.raw_items()), "body": b""}
        status = 200
        if hosts != [self.server.authority]:
            status = 403
        elif origins and origins != [f"http://{self.server.authority}"]:
            status = 403
        elif self.headers.get("Sec-Fetch-Site", "") == "cross-site":
            status = 403
        elif self.headers.get("Transfer-Encoding") is not None:
            status = 400
        else:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                status = 413
            elif length:
                record["body"] = self.rfile.read(length)
        with self.server.lock:
            self.server.requests.append(record)
            if status == 200 and self.path == "/api/feedback":
                self.server.paid_calls += 1
                status = self.server.response_status
            elif status == 200 and self.path not in {"/", "/api/status", "/api/analyze"}:
                status = 404
        body = json.dumps({"from": "guarded-loopback-stub", "path": self.path, "status": status}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = dispatch
    do_POST = dispatch


@contextmanager
def reserved_public_port():
    # Check both loopback listeners, then release before nginx -t: the Windows
    # binary binds listening sockets even during its configuration check.
    with socket.socket() as local, socket.socket() as lan:
        local.bind(("127.0.0.1", 0))
        port = local.getsockname()[1]
        lan.bind(("127.0.0.2", port))
        yield port


class NginxInstance:
    def __init__(self, executable, workspace, name, backend, authority_port_regex=None):
        self.executable = executable
        self.prefix = (workspace / name).resolve()
        self.prefix.mkdir()
        self.runtime = self.prefix / "runtime"
        self.runtime.mkdir()
        (self.prefix / "temp").mkdir()
        for name in ("logs", "client_body_temp", "proxy_temp", "fastcgi_temp", "uwsgi_temp", "scgi_temp"):
            (self.prefix / name).mkdir()
            (self.runtime / name).mkdir()
            (self.prefix / "temp" / name).mkdir()
        self.config = self.prefix / "nginx.conf"
        self.process = None
        self.output = None
        self.pid_file = self.runtime / "nginx.pid"
        with reserved_public_port() as port:
            self.port = port
        self.rendered = render_template(TEMPLATE.read_text(encoding="utf-8"), port, backend.server_port,
                                        self.runtime, authority_port_regex)
        self.config.write_text(self.rendered, encoding="utf-8")
        pid_directive = re.search(r'^\s*pid\s+"?([^";\n]+)"?\s*;', self.rendered, re.MULTILINE)
        if pid_directive:
            self.pid_file = Path(pid_directive[1].strip())
            if not self.pid_file.is_absolute():
                self.pid_file = self.prefix / self.pid_file
        if not self.pid_file.resolve().is_relative_to(self.prefix):
            raise RuntimeError("The test Nginx PID file must stay inside its own prefix")
        self.command = [self.executable, "-p", self.prefix.as_posix() + "/", "-c", self.config.as_posix()]
        checked = subprocess.run([*self.command, "-t"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=10,
                                 creationflags=CREATE_NO_WINDOW)
        if checked.returncode:
            raise AssertionError(f"Real nginx rejected the test configuration:\n{checked.stdout}{checked.stderr}")
        # A fixture canary detects accidental use of Nginx's default static root.
        (self.prefix / "html").mkdir()
        (self.prefix / "html/.env").write_text("NGINX_STATIC_CANARY_MUST_NOT_BE_SERVED", encoding="utf-8")
        arguments = self.command
        if not re.search(r"^\s*daemon\s+", self.rendered, re.MULTILINE):
            arguments = [*arguments, "-g", "daemon off;"]
        self.output = (self.prefix / "process-output.log").open("wb")
        self.process = subprocess.Popen(arguments, cwd=self.prefix, stdout=self.output, stderr=subprocess.STDOUT,
                                        creationflags=CREATE_NO_WINDOW, start_new_session=os.name != "nt")
        try:
            self.wait_until_ready()
        except BaseException:
            self.stop()
            raise

    def wait_until_ready(self):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"Test nginx exited during startup: {self.diagnostics()}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise AssertionError(f"Test nginx did not listen on its reserved high port: {self.diagnostics()}")

    def diagnostics(self):
        logs = []
        for directory in (self.prefix, self.runtime):
            for filename in directory.rglob("*.log"):
                logs.append(filename.read_text(encoding="utf-8", errors="replace")[-3000:])
        return "\n".join(logs)

    def stop(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    subprocess.run([*self.command, "-s", "quit"], cwd=self.prefix, capture_output=True,
                                   timeout=5, creationflags=CREATE_NO_WINDOW, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    self.kill_own_process_tree()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.kill_own_process_tree()
        finally:
            if self.output:
                self.output.close()

    def kill_own_process_tree(self):
        # Popen.poll checks the handle for this process; do not select by image
        # name, scan running servers, or signal another deployment's PID file.
        if self.process.poll() is not None:
            return
        pid = self.process.pid
        if not isinstance(pid, int) or pid <= 1:
            raise RuntimeError("Refusing cleanup without the test-owned process PID")
        if self.pid_file.exists() and int(self.pid_file.read_text().strip()) != pid:
            raise RuntimeError("Refusing process-tree cleanup: the fixture PID does not match Popen")
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           timeout=5, creationflags=CREATE_NO_WINDOW, check=False)
        else:
            if os.getpgid(pid) != pid:
                raise RuntimeError("Refusing cleanup of a process group not created by this test")
            os.killpg(pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
        self.process.wait(timeout=5)


class NginxTemplateTests(unittest.TestCase):
    def test_rendering_resolves_every_placeholder_for_high_and_default_authority_ports(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        for authority in (":48937", "(?::80)?"):
            with self.subTest(authority=authority):
                rendered = render_template(template, 48937, 48938, ROOT / ".tools/nginx-test-runtime", authority)
                self.assertNotRegex(rendered, r"@@[A-Z_]+@@")
                self.assertIn(authority, rendered)
                self.assertNotIn("\\", rendered)

    def test_proxy_and_request_limits_are_explicit(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        # These checks supplement actual wire tests below; they do not emulate
        # map matching or replace nginx -t and the real request path.
        for directive in ("proxy_request_buffering off;", "proxy_buffering off;", "proxy_next_upstream off;",
                          "proxy_cache off;", "proxy_read_timeout 1500s;", "client_max_body_size 16m;"):
            with self.subTest(directive=directive):
                self.assertIn(directive, template)
        self.assertNotRegex(template, r"(?m)^\s*(?:root|alias)\s+")


@unittest.skipUnless(find_nginx(), "Install Nginx or set MDD_NGINX_EXECUTABLE to run real proxy integration tests")
class NginxDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".mdd-nginx-test-", dir=ROOT)
        cls.workspace = Path(cls.temporary.name).resolve()
        cls.addClassCleanup(cls.remove_workspace)
        cls.backend = GuardedBackend()
        cls.thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.stop_backend)
        cls.nginx = NginxInstance(find_nginx(), cls.workspace, "primary", cls.backend)
        cls.addClassCleanup(cls.nginx.stop)

    @classmethod
    def stop_backend(cls):
        cls.backend.shutdown()
        cls.backend.server_close()
        cls.thread.join(timeout=5)

    @classmethod
    def remove_workspace(cls):
        if cls.workspace.parent != ROOT.resolve() or not cls.workspace.name.startswith(".mdd-nginx-test-"):
            raise RuntimeError("Refusing recursive cleanup outside the explicitly created test workspace")
        cls.temporary.cleanup()

    def setUp(self):
        self.backend.reset()

    def request(self, host=None, origin=None, *, method="POST", path="/api/feedback", headers=(),
                body=b'{"analysis_id":"saved-analysis","include_voice":true}', address="127.0.0.1", instance=None):
        instance = instance or self.nginx
        connection = HTTPConnection(address, instance.port, timeout=5)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            connection.putheader("Host", host or f"127.0.0.1:{instance.port}")
            if origin is not None:
                connection.putheader("Origin", origin)
            if method == "POST":
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(len(body)))
            for key, value in headers:
                connection.putheader(key, value)
            connection.endheaders(body if method == "POST" else None)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def assert_no_backend_request(self):
        requests, paid_calls = self.backend.snapshot()
        self.assertEqual(requests, [], "The front proxy must reject this request before contacting the backend")
        self.assertEqual(paid_calls, 0)

    def test_each_trusted_host_and_same_origin_reaches_strict_internal_guard_without_changing_body(self):
        body = '{\n "analysis_id": "saved-✓", "include_voice": false\n}\n'.encode()
        for hostname in ("127.0.0.1", "localhost", "127.0.0.2"):
            with self.subTest(hostname=hostname):
                self.backend.reset()
                host = f"{hostname}:{self.nginx.port}"
                status, result = self.request(host, f"http://{host}", body=body,
                                              address="127.0.0.2" if hostname == "127.0.0.2" else "127.0.0.1")
                self.assertEqual(status, 200, result)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual(len(requests), 1)
                self.assertEqual(paid_calls, 1)
                self.assertEqual(requests[0]["hosts"], [self.backend.authority])
                self.assertEqual(requests[0]["origins"], [f"http://{self.backend.authority}"])
                self.assertEqual(requests[0]["body"], body)

    def test_missing_and_empty_origin_are_forwarded_as_absent(self):
        # Native $http_origin cannot distinguish an empty field from absence.
        # Both cases intentionally remove Origin instead of manufacturing one.
        for origin in (None, ""):
            with self.subTest(origin=origin):
                self.backend.reset()
                status, body = self.request(origin=origin)
                self.assertEqual(status, 200, body)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual(requests[0]["origins"], [])
                self.assertEqual(paid_calls, 1)

    def test_foreign_host_and_origin_variants_are_rejected_before_provider_calls(self):
        port = self.nginx.port
        cases = [
            (f"attacker.example:{port}", None),
            (f"localhost.attacker.example:{port}", None),
            (f"127.0.0.2.attacker.example:{port}", None),
            (f"127x0x0x2:{port}", None),
            (f"LOCALHOST:{port}", None),
            ("localhost", None),
            (f"localhost:{port + 1}", None),
            (f"localhost:{port}", "null"),
            (f"localhost:{port}", f"https://localhost:{port}"),
            (f"localhost:{port}", f"HTTP://localhost:{port}"),
            (f"localhost:{port}", f"http://LOCALHOST:{port}"),
            (f"localhost:{port}", f"http://attacker.example:{port}"),
            (f"localhost:{port}", f"http://127.0.0.1:{port}"),
            (f"127.0.0.2:{port}", f"http://localhost:{port}"),
            (f"localhost:{port}", f"http://localhost:{port}/path"),
            (f"localhost:{port}", f"http://localhost:{port}@attacker.example"),
            (f"localhost:{port}", f"http://localhost:{port}, http://attacker.example"),
        ]
        for host, origin in cases:
            with self.subTest(host=host, origin=origin):
                self.backend.reset()
                status, body = self.request(host, origin)
                self.assertEqual(status, 403, body)
                self.assert_no_backend_request()

    def test_duplicate_headers_and_cross_site_context_are_rejected_before_provider_calls(self):
        host = f"localhost:{self.nginx.port}"
        cases = [
            (None, [("Host", host)], True),
            (f"http://{host}", [("Origin", f"http://{host}")], True),
            (f"http://{host}", [("Origin", "http://attacker.example")], True),
            (f"http://{host}", [("Sec-Fetch-Site", "cross-site")], False),
        ]
        for origin, headers, rejected_at_edge in cases:
            with self.subTest(origin=origin, headers=headers):
                self.backend.reset()
                status, body = self.request(host, origin, headers=headers)
                self.assertIn(status, (400, 403), body)
                if rejected_at_edge:
                    self.assert_no_backend_request()
                else:
                    # Preserve browser context so the backend's existing guard
                    # can reject cross-site requests before any provider action.
                    requests, paid_calls = self.backend.snapshot()
                    self.assertEqual(paid_calls, 0)
                    self.assertEqual(len(requests), 1)
                    self.assertIn(("Sec-Fetch-Site", "cross-site"), requests[0]["headers"])

    def test_upstream_503_is_returned_after_exactly_one_provider_attempt(self):
        self.backend.response_status = 503
        status, body = self.request()
        self.assertEqual(status, 503, body)
        requests, paid_calls = self.backend.snapshot()
        self.assertEqual(len(requests), 1)
        self.assertEqual(paid_calls, 1)
        self.assertEqual(json.loads(body)["from"], "guarded-loopback-stub")

    def test_repository_and_static_paths_are_forwarded_to_backend_instead_of_served_from_disk(self):
        for path in ("/", "/.env", "/.git/config", "/src/mdd/demo.py", "/data/protocol/l2arctic-v1/splits.json",
                     "/runs/any/checkpoint/model.safetensors"):
            with self.subTest(path=path):
                self.backend.reset()
                status, body = self.request(method="GET", path=path)
                self.assertEqual(status, 200 if path == "/" else 404, body)
                self.assertEqual(json.loads(body)["from"], "guarded-loopback-stub")
                self.assertNotIn(b"NGINX_STATIC_CANARY", body)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual([request["path"] for request in requests], [path])
                self.assertEqual(paid_calls, 0)

    def raw_request(self, payload):
        with socket.create_connection(("127.0.0.1", self.nginx.port), timeout=5) as connection:
            connection.sendall(payload)
            response = b""
            while b"\r\n" not in response:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response += chunk
        self.assertTrue(response.startswith(b"HTTP/1."), response)
        return int(response.split(b" ", 2)[1])

    def test_chunked_request_is_rejected_before_reaching_backend(self):
        payload = (f"POST /api/feedback HTTP/1.1\r\nHost: localhost:{self.nginx.port}\r\n"
                   "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                   "2\r\n{}\r\n0\r\n\r\n").encode()
        self.assertEqual(self.raw_request(payload), 400)
        self.assert_no_backend_request()

    def test_oversized_content_length_is_rejected_without_uploading_the_body(self):
        payload = (f"POST /api/analyze HTTP/1.1\r\nHost: localhost:{self.nginx.port}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {MAX_BODY_BYTES + 1}\r\n"
                   "Connection: close\r\n\r\n").encode()
        self.assertEqual(self.raw_request(payload), 413)
        self.assert_no_backend_request()

    def test_default_port_authorities_are_canonicalized_without_binding_port_80(self):
        # Use a separate high listening port while exercising the production
        # default-80 authority expression. No test ever binds an admin port.
        instance = NginxInstance(find_nginx(), self.workspace, "default-authority", self.backend, "(?::80)?")
        try:
            for hostname in ("127.0.0.1", "localhost", "127.0.0.2"):
                for host_suffix, origin_suffix in (("", ""), (":80", ":80"), ("", ":80"), (":80", "")):
                    with self.subTest(hostname=hostname, host_suffix=host_suffix, origin_suffix=origin_suffix):
                        self.backend.reset()
                        status, body = self.request(f"{hostname}{host_suffix}", f"http://{hostname}{origin_suffix}", instance=instance)
                        self.assertEqual(status, 200, body)
                        requests, paid_calls = self.backend.snapshot()
                        self.assertEqual(requests[0]["hosts"], [self.backend.authority])
                        self.assertEqual(requests[0]["origins"], [f"http://{self.backend.authority}"])
                        self.assertEqual(paid_calls, 1)
        finally:
            instance.stop()


if __name__ == "__main__":
    unittest.main()
