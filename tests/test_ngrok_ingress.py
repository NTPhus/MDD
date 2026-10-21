"""Test the HTTPS ngrok ingress with real Nginx and an isolated loopback stub.

Run: python -m unittest discover -s tests -p test_ngrok_ingress.py -v
No ngrok process, tunnel, public network request, or provider is used.
"""

import json
from pathlib import Path
import re
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import test_nginx_deployment as nginx_fixture


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy/nginx/ngrok.conf.template"
PUBLIC_HOST = "mdd-example.ngrok-free.app"


def render_template(template, ingress_port, backend_port, runtime, _authority_port_regex=None):
    replacements = {
        "@@PUBLIC_HOST@@": PUBLIC_HOST,
        "@@PUBLIC_HOST_REGEX@@": PUBLIC_HOST.replace(".", "[.]"),
        "@@INGRESS_PORT@@": str(ingress_port),
        "@@BACKEND_PORT@@": str(backend_port),
        "@@RUNTIME_DIRECTORY@@": Path(runtime).resolve().as_posix().rstrip("/"),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise AssertionError("Rendered ngrok ingress contains unresolved placeholders")
    return template


class NgrokInstance(nginx_fixture.NginxInstance):
    def __init__(self, executable, workspace, backend):
        # Reuse the checked PID, hidden process, native -t and cleanup lifecycle.
        # Only the renderer and input template change during construction; the
        # original fixture module and its files are restored immediately.
        with patch.object(nginx_fixture, "TEMPLATE", TEMPLATE), patch.object(nginx_fixture, "render_template", render_template):
            super().__init__(executable, workspace, "ngrok-ingress", backend)


class NgrokTemplateTests(unittest.TestCase):
    def test_only_five_documented_placeholders_are_rendered(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r"@@[A-Z_]+@@", template)), {
            "@@PUBLIC_HOST@@", "@@PUBLIC_HOST_REGEX@@", "@@INGRESS_PORT@@",
            "@@BACKEND_PORT@@", "@@RUNTIME_DIRECTORY@@",
        })
        rendered = render_template(template, 48937, 48938, ROOT / ".tools/ngrok-test-runtime")
        self.assertNotIn("@@", rendered)
        self.assertNotIn("\\", rendered)

    def test_single_loopback_listener_and_proxy_limits_are_explicit(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        listeners = re.findall(r"(?m)^\s*listen\s+([^;]+);", template)
        self.assertEqual([listener.split()[0] for listener in listeners], ["127.0.0.1:@@INGRESS_PORT@@"])
        self.assertNotRegex(template, r"(?m)^\s*(?:root|alias)\s+")
        for directive in ("proxy_request_buffering off;", "proxy_buffering off;", "proxy_next_upstream off;",
                          "proxy_cache off;", "proxy_store off;", "proxy_read_timeout 1500s;", "client_max_body_size 16m;"):
            with self.subTest(directive=directive):
                self.assertIn(directive, template)


@unittest.skipUnless(nginx_fixture.find_nginx(), "Install Nginx or set MDD_NGINX_EXECUTABLE to run native ingress tests")
class NgrokIngressTests(unittest.TestCase):
    request = nginx_fixture.NginxDeploymentTests.request
    raw_request = nginx_fixture.NginxDeploymentTests.raw_request
    assert_no_backend_request = nginx_fixture.NginxDeploymentTests.assert_no_backend_request

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".mdd-ngrok-test-", dir=ROOT)
        cls.workspace = Path(cls.temporary.name).resolve()
        cls.addClassCleanup(cls.remove_workspace)
        cls.backend = nginx_fixture.GuardedBackend()
        cls.thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.stop_backend)
        cls.nginx = NgrokInstance(nginx_fixture.find_nginx(), cls.workspace, cls.backend)
        cls.addClassCleanup(cls.stop_nginx)

    @classmethod
    def stop_nginx(cls):
        cls.nginx.stop()
        if cls.nginx.process.poll() is None:
            raise AssertionError("The test-owned Nginx process did not stop")
        for address in ("127.0.0.1", "127.0.0.2"):
            try:
                connection = socket.create_connection((address, cls.nginx.port), timeout=0.3)
            except OSError:
                continue
            connection.close()
            raise AssertionError(f"A listener remains on the test-owned high port at {address}")
        cls.server_stopped = True

    @classmethod
    def stop_backend(cls):
        cls.backend.shutdown()
        cls.backend.server_close()
        cls.thread.join(timeout=5)

    @classmethod
    def remove_workspace(cls):
        if cls.workspace.parent != ROOT.resolve() or not cls.workspace.name.startswith(".mdd-ngrok-test-"):
            raise RuntimeError("Refusing recursive cleanup outside the explicitly created ngrok test workspace")
        cls.temporary.cleanup()

    def setUp(self):
        self.backend.reset()

    def test_https_public_origin_and_all_default_443_variants_rewrite_headers_without_changing_body(self):
        body = '{\n "analysis_id": "saved-✓", "include_voice": false\n}\n'.encode()
        for host_suffix, origin_suffix in (("", ""), (":443", ":443"), ("", ":443"), (":443", "")):
            with self.subTest(host_suffix=host_suffix, origin_suffix=origin_suffix):
                self.backend.reset()
                status, result = self.request(PUBLIC_HOST + host_suffix, "https://" + PUBLIC_HOST + origin_suffix, body=body)
                self.assertEqual(status, 200, result)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual(len(requests), 1)
                self.assertEqual(paid_calls, 1)
                self.assertEqual(requests[0]["hosts"], [self.backend.authority])
                self.assertEqual(requests[0]["origins"], ["http://" + self.backend.authority])
                self.assertEqual(requests[0]["body"], body)

    def test_missing_and_empty_origin_are_omitted_instead_of_manufactured(self):
        # Nginx's native $http_origin treats an empty field like an absent one.
        for origin in (None, ""):
            with self.subTest(origin=origin):
                self.backend.reset()
                status, body = self.request(PUBLIC_HOST, origin)
                self.assertEqual(status, 200, body)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual(requests[0]["origins"], [])
                self.assertEqual(paid_calls, 1)

    def test_external_forwarding_headers_are_removed_before_upstream(self):
        names = ("Forwarded", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Forwarded-For")
        status, body = self.request(PUBLIC_HOST, "https://" + PUBLIC_HOST, headers=[(name, "attacker-value") for name in names])
        self.assertEqual(status, 200, body)
        requests, paid_calls = self.backend.snapshot()
        upstream_names = {name.lower() for name, _value in requests[0]["headers"]}
        self.assertTrue(upstream_names.isdisjoint(name.lower() for name in names))
        self.assertEqual(paid_calls, 1)

    def test_other_public_domains_internal_hosts_ports_and_domain_escapes_are_denied(self):
        hosts = (
            "other-example.ngrok-free.app", PUBLIC_HOST.upper(), PUBLIC_HOST + ".attacker.example",
            PUBLIC_HOST.replace(".", "x"), PUBLIC_HOST + ":8443", "localhost",
            f"localhost:{self.nginx.port}", f"127.0.0.1:{self.nginx.port}", self.backend.authority,
        )
        for host in hosts:
            with self.subTest(host=host):
                self.backend.reset()
                status, body = self.request(host)
                self.assertEqual(status, 403, body)
                self.assert_no_backend_request()

    def test_only_exact_https_public_origins_are_accepted(self):
        origins = (
            "null", "http://" + PUBLIC_HOST, "HTTPS://" + PUBLIC_HOST, "https://" + PUBLIC_HOST.upper(),
            "https://other-example.ngrok-free.app", "https://" + PUBLIC_HOST + ":8443",
            "https://" + PUBLIC_HOST + "/path", "https://" + PUBLIC_HOST + ".attacker.example",
            "https://" + PUBLIC_HOST + "@attacker.example", "https://" + PUBLIC_HOST + ", https://attacker.example",
            "http://" + self.backend.authority,
        )
        for origin in origins:
            with self.subTest(origin=origin):
                self.backend.reset()
                status, body = self.request(PUBLIC_HOST, origin)
                self.assertEqual(status, 403, body)
                self.assert_no_backend_request()

    def test_duplicate_host_and_origin_headers_are_denied_before_upstream(self):
        for headers in ([("Host", PUBLIC_HOST)], [("Origin", "https://" + PUBLIC_HOST)], [("Origin", "https://other-example.ngrok-free.app")]):
            with self.subTest(headers=headers):
                self.backend.reset()
                status, body = self.request(PUBLIC_HOST, "https://" + PUBLIC_HOST, headers=headers)
                self.assertIn(status, (400, 403), body)
                self.assert_no_backend_request()

    def test_cross_site_context_is_preserved_and_rejected_before_any_provider_action(self):
        status, body = self.request(PUBLIC_HOST, "https://" + PUBLIC_HOST, headers=[("Sec-Fetch-Site", "cross-site")])
        self.assertEqual(status, 403, body)
        requests, paid_calls = self.backend.snapshot()
        self.assertEqual(paid_calls, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn(("Sec-Fetch-Site", "cross-site"), requests[0]["headers"])

    def test_ingress_is_not_listening_on_the_other_loopback_address(self):
        with self.assertRaises(OSError):
            with socket.create_connection(("127.0.0.2", self.nginx.port), timeout=0.5):
                self.fail("The ngrok ingress must listen only on 127.0.0.1")

    def test_backend_503_causes_exactly_one_attempt_without_retry(self):
        self.backend.response_status = 503
        status, body = self.request(PUBLIC_HOST, "https://" + PUBLIC_HOST)
        self.assertEqual(status, 503, body)
        requests, paid_calls = self.backend.snapshot()
        self.assertEqual(len(requests), 1)
        self.assertEqual(paid_calls, 1)
        self.assertEqual(json.loads(body)["from"], "guarded-loopback-stub")

    def test_static_and_private_paths_use_the_backend_allowlist_instead_of_nginx_files(self):
        for path in ("/", "/.env", "/.git/config", "/src/mdd/demo.py", "/data/protocol/l2arctic-v1/splits.json", "/runs/any/checkpoint/model.safetensors"):
            with self.subTest(path=path):
                self.backend.reset()
                status, body = self.request(PUBLIC_HOST, method="GET", path=path)
                self.assertEqual(status, 200 if path == "/" else 404, body)
                self.assertEqual(json.loads(body)["from"], "guarded-loopback-stub")
                self.assertNotIn(b"NGINX_STATIC_CANARY", body)
                requests, paid_calls = self.backend.snapshot()
                self.assertEqual([request["path"] for request in requests], [path])
                self.assertEqual(paid_calls, 0)

    def test_chunked_input_is_rejected_without_contacting_backend(self):
        payload = (f"POST /api/feedback HTTP/1.1\r\nHost: {PUBLIC_HOST}\r\n"
                   "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                   "2\r\n{}\r\n0\r\n\r\n").encode()
        self.assertEqual(self.raw_request(payload), 400)
        self.assert_no_backend_request()

    def test_oversized_content_length_is_rejected_before_uploading_body(self):
        payload = (f"POST /api/analyze HTTP/1.1\r\nHost: {PUBLIC_HOST}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {nginx_fixture.MAX_BODY_BYTES + 1}\r\n"
                   "Connection: close\r\n\r\n").encode()
        self.assertEqual(self.raw_request(payload), 413)
        self.assert_no_backend_request()


if __name__ == "__main__":
    unittest.main()
