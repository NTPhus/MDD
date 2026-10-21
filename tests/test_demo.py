"""The local demo exposes only curated assets, never arbitrary repository files."""
import base64
from contextlib import redirect_stderr, redirect_stdout
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.demo import DemoApplication, DemoError, DemoServer, MAX_BODY_BYTES, validate_lan_hosts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeEngine:
    def __init__(self):
        self.calls = []

    def analyze(self, model, audio, text, reference=None):
        self.calls.append((model, audio, text, reference))
        return {"predicted_phones": ["HH", "AY"], "reference_phones": ["HH", "AY"]}


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = FakeEngine()
        self.make_fixtures()
        self.app = DemoApplication(self.root, self.engine)

    def tearDown(self):
        self.temporary.cleanup()

    def make_fixtures(self):
        records, manifest, source = [], [], []
        for index in range(11):
            identifier = f"S{index}/arctic_a0001"
            relative = f"l2arctic_release_v5.0/S{index}/wav/arctic_a0001.wav"
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"RIFF-demo-wave")
            row = {"utterance_id": identifier, "audio_path": relative, "speaker_id": f"S{index}",
                   "duration_sec": 2.8, "prompt_text": f"The sample number {index}",
                   "canonical_phones": ["DH", "AH"], "canonical_word_indices": [0, 0],
                   "gold_error_events": [{"kind": "deletion"}], "perceived_phones": ["AH"]}
            records.append(row)
            manifest.append({"utterance_id": identifier, "audio": {"path": relative, "sha256": "unused"}})
            if index < 10:
                source.append(identifier)
        write_json(self.root / "data/protocol/l2arctic-v1/splits.json", {"source_train_ids": source})
        path = self.root / "data/processed/l2arctic-v1/annotations.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        path = self.root / "data/protocol/l2arctic-v1/manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
        base = self.root / "models/pretrained/wav2vec2-base"
        base.mkdir(parents=True)
        (base / "pytorch_model.bin").write_bytes(b"weights")
        run = self.root / "runs/demo-run"
        write_json(run / "config.json", {"stage": "A", "purpose": "technical_smoke_only", "max_steps": 1})
        write_json(run / "completed.json", {"status": "completed", "optimizer_steps": 1, "best_validation_value": 0.1})
        (run / "training_log.jsonl").write_text('{"step":1,"loss":19.5}\n{"step":2,"loss":', encoding="utf-8")
        checkpoint = run / "best_checkpoint"
        write_json(checkpoint / "metadata.json", {"stage": "A", "purpose": "technical_smoke_only", "step": 1})
        write_json(checkpoint / "config.json", {"vocab_size": 83})
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        assets = self.root / "demo"
        assets.mkdir()
        (assets / "index.html").write_text("<h1>Demo</h1>", encoding="utf-8")
        (self.root / ".env").write_text("NEVER_EXPOSE_SECRET=true", encoding="utf-8")

    def test_catalog_only_source_train_and_at_most_eight(self):
        prompts = self.app.prompts()["prompts"]
        self.assertEqual(len(prompts), 8)
        self.assertNotIn("S10/arctic_a0001", {row["id"] for row in prompts})
        self.assertTrue(all("gold_error_events" not in row and "audio_path" not in row for row in prompts))
        self.assertEqual(prompts[0]["audio_url"], "/api/samples/S0%2Farctic_a0001")
        with self.assertRaises(DemoError) as captured:
            self.app.sample_audio("../.env")
        self.assertEqual(captured.exception.status, 404)

    def test_status_reads_complete_records_and_opaque_model_catalog(self):
        status = self.app.status()
        self.assertEqual(status["default_model_id"], "run:demo-run:best")
        run = status["training"][0]
        self.assertEqual((run["step"], run["loss"], run["status"]), (1, 19.5, "completed"))
        self.assertEqual(run["history"], [{"step": 1, "loss": 19.5}])
        self.assertTrue(all("path" not in model and "fingerprint" not in model for model in status["models"]))
        self.assertTrue(next(model for model in status["models"] if model["kind"] == "phoneme")["experimental"])
        self.assertTrue(all(model["ready"] and model["warning"] for model in status["models"]))

    def test_catalog_names_training_phases_and_prefers_adapted_best_over_more_updates(self):
        for stage, best_step, last_step in (("A", 2000, 2000), ("B", 300, 400), ("C", 300, 400)):
            run = self.root / "runs" / f"scientific-{stage}"
            write_json(run / "config.json", {"stage": stage, "purpose": "scientific_training"})
            for checkpoint, step in (("best_checkpoint", best_step), ("checkpoint", last_step)):
                path = run / checkpoint
                write_json(path / "metadata.json", {"stage": stage, "purpose": "scientific_training", "step": step})
                write_json(path / "config.json", {"vocab_size": 83})
                (path / "model.safetensors").write_bytes(b"synthetic weights")
        status = self.app.status()
        models = {model["id"]: model for model in status["models"]}
        self.assertEqual(status["default_model_id"], "run:scientific-B:best")
        self.assertEqual(models["base"]["training_phase"], "pretrained")
        self.assertTrue(models["base"]["can_extract_features"])
        self.assertFalse(models["base"]["can_recognize_phonemes"])
        self.assertFalse(models["base"]["can_compare_phonemes"])
        for stage, phase in (("A", "source"), ("B", "adapted_vi"), ("C", "control")):
            model = models[f"run:scientific-{stage}:best"]
            self.assertEqual(model["training_phase"], phase)
            self.assertTrue(model["can_recognize_phonemes"])
            self.assertTrue(model["can_compare_phonemes"])
            self.assertTrue(model["phase_label"])
        self.assertEqual(models["run:demo-run:best"]["training_phase"], "technical")
        self.assertFalse(models["run:demo-run:best"]["can_compare_phonemes"])

    def test_base_never_publishes_word_errors_or_calls_paid_providers(self):
        self.engine.analyze = Mock(return_value={
            "model": {"id": "base", "kind": "phoneme", "experimental": False},
            "phonemes": [{"phone": "T"}],
            "comparison": {"words": [{"text": word, "status": "difference"}
                                      for word in "The sample number 0".split()],
                           "events": [{"kind": "substitution", "expected_phone": "TH",
                                       "observed_phone": "T", "word_index": 0}]}})
        with patch("scripts.llm_client.LLMClient") as llm, patch("scripts.ai33_tts.AI33TTS") as tts:
            response = self.app.analyze({"model_id": "base", "sample_id": "S0/arctic_a0001",
                                         "reference_text": "The sample number 0"})
            self.assertIsNone(response["comparison"])
            self.assertEqual(response["phonemes"], [])
            self.assertEqual(response["model"]["kind"], "backbone")
            self.assertEqual(response["assessment"]["status"], "uncertain")
            self.assertEqual(response["assessment"]["errors"], [])
            self.assertFalse(response["feedback_available"])
            feedback = self.app.feedback({"analysis_id": response["analysis_id"], "include_voice": True})
            self.assertEqual(feedback["status"], "unavailable")
        llm.assert_not_called()
        tts.assert_not_called()

    def test_analyze_only_selected_audio_and_canonical_reference(self):
        result = self.app.analyze({"model_id": "run:demo-run:best", "sample_id": "S0/arctic_a0001",
                                   "reference_text": "The sample number 0"})
        self.assertEqual(result["model_id"], "run:demo-run:best")
        reference = self.engine.calls[-1][3]
        self.assertEqual(reference["canonical_phones"], ["DH", "AH"])
        self.assertNotIn("perceived_phones", reference)
        self.assertNotIn("gold_error_events", reference)
        self.app.analyze({"model_id": "base", "sample_id": "S0/arctic_a0001", "reference_text": "Another sentence"})
        self.assertIsNone(self.engine.calls[-1][3])
        self.app.analyze({"model_id": "base", "audio_base64": base64.b64encode(b"upload").decode(),
                          "reference_text": " THE  SAMPLE number 0 "})
        self.assertEqual(self.engine.calls[-1][3]["canonical_phones"], ["DH", "AH"])
        self.app.analyze({"model_id": "base", "sample_id": "S0/arctic_a0001", "reference_text": "The sample number 0!"})
        self.assertIsNone(self.engine.calls[-1][3])

    def test_request_validation_and_busy_lock(self):
        valid = {"model_id": "base", "audio_base64": base64.b64encode(b"audio").decode(), "reference_text": "Hello there"}
        for extra, code in [({"audio_base64": "%%%%"}, "invalid_audio"),
                            ({"model_id": "../../.env"}, "model_unavailable"),
                            ({"sample_id": "S0/arctic_a0001"}, "invalid_audio"),
                            ({"reference_text": " "}, "invalid_reference")]:
            with self.subTest(code=code), self.assertRaises(DemoError) as captured:
                self.app.analyze({**valid, **extra})
            self.assertEqual(captured.exception.code, code)
        self.app.inference_lock.acquire()
        try:
            with self.assertRaises(DemoError) as captured:
                self.app.analyze(valid)
            self.assertEqual(captured.exception.code, "busy")
        finally:
            self.app.inference_lock.release()

    def test_http_static_allowlist_origin_and_body_limits(self):
        server = DemoServer(("127.0.0.1", 0), self.app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def request(method, path, body=None, headers=None):
            connection = HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                return response.status, response.read(), dict(response.getheaders())
            finally:
                connection.close()

        try:
            self.assertEqual(request("GET", "/")[0], 200)
            for path in ("/.env", "/../.env", "/%2e%2e/.env", "/models/pretrained/wav2vec2-base/pytorch_model.bin"):
                status, body, _ = request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"NEVER_EXPOSE", body)
            self.assertEqual(request("GET", "/api/status", headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(request("GET", "/api/status", headers={"Host": f"10.10.63.214:{port}"})[0], 403)
            self.assertEqual(request("GET", "/api/status", headers={"Origin": "https://evil.example"})[0], 403)
            self.assertEqual(request("GET", "/api/status", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
            status, body, headers = request("GET", "/api/status", headers={"Origin": f"http://127.0.0.1:{port}"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["inference_device"], "cpu")
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertEqual(request("POST", "/api/analyze", "{}", {"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(request("POST", "/api/analyze", "{", {"Content-Type": "application/json"})[0], 400)
            self.assertEqual(request("POST", "/api/analyze", "", {"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})[0], 413)
            self.assertEqual(request("GET", "/api/samples/S0%2Farctic_a0001")[1], b"RIFF-demo-wave")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_lan_mode_accepts_only_listed_hosts_and_their_exact_same_origin(self):
        server = DemoServer(("0.0.0.0", 0), self.app, lan=True, allow_hosts=["10.10.63.214", "192.168.1.20"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def request(method, path, body=None, headers=None):
            connection = HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                return response.status, response.read(), dict(response.getheaders())
            finally:
                connection.close()

        try:
            for host in ("10.10.63.214", "192.168.1.20", "127.0.0.1", "localhost"):
                with self.subTest(host=host):
                    headers = {"Host": f"{host}:{port}", "Origin": f"http://{host}:{port}"}
                    status, body, response_headers = request("GET", "/api/status", headers=headers)
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["inference_device"], "cpu")
                    self.assertNotIn("Access-Control-Allow-Origin", response_headers)
            lan = {"Host": f"10.10.63.214:{port}", "Origin": f"http://10.10.63.214:{port}"}
            payload = json.dumps({"model_id": "base", "reference_text": "Hello there.",
                                  "audio_base64": base64.b64encode(b"synthetic audio").decode()})
            with patch("scripts.llm_client.LLMClient") as llm, patch("scripts.ai33_tts.AI33TTS") as tts:
                status, body, _ = request("POST", "/api/analyze", payload, {**lan, "Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["model_id"], "base")
            self.assertEqual(json.loads(body)["assessment"]["errors"], [])
            llm.assert_not_called(); tts.assert_not_called()
            for host in (f"10.10.63.215:{port}", f"8.8.8.8:{port}", f"evil.invalid:{port}",
                         f"0.0.0.0:{port}", "10.10.63.214", f"10.10.63.214:{port + 1}"):
                with self.subTest(foreign_host=host):
                    self.assertEqual(request("GET", "/api/status", headers={"Host": host})[0], 403)
            for origin in ("https://evil.invalid", f"http://127.0.0.1:{port}",
                           f"http://192.168.1.20:{port}", f"https://10.10.63.214:{port}", "null"):
                with self.subTest(foreign_origin=origin):
                    self.assertEqual(request("GET", "/api/status", headers={**lan, "Origin": origin})[0], 403)
            self.assertEqual(request("GET", "/api/status", headers={**lan, "Sec-Fetch-Site": "cross-site"})[0], 403)
            self.assertEqual(request("POST", "/api/analyze", payload,
                                     {**lan, "Content-Type": "application/json", "Origin": "https://evil.invalid"})[0], 403)
            for path in ("/.env", "/data/generated/demo/feedback/private.meta.json", "/../.env",
                         "/models/pretrained/wav2vec2-base/pytorch_model.bin"):
                with self.subTest(private_path=path):
                    status, body, _ = request("GET", path, headers=lan)
                    self.assertEqual(status, 404)
                    self.assertNotIn(b"NEVER_EXPOSE", body)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_lan_configuration_requires_explicit_rfc1918_ipv4_addresses(self):
        self.assertEqual(validate_lan_hosts(["10.10.63.214", "172.16.2.3", "192.168.1.20", "10.10.63.214"], lan=True),
                         frozenset({"10.10.63.214", "172.16.2.3", "192.168.1.20"}))
        for hosts, lan in (([], True), (["10.10.63.214"], False), (["*"], True), (["0.0.0.0"], True),
                           (["8.8.8.8"], True), (["127.0.0.1"], True), (["localhost"], True),
                           (["169.254.1.1"], True), (["192.0.2.1"], True), (["100.64.0.1"], True),
                           (["172.15.1.1"], True), (["172.32.1.1"], True), (["::1"], True),
                           (["10.10.63.214:8765"], True), (["10.0.0.0/8"], True)):
            with self.subTest(hosts=hosts, lan=lan), self.assertRaises(ValueError):
                validate_lan_hosts(hosts, lan=lan)
        for address, kwargs in ((("0.0.0.0", 0), {}),
                                (("0.0.0.0", 0), {"lan": True}),
                                (("192.168.1.20", 0), {"lan": True, "allow_hosts": ["192.168.1.20"]})):
            with self.subTest(address=address, kwargs=kwargs), self.assertRaises(ValueError):
                DemoServer(address, self.app, **kwargs)

    def test_cli_rejects_invalid_lan_configuration_before_loading_application(self):
        from scripts import serve_demo
        for arguments in (["--lan"], ["--allow-host", "10.10.63.214"],
                          ["--lan", "--allow-host", "8.8.8.8"], ["--lan", "--allow-host", "::1"],
                          ["--lan", "--allow-host", "10.10.63.214:8765"]):
            with self.subTest(arguments=arguments), patch.object(serve_demo, "DemoApplication") as application, \
                    patch.object(serve_demo, "DemoServer") as server, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as captured:
                    serve_demo.main(arguments)
                self.assertEqual(captured.exception.code, 2)
                application.assert_not_called(); server.assert_not_called()

    def test_cli_lan_mode_passes_repeated_trusted_addresses_to_the_server(self):
        from scripts import serve_demo
        with patch.object(serve_demo, "DemoApplication", return_value=self.app) as application, \
                patch.object(serve_demo, "DemoServer") as server, patch.dict(os.environ, {}), redirect_stdout(io.StringIO()) as output:
            serve_demo.main(["--lan", "--allow-host", "10.10.63.214", "--allow-host", "192.168.1.20",
                             "--port", "8766", "--threads", "1"])
        application.assert_called_once_with(serve_demo.ROOT, threads=1)
        self.assertEqual(server.call_args.args, (("0.0.0.0", 8766), self.app))
        self.assertEqual(server.call_args.kwargs, {"lan": True, "allow_hosts": frozenset({"10.10.63.214", "192.168.1.20"})})
        server.return_value.serve_forever.assert_called_once()
        server.return_value.server_close.assert_called_once()
        self.assertIn("http://10.10.63.214:8766", output.getvalue())


if __name__ == "__main__":
    unittest.main()
