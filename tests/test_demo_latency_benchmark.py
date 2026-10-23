"""Safety and measurement-contract tests; no server or model is started."""
import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_demo_latency import local_url, run_once, summarize_job


class BenchmarkContractTests(unittest.TestCase):
    def test_only_loopback_urls_with_explicit_port_are_accepted(self):
        for url in ("https://127.0.0.1:8765", "http://example.com:8765", "http://127.0.0.1.example.com:8765",
                    "http://127.0.0.1:8765@evil.com", "http://127.0.0.1:8765/api", "http://127.0.0.1"):
            with self.subTest(url=url), self.assertRaises(argparse.ArgumentTypeError):
                local_url(url)
        self.assertEqual(local_url("http://127.0.0.1:8765/"), "http://127.0.0.1:8765")

    def test_summary_separates_observed_reply_from_generation_and_omits_acoustics(self):
        job = {"id": "a" * 32, "status": "ready", "elapsed_sec": 19., "text": "Hello.",
               "speech": {"transcript": "Hello", "audio": {"waveform": [1] * 100},
                          "phonemes": ["HH"], "_acoustic_evidence": {"private": True},
                          "normalization": {"latency_sec": 6., "text": "Hello."}},
               "metadata": {"reply_latency_sec": 10., "generation_runtime": {"device": "cuda"}}}
        result = summarize_job(job, 20., [{"stage": "ready", "observed_elapsed_sec": 20.}], {"elapsed_sec": 12.})
        self.assertEqual(result["reply_ready"]["elapsed_sec"], 12.)
        self.assertEqual(result["metadata"]["reply_latency_sec"], 10.)
        self.assertEqual(result["normalization"], {"latency_sec": 6.})
        self.assertNotIn("speech", result)
        self.assertNotIn("waveform", repr(result))
        self.assertNotIn("private", repr(result))

    def test_poll_failure_cancels_only_its_owned_job(self):
        calls = []
        identifier = "b" * 32
        class Client:
            def request(self, path, payload=None):
                calls.append((path, payload))
                if path == "/api/chat/speech":
                    return {"id": identifier, "status": "transcribing"}
                if path == "/api/chat/cancel":
                    return {"id": identifier, "status": "cancelled"}
                raise OSError("poll unavailable")
        with self.assertRaisesRegex(OSError, "poll unavailable"):
            run_once(Client(), "test-audio", 1, .15, 60, clock=lambda: 1., sleep=lambda _: None)
        self.assertEqual(calls[-1], ("/api/chat/cancel", {"job_id": identifier}))
        self.assertFalse(any("voice" in route for route, _ in calls))


if __name__ == "__main__":
    unittest.main()
