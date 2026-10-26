"""CROTTC setup integrity, phone provenance and subprocess lifecycle contracts."""
import hashlib
import io
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import crottc_pronunciation as crottc


class CROTTCBackendTests(unittest.TestCase):
    def test_selection_missing_manifest_does_not_import_models(self):
        with tempfile.TemporaryDirectory() as folder:
            result = crottc.crottc_selection(folder)
        self.assertFalse(result["ready"])
        self.assertEqual(result["provider"], "crottc-if")
        self.assertEqual(result["device"], "cpu")

    def test_pins_use_official_lfs_hash_for_both_checkpoints(self):
        self.assertEqual(crottc.FILE_PINS["checkpoint/model.ckpt"][1], crottc.MODEL_SHA256)
        self.assertEqual(crottc.FILE_PINS["checkpoint/perceived_ssl.ckpt"][1], crottc.SSL_SHA256)
        self.assertEqual(len(crottc.REVISION), 40)

    def test_same_size_corrupt_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.ckpt"
            path.write_bytes(b"abc")
            pin = (3, hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(crottc.file_record(path, pin)["sha256"], pin[1])
            path.write_bytes(b"xyz")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                crottc.file_record(path, pin)

    def test_small_source_code_verified_as_git_blob(self):
        data = b"print('verified')\n"
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom_interface.py"
            path.write_bytes(data)
            result = crottc.file_record(path, (len(data), blob))
        self.assertEqual(result["pin"], blob)

    def test_downloaded_but_unloaded_model_not_production_ready(self):
        record = {"status": "downloaded", "model_id": crottc.MODEL_ID, "revision": crottc.REVISION}
        with patch.object(Path, "read_text", return_value=json.dumps(record)):
            with self.assertRaisesRegex(ValueError, "smoke-tested"):
                crottc.verify(ROOT)

    def test_phone_sequence_preserves_errors_separately_and_has_no_fabricated_timestamps(self):
        labels = ["sil", "aa", "th", "err", "<eos>", "new-unknown"]
        phones, unresolved, excluded = crottc.phone_evidence(labels)
        self.assertEqual(phones, [{"phone": "AA"}, {"phone": "TH"}])
        self.assertEqual(unresolved, [{"index": 3, "label": "err"}, {"index": 5, "label": "new-unknown"}])
        self.assertEqual([row["label"] for row in excluded], ["sil", "<eos>"])
        self.assertEqual(labels[3], "err")

    def test_cancelled_load_does_not_start_process(self):
        event = threading.Event()
        event.set()
        engine = crottc.CROTTCPronunciation(ROOT)
        with patch.object(engine, "_start") as start:
            with self.assertRaises(InterruptedError):
                engine.load(event)
        start.assert_not_called()

    def test_invalid_audio_never_loads_model(self):
        import numpy as np
        engine = crottc.CROTTCPronunciation(ROOT)
        with patch.object(engine, "load") as load:
            for audio in [[], np.zeros((2, 16000)), np.zeros(480001), [float("nan")] * 400]:
                with self.assertRaises(ValueError):
                    engine.infer(audio)
        load.assert_not_called()

    def test_load_only_request_never_infers(self):
        engine = crottc.CROTTCPronunciation(ROOT)
        engine.process = Mock(poll=Mock(return_value=None))
        engine.events = Queue()
        engine.events.put({"id": 1, "result": {"load_only": True, "device": "cpu"}})
        with patch.object(crottc, "verify"), patch.object(engine, "_start"):
            self.assertIs(engine.load(), engine)
        request = json.loads(engine.process.stdin.write.call_args.args[0])
        self.assertEqual(request["type"], "load")
        self.assertNotIn("waveform", request)

    def test_cancelled_request_closes_owned_worker(self):
        event = threading.Event()
        engine = crottc.CROTTCPronunciation(ROOT)
        engine.process = Mock()
        def waiting(timeout):
            event.set()
            from queue import Empty
            raise Empty
        engine.events = Mock(get=Mock(side_effect=waiting))
        with patch.object(engine, "close") as close:
            with self.assertRaises(InterruptedError):
                engine._request({"type": "load"}, event)
        close.assert_called_once()

    def test_close_only_terminates_owned_process_and_clears_runtime(self):
        engine = crottc.CROTTCPronunciation(ROOT)
        process = Mock(poll=Mock(return_value=None))
        engine.process, engine.runtime = process, {"device": "cpu"}
        engine.close()
        process.stdin.close.assert_called()
        process.wait.assert_called_once()
        process.terminate.assert_not_called()
        self.assertIsNone(engine.process)
        self.assertIsNone(engine.runtime)

    def test_exited_worker_cannot_leave_a_stale_model_ready(self):
        engine = crottc.CROTTCPronunciation(ROOT)
        engine.process = Mock()
        engine.events = Queue()
        engine.events.put(None)
        with patch.object(engine, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "worker stopped"):
                engine._request({"type": "load"}, None)
        close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
