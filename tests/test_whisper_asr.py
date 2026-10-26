"""Offline model pins, cancellation and native timing without loading weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import whisper_asr as whisper


class WhisperASRTests(unittest.TestCase):
    def test_readiness_never_imports_or_downloads_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(whisper, "_load_runtime") as runtime:
            result = whisper.whisper_selection(folder)
        self.assertFalse(result["ready"])
        self.assertEqual(result["model_id"], whisper.MODEL_ID)
        runtime.assert_not_called()

    def test_manifest_revision_and_package_pins_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / whisper.MODEL_RELATIVE
            model.mkdir(parents=True)
            (model / "setup_manifest.json").write_text(json.dumps({"status": "ready", "model_id": whisper.MODEL_ID,
                "revision": "changed", "packages": whisper.PACKAGES}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned model/runtime"):
                whisper.verify(root)

    def test_manifest_never_replaces_required_file_pins(self):
        record = {"status": "ready", "model_id": whisper.MODEL_ID, "revision": whisper.REVISION,
                  "packages": whisper.PACKAGES, "files": deepcopy(whisper.FILES)}
        record["files"]["model.bin"]["sha256"] = "0" * 64
        with patch.object(Path, "read_text", return_value=json.dumps(record)), \
             patch.object(whisper, "file_record") as check_file:
            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                whisper.verify(ROOT)
        # The altered model pin is rejected before its contents can be trusted.
        self.assertNotIn(ROOT / whisper.MODEL_RELATIVE / "model.bin",
                         [call.args[0] for call in check_file.call_args_list])

    def test_optional_dependency_metadata_must_match_all_three_versions(self):
        distributions = [SimpleNamespace(metadata={"Name": name}, version=version)
                         for name, version in whisper.PACKAGES.items()]
        with patch.object(whisper.importlib.metadata, "distributions", return_value=distributions):
            self.assertEqual(whisper.package_versions("unused"), whisper.PACKAGES)
        distributions[-1].version = "0.0"
        with patch.object(whisper.importlib.metadata, "distributions", return_value=distributions):
            with self.assertRaisesRegex(ValueError, "av==19.0.2"):
                whisper.package_versions("unused")

    def test_invalid_cpu_threads_never_reaches_runtime(self):
        for bad in [0, 17, True, "2"]:
            with self.assertRaises(ValueError):
                whisper.WhisperASR(ROOT, cpu_threads=bad)

    def test_checksum_checked_against_pin_not_manifest_alone(self):
        data = b"abc"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.bin"
            path.write_bytes(data)
            expected = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            self.assertEqual(whisper.file_record(path, expected), expected)
            path.write_bytes(b"xyz")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                whisper.file_record(path, expected)

    def test_load_is_cpu_int8_offline_once(self):
        model_class = Mock()
        engine = whisper.WhisperASR(ROOT, cpu_threads=2)
        with patch.object(whisper, "verify") as verify, patch.object(whisper, "_load_runtime", return_value=model_class):
            self.assertIs(engine.load(), engine)
            engine.load()
        verify.assert_called_once_with(ROOT.resolve())
        model_class.assert_called_once_with(str(ROOT / whisper.MODEL_RELATIVE), device="cpu", compute_type="int8",
                                            cpu_threads=2, num_workers=1, local_files_only=True)

    def test_cancellation_before_load_prevents_work(self):
        event = threading.Event()
        event.set()
        with patch.object(whisper, "verify") as verify:
            with self.assertRaises(InterruptedError):
                whisper.WhisperASR(ROOT).load(event)
        verify.assert_not_called()

    def test_cancelled_model_load_not_published(self):
        event = threading.Event()
        engine = whisper.WhisperASR(ROOT)
        def construct(*args, **kwargs):
            event.set()
            return Mock()
        with patch.object(whisper, "verify"), patch.object(whisper, "_load_runtime", return_value=construct):
            with self.assertRaises(InterruptedError):
                engine.load(event)
        self.assertIsNone(engine.model)

    def test_transcribe_preserves_native_text_and_timestamps_without_context(self):
        import numpy as np
        segment = SimpleNamespace(start=.1, end=.9, text=" I don't want 2 teas.", avg_logprob=-.2,
            no_speech_prob=.01, words=[
                SimpleNamespace(word=" I", start=.1, end=.2, probability=.95),
                SimpleNamespace(word=" don", start=.2, end=.3, probability=.91),
                SimpleNamespace(word="'t", start=.3, end=.35, probability=.9),
                SimpleNamespace(word=" want", start=.35, end=.5, probability=.9),
                SimpleNamespace(word=" 2", start=.5, end=.6, probability=.9),
                SimpleNamespace(word=" teas.", start=.6, end=.9, probability=.8)])
        info = SimpleNamespace(language="en", language_probability=1.)
        engine = whisper.WhisperASR(ROOT)
        engine.model = Mock(transcribe=Mock(return_value=(iter([segment]), info)))
        audio = np.zeros(16000, dtype="float32")
        result = engine.transcribe(audio)
        self.assertEqual(result["raw_text"], "I don't want 2 teas.")
        self.assertEqual(result["text"], "I don't want two teas.")
        self.assertEqual([row["text"] for row in result["word_intervals"]], result["text"].split())
        self.assertEqual(result["word_intervals"][1], {"index": 1, "text": "don't", "start_sec": .2, "end_sec": .35})
        self.assertEqual(result["word_timing"], "whisper_native")
        options = engine.model.transcribe.call_args.kwargs
        self.assertTrue(options["word_timestamps"])
        self.assertEqual(options["beam_size"], 5)
        self.assertIsNone(options["initial_prompt"])
        self.assertIsNone(options["hotwords"])
        self.assertFalse(options["condition_on_previous_text"])
        self.assertFalse(result["reference_supplied_to_asr"])

    def test_cardinal_formatting_preserves_values_and_avoids_identifiers_and_times(self):
        text, changes = whisper._format_cardinals("Bring 6 spoons, 5 slabs and 3 red bags.")
        self.assertEqual(text, "Bring six spoons, five slabs and three red bags.")
        self.assertEqual([row["index"] for row in changes], [1, 3, 6])
        raw = "At 7 am and 8 p.m., 06 21 2026 -6 −5 - 3 2.5 6:30 5-6 10/20 3rd +5"
        text, changes = whisper._format_cardinals(raw)
        self.assertEqual(text, raw)
        self.assertEqual(changes, [])

    def test_mismatched_or_overlapping_intervals_abstain(self):
        self.assertEqual(whisper._intervals("hello world", [{"word": "hello earth", "start": 0., "end": 1.}], 1.), [])
        self.assertEqual(whisper._intervals("ice tea", [{"word": "ice tea", "start": 0., "end": 1.}], 1.), [])
        self.assertEqual(whisper._intervals("hello", [{"word": "hello", "start": float("nan"), "end": 1.}], 1.), [])

    def test_input_validation_does_not_attempt_load(self):
        import numpy as np
        engine = whisper.WhisperASR(ROOT)
        with patch.object(engine, "load") as load:
            for bad in [np.zeros((2, 2)), [], [float("nan")]]:
                with self.assertRaises(ValueError):
                    engine.transcribe(bad)
        load.assert_not_called()

    def test_cancel_during_lazy_segments_does_not_return_partial_transcript(self):
        import numpy as np
        event = threading.Event()
        def segments():
            event.set()
            yield SimpleNamespace(text="old", start=0., end=.5, avg_logprob=0., no_speech_prob=0., words=[])
        engine = whisper.WhisperASR(ROOT)
        engine.model = Mock(transcribe=Mock(return_value=(segments(), SimpleNamespace(language="en", language_probability=1.))))
        with self.assertRaises(InterruptedError):
            engine.transcribe(np.zeros(16000), event)


if __name__ == "__main__":
    unittest.main()
