"""Device routing and OOM recovery without allocating a real CUDA tensor."""

import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from src.mdd.demo_inference import LocalInference


class RoutedTensor:
    """Record requested placement while keeping synthetic data on the CPU."""

    def __init__(self, value):
        self.value = value
        self.dtype = value.dtype
        self.moves = []

    def is_floating_point(self):
        return self.value.is_floating_point()

    def to(self, *, device, dtype):
        self.moves.append((device, dtype))
        return self.value.to(dtype=dtype)


class DeviceInferenceTests(unittest.TestCase):
    def test_evaluation_default_remains_cpu_float32(self):
        engine = LocalInference(Path.cwd())
        network = Mock()
        with patch("torch.cuda.is_available") as cuda:
            runtime = engine._place_network(network)
        network.to.assert_called_once_with("cpu")
        cuda.assert_not_called()
        self.assertEqual((runtime["device"], runtime["compute_type"]), ("cpu", "float32"))
        self.assertIsNone(runtime["fallback_reason"])

    def test_unknown_device_or_compute_type_is_rejected(self):
        for options in ({"device": "cuda:1"}, {"compute_type": "int8"},
                        {"device": "cpu", "compute_type": "float16"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LocalInference(Path.cwd(), **options)

    def test_cuda_auto_uses_float16_and_explicit_float32_is_supported(self):
        for requested, dtype in (("auto", torch.float16), ("float32", torch.float32)):
            engine = LocalInference(Path.cwd(), device="auto", compute_type=requested)
            network = Mock()
            with patch("torch.cuda.is_available", return_value=True):
                runtime = engine._place_network(network)
            network.to.assert_called_once_with(device="cuda:0", dtype=dtype)
            self.assertEqual(runtime["device"], "cuda:0")
            self.assertEqual(runtime["compute_type"], str(dtype).split(".")[-1])
            self.assertIsNone(runtime["fallback_reason"])

    def test_missing_cuda_is_explicit_cpu_fallback(self):
        engine = LocalInference(Path.cwd(), device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            runtime = engine._place_network(Mock())
        self.assertEqual(runtime["requested_device"], "cuda")
        self.assertEqual(runtime["device"], "cpu")
        self.assertIn("unavailable", runtime["fallback_reason"])

    def test_loading_oom_moves_partially_loaded_network_back_to_cpu(self):
        engine = LocalInference(Path.cwd(), device="cuda")
        network = Mock()
        network.to.side_effect = [torch.cuda.OutOfMemoryError("no free VRAM"), None]
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as clear:
            runtime = engine._place_network(network)
        self.assertEqual(network.to.call_args_list[-1].kwargs, {"device": "cpu", "dtype": torch.float32})
        clear.assert_called_once_with()
        self.assertEqual(runtime["compute_type"], "float32")
        self.assertIn("loading", runtime["fallback_reason"])

    def test_non_oom_load_errors_are_not_silently_hidden(self):
        engine = LocalInference(Path.cwd(), device="cuda")
        network = Mock()
        network.to.side_effect = RuntimeError("a damaged kernel")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as clear:
            with self.assertRaisesRegex(RuntimeError, "damaged kernel"):
                engine._place_network(network)
        self.assertEqual(network.to.call_count, 1)
        clear.assert_not_called()

    def cuda_fixture(self, network=None):
        engine = LocalInference(Path.cwd(), device="auto")
        inputs = {"input_values": RoutedTensor(torch.ones(1, 160)),
                  "attention_mask": RoutedTensor(torch.ones(1, 160, dtype=torch.long))}
        runtime = {"device": "cuda:0", "compute_type": "float16", "requested_device": "auto",
                   "requested_compute_type": "auto", "fallback_reason": None}
        bundle = {"network": network or Mock(return_value="result"), "runtime": runtime}
        return engine, inputs, bundle

    def test_gpu_forward_casts_only_floating_audio_and_disables_gradients(self):
        calls = []
        def forward(**inputs):
            calls.append({"grad": torch.is_grad_enabled(), "inputs": inputs})
            return "result"
        engine, inputs, bundle = self.cuda_fixture(forward)
        self.assertEqual(engine._forward(bundle, inputs), "result")
        self.assertFalse(calls[0]["grad"])
        self.assertEqual(calls[0]["inputs"]["input_values"].dtype, torch.float16)
        self.assertEqual(calls[0]["inputs"]["attention_mask"].dtype, torch.long)
        self.assertEqual(inputs["input_values"].moves, [("cuda:0", torch.float16)])

    def test_inference_oom_retries_once_on_cpu_and_keeps_cpu_for_next_clip(self):
        network = Mock(side_effect=[RuntimeError("CUDA out of memory"), "recovered", "next clip"])
        engine, inputs, bundle = self.cuda_fixture(network)
        engine._cached = bundle
        with patch("torch.cuda.empty_cache") as clear:
            self.assertEqual(engine._forward(bundle, inputs), "recovered")
            self.assertEqual(engine._forward(bundle, inputs), "next clip")
        self.assertEqual(inputs["input_values"].moves,
                         [("cuda:0", torch.float16), ("cpu", torch.float32), ("cpu", torch.float32)])
        self.assertEqual(engine.runtime_info["device"], "cpu")
        self.assertIn("during", engine.runtime_info["fallback_reason"])
        clear.assert_called_once_with()
        network.to.assert_called_once_with(device="cpu", dtype=torch.float32)

    def test_inference_non_oom_does_not_retry_and_cpu_oom_does_not_retry(self):
        for device, error in (("cuda:0", "bad shape"), ("cpu", "CUDA out of memory")):
            network = Mock(side_effect=RuntimeError(error))
            engine, inputs, bundle = self.cuda_fixture(network)
            bundle["runtime"]["device"] = device
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, error):
                engine._forward(bundle, inputs)
            self.assertEqual(network.call_count, 1)
            network.to.assert_not_called()

    def test_float16_encoder_exports_float32_posteriors_and_actual_runtime(self):
        engine, _, bundle = self.cuda_fixture()
        logits = torch.tensor([[[0., 10., 0.], [9., 0., 0.], [0., 0., 10.]]], dtype=torch.float16)
        bundle.update(network=Mock(return_value=SimpleNamespace(logits=logits)),
                      extractor=lambda waveform, **kwargs: {
                          "input_values": RoutedTensor(torch.as_tensor(waveform)[None])},
                      vocabulary={"[PAD]": 0, "HH": 1, "W": 2}, experimental=False)
        samples = (.2 * np.sin(2 * np.pi * 220 * np.arange(6400) / 16000)).astype(np.float32)
        stream = io.BytesIO()
        sf.write(stream, samples, 16000, format="WAV")
        model = {"id": "B", "label": "B", "kind": "phoneme", "experimental": False}
        with patch.object(engine, "_get_bundle", return_value=bundle):
            result = engine.analyze(model, stream.getvalue(), "", capture_evidence=True)
        self.assertEqual(result["runtime"]["device"], "cuda:0")
        self.assertEqual(result["runtime"]["compute_type"], "float16")
        self.assertEqual(result["_acoustic_evidence"]["log_probs"].dtype, np.float32)
        self.assertEqual([phone["phone"] for phone in result["phonemes"]], ["HH", "W"])
        np.testing.assert_allclose(np.exp(result["_acoustic_evidence"]["log_probs"]).sum(axis=1), 1., rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
