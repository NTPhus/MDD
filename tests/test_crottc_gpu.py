"""CUDA placement and explicit CPU fallback without loading a neural model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crottc_pronunciation import CROTTCPronunciation
from scripts.crottc_worker import (inference_with_device_fallback, load_with_device_fallback,
                                  physical_cuda_memory, select_runtime_device)


class FakeCUDAOutOfMemoryError(RuntimeError):
    pass


def fake_torch(*, available=True, free_mib=3072):
    torch = Mock()
    torch.cuda.is_available.return_value = available
    torch.cuda.mem_get_info.return_value = (free_mib * 1024 ** 2, 4096 * 1024 ** 2)
    torch.cuda.OutOfMemoryError = FakeCUDAOutOfMemoryError
    return torch


class CROTTCGPUTests(unittest.TestCase):
    def setUp(self):
        # Unit tests never inspect the host GPU or launch nvidia-smi.
        probe_patch = patch("scripts.crottc_worker.physical_cuda_memory", return_value=None)
        self.physical_probe = probe_patch.start()
        self.addCleanup(probe_patch.stop)

    def test_default_cpu_does_not_initialize_cuda(self):
        torch = fake_torch()
        device, runtime = select_runtime_device(torch)
        self.assertEqual(device, "cpu")
        self.assertEqual(runtime["compute_type"], "float32")
        self.assertIsNone(runtime["device_fallback_reason"])
        torch.cuda.is_available.assert_not_called()
        torch.cuda.mem_get_info.assert_not_called()

    def test_cuda_with_memory_uses_original_fp32_decoder(self):
        torch = fake_torch()
        loader = Mock(return_value=Mock())
        model, runtime = load_with_device_fallback(torch, loader, "cuda")
        loader.assert_called_once_with("cuda:0")
        self.assertIs(model, loader.return_value)
        self.assertEqual(runtime["requested_device"], "cuda")
        self.assertEqual(runtime["device"], "cuda:0")
        self.assertEqual(runtime["effective_device"], "cuda:0")
        self.assertEqual(runtime["compute_type"], "float32")
        self.assertEqual(runtime["cuda_free_mib"], 3072)

    def test_auto_without_cuda_falls_back_with_reason(self):
        torch = fake_torch(available=False)
        loader = Mock()
        _, runtime = load_with_device_fallback(torch, loader, "auto")
        loader.assert_called_once_with("cpu")
        self.assertEqual(runtime["device_fallback_reason"], "cuda_unavailable")
        torch.cuda.mem_get_info.assert_not_called()

    def test_insufficient_memory_does_not_try_cuda_model(self):
        torch = fake_torch(free_mib=1500)
        loader = Mock()
        _, runtime = load_with_device_fallback(torch, loader, "auto")
        loader.assert_called_once_with("cpu")
        self.assertEqual(runtime["device_fallback_reason"], "insufficient_free_cuda_memory")
        self.assertEqual(runtime["min_free_cuda_mib"], 2560)

    def test_wddm_overstated_torch_budget_uses_physical_free_memory(self):
        torch = fake_torch(free_mib=3308)
        self.physical_probe.return_value = {"free_mib": 1384, "total_mib": 4096}
        loader = Mock()
        _, runtime = load_with_device_fallback(torch, loader, "auto")
        loader.assert_called_once_with("cpu")
        self.assertEqual(runtime["cuda_free_mib"], 3308)
        self.assertEqual(runtime["cuda_physical_free_mib"], 1384)
        self.assertEqual(runtime["cuda_memory_gate_free_mib"], 1384)
        self.assertEqual(runtime["cuda_memory_gate_source"], "min_torch_and_nvidia_smi")
        self.assertEqual(runtime["device_fallback_reason"], "insufficient_free_cuda_memory")

    def test_lower_torch_budget_is_not_overridden_by_physical_probe(self):
        torch = fake_torch(free_mib=1700)
        self.physical_probe.return_value = {"free_mib": 3072, "total_mib": 4096}
        device, runtime = select_runtime_device(torch, "auto")
        self.assertEqual(device, "cpu")
        self.assertEqual(runtime["cuda_memory_gate_free_mib"], 1700)

    def test_load_cuda_oom_retries_cpu_and_preserves_reason(self):
        torch = fake_torch()
        loader = Mock(side_effect=[FakeCUDAOutOfMemoryError("allocation failed"), Mock()])
        _, runtime = load_with_device_fallback(torch, loader, "cuda")
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["cuda:0", "cpu"])
        self.assertEqual(runtime["device_fallback_reason"], "cuda_out_of_memory_during_load")
        torch.cuda.empty_cache.assert_called_once()

    def test_genuine_model_error_is_not_hidden_as_cpu_fallback(self):
        torch = fake_torch()
        loader = Mock(side_effect=RuntimeError("checkpoint tensor shape mismatch"))
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            load_with_device_fallback(torch, loader, "auto")
        loader.assert_called_once_with("cuda:0")
        torch.cuda.empty_cache.assert_not_called()

    def test_inference_cuda_oom_reloads_cpu_once(self):
        torch = fake_torch()
        model = Mock()
        model.brain.inference_batch.side_effect = FakeCUDAOutOfMemoryError("allocation failed")
        cpu_model = Mock()
        cpu_model.brain.inference_batch.return_value = {"hyps": [[1, 2]]}
        loader = Mock(return_value=cpu_model)
        batch = Mock()
        runtime = {"device": "cuda:0", "effective_device": "cuda:0", "compute_type": "float32"}
        result_model, result = inference_with_device_fallback(torch, model, runtime, batch, loader)
        self.assertIs(result_model, cpu_model)
        self.assertEqual(result, {"hyps": [[1, 2]]})
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["device_fallback_reason"], "cuda_out_of_memory_during_inference")
        model.brain.modules.to.assert_called_once_with("cpu")
        loader.assert_called_once_with("cpu")
        self.assertEqual(batch.call_count, 2)

    def test_cpu_allocation_failure_does_not_retry(self):
        torch = fake_torch()
        loader = Mock(side_effect=FakeCUDAOutOfMemoryError("not a CUDA attempt"))
        with self.assertRaises(FakeCUDAOutOfMemoryError):
            load_with_device_fallback(torch, loader, "cpu")
        loader.assert_called_once_with("cpu")

    def test_invalid_device_or_memory_setting_rejected_before_worker(self):
        for device in ("cuda:1", "unknown", "CUDA"):
            with self.assertRaises(ValueError):
                CROTTCPronunciation(ROOT, device=device)
        for size in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                CROTTCPronunciation(ROOT, min_free_cuda_mib=size)

    def test_gpu_child_keeps_existing_cuda_visibility(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            engine = CROTTCPronunciation(folder, device="auto")
            process = Mock(stdout=[])
            with patch("scripts.crottc_pronunciation.subprocess.Popen", return_value=process) as popen:
                engine._start()
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            args = popen.call_args.args[0]
            self.assertEqual(args[args.index("--device") + 1], "auto")
            self.assertEqual(args[args.index("--min-free-cuda-mib") + 1], "2560")

    def test_cpu_child_remains_hidden_from_cuda(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            engine = CROTTCPronunciation(folder)
            process = Mock(stdout=[])
            with patch("scripts.crottc_pronunciation.subprocess.Popen", return_value=process) as popen:
                engine._start()
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")


class PhysicalCUDAMemoryTests(unittest.TestCase):
    def test_probe_is_read_only_single_device_bounded_and_hidden(self):
        with patch("scripts.crottc_worker.subprocess.run", return_value=Mock(stdout="1384, 4096\n")) as run:
            self.assertEqual(physical_cuda_memory(), {"free_mib": 1384, "total_mib": 4096})
        command = run.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")
        self.assertIn("--id=0", command)
        self.assertIn("--query-gpu=memory.free,memory.total", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        if os.name == "nt":
            self.assertEqual(run.call_args.kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)

    def test_probe_failure_or_invalid_output_is_optional(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 2)):
            with self.subTest(error=type(error).__name__):
                with patch("scripts.crottc_worker.subprocess.run", side_effect=error):
                    self.assertIsNone(physical_cuda_memory())
        for output in ("N/A, 4096", "100, 50", "-1, 4096", "1384, 4096\n2000, 4000", ""):
            with self.subTest(output=output):
                with patch("scripts.crottc_worker.subprocess.run", return_value=Mock(stdout=output)):
                    self.assertIsNone(physical_cuda_memory())


if __name__ == "__main__":
    unittest.main()
