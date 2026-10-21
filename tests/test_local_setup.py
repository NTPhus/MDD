"""Local setup must fail clearly without breaking the CUDA notebook's probe."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts import check_training_environment


class LocalSetupReadinessTests(unittest.TestCase):
    def invoke(self, *, local_ready, strict=False, cuda_ready=False, require_cuda=False):
        result = {"ready_for_cpu_pipeline_check": True,
                  "ready_for_local_cpu_training": local_ready,
                  "ready_for_local_cuda_training": cuda_ready}
        arguments = ["check_training_environment.py"]
        if strict:
            arguments.append("--require-local-cpu")
        if require_cuda:
            arguments.append("--require-local-cuda")
        with patch.object(check_training_environment, "report", return_value=result), \
                patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
            return check_training_environment.main()

    def test_local_setup_rejects_missing_cpu_presets_or_unverified_data(self):
        self.assertEqual(self.invoke(local_ready=False, strict=True), 1)

    def test_local_setup_accepts_complete_cpu_environment(self):
        self.assertEqual(self.invoke(local_ready=True, strict=True), 0)

    def test_default_probe_remains_usable_by_cuda_colab_notebook(self):
        self.assertEqual(self.invoke(local_ready=False), 0)

    def test_cuda_flag_selects_cuda_readiness_instead_of_cpu_readiness(self):
        self.assertEqual(self.invoke(local_ready=False, cuda_ready=True, require_cuda=True), 0)
        self.assertEqual(self.invoke(local_ready=True, cuda_ready=False, require_cuda=True), 1)


class CUDAReadinessTests(unittest.TestCase):
    def setUp(self):
        self.result = {"ready_for_cpu_pipeline_check": True, "torch": {
            "cuda_build": "12.6", "cuda_available": True, "cuda_backward_ok": True,
            "bf16_supported": False,
        }}
        self.configs = {name: {"device": "cuda", "mixed_precision": "fp16"}
                        for name in ("smoke", "source", "adapt_vi", "control")}
        self.audit = {"audio_sha256_verified": True,
                      "annotation_and_transcript_sha256_verified": True}

    def ready(self, *, result=None, configs=None, audit=None, device="cuda"):
        return check_training_environment.local_training_ready(
            self.result if result is None else result,
            self.configs if configs is None else configs,
            self.audit if audit is None else audit, device=device)

    def test_complete_cuda_environment_is_ready(self):
        self.assertTrue(self.ready())

    def test_cuda_wheel_gpu_and_backward_are_all_required(self):
        for field, value in (("cuda_build", None), ("cuda_available", False), ("cuda_backward_ok", False)):
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                result["torch"][field] = value
                self.assertFalse(self.ready(result=result))

    def test_missing_inputs_and_unverified_data_reject_cuda_training(self):
        result = copy.deepcopy(self.result)
        result["ready_for_cpu_pipeline_check"] = False
        self.assertFalse(self.ready(result=result))
        for field in self.audit:
            with self.subTest(field=field):
                audit = dict(self.audit, **{field: False})
                self.assertFalse(self.ready(audit=audit))

    def test_wrong_device_missing_config_and_unsupported_bf16_reject_cuda(self):
        for broken_config in ({}, {"device": "cpu", "mixed_precision": "no"},
                              {"device": "cuda", "mixed_precision": "unknown"},
                              {"device": "cuda", "mixed_precision": "bf16"}):
            with self.subTest(config=broken_config):
                configs = copy.deepcopy(self.configs)
                configs["control"] = broken_config
                self.assertFalse(self.ready(configs=configs))
        result = copy.deepcopy(self.result)
        result["torch"]["bf16_supported"] = True
        configs = copy.deepcopy(self.configs)
        configs["control"]["mixed_precision"] = "bf16"
        self.assertTrue(self.ready(result=result, configs=configs))

    def test_cpu_readiness_still_requires_cpu_build_and_fp32_presets(self):
        configs = {name: {"device": "cpu", "mixed_precision": "no"} for name in self.configs}
        self.assertFalse(self.ready(configs=configs, device="cpu"))
        result = copy.deepcopy(self.result)
        result["torch"]["cuda_build"] = None
        self.assertTrue(self.ready(result=result, configs=configs, device="cpu"))
        configs["control"]["mixed_precision"] = "fp16"
        self.assertFalse(self.ready(result=result, configs=configs, device="cpu"))


class CUDAExecutionProbeTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.current_device.return_value = 0
        self.torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="Test NVIDIA GPU", total_memory=8 * 1024**3, major=8, minor=6)
        self.torch.cuda.is_bf16_supported.return_value = True
        self.torch.cuda.mem_get_info.return_value = (6 * 1024**3, 8 * 1024**3)
        self.torch.ones.return_value.grad.__eq__.return_value.all.return_value.item.return_value = True

    def test_success_records_gpu_memory_and_executes_cuda_backward(self):
        result = check_training_environment.cuda_probe(self.torch)
        self.assertTrue(result["cuda_backward_ok"])
        self.assertEqual(result["gpu"]["name"], "Test NVIDIA GPU")
        self.assertEqual(result["gpu"]["total_memory_bytes"], 8 * 1024**3)
        self.assertEqual(result["gpu"]["free_memory_bytes"], 6 * 1024**3)
        self.torch.ones.assert_called_once_with(16, device="cuda:0", requires_grad=True)
        self.torch.ones.return_value.square.return_value.sum.return_value.backward.assert_called_once_with()
        self.torch.cuda.synchronize.assert_called_once_with(0)

    def test_unavailable_cuda_does_not_allocate_gpu_memory(self):
        self.torch.cuda.is_available.return_value = False
        self.assertFalse(check_training_environment.cuda_probe(self.torch)["cuda_backward_ok"])
        self.torch.ones.assert_not_called()

    def test_execution_failure_is_reported_and_never_marks_cuda_ready(self):
        self.torch.ones.side_effect = RuntimeError("driver cannot execute this CUDA build")
        result = check_training_environment.cuda_probe(self.torch)
        self.assertFalse(result["cuda_backward_ok"])
        self.assertIn("RuntimeError: driver cannot execute", result["cuda_probe_error"])


class LocalInputProbeTests(unittest.TestCase):
    def test_model_manifest_controls_the_required_weight_format(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "pytorch_model.bin").write_bytes(b"legacy")
            (model / "download_manifest.json").write_text(
                json.dumps({"weights_file": "model.safetensors"}), encoding="utf-8")
            # An unrelated bin file must not hide the missing manifest-selected format.
            self.assertEqual(check_training_environment.model_weights_path(model), model / "model.safetensors")
            self.assertFalse(check_training_environment.model_weights_path(model).is_file())

    def test_legacy_weights_and_safe_manifest_paths_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            self.assertEqual(check_training_environment.model_weights_path(model), model / "pytorch_model.bin")
            (model / "model.safetensors").write_bytes(b"safe")
            self.assertEqual(check_training_environment.model_weights_path(model), model / "model.safetensors")
            (model / "download_manifest.json").write_text(
                json.dumps({"weights_file": "../outside.bin"}), encoding="utf-8")
            self.assertEqual(check_training_environment.model_weights_path(model), model / "model.safetensors")

    def test_windows_config_report_exposes_missing_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            for suffix in ("smoke", "source", "adapt_vi"):
                (root / "configs" / f"windows_{suffix}.json").write_text(
                    json.dumps({"device": "cuda", "mixed_precision": "fp16"}), encoding="utf-8")
            with patch.object(check_training_environment, "ROOT", root):
                configs = check_training_environment.config_report("windows")
            self.assertEqual(len(configs), 4)
            self.assertEqual(configs["windows_source"]["device"], "cuda")
            self.assertFalse(configs["windows_control"]["exists"])
            self.assertNotIn("device", configs["windows_control"])


if __name__ == "__main__":
    unittest.main()
