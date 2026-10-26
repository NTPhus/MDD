"""Inference device routing without native tensors, weights, or GPU jobs."""
from pathlib import Path
from queue import Queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.chat_runtime import requested_device, select_runtime
from scripts import local_chat_process
from scripts.local_chat_worker import QwenChatEngine


MIB = 1024 ** 2


def fake_torch(*, available=True, free_mib=3072, bf16=True):
    cuda = SimpleNamespace(
        is_available=Mock(return_value=available),
        mem_get_info=Mock(return_value=(free_mib * MIB, 4096 * MIB)),
        is_bf16_supported=Mock(return_value=bf16),
    )
    return SimpleNamespace(cuda=cuda)


class ChatRuntimeTests(unittest.TestCase):
    def assert_cpu(self, runtime, *, requested, reason):
        self.assertEqual(runtime["requested"], requested)
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["dtype"], "float32")
        self.assertEqual(runtime["attention"], "eager")
        self.assertEqual(runtime["reason"], reason)

    def assert_gpu(self, runtime, *, requested, dtype):
        self.assertEqual(runtime["requested"], requested)
        self.assertEqual(runtime["device"], "cuda:0")
        self.assertEqual(runtime["dtype"], dtype)
        self.assertEqual(runtime["attention"], "sdpa")
        self.assertEqual(runtime["reason"], "cuda_available")

    def test_device_defaults_to_auto_and_reads_only_the_chat_setting(self):
        self.assertEqual(requested_device({}), "auto")
        self.assertEqual(requested_device({"CUDA_VISIBLE_DEVICES": "0"}), "auto")
        for device in ("auto", "cpu", "cuda"):
            with self.subTest(device=device):
                self.assertEqual(requested_device({"MDD_CHAT_DEVICE": device}), device)
        with patch.dict("os.environ", {"MDD_CHAT_DEVICE": "cpu"}):
            self.assertEqual(requested_device(), "cpu")

    def test_unknown_configuration_fails_before_any_hardware_probe(self):
        torch = fake_torch()
        with self.assertRaises(ValueError):
            requested_device({"MDD_CHAT_DEVICE": "cuda:42"})
        with self.assertRaises(ValueError):
            select_runtime(torch, requested="gpu")
        torch.cuda.is_available.assert_not_called()
        torch.cuda.mem_get_info.assert_not_called()
        torch.cuda.is_bf16_supported.assert_not_called()

    def test_explicit_cpu_never_touches_cuda_even_if_detection_would_fail(self):
        torch = fake_torch()
        torch.cuda.is_available.side_effect = RuntimeError("driver unavailable")
        self.assert_cpu(select_runtime(torch, requested="cpu"),
                        requested="cpu", reason="cpu_requested")
        torch.cuda.is_available.assert_not_called()
        torch.cuda.mem_get_info.assert_not_called()
        torch.cuda.is_bf16_supported.assert_not_called()

    def test_auto_without_cuda_stays_on_cpu(self):
        torch = fake_torch(available=False)
        self.assert_cpu(select_runtime(torch, requested="auto"),
                        requested="auto", reason="cuda_unavailable")
        torch.cuda.mem_get_info.assert_not_called()
        torch.cuda.is_bf16_supported.assert_not_called()

    def test_auto_reserves_gpu_when_training_or_other_work_uses_memory(self):
        torch = fake_torch(free_mib=2047)
        self.assert_cpu(select_runtime(torch, requested="auto"),
                        requested="auto", reason="cuda_memory_low")
        torch.cuda.mem_get_info.assert_called_once()
        torch.cuda.is_bf16_supported.assert_not_called()

    def test_auto_accepts_exact_minimum_free_memory(self):
        torch = fake_torch(free_mib=2048)
        self.assert_gpu(select_runtime(torch, requested="auto"),
                        requested="auto", dtype="bfloat16")

    def test_gpu_uses_fp16_when_bf16_is_not_supported(self):
        torch = fake_torch(bf16=False)
        self.assert_gpu(select_runtime(torch, requested="auto"),
                        requested="auto", dtype="float16")

    def test_explicit_cuda_bypasses_the_automatic_free_memory_gate(self):
        torch = fake_torch(free_mib=1)
        self.assert_gpu(select_runtime(torch, requested="cuda"),
                        requested="cuda", dtype="bfloat16")

    def test_explicit_cuda_does_not_silently_fallback_when_unavailable(self):
        torch = fake_torch(available=False)
        with self.assertRaises(RuntimeError):
            select_runtime(torch, requested="cuda")
        torch.cuda.mem_get_info.assert_not_called()

    def test_auto_falls_back_after_driver_or_memory_probe_failures(self):
        for probe in ("is_available", "mem_get_info", "is_bf16_supported"):
            for exception in (RuntimeError, OSError):
                with self.subTest(probe=probe, exception=exception.__name__):
                    torch = fake_torch()
                    getattr(torch.cuda, probe).side_effect = exception("CUDA probe failed")
                    self.assert_cpu(select_runtime(torch, requested="auto"),
                                    requested="auto", reason="cuda_probe_failed")

    def test_default_selection_honors_cpu_environment_without_starting_processes(self):
        torch = fake_torch()
        with patch.dict("os.environ", {"MDD_CHAT_DEVICE": "cpu"}), \
                patch("subprocess.Popen") as start_process:
            self.assert_cpu(select_runtime(torch), requested="cpu", reason="cpu_requested")
        start_process.assert_not_called()
        torch.cuda.is_available.assert_not_called()

    def test_offline_engine_stays_cpu_unless_its_caller_selects_another_policy(self):
        with patch.dict("os.environ", {"MDD_CHAT_DEVICE": "auto"}), \
                patch("subprocess.Popen") as start_process:
            offline = QwenChatEngine()
            online = QwenChatEngine(device="auto")
        self.assertEqual(offline.requested_device, "cpu")
        self.assertEqual(offline.device, "cpu")
        self.assertIsNone(offline.model)
        self.assertEqual(online.requested_device, "auto")
        self.assertIsNone(online.model)
        start_process.assert_not_called()

    def test_ipc_keeps_explicit_cpu_requests_cpu_even_with_auto_environment(self):
        with patch.dict("os.environ", {"MDD_CHAT_DEVICE": "auto"}), \
                patch.object(local_chat_process, "_request", return_value={}) as send:
            local_chat_process.complete(
                [{"role": "user", "content": "Hello"}],
                {"device": "cpu", "adapter": None}, 40, 5.,
                on_text=lambda _: None, cancelled=threading.Event())
        self.assertEqual(send.call_args.kwargs["device_policy"], "cpu")
        self.assertEqual(send.call_args.args[0]["model_id"], "qwen-base")

    def test_policy_change_restarts_owned_worker_before_sending_the_request(self):
        old_process = SimpleNamespace(poll=Mock(return_value=None))
        new_process = SimpleNamespace(poll=Mock(return_value=None))
        new_events = Queue()
        new_events.put({"id": "request-id", "type": "done", "metadata": {"finish_reason": "stop"}})
        order = []

        def stop_worker():
            order.append("shutdown")
            local_chat_process._PROCESS = None
            local_chat_process._EVENTS = None
            local_chat_process._DEVICE_POLICY = None

        def start_worker(policy):
            order.append("start:" + policy)
            local_chat_process._PROCESS = new_process
            local_chat_process._EVENTS = new_events
            local_chat_process._DEVICE_POLICY = policy

        def send_request(payload):
            order.append("send")
            self.assertIs(local_chat_process._PROCESS, new_process)
            self.assertEqual(local_chat_process._DEVICE_POLICY, "auto")
            self.assertEqual(payload["id"], "request-id")

        with patch.dict("os.environ", {"MDD_CHAT_DEVICE": "auto"}), \
                patch.object(local_chat_process, "_PROCESS", old_process), \
                patch.object(local_chat_process, "_EVENTS", Queue()), \
                patch.object(local_chat_process, "_DEVICE_POLICY", "cpu"), \
                patch.object(local_chat_process, "_shutdown_worker", side_effect=stop_worker), \
                patch.object(local_chat_process, "_start", side_effect=start_worker), \
                patch.object(local_chat_process, "_send", side_effect=send_request), \
                patch.object(local_chat_process.secrets, "token_hex", return_value="request-id"), \
                patch("subprocess.Popen") as start_process:
            result = local_chat_process._request(
                {"type": "generate", "model_id": "qwen-base"}, 5.,
                on_text=lambda _: None, cancelled=threading.Event())
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(order, ["shutdown", "start:auto", "send"])
        start_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
