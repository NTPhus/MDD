"""Provider selection and CUDA setup use fake ORT sessions, never a real GPU."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import kokoro_worker as worker


class KokoroDeviceTests(unittest.TestCase):
    def config(self,device='cuda'):
        return worker.runtime_config({'device':device},environ={})

    def session(self,providers):
        return SimpleNamespace(get_providers=Mock(return_value=providers),disable_fallback=Mock())

    def ort(self,providers,session):
        return SimpleNamespace(get_available_providers=Mock(return_value=providers),
                               InferenceSession=Mock(return_value=session),preload_dlls=Mock())

    def test_environment_overrides_manifest_and_cpu_is_legacy_default(self):
        self.assertEqual(worker.runtime_config({},environ={})['requested_device'],'cpu')
        settings=worker.runtime_config({'device':'cpu','gpu_mem_limit_mb':768},
            environ={'KOKORO_DEVICE':'cuda','KOKORO_GPU_MEM_LIMIT_MB':'512'})
        self.assertEqual(settings['requested_device'],'cuda')
        self.assertEqual(settings['cuda_provider_options']['gpu_mem_limit'],str(512*1024*1024))
        for record in ({'device':'invalid'},{'device':[]},{'gpu_mem_limit_mb':True},{'gpu_mem_limit_mb':0},
                       {'cuda_dll_directory':'../outside'}):
            with self.subTest(record=record),self.assertRaises(worker.WorkerError):
                worker.runtime_config(record,environ={})

    def test_strict_cuda_fails_before_session_when_provider_is_unavailable(self):
        ort=self.ort(['CPUExecutionProvider'],self.session(['CPUExecutionProvider']))
        with self.assertRaises(worker.WorkerError) as caught:
            worker.create_session(ort,'fixture.onnx',object(),self.config())
        self.assertEqual(caught.exception.code,'kokoro_cuda_unavailable')
        ort.InferenceSession.assert_not_called()

    def test_strict_cuda_rejects_silent_session_fallback(self):
        ort=self.ort(['CUDAExecutionProvider','CPUExecutionProvider'],self.session(['CPUExecutionProvider']))
        with self.assertRaises(worker.WorkerError) as caught:
            worker.create_session(ort,'fixture.onnx',object(),self.config())
        self.assertEqual(caught.exception.code,'kokoro_cuda_fallback')
        self.assertEqual(ort.InferenceSession.call_count,1)

    def test_cuda_receives_bounded_options_disables_fallback_and_reports_provider(self):
        session=self.session(['CUDAExecutionProvider','CPUExecutionProvider'])
        ort=self.ort(['CUDAExecutionProvider','CPUExecutionProvider'],session)
        actual,telemetry=worker.create_session(ort,'fixture.onnx',object(),self.config())
        self.assertIs(actual,session)
        arguments=ort.InferenceSession.call_args.kwargs
        self.assertFalse(arguments['enable_fallback'])
        provider,options=arguments['providers'][0]
        self.assertEqual(provider,'CUDAExecutionProvider')
        self.assertEqual(options['device_id'],'0')
        self.assertEqual(options['gpu_mem_limit'],str(768*1024*1024))
        self.assertEqual(options['arena_extend_strategy'],'kSameAsRequested')
        self.assertEqual(options['cudnn_conv_algo_search'],'HEURISTIC')
        self.assertEqual(options['cudnn_conv_use_max_workspace'],'0')
        self.assertEqual(options['do_copy_in_default_stream'],'1')
        self.assertEqual(options['use_tf32'],'1')
        session.disable_fallback.assert_called_once_with()
        self.assertEqual(telemetry['requested_device'],'cuda')
        self.assertEqual(telemetry['effective_device'],'cuda')
        self.assertEqual(telemetry['execution_provider'],'CUDAExecutionProvider')
        ort.preload_dlls.assert_called_once_with(directory=None)

    def test_auto_uses_cpu_when_cuda_is_not_available(self):
        session=self.session(['CPUExecutionProvider'])
        ort=self.ort(['CPUExecutionProvider'],session)
        _,telemetry=worker.create_session(ort,'fixture.onnx',object(),self.config('auto'))
        self.assertEqual(telemetry['effective_device'],'cpu')
        self.assertEqual(telemetry['fallback_reason'],'cuda_unavailable')
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'],['CPUExecutionProvider'])
        ort.preload_dlls.assert_not_called()

    def test_cuda_initialization_error_is_typed_and_auto_can_fall_back(self):
        ort=self.ort(['CUDAExecutionProvider'],self.session(['CPUExecutionProvider']))
        ort.preload_dlls.side_effect=OSError('private DLL path')
        with self.assertRaises(worker.WorkerError) as caught:
            worker.create_session(ort,'fixture.onnx',object(),self.config())
        self.assertEqual(caught.exception.code,'kokoro_cuda_load_failed')
        self.assertNotIn('private',str(caught.exception))
        _,telemetry=worker.create_session(ort,'fixture.onnx',object(),self.config('auto'))
        self.assertEqual(telemetry['effective_device'],'cpu')
        self.assertEqual(telemetry['fallback_reason'],'cuda_load_failed')


if __name__=='__main__':unittest.main()
