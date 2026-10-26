"""CUDA policy, DLL/allocator failures and lazy decoding without GPU jobs."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import whisper_asr as whisper
from mdd.demo_model_registry import speech_device_policy


class WhisperGPUTests(unittest.TestCase):
    def test_cpu_never_probes_cuda(self):
        with patch.object(whisper, 'verify'), patch.object(whisper, '_load_runtime', return_value=Mock()), \
                patch.object(whisper, '_cuda_available') as probe:
            whisper.WhisperASR(ROOT, device='cpu').load()
        probe.assert_not_called()

    def test_auto_uses_int8_float16_and_offline_weights(self):
        model = Mock()
        engine = whisper.WhisperASR(ROOT, device='auto')
        with patch.object(whisper, 'verify'), patch.object(whisper, '_load_runtime', return_value=model), \
                patch.object(whisper, '_cuda_available', return_value=True):
            engine.load()
        self.assertEqual((engine.device, engine.compute_type), ('cuda', 'int8_float16'))
        self.assertEqual(model.call_args.kwargs['compute_type'], 'int8_float16')
        self.assertTrue(model.call_args.kwargs['local_files_only'])

    def test_auto_fallback_and_strict_cuda_when_unavailable(self):
        with patch.object(whisper, 'verify'), patch.object(whisper, '_load_runtime', return_value=Mock()), \
                patch.object(whisper, '_cuda_available', return_value=False):
            engine = whisper.WhisperASR(ROOT, device='auto').load()
            self.assertEqual(engine.device, 'cpu')
            self.assertEqual(engine.fallback_reason, 'cuda_unavailable')
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                whisper.WhisperASR(ROOT, device='cuda').load()

    def test_missing_cuda_dll_auto_falls_back(self):
        with patch.object(whisper, 'verify'), patch.object(whisper, '_load_runtime', return_value=Mock()), \
                patch.object(whisper, '_cuda_available', side_effect=OSError('cublas DLL missing')):
            engine = whisper.WhisperASR(ROOT, device='auto').load()
        self.assertEqual(engine.device, 'cpu')
        self.assertEqual(engine.fallback_reason, 'cuda_probe_failed')

    def test_load_oom_cpu_retry_is_not_used_for_corrupt_model_errors(self):
        for error, fallback in [(RuntimeError('CUDA out of memory'), True), (ValueError('invalid vocabulary'), False)]:
            factory = Mock(side_effect=[error, Mock()])
            engine = whisper.WhisperASR(ROOT, device='auto')
            with patch.object(whisper, 'verify'), patch.object(whisper, '_load_runtime', return_value=factory), \
                    patch.object(whisper, '_cuda_available', return_value=True):
                if fallback:
                    engine.load()
                    self.assertEqual(engine.device, 'cpu')
                    self.assertEqual(factory.call_count, 2)
                else:
                    with self.assertRaises(ValueError): engine.load()
                    self.assertEqual(factory.call_count, 1)

    def test_lazy_cuda_oom_discards_partial_words_and_retries_complete_audio(self):
        import numpy as np
        segment = SimpleNamespace(start=0., end=.5, text=' partial', avg_logprob=0., no_speech_prob=0., words=[])
        def failed_segments():
            yield segment
            raise RuntimeError('CUDA out of memory')
        info = SimpleNamespace(language='en', language_probability=1.)
        gpu = Mock(transcribe=Mock(return_value=(failed_segments(), info)))
        good = SimpleNamespace(start=0., end=.5, text=' complete', avg_logprob=0., no_speech_prob=0., words=[])
        cpu = Mock(transcribe=Mock(return_value=(iter([good]), info)))
        engine = whisper.WhisperASR(ROOT, device='auto')
        engine.model, engine.device, engine.compute_type = gpu, 'cuda', 'int8_float16'
        factory = Mock(return_value=cpu)
        with patch.object(whisper, '_load_runtime', return_value=factory):
            result = engine.transcribe(np.ones(16000, dtype='float32'))
        self.assertEqual(result['raw_text'], 'complete')
        self.assertEqual(result['device'], 'cpu')
        self.assertEqual(result['fallback_reason'], 'cuda_inference_failed')
        gpu.model.unload_model.assert_called_once()
        self.assertEqual(cpu.transcribe.call_args.args[0].shape, (16000,))

    def test_real_model_errors_and_cancellation_never_become_cpu_retry(self):
        import numpy as np
        for error in (ValueError('bad input'), InterruptedError('cancelled'), RuntimeError('bad tokenizer')):
            engine = whisper.WhisperASR(ROOT, device='auto')
            engine.device = 'cuda'
            engine.model = Mock(transcribe=Mock(side_effect=error))
            with patch.object(whisper, '_load_runtime') as factory:
                with self.assertRaises(type(error)):
                    engine.transcribe(np.ones(16000, dtype='float32'))
            factory.assert_not_called()

    def test_device_policies_keep_global_cpu_and_allow_individual_override(self):
        for component in ('asr','phoneme','crottc'):
            self.assertEqual(speech_device_policy(component, {}), 'auto')
            self.assertEqual(speech_device_policy(component, {'MDD_CHAT_DEVICE':'cpu'}), 'cpu')
        self.assertEqual(speech_device_policy('asr', {'MDD_CHAT_DEVICE':'cpu','MDD_CHAT_ASR_DEVICE':'cuda'}), 'cuda')
        with self.assertRaises(ValueError): speech_device_policy('asr', {'MDD_CHAT_ASR_DEVICE':'bad'})


if __name__ == '__main__': unittest.main()
