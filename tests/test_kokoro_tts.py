"""Exercise local TTS cache and input boundaries without starting a model worker."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import kokoro_tts


class KokoroTTSTests(unittest.TestCase):
    def setUp(self):
        device_patch=patch.dict(os.environ,{'KOKORO_DEVICE':'cpu'})
        device_patch.start()
        self.addCleanup(device_patch.stop)
        kokoro_tts.CACHE.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='unit-kokoro-', dir=kokoro_tts.CACHE)
        self.cache = Path(self.temporary.name).resolve()
        self.assertTrue(self.cache.is_relative_to(kokoro_tts.CACHE.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.record = {
            'status': 'ready', 'model_file': 'fixture-model.onnx', 'voices_file': 'fixture-voices.bin',
            'voices': ['af_heart', 'af_bella', 'bf_emma'], 'default_voice_id': 'af_heart',
            'files': {'fixture-model.onnx': {'sha256': '1' * 64},
                      'fixture-voices.bin': {'sha256': '2' * 64}},
        }
        manifest_patch = patch.object(kokoro_tts, 'manifest', side_effect=lambda: deepcopy(self.record))
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)
        worker_patch = patch.object(kokoro_tts, '_request', side_effect=self.write_audio)
        self.worker = worker_patch.start()
        self.addCleanup(worker_patch.stop)
        start_patch = patch.object(kokoro_tts, '_start', side_effect=AssertionError('A test must not start a model'))
        self.start=start_patch.start()
        self.addCleanup(start_patch.stop)
        self.client = kokoro_tts.KokoroTTS(cache_dir=self.cache)

    def write_audio(self, text, voice, speed, path):
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(b'\x00\x00' * 2400)
        return {'path': str(path), 'sample_rate': 24000, 'audio_duration_sec': .1,
                'synthesis_sec': .01, 'load_sec': .02, 'real_time_factor': .1,
                'private_text': text, 'private_path': 'file:///private/runtime', 'unexpected': {'secret': 'secret'}}

    def metadata(self, path):
        return json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))

    def save_metadata(self, path, value):
        path.with_suffix('.json').write_text(json.dumps(value), encoding='utf-8')

    def test_repeated_text_is_idempotent_and_cache_survives_new_client(self):
        first = self.client.synthesize(' Hello there. ', voice_id='af_heart', speed=1)
        self.assertFalse(self.client.last_timing['cache_hit'])
        with wave.open(str(first), 'rb') as audio:
            self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getnframes()), (1, 24000, 2400))
        self.worker.assert_called_once_with('Hello there.', 'af_heart', 1., first)
        second_client = kokoro_tts.KokoroTTS(cache_dir=self.cache)
        self.assertEqual(second_client.synthesize('Hello there.', voice_id='af_heart', speed=1.), first)
        self.assertTrue(second_client.last_timing['cache_hit'])
        self.assertEqual(second_client.last_timing['synthesis_sec'], 0.)
        self.assertEqual(self.worker.call_count, 1)

    def test_corrupt_audio_regenerates_instead_of_reusing_invalid_cache(self):
        path = self.client.synthesize('Please say hello.')
        original = path.read_bytes()
        path.write_bytes(original[:-4] + b'bad!')
        self.assertEqual(self.client.synthesize('Please say hello.'), path)
        self.assertEqual(self.worker.call_count, 2)
        self.assertFalse(self.client.last_timing['cache_hit'])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.metadata(path)['audio_sha256'], hashlib.sha256(original).hexdigest())

    def test_mismatched_request_or_unreadable_metadata_regenerates(self):
        path = self.client.synthesize('Good morning.')
        value = self.metadata(path)
        value['request']['voice_id'] = 'af_bella'
        self.save_metadata(path, value)
        self.client.synthesize('Good morning.')
        self.assertEqual(self.worker.call_count, 2)
        path.with_suffix('.json').write_text('{not valid json', encoding='utf-8')
        self.client.synthesize('Good morning.')
        self.assertEqual(self.worker.call_count, 3)
        self.assertEqual(self.metadata(path)['request']['voice_id'], 'af_heart')

    def test_voice_speed_locale_and_model_identity_have_separate_cache_entries(self):
        paths = [self.client.synthesize('Hello.'),
                 self.client.synthesize('Hello.', voice_id='af_bella'),
                 self.client.synthesize('Hello.', speed=.85),
                 self.client.synthesize('Hello.', locale='en-GB')]
        self.record['files']['fixture-model.onnx']['sha256'] = '3' * 64
        changed_model = kokoro_tts.KokoroTTS(cache_dir=self.cache)
        paths.append(changed_model.synthesize('Hello.'))
        self.record['files']['fixture-voices.bin']['sha256'] = '4' * 64
        changed_voices = kokoro_tts.KokoroTTS(cache_dir=self.cache)
        paths.append(changed_voices.synthesize('Hello.'))
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(self.worker.call_count, 6)
        self.assertTrue(all(path.parent == self.cache for path in paths))

    def test_force_refresh_regenerates_only_the_selected_cache_entry(self):
        path = self.client.synthesize('Refresh this sentence.')
        self.assertEqual(self.client.synthesize('Refresh this sentence.', force=True), path)
        self.assertEqual(self.worker.call_count, 2)
        self.assertFalse(self.client.last_timing['cache_hit'])

    def test_invalid_input_never_reaches_worker(self):
        cases = [({'text': value}, 'kokoro_invalid_text') for value in (None, 42, [], '', '  ', 'a' * 5001)]
        cases += [({'locale': value}, 'kokoro_language_unsupported') for value in ('vi-VN', 'unknown', None, 42, [])]
        cases += [({'voice_id': value}, 'kokoro_invalid_voice') for value in ('../af_heart', 'af_missing', 'zf_voice', 42, [], False)]
        cases += [({'speed': value}, 'kokoro_invalid_speed') for value in (False, '1.0', [], 0, 1.51, float('nan'), float('inf'))]
        for change, code in cases:
            arguments = {'text': 'Speak this sentence.', **change}
            with self.subTest(change=change), self.assertRaises(kokoro_tts.KokoroError) as caught:
                self.client.synthesize(**arguments)
            self.assertEqual(caught.exception.code, code)
        self.worker.assert_not_called()

    def test_cache_directory_must_stay_inside_provider_root(self):
        with self.assertRaises(kokoro_tts.KokoroError) as caught:
            kokoro_tts.KokoroTTS(cache_dir=kokoro_tts.CACHE.parent / 'kokoro-unsafe-sibling')
        self.assertEqual(caught.exception.code, 'kokoro_invalid_cache')

    def test_worker_cannot_return_audio_path_outside_requested_cache(self):
        def wrong_path(text, voice, speed, path):
            value = self.write_audio(text, voice, speed, path)
            value['path'] = str(kokoro_tts.CACHE.parent / 'outside.wav')
            return value
        self.worker.side_effect = wrong_path
        with self.assertRaises(kokoro_tts.KokoroError) as caught:
            self.client.synthesize('Reject an unexpected output path.')
        self.assertEqual(caught.exception.code, 'kokoro_invalid_output')
        self.assertEqual(list(self.cache.glob('*.json')), [])

    def test_public_timing_contains_only_safe_finite_primitive_metrics(self):
        sentence = 'Private learner sentence.'
        self.client.synthesize(sentence)
        expected = {'total_sec', 'cache_lookup_sec', 'cache_hit', 'new_tasks_submitted',
                    'synthesis_sec', 'load_sec', 'audio_duration_sec', 'real_time_factor',
                    'requested_device','effective_device','execution_provider','provider_options'}
        for cached in (False, True):
            if cached:
                self.client.synthesize(sentence)
            timing = self.client.last_timing
            self.assertEqual(set(timing), expected)
            self.assertIs(timing['cache_hit'], cached)
            for key, value in timing.items():
                if key not in {'cache_hit','requested_device','effective_device','execution_provider','provider_options'}:
                    self.assertIn(type(value), (int, float))
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0)
            serialized = json.dumps(timing)
            self.assertNotIn(sentence, serialized)
            self.assertNotIn('private', serialized)
            self.assertNotIn('secret', serialized)
            self.assertNotIn(str(self.cache), serialized)

    def test_cuda_never_reuses_legacy_cpu_cache_and_reports_actual_provider(self):
        cpu_path=self.client.synthesize('Use the selected processor.')
        self.start.side_effect=None
        self.start.return_value={'effective_device':'cuda'}
        with patch.dict(os.environ,{'KOKORO_DEVICE':'cuda'}):
            gpu_path=self.client.synthesize('Use the selected processor.')
            self.assertNotEqual(gpu_path,cpu_path)
            self.assertEqual(self.worker.call_count,2)
            self.assertEqual(self.client.last_timing['execution_provider'],'CUDAExecutionProvider')
            self.assertEqual(self.client.last_timing['effective_device'],'cuda')
            self.assertEqual(self.metadata(gpu_path)['request']['execution_device'],'cuda')
            self.assertEqual(self.client.synthesize('Use the selected processor.'),gpu_path)
            self.assertTrue(self.client.last_timing['cache_hit'])
            self.assertEqual(self.worker.call_count,2)
        self.assertEqual(self.client.synthesize('Use the selected processor.'),cpu_path)

    def test_requested_cuda_rejects_cpu_worker_even_when_cpu_cache_exists(self):
        self.client.synthesize('This cached sentence belongs to CPU.')
        self.start.side_effect=None
        self.start.return_value={'effective_device':'cpu'}
        with patch.dict(os.environ,{'KOKORO_DEVICE':'cuda'}),self.assertRaises(kokoro_tts.KokoroError) as caught:
            self.client.synthesize('This cached sentence belongs to CPU.')
        self.assertEqual(caught.exception.code,'kokoro_cuda_fallback')
        self.assertEqual(self.worker.call_count,1)

    def test_worker_key_changes_with_device_memory_and_dll_directory(self):
        record=deepcopy(self.record)
        cpu=kokoro_tts._worker_key(record)
        with patch.dict(os.environ,{'KOKORO_DEVICE':'cuda'}):
            gpu=kokoro_tts._worker_key(record)
            with patch.dict(os.environ,{'KOKORO_GPU_MEM_LIMIT_MB':'512'}):
                constrained=kokoro_tts._worker_key(record)
                record['cuda_dll_directory']=str(self.cache.relative_to(kokoro_tts.ROOT))
                directory=kokoro_tts._worker_key(record)
        self.assertEqual(len({cpu,gpu,constrained,directory}),4)

    def test_cold_cuda_load_started_during_provider_check_is_included_once(self):
        ready={'effective_device':'cuda','load_sec':1.5}
        def start():
            kokoro_tts._PROCESS=object()
            kokoro_tts._READY=ready
            return ready
        def synthesize(text,voice,speed,path):
            result=self.write_audio(text,voice,speed,path)
            result['load_sec']=0.
            return result
        self.start.side_effect=start
        self.worker.side_effect=synthesize
        with patch.object(kokoro_tts,'_PROCESS',None),patch.object(kokoro_tts,'_READY',None),patch.dict(
                os.environ,{'KOKORO_DEVICE':'cuda'}):
            self.client.synthesize('Include CUDA startup in the first result.')
            self.assertEqual(self.client.last_timing['load_sec'],1.5)
            self.assertEqual(self.worker.call_count,1)

    def test_cpu_load_is_taken_from_request_without_preflight_double_counting(self):
        def synthesize(text,voice,speed,path):
            result=self.write_audio(text,voice,speed,path)
            result['load_sec']=1.25
            return result
        self.worker.side_effect=synthesize
        self.client.synthesize('CPU worker loading happens inside the request.')
        self.assertEqual(self.client.last_timing['load_sec'],1.25)
        self.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
