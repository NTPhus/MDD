"""Local voice routing, setup readiness, and audio boundaries without TTS requests."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.demo_chat import ChatError, ChatJob, ChatService
from mdd.tts_registry import kokoro_status


class KokoroChatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.service = ChatService(self.root, capability_loader=lambda: {})
        self.addCleanup(self.service.close)
        self.job = ChatJob('qwen-base', status='ready', text='How was your day?')
        self.service._jobs[self.job.identifier] = self.job
        self.payload = {'job_id': self.job.identifier, 'locale': 'en-US'}

    def install_fixture(self):
        model = self.root / 'models/pretrained/kokoro-82m'
        model.mkdir(parents=True)
        runtime = self.root / '.venv-kokoro' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b'fixture runtime')
        self.manifest = {'status': 'ready', 'model_file': 'kokoro-v1.0.int8.onnx',
                         'voices_file': 'voices-v1.0.bin', 'voices': ['af_heart', 'bf_emma'],
                         'default_voice_id': 'af_heart'}
        for key in ('model_file', 'voices_file'):
            (model / self.manifest[key]).write_bytes(b'fixture weights')
        self.manifest_path = model / 'setup_manifest.json'
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def client(self, provider='kokoro'):
        cache = self.service.kokoro_cache_dir if provider == 'kokoro' else self.service.cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        audio = cache / 'fixture.wav'
        audio.write_bytes(b'RIFF\x00\x00\x00\x00WAVEfixture')
        client = SimpleNamespace(synthesize=Mock(return_value=audio), resume=Mock(return_value=audio),
                                 can_resume=Mock(return_value=False), last_timing={})
        return client

    def finished(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            voice = self.service.poll(self.job.identifier)['voice']
            if voice['status'] != 'generating':
                return voice
            time.sleep(.01)
        self.fail('Voice worker did not finish')

    def test_setup_readiness_requires_complete_local_assets_and_filters_catalog(self):
        self.assertFalse(kokoro_status(self.root)['ready'])
        self.install_fixture()
        self.manifest['voices'].extend(['untrusted_voice', '../.env'])
        self.save_manifest()
        status = kokoro_status(self.root)
        self.assertTrue(status['ready'])
        self.assertEqual([voice['id'] for voice in status['voices']], ['af_heart', 'bf_emma'])
        (self.manifest_path.parent / self.manifest['model_file']).unlink()
        self.assertFalse(kokoro_status(self.root)['ready'])

    def test_manifest_cannot_point_outside_model_directory(self):
        self.install_fixture()
        outside = self.manifest_path.parent.parent / 'outside.onnx'
        outside.write_bytes(b'outside fixture')
        self.manifest['model_file'] = '../outside.onnx'
        self.save_manifest()
        self.assertFalse(kokoro_status(self.root)['ready'])

    def test_config_prefers_installed_kokoro_without_loading_any_runtime(self):
        self.install_fixture()
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        with patch.dict(os.environ, {'AI33_VOICE_EN': 'edge_en-US-AvaNeural'}), patch.object(
                self.service, '_module', side_effect=AssertionError('config must not load models')):
            config = self.service.config()
        self.assertTrue(config['kokoro_available'])
        self.assertEqual(config['default_voice_provider'], 'kokoro')
        self.assertEqual(config['kokoro_default_voice_id'], 'af_heart')
        self.assertEqual(config['english_voices'][0]['id'], 'edge_en-US-AvaNeural')
        self.assertEqual(config['voice_locales'], ['en-US'])
        self.assertNotIn(str(self.root), json.dumps(config))

    def test_default_voice_uses_only_local_provider_and_server_owned_text(self):
        self.install_fixture()
        client = self.client()
        with patch.object(self.service, '_module', return_value=SimpleNamespace(KokoroTTS=Mock(return_value=client))) as module:
            self.service.voice(self.payload)
            voice = self.finished()
        self.assertEqual(voice['status'], 'ready')
        self.assertEqual(voice['provider'], 'kokoro')
        self.assertEqual(voice['voice_id'], 'af_heart')
        module.assert_called_once_with('kokoro_tts')
        client.synthesize.assert_called_once_with(self.job.text, locale='en-US', voice_id='af_heart')
        self.assertEqual(self.service.audio(voice['url'].rsplit('/', 1)[1])[1], 'audio/wav')
        self.service.voice(self.payload)
        self.assertEqual(client.synthesize.call_count, 1)
        with self.assertRaises(ChatError) as caught:
            self.service.voice({**self.payload, 'provider': 'ai33'})
        self.assertEqual(caught.exception.code, 'chat_voice_selection')

    def test_explicit_ai33_remains_available_with_local_kokoro_ready(self):
        self.install_fixture()
        client = self.client('ai33')
        self.service.tts_factory = lambda: client
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        self.service.voice({**self.payload, 'provider': 'ai33'})
        voice = self.finished()
        self.assertEqual(voice['status'], 'ready')
        self.assertEqual(voice['provider'], 'ai33')
        client.synthesize.assert_called_once_with(self.job.text, locale='en-US')

    def test_invalid_provider_locale_and_cross_provider_voices_never_start_work(self):
        self.install_fixture()
        self.service.tts_factory = Mock()
        for change in ({'provider': 'remote'}, {'provider': []}, {'provider': None},
                       {'provider': 'kokoro', 'locale': 'vi-VN'},
                       {'provider': 'kokoro', 'voice_id': 'edge_en-US-AvaNeural'},
                       {'provider': 'ai33', 'voice_id': 'af_heart'},
                       {'provider': 'kokoro', 'voice_id': '../.env'}):
            with self.subTest(change=change), self.assertRaises(ChatError):
                self.service.voice({**self.payload, **change})
        self.service.tts_factory.assert_not_called()

    def test_local_failure_does_not_start_ai33_or_expose_private_errors(self):
        self.install_fixture()
        client = self.client()
        client.synthesize.side_effect = RuntimeError('private path or response')
        client.last_timing = {'synthesis_sec': .15, 'cache_hit': False, 'private': 'secret'}
        with patch.object(self.service, '_module', return_value=SimpleNamespace(KokoroTTS=Mock(return_value=client))) as module:
            self.service.voice({**self.payload, 'provider': 'kokoro', 'speed': .85})
            voice = self.finished()
        module.assert_called_once_with('kokoro_tts')
        self.assertEqual(voice['status'], 'error')
        self.assertEqual(voice['provider'], 'kokoro')
        self.assertFalse(voice['retryable'])
        self.assertEqual(voice['timing']['synthesis_sec'], .15)
        self.assertNotIn('private', json.dumps(voice))
        self.assertNotIn('secret', json.dumps(voice))

    def test_audio_reader_accepts_only_two_provider_cache_roots(self):
        for provider in ('kokoro', 'ai33'):
            self.assertEqual(self.service._read_voice(self.client(provider).synthesize.return_value)[1], 'audio/wav')
        outside = self.root / 'data/generated/tts/outside.wav'
        outside.write_bytes(b'RIFF\x00\x00\x00\x00WAVEfixture')
        with self.assertRaises(ChatError):
            self.service._read_voice(outside)


if __name__ == '__main__':
    unittest.main()
