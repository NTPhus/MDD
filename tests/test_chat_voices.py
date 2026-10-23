"""Voice selection, cache isolation, and sample downloads without paid requests."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.chat_voices import load_voices, normalize_voice, preview_stem
from mdd.demo_chat import ChatError, ChatJob, ChatService
from scripts.ai33_tts import AI33Error, AI33TTS
from tests.test_ai33_tts import Response


class ChatVoiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {'AI33_API_KEY': 'test-only', 'AI33_VOICE_EN': 'edge_en-US-AvaNeural'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.voice = {'id': 'edge_en-GB-SoniaNeural', 'name': 'Sonia', 'locale': 'en-GB', 'gender': 'female',
                      'provider': 'edge', 'preview_url': 'https://cdn.ai33.pro/persists/samples/sonia.mp3'}
        (self.root / 'configs').mkdir()
        (self.root / 'configs/ai33_voices.json').write_text(json.dumps({'voices': [self.voice]}))
        self.service = ChatService(self.root, capability_loader=lambda: {'tts_configured': True, 'english_voice_configured': True})
        self.audio = self.service.cache_dir / 'reply.mp3'
        self.audio.parent.mkdir(parents=True)
        self.audio.write_bytes(b'ID3fixture-audio')
        self.client = Mock()
        self.client.synthesize.return_value = self.audio
        self.service.tts_factory = lambda: self.client
        self.job = ChatJob('qwen-base', status='ready', text='What do you enjoy about your work?')
        self.service._jobs[self.job.identifier] = self.job
        self.payload = {'job_id': self.job.identifier, 'locale': 'en-US', 'voice_id': self.voice['id'], 'speed': .85}

    def finish_voice(self, field='voice'):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.service.poll(self.job.identifier)
            if result[field]['status'] != 'generating':
                return result[field]
            time.sleep(.01)
        self.fail('Voice worker did not finish')

    def test_selected_voice_speed_reach_tts_and_repeat_does_not_create_another_task(self):
        self.service.voice(self.payload)
        result = self.finish_voice()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['voice_id'], self.voice['id'])
        self.assertEqual(result['speed'], .85)
        self.assertEqual(set(result['timing']), {'backend_total_sec'})
        self.assertGreaterEqual(result['timing']['backend_total_sec'], 0)
        self.client.synthesize.assert_called_once_with(self.job.text, locale='en-US', voice_id=self.voice['id'], speed=.85)
        self.service.voice(self.payload)
        self.assertEqual(self.client.synthesize.call_count, 1)
        with self.assertRaisesRegex(ChatError, 'tốc độ khác'):
            self.service.voice({**self.payload, 'speed': 1})

    def test_public_voice_timing_has_only_allowed_primitive_metrics(self):
        self.client.last_timing = {
            'total_sec': 5.2, 'cache_lookup_sec': .01, 'submit_sec': .4, 'wait_sec': 4.6, 'download_sec': .1,
            'cache_hit': False, 'resumed_task': True, 'new_tasks_submitted': 0,
            'request_count': 3, 'poll_count': 2, 'download_attempts': 1,
            'audio_url': 'https://cdn.example/audio?signature=secret', 'task_id': 'private-task',
            'key': 'test-only', 'text': 'Private conversation', 'nested': {'secret': 'secret'},
        }
        self.service.voice(self.payload)
        result = self.finish_voice()
        self.assertEqual(result['status'], 'ready')
        timing = result['timing']
        self.assertEqual(timing['total_sec'], 5.2)
        self.assertEqual(timing['request_count'], 3)
        self.assertTrue(timing['resumed_task'])
        self.assertFalse(timing['cache_hit'])
        self.assertGreaterEqual(timing['backend_total_sec'], 0)
        self.assertEqual(set(timing), set(self.client.last_timing) - {'audio_url', 'task_id', 'key', 'text', 'nested'} | {'backend_total_sec'})
        self.assertNotIn('secret', json.dumps(timing))
        self.assertNotIn('test-only', json.dumps(timing))
        self.assertNotIn('private-task', json.dumps(timing))

    def test_voice_error_keeps_safe_timing_and_rejects_nonfinite_or_wrong_types(self):
        self.client.last_timing = {
            'total_sec': float('nan'), 'cache_lookup_sec': True, 'submit_sec': .02,
            'wait_sec': float('inf'), 'download_sec': -1, 'cache_hit': 'yes', 'resumed_task': False,
            'new_tasks_submitted': 2, 'request_count': True, 'poll_count': -1, 'download_attempts': 0,
            'error': 'test-only',
        }
        self.client.synthesize.side_effect = AI33Error('private raw provider response')
        self.client.can_resume.return_value = False
        self.service.voice(self.payload)
        result = self.finish_voice()
        self.assertEqual(result['status'], 'error')
        self.assertEqual(set(result['timing']), {'submit_sec', 'resumed_task', 'download_attempts', 'backend_total_sec'})
        self.assertEqual(result['timing']['submit_sec'], .02)
        self.assertNotIn('private raw', json.dumps(result))

    def test_invalid_selection_is_rejected_before_provider_call(self):
        for change in ({'voice_id': '../.env'}, {'voice_id': 'unknown'}, {'speed': True}, {'speed': float('nan')},
                       {'speed': 2}, {'locale': 'vi-VN'}, {'text': 'client-owned text'}):
            with self.subTest(change=change), self.assertRaises(ChatError):
                self.service.voice({**self.payload, **change})
        self.client.synthesize.assert_not_called()

    def test_sentence_and_reply_keep_independent_selected_voices_and_server_text(self):
        self.job.speech = {'reference_text': 'I am a software engineer.'}
        self.service.voice(self.payload)
        self.assertEqual(self.finish_voice()['voice_id'], self.voice['id'])
        sentence = {**self.payload, 'target': 'sentence', 'voice_id': 'edge_en-US-AvaNeural', 'speed': 1.15}
        self.service.voice(sentence)
        self.assertEqual(self.finish_voice('sentence_voice')['voice_id'], sentence['voice_id'])
        self.client.synthesize.assert_called_with(self.job.speech['reference_text'], locale='en-US',
                                                 voice_id=sentence['voice_id'], speed=1.15)
        self.assertEqual(self.job.voice['voice_id'], self.voice['id'])
        self.service.voice(sentence)
        self.assertEqual(self.client.synthesize.call_count, 2)

    def test_retry_preserves_selected_voice_speed_and_only_resumes_existing_task(self):
        self.client.synthesize.side_effect = AI33Error('Provider unavailable')
        self.client.can_resume.return_value = True
        self.client.resume.return_value = self.audio
        self.service.voice(self.payload)
        self.assertTrue(self.finish_voice()['retryable'])
        self.service.voice(self.payload)
        self.assertEqual(self.finish_voice()['status'], 'ready')
        self.client.synthesize.assert_called_once()
        self.client.can_resume.assert_called_once_with(self.job.text, locale='en-US', voice_id=self.voice['id'], speed=.85)
        self.client.resume.assert_called_once_with(self.job.text, locale='en-US', voice_id=self.voice['id'], speed=.85)

    def test_preview_serves_existing_sample_without_tts_or_credentials_in_public_config(self):
        path = preview_stem(self.service.cache_dir, self.voice['id']).with_suffix('.mp3')
        path.parent.mkdir(parents=True)
        path.write_bytes(b'ID3preview')
        self.assertEqual(self.service.voice_preview(self.voice['id']), (b'ID3preview', 'audio/mpeg'))
        self.client.synthesize.assert_not_called()
        catalog = self.service.english_voices()
        self.assertIn('/api/chat/voice-preview/', catalog[-1]['preview_url'])
        self.assertNotIn('test-only', json.dumps(catalog))
        with self.assertRaises(ChatError):
            self.service.voice_preview('../.env')

    def test_preview_catalog_excludes_untrusted_or_signed_urls(self):
        for url in ('http://cdn.ai33.pro/a.mp3', 'https://localhost/a.mp3', 'https://cdn.ai33.pro/a?token=secret',
                    'https://cdn.ai33.pro:invalid/a', 'https://user:secret@cdn.ai33.pro/a.mp3'):
            voice = normalize_voice({**self.voice, 'preview_url': url}, 'edge')
            self.assertEqual(voice['preview_url'], '')
        self.assertEqual(len(load_voices(self.root)), 1)

    def test_tts_cache_separates_speed_and_voice_and_accepts_audio_magic_with_plain_mime(self):
        client = AI33TTS(self.root / 'cache')
        _, a, _, _ = client._cache_request('Hello', 'en-US', 'voice-a', .85)
        _, b, _, _ = client._cache_request('Hello', 'en-US', 'voice-a', 1)
        _, c, _, _ = client._cache_request('Hello', 'en-US', 'voice-b', .85)
        self.assertEqual(len({a, b, c}), 3)
        with patch.object(client.downloads, 'open', return_value=Response(b'ID3audio', 'text/plain')):
            path, _ = client.download('https://cdn.ai33.pro/sample.mp3', self.root / 'sample')
            self.assertEqual(path.read_bytes(), b'ID3audio')
        with patch.object(client.downloads, 'open', return_value=Response(b'Not an audio file', 'text/plain')):
            with self.assertRaises(AI33Error):
                client.download('https://cdn.ai33.pro/sample.mp3', self.root / 'invalid')


if __name__ == '__main__':
    unittest.main()
