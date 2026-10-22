"""Confirmed text regenerates the reply without decoding the recording again."""
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdd.demo_chat import ChatError, spoken_text_key
from mdd.demo import DemoApplication, DemoServer
import test_speech_chat as fixture


class ChatReferenceApiTests(unittest.TestCase):
    payload = fixture.SpeechChatTests.payload

    def setUp(self):
        fixture.SpeechChatTests.setUp(self)
        self.reference_calls = []
        def regenerate(request, *, on_stage, on_speech, on_text, cancelled):
            self.reference_calls.append(deepcopy(request))
            on_speech({**request['speech'], 'spoken_response': 'What do you like about it?'})
            on_text('What do you like about it?')
            return {'reply_model_id': 'qwen-base', 'reference_revision': 999}
        self.service.reference_generator = regenerate

    def recording(self, *, history=None):
        def run(request, *, on_stage, on_speech, on_text, cancelled):
            self.calls.append(deepcopy(request))
            on_speech({'transcript': 'My fravorit food is pasta', 'reference_text': 'My fravorit food is pasta',
                       'reference_source': 'asr', 'comparison': None,
                       '_acoustic_evidence': {'data': 'private-original-posterior'},
                       'phonemes': [{'phone': 'F', 'start_sec': 0, 'end_sec': .2, 'confidence': .8}],
                       'spoken_response': 'Pasta sounds delicious.'})
            on_text('Pasta sounds delicious.')
            return {'topic_id': request.get('topic_id'), 'reply_model_id': 'qwen-base'}
        self.service.speech_generator = run
        job = self.service.submit_speech(self.payload(topic_id='food', history=history or []))
        return fixture.finished(self.service, job['id'])

    def reassess(self, root, speech, text):
        return {**deepcopy(speech), 'reference_text': text, 'reference_source': 'user_confirmed',
                'reference_resolution': {'status': 'confirmed'},
                'comparison': {'words': [], 'events': [], 'counts': {}},
                'coaching': {'status': 'unavailable', 'feedback': {}}}

    def confirm(self, source, text='My favorite food is pasta'):
        with patch('mdd.chat_reference.reassess_reference', side_effect=self.reassess):
            return self.service.reference({'job_id': source['id'], 'reference_text': text})

    def test_confirmation_clears_old_reply_and_uses_immutable_pre_turn_history(self):
        history = [{'role': 'user', 'content': 'I enjoy Italian food.'},
                   {'role': 'assistant', 'content': 'Which dish do you prefer?'}]
        source = self.recording(history=history)
        history[0]['content'] = 'Changed by browser'
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def waiting(request, *, on_stage, on_speech, on_text, cancelled):
            self.reference_calls.append(deepcopy(request)); entered.set(); release.wait(2)
            on_speech({**request['speech'], 'spoken_response': 'What do you like about it?'})
            on_text('What do you like about it?')
            return {'reply_model_id': 'qwen-base', 'reference_revision': 999}
        self.service.reference_generator = waiting
        accepted = self.confirm(source)
        self.assertTrue(entered.wait(1))
        self.assertEqual((accepted['id'], accepted['status'], accepted['text']), (source['id'], 'generating', ''))
        self.assertEqual(accepted['voice']['status'], 'not_requested')
        self.assertEqual(accepted['reference_update'], {'changed': True, 'regenerated_reply': True})
        self.assertEqual(accepted['speech']['conversation_text'], 'My favorite food is pasta')
        self.assertNotIn('spoken_response', accepted['speech'])
        request = self.reference_calls[0]
        self.assertEqual(request['operation'], 'regenerate_reference')
        self.assertEqual(request['dialogue_text'], 'My favorite food is pasta')
        self.assertEqual(request['history'], self.calls[0]['history'])
        self.assertEqual(request['history'][0]['content'], 'I enjoy Italian food.')
        self.assertNotIn(source['speech']['transcript'], json.dumps(request['history']))
        self.assertNotIn(source['text'], json.dumps(request['history']))
        self.assertEqual((request['topic_id'], request['model_id']), ('food', 'qwen-base'))
        self.assertTrue(request['rewrite_enabled'])
        self.assertNotIn('audio_base64', request)
        self.assertNotIn('_acoustic_evidence', request['speech'])
        self.assertEqual(request['speech']['transcript'], source['speech']['transcript'])
        self.assertEqual(request['speech']['phonemes'], source['speech']['phonemes'])
        release.set()
        final = fixture.finished(self.service, source['id'])
        self.assertEqual(final['text'], 'What do you like about it?')
        self.assertEqual(final['metadata']['reference_revision'], 1)
        self.assertTrue(final['metadata']['regenerated_from_reference'])
        self.assertEqual(final['speech']['transcript'], source['speech']['transcript'])
        self.assertNotIn('private-original-posterior', json.dumps(final))
        self.assertEqual(self.service._jobs[source['id']].speech['_acoustic_evidence'],
                         {'data': 'private-original-posterior'})
        self.assertEqual(len(self.calls), 1)

    def test_regeneration_acceptance_is_empty_even_if_worker_finishes_immediately(self):
        source = self.recording()
        accepted = self.confirm(source)
        self.assertEqual((accepted['status'], accepted['text']), ('generating', ''))
        final = fixture.finished(self.service, source['id'])
        self.assertEqual(final['text'], 'What do you like about it?')
        self.assertEqual(final['speech']['conversation_text'], 'My favorite food is pasta')
        self.assertEqual(final['metadata']['reference_revision'], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.reference_calls), 1)

    def test_same_asr_words_confirm_sounds_without_regeneration_even_when_model_busy(self):
        source = self.recording()
        job = self.service._jobs[source['id']]
        job.voice = {'status': 'ready', 'url': '/api/chat/audio/' + 'b' * 32}
        job.voice_locale = 'en-US'
        self.service._voices['b' * 32] = (source['id'], self.root / 'old.mp3')
        old_voice, old_created = deepcopy(job.voice), job.created
        self.assertTrue(self.service._generation_slot.acquire(blocking=False))
        try:
            same = self.confirm(source, source['speech']['transcript'])
        finally:
            self.service._generation_slot.release()
        self.assertEqual(same['reference_update'], {'changed': False, 'regenerated_reply': False})
        self.assertEqual((same['status'], same['text'], same['voice']), ('ready', source['text'], old_voice))
        self.assertEqual(same['metadata']['reference_revision'], 0)
        self.assertEqual(same['speech']['reference_source'], 'user_confirmed')
        self.assertEqual(same['speech']['conversation_text'], source['speech']['transcript'])
        self.assertEqual(job.created, old_created)
        self.assertEqual(job.reply_revisions, [])
        self.assertEqual(job.voice_locale, 'en-US')
        self.assertIn('b' * 32, self.service._voices)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.reference_calls, [])

    def test_punctuation_case_and_spacing_edits_keep_current_reply_and_revision(self):
        source = self.recording()
        self.confirm(source)
        first = fixture.finished(self.service, source['id'])
        for text in ('MY FAVORITE FOOD IS PASTA!', '  My favorite  food is pasta.  ', 'My favorite food is pasta?'):
            with self.subTest(text=text):
                same = self.confirm(first, text)
                self.assertEqual(same['reference_update'], {'changed': False, 'regenerated_reply': False})
                self.assertEqual((same['status'], same['text'], same['voice']),
                                 (first['status'], first['text'], first['voice']))
                self.assertEqual(same['metadata']['reference_revision'], 1)
                self.assertEqual(same['speech']['transcript'], source['speech']['transcript'])
        self.assertEqual(len(self.reference_calls), 1)
        self.assertEqual(len(self.service._jobs[source['id']].reply_revisions), 1)

    def test_real_word_change_after_same_input_still_regenerates_once(self):
        source = self.recording()
        same = self.confirm(source, source['speech']['transcript'])
        self.assertFalse(same['reference_update']['regenerated_reply'])
        accepted = self.confirm(same, 'My favorite food is not pasta')
        self.assertTrue(accepted['reference_update']['regenerated_reply'])
        self.assertEqual(accepted['metadata']['reference_revision'], 1)
        final = fixture.finished(self.service, source['id'])
        self.assertEqual(final['speech']['conversation_text'], 'My favorite food is not pasta')
        self.assertEqual(len(self.reference_calls), 1)
        self.assertEqual(len(self.calls), 1)

    def test_spoken_word_comparison_preserves_numbers_negation_and_apostrophes(self):
        equal = (("I’m here.", "i'm   here!"), ('well-known name', 'Well known name'),
                 ('１２ cups', '12 cups'), ('Café Quendor', 'café quendor'))
        different = (('I like tea', 'I do not like tea'), ('one coffee', 'two coffees'),
                     ("I'm here", 'I am here'), ('12 cups', '13 cups'))
        for left, right in equal:
            self.assertEqual(spoken_text_key(left), spoken_text_key(right))
        for left, right in different:
            self.assertNotEqual(spoken_text_key(left), spoken_text_key(right))

    def test_unknown_ids_and_injected_audio_or_labels_are_rejected(self):
        for payload in ({'job_id': 'f' * 32, 'reference_text': 'Favorite'},
                        {'job_id': '../.env', 'reference_text': 'Favorite'},
                        {'job_id': [], 'reference_text': 'Favorite'},
                        {'job_id': 'f' * 32, 'reference_text': 'Favorite', 'phonemes': ['FAKE']},
                        {'job_id': 'f' * 32, 'reference_text': 'Favorite', 'audio_base64': 'FAKE'}):
            with self.subTest(payload=payload), self.assertRaises(ChatError):
                self.service.reference(payload)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.reference_calls, [])

    def test_repeated_edits_increment_revision_without_appending_replaced_turns(self):
        history = [{'role': 'user', 'content': 'I enjoy Italian food.'},
                   {'role': 'assistant', 'content': 'Which dish do you prefer?'}]
        source = self.recording(history=history)
        self.confirm(source)
        first = fixture.finished(self.service, source['id'])
        accepted = self.confirm(first, 'My favorite food is ravioli')
        self.assertEqual(accepted['metadata']['reference_revision'], 2)
        self.assertEqual(accepted['text'], '')
        final = fixture.finished(self.service, source['id'])
        self.assertEqual(final['metadata']['reference_revision'], 2)
        self.assertEqual(final['speech']['transcript'], source['speech']['transcript'])
        self.assertEqual(final['speech']['conversation_text'], 'My favorite food is ravioli')
        self.assertEqual([call['history'] for call in self.reference_calls], [history, history])
        self.assertEqual([call['reference_revision'] for call in self.reference_calls], [1, 2])
        self.assertEqual(len(self.calls), 1)

    def test_regenerator_error_retains_confirmed_evidence_without_restoring_stale_reply(self):
        source = self.recording()
        self.service.reference_generator = Mock(side_effect=RuntimeError('Test generation failure'))
        accepted = self.confirm(source)
        self.assertEqual((accepted['status'], accepted['text']), ('generating', ''))
        final = fixture.finished(self.service, source['id'])
        self.assertEqual((final['status'], final['text']), ('error', ''))
        self.assertEqual(final['speech']['reference_source'], 'user_confirmed')
        self.assertEqual(final['speech']['transcript'], source['speech']['transcript'])
        self.assertEqual(final['metadata']['reference_revision'], 1)
        self.assertEqual(final['voice']['status'], 'not_requested')

    def test_invalid_target_does_not_replace_previous_reply_or_evidence(self):
        source = self.recording()
        before = deepcopy(self.service._jobs[source['id']].speech)
        with patch('mdd.chat_reference.reassess_reference', side_effect=ValueError('Invalid target text')):
            with self.assertRaises(ChatError) as raised:
                self.service.reference({'job_id': source['id'], 'reference_text': ''})
        self.assertEqual(raised.exception.status, 400)
        current = self.service.poll(source['id'])
        self.assertEqual((current['text'], current['speech'], current['metadata']),
                         (source['text'], source['speech'], source['metadata']))
        self.assertEqual(self.service._jobs[source['id']].speech, before)
        self.assertEqual(self.reference_calls, [])

    def test_busy_rejection_does_not_commit_reference_or_clear_reply(self):
        source = self.recording()
        self.assertTrue(self.service._generation_slot.acquire(blocking=False))
        try:
            with self.assertRaises(ChatError) as raised:
                self.confirm(source)
            self.assertEqual(raised.exception.status, 409)
            current = self.service.poll(source['id'])
            self.assertEqual((current['text'], current['speech'], current['metadata'], current['voice']),
                             (source['text'], source['speech'], source['metadata'], source['voice']))
            self.assertEqual(self.service._jobs[source['id']].generation_revision, 0)
            self.assertEqual(self.reference_calls, [])
        finally:
            self.service._generation_slot.release()

    def test_pending_and_cancelled_recordings_cannot_be_reassessed(self):
        source = self.recording()
        for state in ('coaching', 'cancelled', 'error'):
            self.service._jobs[source['id']].status = state
            with self.subTest(state=state), self.assertRaises(ChatError) as raised:
                self.service.reference({'job_id': source['id'], 'reference_text': 'Favorite'})
            self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.reference_calls, [])

    def test_concurrent_target_change_is_not_overwritten_by_an_older_request(self):
        source = self.recording()
        def raced(root, speech, text):
            self.service._jobs[source['id']].speech = {**speech, 'reference_text': 'A newer target'}
            return self.reassess(root, speech, text)
        with patch('mdd.chat_reference.reassess_reference', side_effect=raced):
            with self.assertRaises(ChatError) as raised:
                self.service.reference({'job_id': source['id'], 'reference_text': 'Favorite'})
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(self.service.poll(source['id'])['speech']['reference_text'], 'A newer target')
        self.assertEqual(self.service.poll(source['id'])['text'], source['text'])
        self.assertEqual(self.reference_calls, [])

    def test_cancel_regeneration_ignores_late_text_stage_and_speech(self):
        source = self.recording()
        entered = threading.Event()
        def waiting(request, *, on_stage, on_speech, on_text, cancelled):
            self.reference_calls.append(deepcopy(request)); entered.set(); cancelled.wait(2)
            on_stage('coaching'); on_text('Late text must be ignored.')
            on_speech({'transcript': 'Late transcript must be ignored.'})
            return {'reply_model_id': 'qwen-base'}
        self.service.reference_generator = waiting
        self.confirm(source); self.assertTrue(entered.wait(1))
        self.service.cancel({'job_id': source['id']})
        final = fixture.finished(self.service, source['id'])
        self.assertEqual((final['status'], final['text']), ('cancelled', ''))
        self.assertEqual(final['speech']['transcript'], source['speech']['transcript'])
        self.assertEqual(final['speech']['conversation_text'], 'My favorite food is pasta')
        self.assertEqual(final['metadata']['reference_revision'], 1)
        self.assertIn('_acoustic_evidence', self.service._jobs[source['id']].speech)

    def test_confirmation_invalidates_existing_voice_url(self):
        source = self.recording()
        old_id = 'b' * 32
        job = self.service._jobs[source['id']]
        job.voice = {'status': 'ready', 'url': '/api/chat/audio/' + old_id}
        self.service._voices[old_id] = (source['id'], self.root / 'old.mp3')
        accepted = self.confirm(source)
        self.assertEqual(accepted['voice']['status'], 'not_requested')
        self.assertNotIn(old_id, self.service._voices)
        self.assertEqual(fixture.finished(self.service, source['id'])['voice']['status'], 'not_requested')

    def test_old_tts_completion_cannot_publish_voice_for_regenerated_reply(self):
        source = self.recording()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        audio = self.service.cache_dir / 'old.mp3'
        audio.parent.mkdir(parents=True, exist_ok=True); audio.write_bytes(b'ID3test')
        def synthesize(text, *, locale):
            self.assertEqual(text, source['text']); entered.set(); release.wait(2)
            return audio
        tts = Mock(); tts.synthesize.side_effect = synthesize
        self.service.tts_factory = lambda: tts
        self.service.capability_loader = lambda: {'tts_configured': True, 'english_voice_configured': True}
        self.service.voice({'job_id': source['id'], 'locale': 'en-US'})
        self.assertTrue(entered.wait(1))
        self.confirm(source)
        final = fixture.finished(self.service, source['id'])
        self.assertEqual(final['text'], 'What do you like about it?')
        release.set()
        self.assertTrue(self.service._voice_slot.acquire(timeout=2))
        self.service._voice_slot.release()
        self.assertEqual(self.service.poll(source['id'])['voice']['status'], 'not_requested')
        self.assertEqual(self.service._voices, {})
        tts.synthesize.assert_called_once_with(source['text'], locale='en-US')

    def test_http_confirmation_keeps_same_origin_and_json_guards(self):
        source = self.recording()
        server = DemoServer(('127.0.0.1', 0), DemoApplication(self.root, chat_service=self.service))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            def request(origin):
                connection = HTTPConnection('127.0.0.1', server.server_port)
                body = {'job_id': source['id'], 'reference_text': 'My favorite food is pasta'}
                connection.request('POST', '/api/chat/reference', json.dumps(body),
                                   {'Content-Type': 'application/json', 'Origin': origin})
                response = connection.getresponse(); value = json.loads(response.read()); connection.close()
                return response.status, value
            self.assertEqual(request('https://evil.invalid')[0], 403)
            with patch('mdd.chat_reference.reassess_reference', side_effect=self.reassess):
                status, accepted = request(f'http://127.0.0.1:{server.server_port}')
            self.assertEqual(status, 200)
            self.assertEqual((accepted['status'], accepted['text']), ('generating', ''))
            final = fixture.finished(self.service, source['id'])
            self.assertEqual(final['speech']['reference_source'], 'user_confirmed')
            self.assertEqual(final['text'], 'What do you like about it?')
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(len(self.reference_calls), 1)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__': unittest.main()
