"""CROTTC routing and optional analysis must not hold up or corrupt dialogue."""
import base64
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from mdd.demo_model_registry import pronunciation_selection
from scripts.local_speech_engine import LocalSpeechEngine


class PronunciationSelectionTests(unittest.TestCase):
    def test_default_keeps_b_after_pilot_and_crottc_is_an_explicit_choice(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch('scripts.crottc_pronunciation.crottc_selection') as experimental:
            self.assertEqual(pronunciation_selection(ROOT)['id'], 'cascade-B')
            experimental.assert_not_called()

    def test_installed_crottc_is_selected_and_legacy_b_is_explicit(self):
        with patch('scripts.crottc_pronunciation.crottc_selection', return_value={
                'id': 'crottc-if-l2-arctic', 'ready': True}):
            self.assertEqual(pronunciation_selection(ROOT, 'auto')['provider'], 'crottc-if')
            self.assertEqual(pronunciation_selection(ROOT, 'wav2vec2-b')['id'], 'cascade-B')

    def test_explicit_missing_model_does_not_silently_use_b(self):
        with patch('scripts.crottc_pronunciation.crottc_selection', return_value={'ready': False}):
            self.assertFalse(pronunciation_selection(ROOT, 'crottc-if')['ready'])
            self.assertEqual(pronunciation_selection(ROOT, 'auto')['id'], 'cascade-B')
        with self.assertRaises(ValueError):
            pronunciation_selection(ROOT, 'unknown')

    def test_crottc_load_is_reused_and_receives_own_cancellation_event(self):
        engine = object.__new__(LocalSpeechEngine)
        engine._phoneme_model = Mock(return_value={'id': 'crottc-if-l2-arctic', 'provider': 'crottc-if'})
        engine.phonemes = Mock()
        cancelled = threading.Event()
        with patch('scripts.crottc_pronunciation.CROTTCPronunciation') as wrapper:
            engine._load_pronunciation(cancelled); engine._load_pronunciation(cancelled)
        wrapper.assert_called_once_with(ROOT, timeout=45., device='auto')
        self.assertEqual(wrapper.return_value.load.call_count, 2)
        wrapper.return_value.load.assert_called_with(cancelled)
        engine.phonemes._get_bundle.assert_not_called()
        engine._pronunciation_cancelled = cancelled
        engine.close()
        self.assertTrue(cancelled.is_set())
        wrapper.return_value.close.assert_called_once()


class PronunciationRequestTests(unittest.TestCase):
    def fixture(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_speech_chat import SpeechChatTests
        fixture = SpeechChatTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.service.close)
        return fixture

    def test_selected_model_is_resolved_by_server_and_kept_in_job_context(self):
        from test_speech_chat import finished
        fixture = self.fixture()
        with patch('mdd.demo_model_registry.pronunciation_selection', return_value={
                'id': 'crottc-if-l2-arctic', 'provider': 'crottc-if', 'ready': True}) as choose:
            job = fixture.service.submit_speech(fixture.payload(pronunciation_model_id='crottc-if-l2-arctic'))
            final = finished(fixture.service, job['id'])
        choose.assert_called_with(fixture.root, 'crottc-if')
        self.assertEqual(final['status'], 'ready')
        self.assertEqual(fixture.calls[0]['pronunciation_backend'], 'crottc-if')
        self.assertEqual(fixture.service._jobs[job['id']].speech_context['pronunciation_model_id'], 'crottc-if-l2-arctic')

    def test_unavailable_or_unadvertised_models_cannot_select_arbitrary_weights(self):
        from mdd.demo_chat import ChatError
        fixture = self.fixture()
        for value in ('../../model', 'other', None, 42):
            with self.subTest(value=value), self.assertRaises(ChatError) as error:
                fixture.service.submit_speech(fixture.payload(pronunciation_model_id=value))
            self.assertEqual(error.exception.code, 'invalid_pronunciation_model')
        with patch('mdd.demo_model_registry.pronunciation_selection', return_value={'ready': False}):
            with self.assertRaises(ChatError) as error:
                fixture.service.submit_speech(fixture.payload(pronunciation_model_id='crottc-if-l2-arctic'))
        self.assertEqual(error.exception.code, 'pronunciation_model_unavailable')
        self.assertEqual(fixture.calls, [])


class DeferredPronunciationTests(unittest.TestCase):
    def test_background_load_keeps_original_turn_cancellation_event(self):
        import numpy as np
        entered, release, event_a, event_b = (threading.Event() for _ in range(4))
        engine = object.__new__(LocalSpeechEngine)
        engine._defer_pronunciation = True
        engine._pronunciation_cancelled = event_a
        engine.asr_backend = 'whisper'; engine.asr_word_intervals = []
        engine.transcribe = Mock(return_value='cat')
        engine._phoneme_model = Mock(return_value={'id': 'crottc-if-l2-arctic', 'provider': 'crottc-if'})
        engine.phonemes = Mock()
        engine.phonemes._audio.return_value = (np.ones(16000, dtype=np.float32), {'duration_sec': 1.})
        wrapper = Mock()
        def load(event):
            self.assertIs(event, event_a)
            entered.set(); release.wait(2)
        wrapper.load.side_effect = load
        wrapper.infer.return_value = {'provider': 'crottc-if', 'phonemes': [{'phone': 'K'}, {'phone': 'AE'}, {'phone': 'T'}],
            'duration_sec': 1., 'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64}
        engine._crottc_engine = Mock(return_value=wrapper)
        reference = {'prompt_text': 'cat', 'canonical_phones': ['K', 'AE', 'T'], 'canonical_word_indices': [0, 0, 0]}
        with patch('mdd.demo_reference.dictionary_reference', return_value=(reference, [])), \
                patch('mdd.chat_reference.reference_resolution', return_value={'status': 'asr'}), \
                patch('mdd.chat_pronunciation.target_words', return_value=[]), \
                patch('mdd.chat_assessment.assessment_fields', return_value={'target_words': []}):
            engine.analyze(b'fixture')
            self.assertTrue(entered.wait(1))
            engine._pronunciation_cancelled = event_b
            release.set()
            engine._pending_pronunciation.result(timeout=2)
        self.assertIs(wrapper.infer.call_args.kwargs['cancelled'], event_a)
        self.assertFalse(event_b.is_set())

    def test_reply_is_published_before_slow_pronunciation_finishes(self):
        engine = object.__new__(LocalSpeechEngine)
        future, published = Future(), threading.Event()
        observed, errors, metadata = [], [], []
        def analyze(*args):
            engine._pending_pronunciation = future
            return {'transcript': 'I visited a cafe.', 'reference_text': 'I visited a cafe.',
                    'reference_source': 'reading_prompt', 'reference': None, 'comparison': None,
                    'missing_words': [], 'phonemes': [], 'assessment': {'status': 'pending'},
                    'sound_candidates': {'status': 'pending', 'words': []}}
        engine.analyze = analyze
        def generate(request, cancelled, emit):
            emit({'type': 'chunk', 'text': 'What did you enjoy there?'})
            return {'finish_reason': 'stop'}
        qwen = Mock(); qwen.generate.side_effect = generate
        def emit(event):
            observed.append(deepcopy(event))
            if event.get('speech', {}).get('reply_ready'):
                published.set()
        request = {'model_id': 'qwen-base', 'history': [],
                   'reference_text': 'I visited a cafe.', 'reference_source': 'reading_prompt',
                   'audio_base64': base64.b64encode(b'fixture').decode()}
        def run():
            try:
                metadata.append(engine.run(request, threading.Event(), emit, qwen))
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.assertTrue(published.wait(2), errors)
            self.assertTrue(worker.is_alive())
            self.assertFalse(future.done())
            future.set_result({'assessment': {'status': 'ready'}, 'comparison': {'words': [], 'events': []},
                               'sound_candidates': {'status': 'available', 'words': []},
                               '_pronunciation_timings': {'phoneme_sec': 4.}})
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            speech = [item['speech'] for item in observed if item['type'] == 'speech']
            first_reply = next(item for item in speech if item.get('reply_ready'))
            self.assertEqual(first_reply['assessment']['status'], 'pending')
            self.assertEqual(speech[-1]['assessment']['status'], 'ready')
            self.assertEqual(first_reply['spoken_response'], speech[-1]['spoken_response'])
            self.assertEqual(metadata[0]['feedback_order'], 'reply_first')
        finally:
            if not future.done():
                future.set_result({})
            worker.join(2)

    def test_failed_model_timeout_is_not_confused_with_pending_future(self):
        engine = object.__new__(LocalSpeechEngine)
        engine._pending_pronunciation = Future()
        engine._pending_pronunciation.set_exception(TimeoutError('model timeout'))
        speech = {'spoken_response': 'What happened next?', 'reply_ready': True}
        engine._finish_pronunciation(speech, threading.Event(), lambda event: None)
        self.assertEqual(speech['assessment']['status'], 'unavailable')
        self.assertEqual(speech['spoken_response'], 'What happened next?')
        self.assertTrue(speech['reply_ready'])
        self.assertIsNone(engine._pending_pronunciation)


if __name__ == '__main__':
    unittest.main()
