"""Whisper selection and transcript provenance in the existing speech pipeline."""
import base64
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from mdd.demo_model_registry import speech_asr_selection
from mdd.demo_chat import ChatService
from scripts.local_speech_engine import LocalSpeechEngine, UnrecognizedSpeech


def selection():
    return {'id': 'whisper-small-en', 'ready': True, 'provider': 'whisper', 'revision': 'pinned',
            'device': 'cpu', 'compute_type': 'int8'}


class WhisperSelectionTests(unittest.TestCase):
    def test_unconfigured_system_always_selects_whisper_without_automatic_ctc_fallback(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch('mdd.demo_model_registry.asr_selection') as legacy:
            for ready in (True, False):
                with self.subTest(whisper_ready=ready), \
                        patch('scripts.whisper_asr.whisper_selection', return_value={**selection(), 'ready': ready}):
                    result = speech_asr_selection(ROOT)
                    self.assertEqual(result['id'], 'whisper-small-en')
                    self.assertEqual(result['ready'], ready)
            legacy.assert_not_called()

    def test_ready_whisper_is_default_and_explicit_ctc_keeps_trained_selection(self):
        with patch('scripts.whisper_asr.whisper_selection', return_value=selection()) as whisper, \
                patch('mdd.demo_model_registry.asr_selection', return_value={
                    'id': 'trained-ctc', 'path': ROOT, 'sha256': 'fixture', 'trained': True}):
            self.assertEqual(speech_asr_selection(ROOT, 'auto')['provider'], 'whisper')
            self.assertEqual(speech_asr_selection(ROOT, 'wav2vec2')['id'], 'trained-ctc')
            whisper.assert_called_once_with(ROOT)

    def test_explicit_missing_whisper_does_not_misreport_a_ctc_fallback(self):
        with patch('scripts.whisper_asr.whisper_selection', return_value={**selection(), 'ready': False}):
            result = speech_asr_selection(ROOT, 'whisper-small-en')
        self.assertEqual(result['provider'], 'whisper')
        self.assertFalse(result['ready'])
        with self.assertRaises(ValueError):
            speech_asr_selection(ROOT, 'unknown')


class WhisperTranscriptionTests(unittest.TestCase):
    def engine(self, result):
        engine = object.__new__(LocalSpeechEngine)
        engine.asr_word_intervals = [{'index': 99, 'text': 'old recording'}]
        engine.asr_backend = 'whisper'
        engine.load_asr = Mock()
        engine.whisper_asr = SimpleNamespace(transcribe=Mock(return_value=result))
        return engine

    def test_selected_text_and_its_native_times_replace_old_ctc_positions(self):
        intervals = [{'index': 0, 'text': 'Six', 'start_sec': 0., 'end_sec': .3}]
        engine = self.engine({'text': 'Six', 'raw_text': '6', 'word_intervals': intervals,
                              'model_id': 'whisper-small-en', 'word_timing': 'whisper_native',
                              'device': 'cpu', 'compute_type': 'int8'})
        self.assertEqual(engine.transcribe([0., .1]), 'Six')
        self.assertEqual(engine.asr_word_intervals, intervals)
        self.assertEqual(engine.asr_raw_transcript, '6')
        self.assertEqual(engine.asr_word_timing, 'whisper_native')

    def test_empty_output_and_load_failure_never_keep_previous_word_times(self):
        engine = self.engine({'text': '', 'word_intervals': []})
        with self.assertRaises(UnrecognizedSpeech):
            engine.transcribe([.1])
        self.assertEqual(engine.asr_word_intervals, [])
        engine.asr_word_intervals = [{'text': 'old'}]
        engine.load_asr.side_effect = ValueError('checksum')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            engine.transcribe([.1])
        self.assertEqual(engine.asr_word_intervals, [])

    def test_whisper_load_reuses_wrapper_and_does_not_load_wav2vec_weights(self):
        engine = object.__new__(LocalSpeechEngine)
        engine.asr = engine.processor = object()
        with patch('mdd.demo_model_registry.speech_asr_selection', return_value=selection()), \
                patch('scripts.whisper_asr.WhisperASR') as wrapper:
            engine.load_asr(); engine.load_asr()
        wrapper.assert_called_once_with(ROOT, cpu_threads=2, device='cpu')
        self.assertEqual(wrapper.return_value.load.call_count, 2)
        self.assertIsNone(engine.asr)
        self.assertEqual(engine.asr_backend, 'whisper')

    def test_whisper_prewarm_skips_unused_bert_without_running_audio(self):
        engine = object.__new__(LocalSpeechEngine)
        engine.asr_backend = 'whisper'
        engine.asr_model_id = 'whisper-small-en'
        engine.bert_recovery = None
        engine.load_asr = Mock()
        engine._phoneme_model = Mock(return_value={'id': 'cascade-B'})
        engine.phonemes = SimpleNamespace(_get_bundle=Mock())
        engine._prime_dictionaries = Mock()
        qwen = SimpleNamespace(load=Mock(), runtime={'device': 'cuda:0'})
        with patch('mdd.demo_model_registry.recovery_selection', return_value={'provider': 'bert'}), \
                patch('scripts.bert_transcript_recovery.BertTranscriptRecovery') as bert:
            result = engine.warmup(threading.Event(), qwen)
        bert.assert_not_called()
        self.assertIsNone(result['recovery_provider'])
        self.assertTrue(result['load_only'])


class WhisperChatRoutingTests(unittest.TestCase):
    def test_no_lexical_bert_or_qwen_recovery_but_grammar_and_phonemes_remain_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'models/pretrained/Qwen3-0.6B'
            base.mkdir(parents=True)
            for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
                (base / name).write_text('fixture')
            phoneme = root / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint/model.safetensors'
            phoneme.parent.mkdir(parents=True); phoneme.write_bytes(b'fixture')
            calls = []
            def generate(request, *, on_stage, on_speech, on_text, cancelled):
                calls.append(request)
                on_speech({'transcript': 'Six bags', 'asr_model': 'whisper-small-en'})
                on_text('Where are you going?')
                return {'reply_model_id': 'qwen-base'}
            service = ChatService(root, speech_generator=generate)
            try:
                with patch('mdd.demo_model_registry.speech_asr_selection', return_value=selection()):
                    config = service.config()
                    self.assertTrue(config['speech_ready'])
                    self.assertEqual(config['asr_model_id'], 'whisper-small-en')
                    self.assertIsNone(config['transcript_recovery'])
                    self.assertIsNone(config['transcript_normalization'])
                    self.assertEqual(config['pronunciation_reference_policy'], 'asr')
                    data = io.BytesIO()
                    with wave.open(data, 'wb') as audio:
                        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
                        audio.writeframes(b'\x00\x01' * 16000)
                    job = service.submit_speech({'model_id': 'qwen-base', 'request_id': 'c' * 32,
                        'audio_base64': base64.b64encode(data.getvalue()).decode(), 'history': []})
                    deadline = time.monotonic() + 3
                    while service.poll(job['id'])['status'] not in {'ready', 'error'} and time.monotonic() < deadline:
                        time.sleep(.01)
                self.assertEqual(service.poll(job['id'])['status'], 'ready')
                self.assertFalse(calls[0]['recovery_enabled'])
                self.assertFalse(calls[0]['normalization_enabled'])
                self.assertTrue(calls[0]['rewrite_enabled'])
            finally:
                service.close()


class WhisperEvaluationTests(unittest.TestCase):
    def test_disabled_recovery_still_checks_reference_and_prevents_tts(self):
        from copy import deepcopy
        from scripts.evaluate_vi_voice_pipeline import verify_job_recovery
        config = {'transcript_recovery': None, 'pronunciation_reference_policy': 'asr'}
        job = {'sentence_voice': {'status': 'not_requested'}, 'speech': {
            'transcript': 'Six bags', 'reference_text': 'Six bags',
            'reference_source': 'asr', 'pronunciation_reference_ready': True}}
        verify_job_recovery(job, config)
        for field, value in (('reference_text', 'Seven bags'), ('reference_source', 'bert_inferred'),
                             ('pronunciation_reference_ready', False)):
            with self.subTest(field=field):
                changed = deepcopy(job); changed['speech'][field] = value
                with self.assertRaisesRegex(ValueError, 'selected ASR transcript'):
                    verify_job_recovery(changed, config)
        changed = deepcopy(job)
        changed['speech']['interpretation'] = {'applied_as_pronunciation_reference': True}
        with self.assertRaisesRegex(ValueError, 'selected ASR transcript'):
            verify_job_recovery(changed, config)
        job['sentence_voice']['status'] = 'ready'
        with self.assertRaisesRegex(ValueError, 'Unexpected sentence TTS'):
            verify_job_recovery(job, config)


if __name__ == '__main__':
    unittest.main()
