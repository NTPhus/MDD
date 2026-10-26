"""Use recovered BERT words with retained acoustic evidence, without confirmation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from mdd import demo_reference
from mdd.acoustic_evidence import pack_evidence
from scripts.local_speech_engine import LocalSpeechEngine


LEXICON = b'fraud F R AO1 D\nforces F AO1 R S IH0 Z\n'


class BertPronunciationReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        path = self.root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict'
        path.parent.mkdir(parents=True)
        path.write_bytes(LEXICON)
        for mocked in (patch.object(demo_reference, 'DICTIONARY_SHA256', hashlib.sha256(LEXICON).hexdigest()),
                       patch('scripts.local_speech_engine.ROOT', self.root)):
            mocked.start()
            self.addCleanup(mocked.stop)
        self.engine = object.__new__(LocalSpeechEngine)

    def speech(self):
        # The original decoder hears AW instead of AO and omits R in forces.
        # Reusing this matrix must retain those acoustic discrepancies after
        # replacing the unknown ASR spelling with BERT's known word.
        observed = 'F R AW D F AO S IH Z'.split()
        vocabulary = ['[PAD]', *sorted(set(observed + 'F R AO D F AO R S IH Z'.split()))]
        matrix = np.full((len(observed) * 3, len(vocabulary)), -12., dtype=np.float32)
        for index, phone in enumerate(observed):
            matrix[index * 3:index * 3 + 2, vocabulary.index(phone)] = 0.
            matrix[index * 3 + 2, 0] = 0.
        matrix -= np.log(np.exp(matrix).sum(axis=1, keepdims=True))
        return {'transcript': 'Frout forces', 'reference_text': 'Frout forces', 'reference_source': 'asr',
                'phonemes': [{'phone': phone} for phone in observed], 'audio': {'duration_sec': 2.},
                '_acoustic_evidence': pack_evidence({'log_probs': matrix,
                    'id_to_phone': dict(enumerate(vocabulary)), 'blank_id': 0, 'duration_sec': 2.,
                    'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64}),
                'reference': None, 'comparison': None, 'missing_words': ['Frout'], 'timings': {},
                'phonetic_candidates': {'status': 'available', 'words': [{'index': 0, 'word': 'Frout'}]},
                'interpretation': {'original': 'Frout forces', 'text': 'fraud forces', 'status': 'accepted',
                                  'provider': 'bert', 'reference_confirmed': False,
                                  'source': 'bert_inferred', 'latency_sec': .1}}

    def test_recovered_reference_exposes_sound_differences_from_the_unchanged_recording(self):
        speech = self.speech()
        original = deepcopy(speech)
        with patch.object(self.engine, 'transcribe') as transcribe:
            result = self.engine._use_bert_reference(speech)
        transcribe.assert_not_called()
        self.assertEqual(result['transcript'], original['transcript'])
        self.assertEqual(result['audio'], original['audio'])
        self.assertEqual(result['phonemes'], original['phonemes'])
        self.assertEqual(result['_acoustic_evidence'], original['_acoustic_evidence'])
        self.assertEqual(result['phonetic_candidates'], original['phonetic_candidates'])
        self.assertEqual(result['reference_text'], 'fraud forces')
        self.assertEqual(result['reference_source'], 'bert_inferred')
        self.assertEqual(result['missing_words'], [])
        self.assertEqual(result['reference_resolution']['target_text'], 'fraud forces')
        self.assertEqual(result['reference_resolution']['source'], 'bert_inferred')
        self.assertEqual(result['reference_resolution']['status'], 'inferred')
        self.assertTrue(result['interpretation']['applied_as_pronunciation_reference'])
        self.assertFalse(result['interpretation']['reference_confirmed'])
        self.assertEqual(result['assessment']['status'], 'ready')
        self.assertFalse(result['assessment']['calibration']['applied'])
        self.assertEqual([word['text'] for word in result['target_words']], ['fraud', 'forces'])
        self.assertEqual([word['text'] for word in result['comparison']['words']], ['fraud', 'forces'])
        self.assertTrue(all(word['status'] == 'uncertain' for word in result['comparison']['words']))
        hints = result['sound_candidates']
        self.assertEqual([word['text'] for word in hints['words']], ['fraud', 'forces'])
        self.assertEqual(hints['source'], 'bert_inferred')
        self.assertTrue(hints['provisional'])
        self.assertFalse(hints['reference_confirmed'])

    def test_unchanged_bert_output_is_also_the_selected_reference(self):
        speech = self.speech()
        speech['interpretation'].update(status='unchanged', text='Fraud forces', source='format_rules')
        result = self.engine._use_bert_reference(speech)
        self.assertEqual(result['reference_text'], 'Fraud forces')
        self.assertEqual(result['reference_source'], 'bert_inferred')

    def test_ambiguous_unavailable_unbound_or_falsely_confirmed_outputs_cannot_relabel_audio(self):
        for update in ({'provider': 'qwen'}, {'status': 'ambiguous'}, {'status': 'unavailable'}, {'original': 'Different audio'},
                       {'reference_confirmed': True}, {'text': ''}):
            speech = self.speech()
            speech['interpretation'].update(update)
            before = deepcopy(speech)
            with self.subTest(update=update):
                self.assertEqual(self.engine._use_bert_reference(speech), before)

    def test_explicit_reading_and_user_confirmed_references_take_priority(self):
        for source in ('reading_prompt', 'user_confirmed'):
            speech = self.speech()
            speech.update(reference_source=source, reference_text='Some explicit reference')
            before = deepcopy(speech)
            with self.subTest(source=source):
                self.assertEqual(self.engine._use_bert_reference(speech), before)

    def test_unknown_recovered_words_remain_unassessable_instead_of_becoming_sound_errors(self):
        speech = self.speech()
        speech['interpretation']['text'] = 'fraud blargzz'
        result = self.engine._use_bert_reference(speech)
        self.assertEqual(result['reference_text'], 'fraud blargzz')
        self.assertEqual(result['missing_words'], ['blargzz'])
        self.assertEqual(result['assessment']['status'], 'unavailable')
        self.assertEqual(result['assessment']['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['sound_candidates']['count'], 0)
        self.assertEqual(result['sound_candidates']['source'], 'bert_inferred')

    def test_missing_retained_evidence_cannot_fall_back_to_raw_edit_errors(self):
        speech = self.speech()
        speech['_acoustic_evidence'] = None
        with self.assertRaises(ValueError):
            self.engine._use_bert_reference(speech)

    def test_pending_raw_reference_is_not_ready_then_bert_reference_drives_reply(self):
        speech = self.speech()
        inferred = speech.pop('interpretation')
        self.engine.analyze = Mock(return_value=deepcopy(speech))
        self.engine.bert_recovery = Mock()
        self.engine.bert_recovery.recover.return_value = inferred
        events, requests = [], []
        qwen = Mock()
        def generate(request, cancelled, emit):
            requests.append(deepcopy(request))
            if request.get('generation_profile') == 'conversation':
                text = 'How did you discover that?'
            else:
                original = json.loads(request['messages'][-1]['content'])['transcript']
                text = json.dumps({'grammar_corrected': original, 'natural': original,
                                   'explanation': 'Keep this wording.'})
            emit({'type': 'chunk', 'text': text})
            return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        metadata = self.engine.run({'model_id': 'qwen-base', 'audio_base64': 'YWJj', 'history': [],
                                   'rewrite_enabled': True, 'recovery_enabled': True,
                                   'recovery_provider': 'bert', 'reference_source': 'asr'}, threading.Event(),
                                  lambda event: events.append(deepcopy(event)), qwen)
        speech_events = [event['speech'] for event in events if event['type'] == 'speech']
        self.assertFalse(speech_events[0]['pronunciation_reference_ready'])
        self.assertEqual(speech_events[0]['interpretation']['status'], 'pending')
        self.assertTrue(all(event['pronunciation_reference_ready'] for event in speech_events[1:]))
        final = speech_events[-1]
        self.assertEqual(final['reference_source'], 'bert_inferred')
        self.assertEqual(final['reference_text'], 'fraud forces')
        self.assertEqual(final['conversation_text'], 'fraud forces')
        self.assertEqual(requests[0]['messages'][-1]['content'], 'fraud forces')
        self.assertEqual(metadata['conversation_source'], 'asr_interpreted')
        self.assertEqual(metadata['pronunciation_reference_source'], 'bert_inferred')
        self.assertEqual(final['_acoustic_evidence'], speech['_acoustic_evidence'])
        self.engine.analyze.assert_called_once_with(b'abc', None)

    def test_cancellation_after_recovery_never_publishes_raw_as_a_ready_reference(self):
        speech = self.speech()
        inferred = speech.pop('interpretation')
        self.engine.analyze = Mock(return_value=speech)
        cancelled, events = threading.Event(), []
        self.engine.bert_recovery = Mock()
        def recover(*args, **kwargs):
            cancelled.set()
            return inferred
        self.engine.bert_recovery.recover.side_effect = recover
        self.engine._respond = Mock()
        result = self.engine.run({'audio_base64': 'YWJj', 'recovery_enabled': True,
                                  'recovery_provider': 'bert'}, cancelled, events.append, Mock())
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.engine._respond.assert_not_called()
        self.assertFalse(any(event.get('speech', {}).get('pronunciation_reference_ready') for event in events))


if __name__ == '__main__':
    unittest.main()
