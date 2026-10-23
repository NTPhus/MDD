"""Intent recovery drives dialogue without relabeling the recorded acoustics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from scripts.conversation_context import prepare_dialogue_context
from scripts.local_speech_engine import LocalSpeechEngine


def interpretation(raw, text, status='accepted'):
    return {'version': 'contextual_intent_v1', 'original': raw, 'text': text,
            'status': status, 'changed': text != raw, 'requires_confirmation': text != raw,
            'source': 'qwen_inferred', 'reference_confirmed': False, 'latency_sec': .02}


class InterpretationContextTests(unittest.TestCase):
    def test_inferred_words_can_drive_dialogue_without_becoming_confirmed(self):
        raw, recovered = 'my fravorit is tea', 'My favorite is tea.'
        request = {'history': [], 'reference_text': None}
        before = deepcopy(request)
        result = prepare_dialogue_context(request, raw, missing_words=['fravorit'],
                                         interpretation=interpretation(raw, recovered))
        self.assertEqual(result['text'], recovered)
        self.assertEqual(result['request']['conversation_source'], 'asr_interpreted')
        self.assertTrue(result['clarity']['requires_confirmation'])
        self.assertFalse(result['clarity']['needs_clarification'])
        self.assertEqual(request, before)
        self.assertIsNone(result['request']['reference_text'])

    def test_ambiguity_asks_even_when_all_tokens_exist_in_a_dictionary(self):
        raw = 'Banana software yesterday purple.'
        result = prepare_dialogue_context({}, raw, interpretation=interpretation(raw, raw, 'ambiguous'))
        self.assertEqual(result['text'], raw)
        self.assertTrue(result['clarity']['needs_clarification'])
        self.assertEqual(result['clarity']['reason'], 'unclear_interpretation')

    def test_failed_recovery_keeps_raw_and_existing_clarification(self):
        raw = 'and he dishurt my m'
        result = prepare_dialogue_context({}, raw, missing_words=['dishurt', 'm'],
                                         interpretation=interpretation(raw, raw, 'unavailable'))
        self.assertEqual(result['text'], raw)
        self.assertEqual(result['request']['conversation_source'], 'asr')
        self.assertTrue(result['clarity']['needs_clarification'])

    def test_copying_unresolved_words_does_not_bypass_clarification(self):
        raw = 'and he dishurt my m'
        for state in ('unchanged', 'accepted'):
            value = interpretation(raw, raw, state)
            value['unresolved_words'] = ['dishurt', 'm']
            with self.subTest(status=state):
                result = prepare_dialogue_context({}, raw, missing_words=['dishurt', 'm'], interpretation=value)
                self.assertTrue(result['clarity']['needs_clarification'])
                self.assertEqual(result['clarity']['reason'], 'unclear_interpretation')

    def test_unbound_and_falsely_confirmed_interpretations_are_rejected(self):
        raw, text = 'my fravorit is tea', 'My favorite is tea.'
        for changes in ({'original': 'another recording'}, {'reference_confirmed': True}, {'text': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare_dialogue_context({}, raw, interpretation={**interpretation(raw, text), **changes})
        with self.assertRaises(ValueError):
            prepare_dialogue_context({}, raw, source='reading_prompt', interpretation=interpretation(raw, text))

    def test_user_confirmation_takes_priority_over_old_inference(self):
        raw, text = 'my fravorit is tea', 'My favorite is coffee.'
        result = prepare_dialogue_context({}, raw, source='user_confirmed', dialogue_text=text,
            interpretation=interpretation(raw, 'My favorite is tea.'))
        self.assertEqual(result['text'], text)
        self.assertEqual(result['request']['conversation_source'], 'user_confirmed')
        self.assertFalse(result['clarity']['requires_confirmation'])


class RecoveryPipelineTests(unittest.TestCase):
    def run_pipeline(self, raw, inferred, *, reference_text=None):
        engine = object.__new__(LocalSpeechEngine)
        evidence = {'transcript': raw, 'reference_text': reference_text or raw, 'reference': None,
                    'comparison': None, 'phonemes': [{'phone': 'T'}], 'missing_words': ['fravorit'],
                    '_acoustic_evidence': {'private': 'unchanged'}, 'timings': {}}
        engine.analyze = Mock(return_value=deepcopy(evidence))
        events, requests = [], []
        qwen = Mock()
        def generate(request, cancelled, emit):
            requests.append(deepcopy(request))
            if request.get('generation_profile') == 'conversation':
                text = 'Do you prefer it hot or iced?'
            else:
                original = json.loads(request['messages'][-1]['content'])['transcript']
                text = json.dumps({'grammar_corrected': original, 'natural': original,
                                   'explanation': 'Keep this wording.'})
            emit({'type': 'chunk', 'text': text})
            return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        with patch('scripts.transcript_recovery.recover', return_value=inferred) as recover:
            metadata = engine.run({'model_id': 'qwen-base', 'audio_base64': 'YWJj', 'history': [],
                                   'rewrite_enabled': True, 'recovery_enabled': True,
                                   'normalization_enabled': True, 'reference_text': reference_text},
                                  threading.Event(), lambda e: events.append(deepcopy(e)), qwen)
        return evidence, events, requests, recover, metadata

    def test_uses_inference_for_reply_and_grammar_with_raw_reference_untouched(self):
        raw, text = 'my fravorit is tea', 'My favorite is tea.'
        evidence, events, requests, recover, metadata = self.run_pipeline(raw, interpretation(raw, text))
        recover.assert_called_once()
        speech = [e['speech'] for e in events if e['type'] == 'speech'][-1]
        self.assertEqual(speech['transcript'], evidence['transcript'])
        self.assertEqual(speech['reference_text'], evidence['reference_text'])
        self.assertEqual(speech['_acoustic_evidence'], evidence['_acoustic_evidence'])
        self.assertEqual(speech['reference_source'], 'asr')
        self.assertEqual(speech['conversation_text'], text)
        self.assertEqual(requests[0]['messages'][-1]['content'], text)
        self.assertEqual(metadata['conversation_source'], 'asr_interpreted')
        self.assertEqual(metadata['interpretation_status'], 'accepted')
        # No strict-normalization call, and no recovery JSON leaks into the reply.
        self.assertEqual(len(requests), 2)
        self.assertEqual(''.join(e['text'] for e in events if e['type'] == 'chunk'), 'Do you prefer it hot or iced?')
        reply_index = next(i for i, e in enumerate(events) if e.get('speech', {}).get('reply_ready'))
        grammar_index = next(i for i, e in enumerate(events)
                             if e.get('speech', {}).get('rewrite', {}).get('status') == 'ready')
        self.assertLess(reply_index, grammar_index)

    def test_reading_mode_never_recovers_or_changes_the_reference(self):
        raw, reference = 'my fravorit is tea', 'My favorite is tea.'
        _, events, requests, recover, metadata = self.run_pipeline(raw, interpretation(raw, reference), reference_text=reference)
        recover.assert_not_called()
        speech = [e['speech'] for e in events if e['type'] == 'speech'][-1]
        self.assertEqual(speech['reference_text'], reference)
        self.assertEqual(speech['reference_source'], 'reading_prompt')
        self.assertNotIn('interpretation', speech)


if __name__ == '__main__':
    unittest.main()
