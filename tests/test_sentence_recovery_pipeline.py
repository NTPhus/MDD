"""Conversation repair must reach dialogue without changing recorded evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.local_speech_engine import LocalSpeechEngine


class SentenceRecoveryPipelineTests(unittest.TestCase):
    raw = 'Plee call Jamie and ask her to bring the bred fromer shop'
    text = 'Please call Jamie and ask her to bring the bread from the shop.'

    def run_pipeline(self, state='accepted', *, reading=False, cancelled=False):
        engine = object.__new__(LocalSpeechEngine)
        evidence = {'transcript': self.raw, 'reference_text': self.raw,
                    'reference_source': 'asr', 'reference': None, 'comparison': None,
                    'phonemes': [], 'missing_words': ['Plee', 'fromer'], 'timings': {},
                    '_acoustic_evidence': {'unchanged': 'original posteriors'}}
        engine.analyze = Mock(return_value=deepcopy(evidence))
        engine._use_bert_reference = Mock(side_effect=lambda result: result)
        word = {'provider': 'bert', 'original': self.raw, 'text': self.raw, 'status': 'unchanged',
                'reference_confirmed': False, 'latency_sec': .1, 'evidence': [
                    {'original': 'Plee', 'reason': 'unknown_word', 'accepted': False}]}
        sentence = {'provider': 'qwen', 'original': self.raw, 'text': self.text, 'status': state,
                    'reference_confirmed': False, 'requires_confirmation': True, 'latency_sec': .2}
        calls, events = [], []
        qwen = Mock()
        def generate(request, cancel, emit):
            calls.append(deepcopy(request))
            if request.get('generation_profile') == 'conversation':
                text = 'What else should she bring?'
            else:
                source = json.loads(request['messages'][-1]['content'])['transcript']
                text = json.dumps({'grammar_corrected': source, 'natural': source, 'explanation': 'Keep this wording.'})
            emit({'type': 'chunk', 'text': text})
            return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        cancel = threading.Event()
        def repair(*args, **kwargs):
            if cancelled: cancel.set()
            return deepcopy(sentence)
        with patch('scripts.bert_transcript_recovery.BertTranscriptRecovery') as bert, \
                patch('scripts.sentence_recovery.needs_recovery', return_value=True), \
                patch('scripts.sentence_recovery.recover', side_effect=repair) as recover:
            bert.return_value.recover.return_value = deepcopy(word)
            metadata = engine.run({'model_id': 'qwen-base', 'audio_base64': 'YWJj', 'history': [],
                                   'recovery_enabled': True, 'recovery_provider': 'bert',
                                   'rewrite_enabled': True, 'reference_text': self.text if reading else None},
                                  cancel, lambda event: events.append(deepcopy(event)), qwen)
        return evidence, events, calls, recover, metadata

    def test_whole_sentence_repair_drives_dialogue_and_writing_with_original_audio_target_retained(self):
        evidence, events, calls, recover, metadata = self.run_pipeline()
        recover.assert_called_once()
        self.assertEqual(recover.call_args.args[0], self.raw)
        speech = [event['speech'] for event in events if event['type'] == 'speech'][-1]
        self.assertEqual(speech['transcript'], self.raw)
        self.assertEqual(speech['reference_text'], evidence['reference_text'])
        self.assertEqual(speech['_acoustic_evidence'], evidence['_acoustic_evidence'])
        self.assertEqual(speech['interpretation']['provider'], 'bert')
        self.assertEqual(speech['conversation_text'], self.text)
        self.assertEqual(calls[0]['messages'][-1]['content'], self.text)
        self.assertFalse(speech['transcript_clarity']['needs_clarification'])
        self.assertTrue(speech['transcript_clarity']['requires_confirmation'])
        self.assertEqual(metadata['conversation_recovery_status'], 'accepted')
        self.assertEqual(speech['timings']['sentence_recovery_sec'], .2)
        self.assertTrue(any(event.get('speech', {}).get('conversation_recovery', {}).get('status') == 'pending'
                            and event['speech']['pronunciation_reference_ready'] is False for event in events))

    def test_rejected_repair_does_not_override_partial_word_recovery(self):
        _, events, calls, _, _ = self.run_pipeline('ambiguous')
        speech = [event['speech'] for event in events if event['type'] == 'speech'][-1]
        self.assertEqual(speech['conversation_text'], self.raw)
        self.assertTrue(speech['transcript_clarity']['needs_clarification'])
        self.assertFalse(any(call.get('generation_profile') == 'conversation' for call in calls))

    def test_reading_reference_bypasses_sentence_repair(self):
        _, _, _, recover, _ = self.run_pipeline(reading=True)
        recover.assert_not_called()

    def test_cancellation_after_repair_does_not_generate_dialogue(self):
        _, _, calls, _, metadata = self.run_pipeline(cancelled=True)
        self.assertEqual(metadata['finish_reason'], 'cancelled')
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
