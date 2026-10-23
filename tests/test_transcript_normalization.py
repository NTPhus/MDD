"""Normalization keeps content and never becomes an audio reference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from scripts.transcript_normalization import format_only, normalization_request, parse_normalization, preservation_reason, normalize
from scripts.conversation_context import prepare_dialogue_context
from scripts.local_speech_engine import LocalSpeechEngine


class NormalizationTests(unittest.TestCase):
    def parse(self, original, candidate):
        return parse_normalization(json.dumps({'normalized_text': candidate}), True, original)

    def test_formatting_preserves_grammar_repetition_and_speaker(self):
        original = 'i i usually wakes up at seven am'
        result = self.parse(original, 'I, I usually wakes up at seven am.')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['text'], 'I, I usually wakes up at seven am.')
        self.assertFalse(result['reference_confirmed'])
        self.assertEqual(format_only('  i\nwork  here '), 'I work here')

    def test_closed_spelling_rules_allow_only_clear_boundary_and_apostrophe_repairs(self):
        for raw, candidate in [('I am a soft ware engineer', 'I am a software engineer.'),
                               ('i dont repair hard ware', "I don't repair hardware."),
                               ('their web site', 'Their website.')]:
            with self.subTest(raw=raw): self.assertEqual(self.parse(raw, candidate)['status'], 'accepted')
        for raw, candidate in [("I don't know", 'I dont know'), ('my software', 'my soft ware'),
                               ('the therapist', 'the rapist'), ('its mine', "It's mine"),
                               ('we were happy', "We we're happy"), ('I am Curly', 'I am Currently')]:
            with self.subTest(raw=raw): self.assertEqual(self.parse(raw, candidate)['status'], 'rejected')

    def test_changes_to_facts_grammar_polarity_quantity_or_order_are_rejected(self):
        cases = [('I might go', 'I will go'), ('I want two coffees', 'I want three coffees'),
                 ('I do not like tea', 'I like tea'), ('I no brush teeth', "I didn't brush my teeth"),
                 ('Yesterday I wake up', 'Yesterday I woke up'), ('Japan', 'I would like to visit Japan'),
                 ('My fravorit is tea', 'My favorite is tea'), ('I saw Alice', 'I saw Bob'),
                 ('I work as an engineer', 'As an engineer I work'), ('it is -5', 'It is 5'),
                 ('It costs $5', 'It costs 5'), ('I go', 'I go?')]
        for raw, candidate in cases:
            with self.subTest(raw=raw):
                result = self.parse(raw, candidate)
                self.assertEqual(result['status'], 'rejected')
                self.assertEqual(result['text'], format_only(raw))

    def test_invalid_or_truncated_json_falls_back_without_exposing_model_text(self):
        for response in ('bad', '{}', '{"normalized_text":null}', '{"normalized_text":"Hi","explanation":"private"}'):
            with self.subTest(response=response):
                result = parse_normalization(response, True, 'i work')
                self.assertEqual(result['text'], 'I work')
                self.assertNotIn('private', json.dumps(result))
        self.assertEqual(parse_normalization('{"normalized_text":"I work."}', False, 'i work')['reason'], 'incomplete_output')

    def test_context_cannot_authorize_hypothetical_word_replacement(self):
        raw = 'Curly am ma a sockway engineer'
        candidate = 'Currently, I am a software engineer.'
        result = self.parse(raw, candidate)
        self.assertEqual(result['status'], 'rejected')
        with self.assertRaises(ValueError): prepare_dialogue_context({}, raw, normalized_text=candidate)
        prepared = prepare_dialogue_context({}, raw, normalized_text=result['text'], missing_words=['ma', 'sockway'])
        self.assertEqual(prepared['text'], raw)
        self.assertEqual(prepared['request']['conversation_source'], 'asr_normalized')
        self.assertTrue(prepared['clarity']['needs_clarification'])

    def test_normalization_request_uses_base_qwen_and_never_supplies_an_audio_reference(self):
        request = normalization_request('i work', [{'role': 'assistant', 'content': 'What do you do?'}])
        self.assertEqual(request['model_id'], 'qwen-base')
        self.assertNotIn('generation_profile', request)
        self.assertEqual(json.loads(request['messages'][-1]['content'])['transcript'], 'i work')
        qwen = Mock(); cancelled = threading.Event(); cancelled.set()
        self.assertEqual(normalize('i work', qwen, cancelled)['reason'], 'cancelled')
        qwen.generate.assert_not_called()

    def test_pipeline_uses_normalized_text_for_dialogue_only(self):
        raw, normalized = 'i am a soft ware engineer', 'I am a software engineer.'
        result = {'transcript': raw, 'reference_text': raw, 'reference': None, 'comparison': None,
                  'phonemes': [{'phone': 'S'}], 'missing_words': [], '_acoustic_evidence': {'private': True}}
        original = deepcopy(result)
        engine = object.__new__(LocalSpeechEngine)
        engine.analyze = Mock(return_value=result)
        events, requests = [], []
        qwen = Mock()
        def generate(request, cancelled, emit):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                output = json.dumps({'normalized_text': normalized})
            elif len(requests) == 2:
                self.assertEqual(request['messages'][-1]['content'], normalized)
                output = 'What do you enjoy most about your work?'
            else:
                self.assertEqual(json.loads(request['messages'][-1]['content'])['transcript'], normalized)
                output = json.dumps({'grammar_corrected': normalized, 'natural': normalized, 'explanation': 'No grammar changes.'})
            emit({'type': 'chunk', 'text': output}); return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        metadata = engine.run({'model_id': 'qwen-base', 'audio_base64': 'YWJj', 'history': [],
            'rewrite_enabled': True, 'normalization_enabled': True}, threading.Event(),
            lambda event: events.append(deepcopy(event)), qwen)
        speech = [event['speech'] for event in events if event['type'] == 'speech'][-1]
        self.assertEqual(len(requests), 3)
        self.assertEqual(speech['transcript'], original['transcript'])
        self.assertEqual(speech['reference_text'], original['reference_text'])
        self.assertEqual(speech['_acoustic_evidence'], original['_acoustic_evidence'])
        self.assertEqual(speech['conversation_text'], normalized)
        self.assertEqual(speech['reference_source'], 'asr')
        self.assertEqual(metadata['conversation_source'], 'asr_normalized')
        self.assertEqual(speech['spoken_response'], 'What do you enjoy most about your work?')


if __name__ == '__main__': unittest.main()
