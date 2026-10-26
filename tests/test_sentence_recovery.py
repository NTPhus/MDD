import json
import threading
import unittest
from unittest.mock import Mock

from scripts import sentence_recovery as recovery


class SentenceRecoveryTests(unittest.TestCase):
    def parse(self, raw, text, **kwargs):
        return recovery.parse_recovery(json.dumps({'status': 'recovered', 'text': text}),
                                       kwargs.get('finished', True), raw)

    def test_trigger_only_when_needed(self):
        self.assertFalse(recovery.needs_recovery('I work at home', {'status': 'unchanged'}))
        self.assertFalse(recovery.needs_recovery('My friend Meilin is here', {'status': 'unchanged'}, ['Meilin']))
        self.assertTrue(recovery.needs_recovery('a tiket for tomorow', {'status': 'unavailable'}))
        self.assertTrue(recovery.needs_recovery('a tiket for tomorow', {'status': 'accepted', 'unresolved_words': ['tiket', 'tomorow']}))
        self.assertTrue(recovery.needs_recovery('a tiket for tomorow', {'status': 'accepted'}, ['tiket', 'tomorow']))
        self.assertTrue(recovery.needs_recovery('some tikket', {'status': 'accepted', 'evidence': [
            {'reason': 'unknown_word', 'accepted': False}]}))
        self.assertFalse(recovery.needs_recovery('', {'status': 'unavailable'}))
        self.assertFalse(recovery.needs_recovery('hello', {'status': 'unavailable', 'reason': 'cancelled'}))

    def test_suspicious_short_word_splits(self):
        self.assertTrue(recovery.needs_recovery('unclear', {'status': 'accepted', 'evidence': [
            {'accepted': True, 'selected': 'un clear'}]}))
        self.assertFalse(recovery.needs_recovery('soft ware', {'status': 'accepted', 'evidence': [
            {'accepted': True, 'selected': 'software'}]}))

    def test_boundaries_spelling_and_function_words_can_be_recovered(self):
        result = self.parse('bring the things fromer stor', 'Bring these things from the store.')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['source'], 'qwen_sentence_inferred')
        self.assertTrue(result['requires_confirmation'])
        self.assertFalse(result['reference_confirmed'])
        self.assertFalse(result['lexical_preservation'])
        self.assertEqual(result['original'], 'bring the things fromer stor')
        self.assertEqual(self.parse('the soft ware enginer works frome home',
                                   'The software engineer works from home.')['status'], 'accepted')

    def test_readability_grammar_is_independent_from_acoustic_reference(self):
        self.assertEqual(self.parse('Yesterday I go home', 'Yesterday I went home.')['status'], 'accepted')
        self.assertEqual(self.parse('she bring two tiket', 'She brings two tickets.')['status'], 'accepted')
        self.assertEqual(self.parse('we need of small bag', 'We need a small bag.')['status'], 'accepted')

    def test_format_only_is_unchanged_not_false_recovery(self):
        result = self.parse('hello there', 'Hello there.')
        self.assertEqual(result['status'], 'unchanged')
        self.assertFalse(result['requires_confirmation'])
        self.assertTrue(result['lexical_preservation'])

    def test_protected_information(self):
        cases = [
            ('I need two tiket', 'I need three tickets.', 'number_change'),
            ('I need 25 tiket', 'I need 20 tickets.', 'number_change'),
            ('I do not need tiket', 'I do need tickets.', 'negation_change'),
            ('I wont buy tiket', 'I will buy tickets.', 'negation_change'),
            ('I met Alice at home', 'I met Alicia at home.', 'name_change'),
            ('he gives her a book', 'He gives him a book.', 'pronoun_change'),
            ('the cost is $20', 'The cost is 20.', 'symbol_change'),
        ]
        for raw, text, reason in cases:
            with self.subTest(reason=reason):
                result = self.parse(raw, text)
                self.assertEqual(result['status'], 'ambiguous')
                self.assertEqual(result['reason'], reason)
                self.assertNotEqual(result['text'], text)

    def test_no_summary_hallucination_or_unrelated_paraphrase(self):
        cases = [
            ('I walk to work and buy bread then meet my brother and get the bus home', 'I had a busy day.'),
            ('I work at home', 'I work at home and bought a new car yesterday.'),
            ('I repair bicycles', 'I fix bikes.'),
            ('I like coffee in the morning', 'I like tea in the morning.'),
        ]
        for raw, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.parse(raw, text)['status'], 'ambiguous')

    def test_rejected_proposal_is_explicit_unselected_diagnostics(self):
        raw, proposed = 'I like coffee in the morning', 'I like tea in the morning.'
        result = self.parse(raw, proposed)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['text'], raw)
        self.assertEqual(result['proposed_text'], proposed)
        self.assertIs(result['proposal_selected'], False)
        self.assertEqual(result['validation']['reason'], 'unsupported_spelling_change')
        span = result['validation']['rejected_span']
        self.assertEqual(span['original'], 'coffee')
        self.assertEqual(span['proposed'], 'tea')
        self.assertLess(span['spelling_similarity'], span['min_spelling_similarity'])
        self.assertNotIn('proposed_text', self.parse('hello', 'Hello.'))
        self.assertNotIn('proposed_text', self.parse('hello', 'Hello\x00'))

    def test_unclear_invalid_and_incomplete_output(self):
        self.assertEqual(recovery.parse_recovery('{', True, 'hello')['status'], 'unavailable')
        self.assertEqual(recovery.parse_recovery('[]', True, 'hello')['status'], 'unavailable')
        self.assertEqual(recovery.parse_recovery('{"status":[],"text":"hi"}', True, 'hello')['status'], 'unavailable')
        self.assertEqual(self.parse('hello', 'Hello', finished=False)['reason'], 'incomplete_output')
        self.assertEqual(self.parse('', 'Hello')['reason'], 'invalid_text')
        self.assertEqual(self.parse('hello', 'Hello\x00')['reason'], 'invalid_text')
        result = recovery.parse_recovery('{"status":"unclear","text":""}', True, 'unclear phrase')
        self.assertEqual(result['status'], 'ambiguous')
        self.assertTrue(result['requires_confirmation'])

    def test_request_is_local_base_greedy_task_and_bounded(self):
        request = recovery.recovery_request('x' * 1000, [{'role': 'assistant', 'content': 'earlier'}])
        self.assertEqual(request['model_id'], 'qwen-base')
        self.assertNotIn('generation_profile', request)
        self.assertLessEqual(request['max_tokens'], 512)
        self.assertLessEqual(request['timeout'], 25.)
        payload = json.loads(request['messages'][-1]['content'])
        self.assertEqual(payload['transcript'], 'x' * 1000)
        self.assertEqual(payload['conversation'], [{'role': 'assistant', 'content': 'earlier'}])
        all_messages = json.dumps(request['messages']).lower()
        self.assertNotIn('stella', all_messages)
        self.assertNotIn('snow peas', all_messages)

    def test_cancelled_does_not_call_model_or_publish_proposal(self):
        cancelled = threading.Event()
        cancelled.set()
        model = Mock()
        result = recovery.recover('hello', model, cancelled)
        model.generate.assert_not_called()
        self.assertEqual(result['reason'], 'cancelled')
        self.assertEqual(result['status'], 'unavailable')

    def test_model_failure_and_finish_reason(self):
        event = threading.Event()
        model = Mock()
        model.generate.side_effect = RuntimeError('private path')
        result = recovery.recover('hello', model, event)
        self.assertEqual(result['reason'], 'model_unavailable')
        self.assertNotIn('private', json.dumps(result))
        def generate(request, cancelled, emit):
            emit({'type': 'chunk', 'text': '{"status":"recovered","text":"Hello."}'})
            return {'finish_reason': 'length'}
        model.generate.side_effect = generate
        self.assertEqual(recovery.recover('hello', model, event)['reason'], 'incomplete_output')

    def test_model_completion_cancellation_wins(self):
        event = threading.Event()
        model = Mock()
        def generate(request, cancelled, emit):
            emit({'type': 'chunk', 'text': '{"status":"recovered","text":"Hello."}'})
            cancelled.set()
            return {'finish_reason': 'stop'}
        model.generate.side_effect = generate
        result = recovery.recover('hello', model, event)
        self.assertEqual(result['reason'], 'cancelled')
        self.assertEqual(result['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
