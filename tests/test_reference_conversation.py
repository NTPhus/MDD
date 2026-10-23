"""Reference edits affect dialogue context, never the recorded ASR evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conversation_context import prepare_dialogue_context
from scripts.local_speech_engine import LocalSpeechEngine, dialogue_request
from scripts.conversation_reply import REFERENCE_APOLOGY


class ReferenceConversationContextTests(unittest.TestCase):
    raw = 'It was the divicote true getesir'
    corrected = 'It was difficult to get here'

    def test_confirmed_edit_drives_reply_without_turning_into_reading_practice(self):
        history = [{'role': 'user', 'content': 'I am visiting today.'},
                   {'role': 'assistant', 'content': 'How was your journey?'}]
        request = {'model_id': 'qwen-base', 'topic_id': 'daily-life', 'history': history,
                   'reference_text': self.corrected, 'needs_clarification': True}
        before = deepcopy(request)
        context = prepare_dialogue_context(request, self.raw, dialogue_text=self.corrected,
                                          source='user_confirmed', missing_words=['divicote', 'getesir'])
        model_request = dialogue_request(context['request'], context['text'])
        self.assertEqual(request, before)
        self.assertEqual(context['text'], self.corrected)
        self.assertEqual(context['request']['conversation_source'], 'user_confirmed')
        self.assertIsNone(context['request']['reference_text'])
        self.assertFalse(context['clarity']['needs_clarification'])
        self.assertEqual(model_request['messages'][-1], {'role': 'user', 'content': self.corrected})
        self.assertEqual(model_request['messages'][-len(history)-1:-1], history)
        self.assertNotIn(self.raw, json.dumps(model_request['messages']))
        self.assertNotIn('I just practised reading', model_request['messages'][-1]['content'])

    def test_confirmed_cafe_order_enters_the_active_order_frame(self):
        target = 'I would like one coffee and one milk tea please'
        context = prepare_dialogue_context({'topic_id': 'food', 'history': [], 'reference_text': target},
            'I would like one copy and one milk tire please', dialogue_text=target, source='user_confirmed')
        request = dialogue_request(context['request'], context['text'])
        self.assertEqual(request['conversation_state']['items'], ['coffee', 'milk tea'])
        self.assertEqual(request['conversation_state']['next_action'], 'ASK_TEMPERATURE')
        self.assertEqual(request['messages'][-1]['content'], target)

    def test_confirmed_oov_names_are_not_misclassified_as_unclear_asr(self):
        names = 'I met Xalrien and Zorvaya yesterday'
        for source in ('asr', 'user_confirmed'):
            with self.subTest(source=source):
                context = prepare_dialogue_context({'history': [], 'needs_clarification': True}, names,
                    dialogue_text=names if source == 'user_confirmed' else None,
                    source=source, missing_words=['Xalrien', 'Zorvaya'])
                self.assertEqual(context['clarity']['needs_clarification'], source == 'asr')
                self.assertEqual(context['request']['needs_clarification'], source == 'asr')

    def test_raw_asr_clarity_rule_and_reading_semantics_remain_separate(self):
        request = {'history': [], 'reference_text': 'A fixed reading prompt'}
        original = deepcopy(request)
        asr = prepare_dialogue_context(request, self.raw, missing_words=['divicote', 'getesir'])
        reading = prepare_dialogue_context(request, self.raw, source='reading_prompt', missing_words=['divicote', 'getesir'])
        self.assertEqual(asr['text'], self.raw)
        self.assertTrue(asr['clarity']['needs_clarification'])
        self.assertFalse(reading['clarity']['needs_clarification'])
        self.assertEqual(reading['request']['reference_text'], request['reference_text'])
        self.assertEqual(request, original)

    def test_edits_require_bounded_confirmed_text_and_cannot_be_auto_rewrites(self):
        for text in (None, '', ' ', 'x' * 1001, 'bad\x00text'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prepare_dialogue_context({}, self.raw, dialogue_text=text, source='user_confirmed')
        with self.assertRaises(ValueError):
            prepare_dialogue_context({}, self.raw, dialogue_text=self.corrected, source='asr')

    def test_history_is_copied_not_removed_by_string_matching(self):
        # A previous genuinely spoken same sentence must not be dropped. The
        # service binds the current turn by job/revision, not text equality.
        history = [{'role': 'user', 'content': self.raw}, {'role': 'assistant', 'content': 'What happened then?'}]
        original = deepcopy(history)
        context = prepare_dialogue_context({'history': history}, self.raw,
                                          dialogue_text=self.corrected, source='user_confirmed')
        self.assertEqual(context['request']['history'], original)
        context['request']['history'][0]['content'] = 'changed in caller view'
        self.assertEqual(history, original)


class ReferenceRegenerationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw = 'It was the divicote true getesir'
        self.target = 'It was difficult to get here'
        self.history = [{'role': 'user', 'content': 'I am visiting today.'},
                        {'role': 'assistant', 'content': 'How was your journey?'}]
        self.speech = {'transcript': self.raw, 'reference_text': self.target,
            'conversation_text': self.target, 'reference_source': 'user_confirmed',
            'missing_words': ['divicote', 'getesir'],
            'phonemes': [{'phone': 'D', 'confidence': .8}],
            'assessment': {'status': 'experimental', 'calibration_status': 'ineligible'},
            'target_words': [{'text': 'difficult', 'status': 'uncertain'}],
            'comparison': None, 'spoken_response': 'Stale old reply.',
            'coaching': {'status': 'ready', 'feedback': {'stale': []}},
            'rewrite': {'status': 'ready', 'natural': 'A stale rewrite.'}}
        self.request = {'operation': 'regenerate_reference', 'dialogue_text': self.target,
            'speech': deepcopy(self.speech), 'history': deepcopy(self.history),
            'model_id': 'qwen-base', 'topic_id': None, 'reference_revision': 1}
        self.engine = object.__new__(LocalSpeechEngine)
        self.engine.analyze = Mock(side_effect=AssertionError('No new audio analysis allowed'))
        self.engine.transcribe = Mock(side_effect=AssertionError('No new ASR allowed'))
        self.engine.phonemes = Mock()
        self.events = []
        self.cancelled = threading.Event()

    def qwen(self, replies):
        qwen = Mock()
        def generate(request, cancelled, emit):
            if request.get('generation_profile') == 'conversation':
                text = next(replies)
            else:
                payload = json.loads(request['messages'][-1]['content'])
                target = payload['transcript']
                text = json.dumps({'grammar_corrected': target, 'natural': target,
                                   'explanation': 'Keep the confirmed wording.'})
            for chunk in text if isinstance(text, (list, tuple)) else [text]:
                emit({'type': 'chunk', 'text': chunk})
            return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        return qwen

    def run_regeneration(self, qwen):
        return self.engine.regenerate(self.request, self.cancelled,
                                      lambda event: self.events.append(deepcopy(event)), qwen)

    def test_confirmed_words_drive_new_dialogue_and_writing_help_without_audio_inference(self):
        request_before = deepcopy(self.request)
        qwen = self.qwen(iter(['Was there a lot of traffic?']))
        metadata = self.run_regeneration(qwen)
        requests = [call.args[0] for call in qwen.generate.call_args_list]
        self.assertEqual(len(requests), 2)
        dialogue, rewrite = requests
        self.assertEqual(dialogue['messages'][-1], {'role': 'user', 'content': self.target})
        self.assertEqual(dialogue['messages'][-len(self.history)-1:-1], self.history)
        self.assertNotIn(self.raw, json.dumps(dialogue['messages']))
        self.assertNotIn('Stale old reply', json.dumps(dialogue['messages']))
        self.assertEqual(json.loads(rewrite['messages'][-1]['content'])['transcript'], self.target)
        first_chunk = next(index for index, event in enumerate(self.events) if event['type'] == 'chunk')
        generating = next(index for index, event in enumerate(self.events)
                          if event['type'] == 'stage' and event['stage'] == 'generating')
        feedback = next(index for index, event in enumerate(self.events)
                        if event['type'] == 'speech' and event['speech']['rewrite']['status'] == 'ready'
                        and event['speech']['coaching']['status'] in ('ready', 'unavailable'))
        self.assertLess(first_chunk, feedback)
        self.assertLess(generating, first_chunk)
        published = next(event['speech'] for event in self.events
                         if event['type'] == 'speech' and event['speech'].get('reply_ready'))
        self.assertEqual(published['rewrite']['status'], 'pending')
        self.assertEqual(published['coaching']['status'], 'pending')
        self.assertEqual(published['spoken_response'], self.events[feedback]['speech']['spoken_response'])
        final = self.events[-1]['speech']
        for field in ('transcript', 'phonemes', 'assessment', 'target_words', 'reference_text'):
            self.assertEqual(final[field], self.speech[field])
        self.assertEqual(final['conversation_text'], self.target)
        self.assertEqual(final['spoken_response'], REFERENCE_APOLOGY + ' Was there a lot of traffic?')
        self.assertEqual(final['spoken_response'], ''.join(e['text'] for e in self.events if e['type'] == 'chunk'))
        self.assertEqual(metadata['reference_apology'], {'text': REFERENCE_APOLOGY, 'source': 'application'})
        self.assertEqual(final['rewrite']['original'], self.target)
        self.assertFalse(final['transcript_clarity']['needs_clarification'])
        self.assertEqual(metadata['conversation_source'], 'user_confirmed')
        self.assertTrue(metadata['regenerated_from_reference'])
        self.assertEqual(self.request, request_before)
        self.assertNotIn('transcribing', [e['stage'] for e in self.events if e['type'] == 'stage'])
        self.engine.analyze.assert_not_called(); self.engine.transcribe.assert_not_called()
        self.assertEqual(self.engine.phonemes.mock_calls, [])

    def test_confirmed_edit_is_used_for_bounded_retry_instead_of_old_asr(self):
        qwen = self.qwen(iter(['Hi!', 'Was the traffic heavy?']))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Was the traffic heavy?')
        self.assertEqual(metadata['reply_source'], 'qwen_retry')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], 'Hi!')
        dialogue_calls = [call for call in qwen.generate.call_args_list
                          if call.args[0].get('generation_profile') == 'conversation']
        self.assertEqual(len(dialogue_calls), 2)
        for call in dialogue_calls:
            self.assertEqual(call.args[0]['messages'][-1]['content'], self.target)
            self.assertNotIn(self.raw, json.dumps(call.args[0]['messages']))
        self.assertEqual(self.events[-1]['speech']['transcript'], self.raw)

    def test_confirmed_oov_names_do_not_trigger_asr_clarification(self):
        names = 'I met Xalrien and Zorvaya yesterday'
        self.request['dialogue_text'] = names
        self.request['speech'].update(conversation_text=names, reference_text=names,
                                     missing_words=['Xalrien', 'Zorvaya'])
        qwen = self.qwen(iter(['How did you meet them?']))
        self.run_regeneration(qwen)
        first = next(call.args[0] for call in qwen.generate.call_args_list
                     if call.args[0].get('generation_profile') == 'conversation')
        self.assertEqual(first['messages'][-1]['content'], names)
        self.assertFalse(self.events[-1]['speech']['transcript_clarity']['needs_clarification'])

    def test_unconfirmed_or_mismatched_edit_is_rejected_before_generation(self):
        for change in ({'reference_source': 'asr'}, {'conversation_text': 'A different text'}):
            self.request['speech'] = {**deepcopy(self.speech), **change}
            qwen = Mock()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_regeneration(qwen)
            qwen.generate.assert_not_called()
        self.engine.analyze.assert_not_called()

    def test_cancelled_reference_never_starts_qwen_or_audio_analysis(self):
        self.cancelled.set()
        qwen = Mock()
        self.assertEqual(self.run_regeneration(qwen), {'finish_reason': 'cancelled'})
        qwen.generate.assert_not_called(); self.engine.analyze.assert_not_called()

    def test_cancel_after_reply_snapshot_prevents_optional_feedback_generation(self):
        qwen = self.qwen(iter(['This response must never be generated.']))
        def emit(event):
            self.events.append(deepcopy(event))
            if event['type'] == 'speech' and event['speech'].get('reply_ready'):
                self.cancelled.set()
        result = self.engine.regenerate(self.request, self.cancelled, emit, qwen)
        self.assertEqual(result, {'finish_reason': 'cancelled'})
        self.assertEqual(qwen.generate.call_count, 1)
        self.assertEqual(qwen.generate.call_args.args[0]['generation_profile'], 'conversation')
        self.assertTrue(any(event['type'] == 'chunk' for event in self.events))
        self.assertNotIn('correcting', [event.get('stage') for event in self.events])

    def test_cancel_at_generating_stage_does_not_call_conversation_model(self):
        qwen = self.qwen(iter(['This response must never be generated.']))
        def emit(event):
            self.events.append(deepcopy(event))
            if event.get('stage') == 'generating':
                self.cancelled.set()
        result = self.engine.regenerate(self.request, self.cancelled, emit, qwen)
        self.assertEqual(result, {'finish_reason': 'cancelled'})
        qwen.generate.assert_not_called()
        self.assertEqual([event for event in self.events if event['type'] == 'chunk'], [])

    def test_unavailable_writing_help_does_not_delay_or_leak_into_published_reply(self):
        qwen = Mock()
        def generate(request, cancelled, emit):
            if request.get('generation_profile') != 'conversation':
                emit({'type': 'chunk', 'text': 'invalid private feedback'})
                raise RuntimeError('Optional writer unavailable')
            terminal = self.events[-2] if self.events[-1].get('stage') == 'generating' else self.events[-1]
            self.assertEqual(terminal['type'], 'speech')
            self.assertEqual(terminal['speech']['rewrite']['status'], 'pending')
            self.assertEqual(terminal['speech']['coaching']['status'], 'pending')
            emit({'type': 'chunk', 'text': 'Did the journey take a long time?'})
            return {'finish_reason': 'stop'}
        qwen.generate.side_effect = generate
        self.run_regeneration(qwen)
        visible = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Did the journey take a long time?')
        self.assertNotIn('invalid private feedback', visible)
        self.assertEqual(self.events[-1]['speech']['rewrite']['status'], 'unavailable')
        self.assertEqual(self.events[-1]['speech']['transcript'], self.raw)
        self.assertEqual(self.events[-1]['speech']['reference_text'], self.target)

    def test_model_apology_then_echo_and_new_question_streams_exactly_one_supplied_apology(self):
        raw = REFERENCE_APOLOGY + ' ' + REFERENCE_APOLOGY + ' You said: ' + self.target + '. Was there a lot of traffic? ' + REFERENCE_APOLOGY
        qwen = self.qwen(iter([list(raw)]))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Was there a lot of traffic?')
        self.assertEqual(visible.count(REFERENCE_APOLOGY), 1)
        self.assertEqual(self.events[-1]['speech']['spoken_response'], visible)
        self.assertNotIn(self.target, visible)
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], raw)
        self.assertEqual(metadata['reply_generation_attempts'], 1)

    def test_raw_asr_sentence_copy_is_removed_after_model_apology_too(self):
        raw = REFERENCE_APOLOGY + ' You mentioned ' + self.raw + '. Did you take a bus?'
        qwen = self.qwen(iter([list(raw)]))
        self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Did you take a bus?')
        self.assertNotIn(self.raw, visible)
        self.assertEqual(self.events[-1]['speech']['transcript'], self.raw)

    def test_unknown_wrapper_embedded_target_never_reaches_live_text_or_voice(self):
        raw = 'I now understand that you said "' + self.target + '". Was the traffic heavy?'
        qwen = self.qwen(iter([list(raw)]))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Was the traffic heavy?')
        self.assertNotIn(self.target, visible)
        self.assertEqual(self.events[-1]['speech']['spoken_response'], visible)
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], raw)

    def test_apology_embedded_copy_and_same_sentence_new_question_preserve_only_new_question(self):
        raw = REFERENCE_APOLOGY + ' I now understand you meant "' + self.target + '", and did you take a taxi?'
        qwen = self.qwen(iter([list(raw)]))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' did you take a taxi?')
        self.assertEqual(visible.count(REFERENCE_APOLOGY), 1)
        self.assertEqual(metadata['reply_generation_attempts'], 1)

    def test_model_apology_alone_cannot_mask_no_answer_or_prevent_retry(self):
        qwen = self.qwen(iter([REFERENCE_APOLOGY, 'Did you get caught in traffic?']))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        self.assertEqual(metadata['reply_source'], 'qwen_retry')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Did you get caught in traffic?')
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], REFERENCE_APOLOGY)

    def test_short_confirmed_sentence_is_not_repeated_as_a_question(self):
        self.request['dialogue_text'] = 'I like pizza'
        self.request['speech'].update(conversation_text='I like pizza', reference_text='I like pizza')
        qwen = self.qwen(iter([list('I like pizza? Which toppings do you enjoy?')]))
        self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, REFERENCE_APOLOGY + ' Which toppings do you enjoy?')

    def test_both_apology_echo_failures_remain_explicit_fallback_with_one_apology(self):
        raw = 'Sorry for the misunderstanding, you said ' + self.target + '.'
        qwen = self.qwen(iter([list(raw), list(raw)]))
        metadata = self.run_regeneration(qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(metadata['reply_source'], 'conversation_unavailable')
        self.assertEqual(metadata['fallback_reason'], 'no_valid_model_reply')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        self.assertTrue(visible.startswith(REFERENCE_APOLOGY + ' '))
        self.assertEqual(visible.count(REFERENCE_APOLOGY), 1)
        self.assertIn('trouble continuing', visible)
        self.assertNotIn(self.target, visible)
        self.assertEqual(self.events[-1]['speech']['spoken_response'], visible)
        self.assertEqual([attempt['raw_text'] for attempt in metadata['reply_attempts']], [raw, raw])

    def test_cancellation_while_model_only_repeats_never_emits_supplied_apology(self):
        qwen = Mock()
        def generate(request, cancelled, emit):
            if request.get('generation_profile') != 'conversation':
                emit({'type': 'chunk', 'text': json.dumps({'grammar_corrected': self.target,
                      'natural': self.target, 'explanation': 'Keep the confirmed wording.'})})
                return {'finish_reason': 'stop'}
            for character in REFERENCE_APOLOGY + ' ' + self.target + '.':
                emit({'type': 'chunk', 'text': character})
            cancelled.set()
            return {'finish_reason': 'cancelled'}
        qwen.generate.side_effect = generate
        self.assertEqual(self.run_regeneration(qwen), {'finish_reason': 'cancelled'})
        self.assertEqual([e for e in self.events if e['type'] == 'chunk'], [])
        self.assertEqual(qwen.generate.call_count, 1)

    def test_initial_audio_reply_does_not_receive_regeneration_apology(self):
        result = deepcopy(self.speech)
        result.update(reference_source='asr', reference_text=self.raw, missing_words=[])
        result.pop('conversation_text')
        self.engine.analyze = Mock(return_value=result)
        qwen = self.qwen(iter(['Did the journey take a long time?']))
        metadata = self.engine.run({'audio_base64': 'YWJj', 'model_id': 'qwen-base',
                                   'history': self.history, 'rewrite_enabled': True},
                                  self.cancelled, lambda e: self.events.append(deepcopy(e)), qwen)
        visible = ''.join(e['text'] for e in self.events if e['type'] == 'chunk')
        self.assertEqual(visible, 'Did the journey take a long time?')
        self.assertNotIn('reference_apology', metadata)


if __name__ == '__main__':
    unittest.main()
