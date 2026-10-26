"""Sentence repairs must preserve the recording and provide bounded repeat targets."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_speech_chat as speech_fixture
from mdd.demo_chat import ChatError
from scripts.local_speech_engine import LocalSpeechEngine, REPEAT_REQUEST, UnrecognizedSpeech
from scripts.sentence_correction import correction_request, parse_correction


class SentenceCorrectionTests(unittest.TestCase):
    def parse(self, original, corrected, explanation="A short sentence completion.", finished=True, history=None, **extra):
        return parse_correction(json.dumps({"corrected": corrected, "explanation": explanation, **extra}),
                                finished, original, history=history)

    def test_complete_ungrammatical_thought_keeps_every_spoken_word(self):
        original = "Yesterday I go to the park with my friend"
        corrected = original + "."
        result = self.parse(original, corrected, "Use went for a past action.")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["original"], original)
        self.assertEqual(result["corrected"], corrected)
        self.assertTrue(result["changed"])
        self.assertEqual(result["model_id"], "qwen-base")
        self.assertEqual(result["source"], "qwen")
        self.assertEqual(result["mode"], "completion_only")
        self.assertEqual(result["explanation"], "Your spoken words are unchanged.")
        self.assertEqual(result["model_explanation"], "Use went for a past action.")
        self.assertNotIn('went', result['corrected'])

    def test_already_complete_sentence_is_not_invented_as_an_error(self):
        result = self.parse("I like coffee.", "I like coffee.", "This sentence is already correct.")
        self.assertFalse(result["changed"])
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["original"], "I like coffee.")

    def test_correction_request_selects_base_model_and_keeps_transcript_as_user_data(self):
        transcript = "Yesterday I go park with my friend"
        request = correction_request(transcript)
        self.assertEqual(request["model_id"], "qwen-base")
        self.assertIsInstance(request["max_tokens"], int)
        self.assertGreater(request["max_tokens"], 0)
        self.assertGreater(request["timeout"], 0)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][-1]["role"], "user")
        self.assertIn(transcript, request["messages"][-1]["content"])
        instructions = request["messages"][0]["content"].lower()
        self.assertIn("not correct grammar", instructions)
        self.assertIn("preserve every spoken word", instructions)
        self.assertIn("json", instructions)

    def test_verb_inflections_and_person_changes_are_rejected(self):
        for original, corrected in (('I liking coffee', 'I like coffee.'), ('Yesterday I make coffee', 'Yesterday I made coffee.'),
                                     ('Yesterday he visit the park', 'Yesterday he visited the park.')):
            with self.subTest(original=original), self.assertRaises(ValueError): self.parse(original, corrected)
        with self.assertRaises(ValueError): self.parse('I like coffee.', 'She likes coffee.')

    def test_contractions_are_not_rewritten_as_grammar_normalization(self):
        for original, corrected in (("I'm a student.", 'I am a student.'), ("I don't like coffee.", 'I do not like coffee.'),
                                     ("He can't go.", 'He cannot go.')):
            with self.subTest(original=original), self.assertRaises(ValueError): self.parse(original, corrected)
        with self.assertRaises(ValueError): self.parse('A beautiful city.', 'I live in a beautiful city.')

    def test_short_answer_gets_only_the_frame_supported_by_previous_question(self):
        for question, original, completed in (
            ('Where would you love to visit?', 'Japan', 'I would love to visit Japan.'),
            ('That sounds fun! Where would you like to go?', 'Japan', 'I would like to go to Japan.'),
            ("Let's plan a trip together! Where would you love to travel, and what makes that place special to you?",
             'Japan', 'I would love to travel to Japan.'),
            ('What is your name?', 'Alex', 'My name is Alex.'),
            ('Where are you from?', 'Vietnam', 'I am from Vietnam.'),
            ('What is your favorite drink?', 'coffee', 'My favorite drink is coffee.'),
        ):
            with self.subTest(question=question):
                result = self.parse(original, completed, history=[{'role': 'assistant', 'content': question}])
                self.assertEqual(result['corrected'], completed)
                self.assertIn('previous question', result['explanation'])
                self.assertEqual(result['changes'][-1]['after'].split()[-1], original + '.')

    def test_unrelated_or_old_questions_cannot_supply_an_answer_frame(self):
        for history in (None, [{'role': 'assistant', 'content': 'What do you think?'}],
                        [{'role': 'assistant', 'content': 'Where would you love to visit?'},
                         {'role': 'user', 'content': 'I like coffee.'}],
                        [{'role': 'assistant', 'content': 'What is your name?'}]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                self.parse('Japan', 'I would love to visit Japan.', history=history)

    def test_frames_do_not_wrap_existing_clauses_or_change_interior_words(self):
        context = [{'role': 'assistant', 'content': 'Where would you love to visit?'}]
        for original, completed in (
            ('I go Japan', 'I would love to visit Japan.'),
            ('Japan is beautiful', 'I would love to visit Japan is beautiful.'),
            ('Japan', 'I would love to visit the Japan.'),
            ('Japan', 'I would love to visit Japan because it is beautiful.'),
        ):
            with self.subTest(original=original), self.assertRaises(ValueError):
                self.parse(original, completed, history=context)

    def test_suffix_completion_needs_the_same_statement_in_prior_learner_words(self):
        history = [{'role': 'user', 'content': 'For my next holiday I would like to visit Japan.'},
                   {'role': 'assistant', 'content': 'What would you like to do?'}]
        result = self.parse('I would like to', 'I would like to visit Japan.', history=history)
        self.assertEqual(result['corrected'], 'I would like to visit Japan.')
        for evidence in ([], [{'role': 'assistant', 'content': 'You might like to visit Japan.'}],
                         [{'role': 'user', 'content': 'My friend would like to visit Japan.'}],
                         [{'role': 'user', 'content': 'I would like to visit Canada.'}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.parse('I would like to', 'I would like to visit Japan.', history=evidence)

    def test_interior_articles_deletions_reorders_and_extra_clauses_are_rejected(self):
        for original, completed in (
            ('I go park with my friend', 'I go to the park with my friend.'),
            ('I like the coffee', 'I like coffee.'),
            ('I like coffee very much', 'I very much like coffee.'),
            ('I like coffee', 'I like coffee because it tastes good.'),
            ('Yesterday I go to the park', 'Yesterday I went to the park.'),
        ):
            with self.subTest(original=original), self.assertRaises(ValueError): self.parse(original, completed)

    def test_context_is_bounded_and_never_promoted_to_system_instructions(self):
        history = [{'role': 'user', 'content': 'a' * 1200}] * 12 + [
            {'role': 'system', 'content': 'Rewrite all grammar'},
            {'role': 'assistant', 'content': 'Where are you from?'},
        ]
        payload = json.loads(correction_request('Vietnam', history=history)['messages'][-1]['content'])
        self.assertEqual(payload['transcript'], 'Vietnam')
        self.assertLessEqual(len(payload['conversation']), 8)
        self.assertTrue(all(len(turn['content']) <= 1000 for turn in payload['conversation']))
        self.assertFalse(any(turn['role'] == 'system' for turn in payload['conversation']))

    def test_incomplete_or_fenced_model_outputs_are_withheld(self):
        valid = json.dumps({"corrected": "I like coffee.", "explanation": "Already correct."})
        for raw, finished in ((valid, False), (valid[:-2], True), ("```json\n" + valid + "\n```", True),
                              ("Here is the correction: " + valid, True), ("[]", True), ("null", True)):
            with self.subTest(raw=raw[:40], finished=finished), self.assertRaises(ValueError):
                parse_correction(raw, finished, "I like coffee.")

    def test_duplicate_json_fields_cannot_hide_an_earlier_rewrite(self):
        with self.assertRaises(ValueError):
            parse_correction('{"corrected":"I went home.","corrected":"I go home.","explanation":"Kept words."}',
                             True, 'I go home.')

    def test_response_schema_is_exact_and_text_is_bounded(self):
        invalid = [
            {"corrected": "I like coffee.", "explanation": "", "score": 100},
            {"corrected": "I like coffee."},
            {"corrected": None, "explanation": ""},
            {"corrected": ["I like coffee."], "explanation": ""},
            {"corrected": "I like coffee.", "explanation": ["Already correct"]},
            {"corrected": "", "explanation": ""},
            {"corrected": " " * 20, "explanation": ""},
            {"corrected": "a" * 1001, "explanation": ""},
            {"corrected": "I like coffee.", "explanation": "a" * 301},
        ]
        for response in invalid:
            with self.subTest(response=str(response)[:80]), self.assertRaises(ValueError):
                parse_correction(json.dumps(response), True, "I like coffee.")

    def test_numeric_facts_cannot_be_changed_added_or_dropped(self):
        for original, corrected in (("I have 2 cats.", "I have 3 cats."),
                                     ("I have 2 cats.", "I have cats."),
                                     ("I have cats.", "I have 2 cats."),
                                     ("I paid 12.50 for lunch.", "I paid 1250 for lunch.")):
            with self.subTest(original=original, corrected=corrected), self.assertRaises(ValueError):
                self.parse(original, corrected)

    def test_negation_cannot_be_added_or_removed(self):
        for original, corrected in (("I like coffee.", "I do not like coffee."),
                                     ("I do not like coffee.", "I like coffee."),
                                     ("I never drink coffee.", "I drink coffee."),
                                     ("I drink coffee.", "I never drink coffee.")):
            with self.subTest(original=original, corrected=corrected), self.assertRaises(ValueError):
                self.parse(original, corrected)

    def test_unrelated_rewrites_and_new_content_are_rejected(self):
        for corrected in ("The stock market will rise tomorrow.",
                          "I like coffee because it reminds me of my grandmother in London."):
            with self.subTest(corrected=corrected), self.assertRaises(ValueError):
                self.parse("I like coffee.", corrected)


class SentenceCorrectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.original = "Yesterday I go park with my friend"
        self.corrected = self.original + "."
        self.result = {
            "transcript": self.original, "reference_text": self.original, "reference_source": "asr",
            "comparison": None, "reference": None, "phonemes": [{"phone": "F"}, {"phone": "R"}],
        }
        # Bypass only model loading. The production run method and its request,
        # callback, cancellation and evidence handling all run unchanged.
        self.engine = object.__new__(LocalSpeechEngine)
        self.engine.analyze = Mock(side_effect=lambda *args: deepcopy(self.result))
        self.request = {"model_id": "qwen-trained", "audio_base64": "YWJj", "history": []}
        self.cancelled = threading.Event()
        self.events = []
        self.emit = lambda event: self.events.append(deepcopy(event))

    def qwen_with_outputs(self, outputs):
        qwen = Mock()
        remaining = iter(outputs)

        def generate(request, cancelled, callback):
            text, finish_reason = next(remaining)
            callback({"type": "chunk", "text": text})
            return {"finish_reason": finish_reason}

        qwen.generate.side_effect = generate
        return qwen

    def correction_json(self):
        return json.dumps({"corrected": self.corrected, "explanation": "Keep the spoken words unchanged."})

    def test_no_recognized_words_asks_for_another_recording_without_model_guess(self):
        self.engine.analyze.side_effect = UnrecognizedSpeech('No usable words')
        qwen = Mock()
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        qwen.generate.assert_not_called()
        output = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(output, REPEAT_REQUEST)
        self.assertEqual(metadata['fallback_reason'], 'unrecognized_speech')
        final = self.events[-1]['speech']
        self.assertEqual(final['transcript'], '')
        self.assertFalse(final['pronunciation_reference_ready'])
        self.assertEqual(final['assessment']['status'], 'unavailable')
        self.assertEqual(final['rewrite']['status'], 'not_needed')

    def test_model_or_audio_failure_is_not_misrepresented_as_unclear_speech(self):
        self.engine.analyze.side_effect = RuntimeError('Model failed')
        with self.assertRaises(RuntimeError):
            self.engine.run(self.request, self.cancelled, self.emit, Mock())
        self.assertFalse(any(event['type'] == 'chunk' for event in self.events))

    def test_cancelled_empty_recognition_cannot_publish_a_late_repeat_prompt(self):
        self.engine.analyze.side_effect = UnrecognizedSpeech('No usable words')
        self.cancelled.set()
        result = self.engine.run(self.request, self.cancelled, self.emit, Mock())
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertFalse(any(event['type'] == 'chunk' for event in self.events))

    def test_unclear_transcript_fallback_is_explicit_and_never_claimed_as_a_qwen_reply(self):
        transcript = 'It was the divicote true getesir'
        self.result.update(transcript=transcript, reference_text=transcript,
                           missing_words=['divicote', 'getesir'])
        self.request['rewrite_enabled'] = True
        qwen = self.qwen_with_outputs([('not JSON', 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        output = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        final = self.events[-1]['speech']
        self.assertEqual(metadata['reply_source'], 'clarification_fallback')
        self.assertEqual(metadata['reply_generation_attempts'], 0)
        self.assertEqual(metadata['reply_attempts'], [])
        self.assertFalse(metadata['reply_model_used'])
        self.assertEqual(qwen.generate.call_count, 1, 'Only optional grammar feedback uses Qwen.')
        self.assertIn('Could you say it again, please?', output)
        self.assertNotIn('Hi!', output)
        self.assertTrue(final['transcript_clarity']['needs_clarification'])
        self.assertEqual(final['transcript'], transcript)
        self.assertEqual(final['reference_text'], transcript)
        self.assertEqual(final['spoken_response'], output)

    def test_unresolved_words_never_receive_the_reported_great_day_reply(self):
        self.result.update(transcript='It was the divicote true getesir',
                           reference_text='It was the divicote true getesir',
                           missing_words=['divicote', 'getesir'])
        self.request['rewrite_enabled'] = True
        bad = 'It was a great day! How does it feel for you today?'
        qwen = self.qwen_with_outputs([('not JSON', 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        output = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertNotIn('great day', output)
        self.assertIn('Could you say it again, please?', output)
        self.assertEqual(metadata['reply_source'], 'clarification_fallback')
        self.assertEqual(metadata['reply_attempts'], [])
        self.assertEqual(qwen.generate.call_count, 1)

    def test_exact_ambiguous_bert_example_asks_repeat_before_optional_feedback(self):
        raw = 'I will wi to order a cupf coffee and one ice ta please'
        self.result.update(transcript=raw, reference_text=raw, missing_words=['wi', 'cupf'])
        self.engine.bert_recovery = Mock()
        self.engine.bert_recovery.recover.return_value = {
            'provider': 'bert', 'status': 'ambiguous', 'original': raw, 'text': raw,
            'reference_confirmed': False, 'reason': 'candidate_ambiguity', 'latency_sec': .1}
        self.request.update(recovery_enabled=True, recovery_provider='bert', rewrite_enabled=True,
                            reference_source='asr', topic_id='food')
        qwen = self.qwen_with_outputs([(json.dumps({'status': 'unclear', 'text': ''}), 'stop'),
                                       ('not JSON', 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        text = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        final = self.events[-1]['speech']
        self.assertEqual(text, "Sorry, I didn't catch that. Could you say it again, please?")
        self.assertEqual(final['spoken_response'], text)
        self.assertEqual(final['reference_text'], raw)
        self.assertTrue(final['transcript_clarity']['needs_clarification'])
        self.assertEqual(metadata['reply_source'], 'clarification_fallback')
        self.assertEqual(metadata['follow_up_source'], 'application')
        self.assertEqual(metadata['reply_generation_attempts'], 0)
        self.assertIsNone(metadata['conversation_state'], 'An unclear order cannot advance cafe state.')
        self.assertEqual(qwen.generate.call_count, 2, 'One sentence recovery and one optional writing pass.')
        self.assertEqual(json.loads(qwen.generate.call_args_list[0].args[0]['messages'][-1]['content'])['transcript'], raw)
        self.assertTrue(all(call.args[0].get('generation_profile') != 'conversation'
                            for call in qwen.generate.call_args_list))
        first_reply = next(i for i, event in enumerate(self.events) if event['type'] == 'chunk')
        final_feedback = max(i for i, event in enumerate(self.events) if event['type'] == 'speech')
        self.assertLess(first_reply, final_feedback)

    def test_bot_reply_precedes_optional_grammar_help_and_never_relabels_audio(self):
        self.original = 'Yesterday I go to the park'
        self.result.update(transcript=self.original, reference_text=self.original)
        self.request['rewrite_enabled'] = True
        rewrite = json.dumps({'grammar_corrected': 'Yesterday I went to the park.',
                              'natural': 'I went to the park yesterday.', 'explanation': 'Use the past tense.'})
        qwen = self.qwen_with_outputs([('Did you enjoy your time there?', 'stop'), (rewrite, 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        final = self.events[-1]['speech']
        first_chunk = next(index for index, event in enumerate(self.events) if event['type'] == 'chunk')
        written = next(index for index, event in enumerate(self.events)
                       if event['type'] == 'speech' and event['speech']['rewrite']['status'] == 'ready')
        self.assertLess(first_chunk, written)
        completed_feedback = next(index for index, event in enumerate(self.events)
            if event['type'] == 'speech' and event['speech']['rewrite']['status'] == 'ready'
            and event['speech']['coaching']['status'] in ('ready', 'unavailable'))
        self.assertLess(first_chunk, completed_feedback)
        published = next(event['speech'] for event in self.events
                         if event['type'] == 'speech' and event['speech'].get('reply_ready'))
        self.assertEqual(published['rewrite']['status'], 'pending')
        self.assertEqual(published['coaching']['status'], 'pending')
        self.assertEqual(published['spoken_response'], 'Did you enjoy your time there?')
        self.assertEqual(final['transcript'], self.original)
        self.assertEqual(final['reference_text'], self.original)
        self.assertEqual(final['rewrite']['grammar_corrected'], 'Yesterday I went to the park.')
        self.assertEqual(final['spoken_response'], 'Did you enjoy your time there?')
        self.assertEqual(final['correction']['status'], 'not_needed')
        self.assertEqual(metadata['rewrite_mode'], 'grammar_and_natural')
        self.assertNotIn('generation_profile', qwen.generate.call_args_list[1].args[0])

    def test_reply_precedes_completion_and_sound_coach_without_relabeling_the_audio(self):
        reference = {"canonical_phones": ["F", "R"], "canonical_word_indices": [7, 7]}
        self.result.update(reference=reference, comparison={"words": [{"text": "friend", "status": "difference"}]})
        feedback = {"friend": [{"issue": "Possible sound difference.", "suggestion": "Practise friend slowly."}]}
        qwen = self.qwen_with_outputs([("That sounds enjoyable! What did you do at the park?", "stop"), (self.correction_json(), "stop"), (json.dumps(feedback), "stop")])
        with patch("scripts.local_speech_engine.native_condition", return_value={"input_alignment_status": "ok"}) as native, \
             patch("scripts.l2arctic_plus_data.messages", return_value=[{"role": "user", "content": "Sound evidence"}]), \
             patch("scripts.local_speech_engine.grounded_coaching", return_value=feedback):
            metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        stages = [event["stage"] for event in self.events if event["type"] == "stage"]
        self.assertEqual(stages, ["transcribing", "generating", "correcting", "coaching"])
        corrected_index = next(index for index, event in enumerate(self.events)
                               if event["type"] == "speech" and event["speech"]["correction"]["status"] == "ready")
        first_reply_index = next(index for index, event in enumerate(self.events) if event["type"] == "chunk")
        self.assertLess(first_reply_index, corrected_index)
        coached_index = next(index for index, event in enumerate(self.events)
                             if event['type'] == 'speech' and event['speech']['coaching']['status'] == 'ready')
        self.assertLess(first_reply_index, coached_index)
        final = self.events[-1]["speech"]
        self.assertEqual(final["transcript"], self.original)
        self.assertEqual(final["reference_text"], self.original)
        self.assertEqual(final["correction"]["corrected"], self.corrected)
        native.assert_called_once_with(self.original, reference, ["F", "R"])
        requests = [call.args[0] for call in qwen.generate.call_args_list]
        self.assertEqual([request["model_id"] for request in requests], ["qwen-base", "qwen-base", "qwen-trained"])
        self.assertEqual(metadata["correction_status"], "ready")
        self.assertEqual(final["correction"]["mode"], "completion_only")
        self.assertIn("go park", final["correction"]["corrected"])
        self.assertEqual(final['spoken_response'], 'That sounds enjoyable! What did you do at the park?')
        self.assertNotIn('Practise friend', final['spoken_response'])
        self.assertEqual(final['coaching']['feedback'], feedback)

    def test_topic_question_is_shared_with_completion_and_reply_without_changing_audio(self):
        self.original, self.corrected = "Japan", "I would love to travel to Japan."
        self.result.update(transcript=self.original, reference_text=self.original)
        self.request.update(topic_id='travel')
        qwen = self.qwen_with_outputs([('Japan has many places to explore. What interests you most about Japan?', 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        final = self.events[-1]['speech']
        self.assertEqual(final['correction']['corrected'], self.corrected)
        self.assertEqual(final['transcript'], 'Japan')
        self.assertEqual(final['reference_text'], 'Japan')
        second, first = [call.args[0] for call in qwen.generate.call_args_list]
        self.assertIn('Where would you love to travel', json.dumps(first['messages']))
        self.assertEqual(second['messages'][-1]['content'], 'Japan')
        self.assertEqual(metadata['dialogue_policy'], 'natural_conversation')

    def test_grammar_rewrite_is_rejected_and_bot_keeps_talking_about_raw_utterance(self):
        invalid = json.dumps({'corrected': 'Yesterday I went to the park with my friend.',
                              'explanation': 'Use past tense.'})
        qwen = self.qwen_with_outputs([('What did you do at the park?', 'stop'), (invalid, 'stop')])
        self.engine.run(self.request, self.cancelled, self.emit, qwen)
        final = self.events[-1]['speech']
        self.assertEqual(final['correction']['status'], 'unavailable')
        self.assertEqual(final['transcript'], self.original)
        self.assertEqual(qwen.generate.call_args_list[0].args[0]['messages'][-1]['content'], self.original)

    def test_echoing_model_output_is_removed_before_live_reply_and_voice(self):
        raw = f'You said: "{self.original}." How was your visit?'
        qwen = self.qwen_with_outputs([(raw, 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(emitted, 'How was your visit?')
        self.assertEqual(self.events[-1]['speech']['spoken_response'], emitted)
        self.assertEqual(metadata['raw_reply_text'], raw)
        self.assertEqual(len(metadata['omitted_leading_echoes']), 1)

    def test_repeated_echo_only_output_reports_model_failure_without_blame_on_recording(self):
        qwen = self.qwen_with_outputs([(self.original + '.', 'stop'), (self.original + '.', 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk').strip()
        self.assertNotIn(self.original, emitted)
        self.assertEqual(metadata['reply_source'], 'conversation_unavailable')
        self.assertEqual(metadata['fallback_reason'], 'no_valid_model_reply')
        self.assertIn('trouble continuing', emitted)
        self.assertNotIn('say it', emitted)
        self.assertEqual(metadata['raw_reply_text'], self.original + '.')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        self.assertEqual(qwen.generate.call_count, 3)

    def test_echo_only_reply_gets_one_contextual_retry_before_requesting_another_recording(self):
        response = 'Did you find a quiet spot to sit?'
        self.request['history'] = [{'role': 'user', 'content': 'I need some fresh air.'},
                                   {'role': 'assistant', 'content': 'Did you get outside?'}]
        qwen = self.qwen_with_outputs([(self.original + '.', 'stop'), (response, 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(emitted, response)
        self.assertEqual(self.events[-1]['speech']['spoken_response'], response)
        self.assertEqual(metadata['reply_source'], 'qwen_retry')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        first, retry = [call.args[0] for call in qwen.generate.call_args_list[:2]]
        self.assertEqual(retry['messages'][1:], first['messages'][1:])
        self.assertEqual(retry['messages'][-1]['content'], self.original)
        self.assertEqual(retry['model_id'], 'qwen-base')
        self.assertLessEqual(retry['max_tokens'], 96)
        self.assertLessEqual(retry['timeout'], 35)
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], self.original + '.')
        self.assertEqual(metadata['reply_attempts'][1]['raw_text'], response)

    def test_hi_only_for_reported_unclear_speech_retries_before_text_or_voice(self):
        self.original = self.corrected = 'It was the divicote true getesir'
        self.result.update(transcript=self.original, reference_text=self.original)
        response = 'Could you explain what you mean by the last few words?'
        qwen = self.qwen_with_outputs([('Hi!', 'stop'), (response, 'stop'), (self.correction_json(), 'stop')])
        original_generate = qwen.generate.side_effect

        def generate(request, cancelled, callback):
            metadata = original_generate(request, cancelled, callback)
            if qwen.generate.call_count == 1:
                self.assertFalse(any(event['type'] == 'chunk' for event in self.events),
                                 'The greeting-only first attempt must stay private before retry.')
            return metadata

        qwen.generate.side_effect = generate
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        visible = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(visible, response)
        self.assertEqual(self.events[-1]['speech']['spoken_response'], response)
        self.assertEqual(self.events[-1]['speech']['transcript'], self.original)
        self.assertEqual(metadata['reply_source'], 'qwen_retry')
        self.assertEqual(metadata['reply_generation_attempts'], 2)
        self.assertEqual(metadata['reply_attempts'][0]['raw_text'], 'Hi!')
        self.assertEqual(metadata['reply_attempts'][1]['raw_text'], response)
        self.assertTrue(any(item['reason'] == 'no_conversation_contribution' for item in metadata['blocked_questions']))
        self.assertEqual(qwen.generate.call_count, 3, 'One completion fixture and at most two dialogue attempts.')

    def test_cancelling_an_echo_retry_does_not_emit_a_fallback_or_voice(self):
        qwen = self.qwen_with_outputs([(self.original + '.', 'stop'), (self.correction_json(), 'stop')])
        original_generate = qwen.generate.side_effect
        def generate(request, cancelled, callback):
            if qwen.generate.call_count == 2:
                cancelled.set()
                return {'finish_reason': 'cancelled'}
            return original_generate(request, cancelled, callback)
        qwen.generate.side_effect = generate
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        self.assertEqual(metadata['finish_reason'], 'cancelled')
        self.assertFalse(any(event['type'] == 'chunk' for event in self.events))

    def test_previous_assistant_welcome_is_not_replayed_as_the_new_answer(self):
        previous = 'Welcome to our little cafe! What would you like to order today?'
        response = 'Would you prefer a hot drink or something cold?'
        self.request['history'] = [{'role': 'user', 'content': 'I need a break.'},
                                   {'role': 'assistant', 'content': previous}]
        qwen = self.qwen_with_outputs([(previous, 'stop'), (response, 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk')
        self.assertEqual(emitted, response)
        self.assertEqual(metadata['reply_source'], 'qwen_retry')
        self.assertEqual(self.events[-1]['speech']['spoken_response'], response)

    def test_provided_order_cannot_be_followed_by_another_item_selection_question(self):
        order = 'I would like to order one coffee and one milk tea please'
        self.original = self.corrected = order
        self.result.update(transcript=order, reference_text=order)
        self.request['topic_id'] = 'food'
        response = 'Would you like the drinks hot or iced?'
        for bad in ('Welcome! What would you like to order, and why is it your favorite?',
                    'What can I get for you?'):
            with self.subTest(bad=bad):
                self.events.clear()
                qwen = self.qwen_with_outputs([(bad, 'stop'), (response, 'stop'), (self.correction_json(), 'stop')])
                metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
                emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk').strip()
                self.assertEqual(emitted, response)
                self.assertEqual(self.events[-1]['speech']['spoken_response'], response)
                self.assertEqual(metadata['reply_generation_attempts'], 2)
                self.assertTrue(metadata['blocked_questions'] or metadata['omitted_echoes'])
                request = qwen.generate.call_args_list[0].args[0]
                self.assertEqual(request['conversation_state']['next_action'], 'ASK_TEMPERATURE')
                self.assertEqual(request['messages'][-1]['content'], order)
                self.assertEqual([message['role'] for message in request['messages']], ['system', 'user'])

    def test_valid_new_detail_question_is_kept_without_retrying_a_blocked_extra_question(self):
        order = 'I would like to order one coffee and one milk tea please'
        self.original = self.corrected = order
        self.result.update(transcript=order, reference_text=order)
        self.request['topic_id'] = 'food'
        good = 'Would you like the drinks hot or iced?'
        qwen = self.qwen_with_outputs([(good + ' What can I get for you?', 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk').strip()
        self.assertEqual(emitted, good)
        self.assertEqual(metadata['reply_generation_attempts'], 1)
        self.assertEqual(qwen.generate.call_count, 2)

    def test_invented_item_limit_is_not_spoken_before_a_valid_size_question(self):
        order = 'I would like to order one coffee and one milk tea please'
        self.original = self.corrected = order
        self.result.update(transcript=order, reference_text=order)
        self.request['topic_id'] = 'food'
        question = "Could you please specify the size of the coffee and milk tea you'd like?"
        qwen = self.qwen_with_outputs([("I'm sorry, but I can only provide one item at a time. " + question, 'stop'), (self.correction_json(), 'stop')])
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        emitted = ''.join(event['text'] for event in self.events if event['type'] == 'chunk').strip()
        self.assertEqual(emitted, question)
        self.assertEqual(self.events[-1]['speech']['spoken_response'].strip(), question)
        self.assertTrue(metadata['blocked_questions'])
        self.assertEqual(metadata['reply_generation_attempts'], 1)

    def test_invalid_or_truncated_correction_is_withheld_but_conversation_continues(self):
        for raw, reason in (("not json", "stop"), (self.correction_json(), "length")):
            with self.subTest(reason=reason):
                self.events.clear()
                qwen = self.qwen_with_outputs([("How was your visit?", "stop"), (raw, reason)])
                metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
                final = self.events[-1]["speech"]
                self.assertEqual(final["correction"]["status"], "unavailable")
                self.assertNotIn("corrected", final["correction"])
                self.assertEqual(final["transcript"], self.original)
                self.assertEqual(metadata["correction_status"], "unavailable")
                self.assertEqual(qwen.generate.call_count, 2)

    def test_known_reading_and_repeat_targets_skip_grammar_repair(self):
        for source in ("reading_prompt", "corrected_sentence"):
            with self.subTest(source=source):
                self.events.clear()
                self.engine.analyze.reset_mock()
                self.result["reference_text"] = self.corrected
                qwen = self.qwen_with_outputs([("What did you like about the park?", "stop")])
                request = {**self.request, "reference_text": self.corrected, "reference_source": source}
                metadata = self.engine.run(request, self.cancelled, self.emit, qwen)
                self.engine.analyze.assert_called_once_with(b"abc", self.corrected)
                final = self.events[-1]["speech"]
                self.assertEqual(final["reference_source"], source)
                self.assertEqual(final["correction"]["status"], "not_needed")
                self.assertEqual(metadata["correction_status"], "not_needed")
                self.assertEqual(qwen.generate.call_count, 1)
                self.assertNotIn("correcting", [event.get("stage") for event in self.events])

    def test_cancellation_during_correction_keeps_published_reply_and_withholds_repair(self):
        qwen = Mock()

        def cancel(request, cancelled, callback):
            if request.get('generation_profile') == 'conversation':
                callback({'type': 'chunk', 'text': 'How was your visit?'})
                return {'finish_reason': 'stop'}
            callback({"type": "chunk", "text": self.correction_json()})
            cancelled.set()
            return {"finish_reason": "stop"}

        qwen.generate.side_effect = cancel
        metadata = self.engine.run(self.request, self.cancelled, self.emit, qwen)
        self.assertEqual(metadata["finish_reason"], "cancelled")
        self.assertEqual(qwen.generate.call_count, 2)
        self.assertEqual(''.join(event['text'] for event in self.events if event['type'] == 'chunk'), 'How was your visit?')
        published = self.events[-2]['speech']
        self.assertTrue(published['reply_ready'])
        self.assertEqual(published['correction']['status'], 'pending')
        self.assertFalse(any(event["type"] == "speech" and event["speech"]["correction"]["status"] == "ready"
                             for event in self.events))


class CorrectedSentenceRepeatTests(unittest.TestCase):
    # Reuse the real service fixture without importing its TestCase class into
    # this module, which would run the original class again during discovery.
    setUp = speech_fixture.SpeechChatTests.setUp
    payload = speech_fixture.SpeechChatTests.payload

    def correction_source(self, *, correction=None, topic_id="daily-life", request_id="a" * 32):
        correction = correction if correction is not None else {
            "status": "ready", "original": "Yesterday I go park with my friend",
            "corrected": "Yesterday I go park with my friend.", "mode": "completion_only",
            "explanation": "Keep the spoken words unchanged.", "changed": False,
            "changes": [],
            "model_id": "qwen-base", "source": "qwen",
        }

        def run(request, *, on_stage, on_speech, on_text, cancelled):
            self.calls.append(request)
            on_stage("correcting")
            on_speech({"transcript": "Yesterday I go park with my friend", "correction": correction})
            on_text("What did you enjoy about your visit?")
            return {"input": "audio", "topic_id": request.get("topic_id")}

        self.service.speech_generator = run
        source = self.service.submit_speech(self.payload(topic_id=topic_id, request_id=request_id))
        return speech_fixture.finished(self.service, source["id"])

    def test_repeat_uses_server_correction_as_reference_without_overwriting_transcript(self):
        source = self.correction_source()
        repeated = self.service.submit_speech(self.payload(request_id="b" * 32, topic_id="daily-life",
                                                            correction_job_id=source["id"]))
        final = speech_fixture.finished(self.service, repeated["id"])
        self.assertEqual(final["status"], "ready")
        self.assertEqual(self.calls[1]["reference_text"], source["speech"]["correction"]["corrected"])
        self.assertEqual(self.calls[1]["reference_source"], "corrected_sentence")
        self.assertEqual(source["speech"]["transcript"], "Yesterday I go park with my friend")
        self.assertEqual(self.service.poll(source["id"])["speech"]["transcript"], source["speech"]["transcript"])

    def test_unknown_or_invalid_correction_job_ids_are_rejected(self):
        for identifier in ("f" * 32, "../.env", "", 123, ["f" * 32]):
            with self.subTest(identifier=identifier), self.assertRaises(ChatError):
                self.service.submit_speech(self.payload(correction_job_id=identifier))
        self.assertEqual(self.calls, [])

    def test_legacy_grammar_suggestion_cannot_be_used_as_a_new_completion_target(self):
        source = self.correction_source(correction={'status': 'ready', 'corrected': 'Yesterday I went to the park.'})
        with self.assertRaises(ChatError):
            self.service.submit_speech(self.payload(request_id='b' * 32, topic_id='daily-life', correction_job_id=source['id']))
        self.assertEqual(len(self.calls), 1)

    def test_repeat_cannot_change_topic_or_supply_a_conflicting_reading_target(self):
        source = self.correction_source()
        for extra in ({"topic_id": "travel"}, {"topic_id": "daily-life", "practice_id": "idea"}):
            with self.subTest(extra=extra), self.assertRaises(ChatError):
                self.service.submit_speech(self.payload(request_id="b" * 32, correction_job_id=source["id"], **extra))
        self.assertEqual(len(self.calls), 1)

    def test_repeat_cannot_switch_coaching_models(self):
        source = self.correction_source()
        run = self.root / "runs/cascade-l2arctic-plus-seed42/Q"
        adapter = run / "adapter"; adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        weights = b"fixture adapter"
        (adapter / "adapter_model.safetensors").write_bytes(weights)
        (run / "training_provenance.json").write_text(json.dumps({
            "status": "completed", "purpose": "scientific_training",
            "adapter_sha256": hashlib.sha256(weights).hexdigest(),
        }))
        self.assertTrue(self.service.config()["models"][1]["ready"])
        with self.assertRaises(ChatError):
            self.service.submit_speech(self.payload(request_id="b" * 32, model_id="qwen-trained",
                                                    topic_id="daily-life", correction_job_id=source["id"]))
        self.assertEqual(len(self.calls), 1)

    def test_pending_unavailable_or_malformed_correction_cannot_be_a_repeat_target(self):
        for index, correction in enumerate(({"status": "pending"}, {"status": "unavailable"},
                                             {"status": "ready", "corrected": ""},
                                             {"status": "ready", "corrected": ["arbitrary"]},
                                             {"status": "ready", "corrected": "x" * 1001})):
            with self.subTest(correction=correction):
                source = self.correction_source(correction=correction, request_id=(str(index + 1) * 32))
                with self.assertRaises(ChatError):
                    self.service.submit_speech(self.payload(request_id=(format(index + 6, "x") * 32), topic_id="daily-life",
                                                            correction_job_id=source["id"]))
        self.assertEqual(len(self.calls), 5)

    def test_source_must_finish_before_its_correction_is_available_for_repeat(self):
        started, release = threading.Event(), threading.Event()

        def waiting(request, *, on_stage, on_speech, on_text, cancelled):
            on_stage("correcting")
            on_speech({"transcript": "I like coffee", "correction": {
                "status": "ready", "corrected": "I like coffee.", "original": "I like coffee"}})
            started.set(); release.wait(2)
            on_text("Why do you enjoy it?")
            return {"input": "audio"}

        self.service.speech_generator = waiting
        source = self.service.submit_speech(self.payload())
        try:
            self.assertTrue(started.wait(1))
            self.assertEqual(self.service.poll(source["id"])["status"], "correcting")
            with self.assertRaises(ChatError):
                self.service.submit_speech(self.payload(request_id="b" * 32, correction_job_id=source["id"]))
        finally:
            release.set()
            speech_fixture.finished(self.service, source["id"])


if __name__ == "__main__":
    unittest.main()
