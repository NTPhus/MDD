"""Grounded cafe replies must be checked before any text/voice callback."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conversation_policy import GroundedReply
from scripts.conversation_state import cafe_state


class GroundedReplyTests(unittest.TestCase):
    order = 'I would like to order one coffee and one milk tea please'
    temperature_question = 'Would you like the drinks hot or iced?'

    def order_state(self):
        state = cafe_state({'topic_id': 'food', 'history': []}, self.order)
        self.assertIsNotNone(state, 'The exact reported order must enter cafe context.')
        return state

    def iced_state(self):
        state = cafe_state({'topic_id': 'food', 'history': [
            {'role': 'user', 'content': self.order},
            {'role': 'assistant', 'content': self.temperature_question},
        ]}, 'Both drinks are iced, please.')
        self.assertIsNotNone(state)
        return state

    def run_stream(self, state, chunks, *, remain_silent=False):
        emitted = []
        policy = GroundedReply(state, emitted.append)
        for chunk in chunks:
            policy.push(chunk)
            if remain_silent:
                self.assertEqual(emitted, [], 'Rejected or incomplete policy decisions must never reach a live callback.')
        policy.finish()
        if remain_silent:
            self.assertEqual(emitted, [])
        return ''.join(emitted), policy

    def boundaries(self, text):
        for boundary in range(len(text) + 1):
            yield [text[:boundary], text[boundary:]]
        yield list(text)

    def test_exact_order_does_not_restart_ordering_at_any_chunk_boundary(self):
        reply = 'Welcome! What would you like to order, and why is it your favorite?'
        state = self.order_state()
        for chunks in self.boundaries(reply):
            with self.subTest(chunks=chunks):
                output, policy = self.run_stream(state, chunks, remain_silent=True)
                self.assertEqual(output, '')
                self.assertTrue(policy.blocked)
                self.assertFalse(policy.accepted_question)
                self.assertTrue(policy.needs_retry)

    def test_next_missing_temperature_question_is_accepted_without_rewriting(self):
        state = self.order_state()
        for chunks in self.boundaries(self.temperature_question):
            with self.subTest(chunks=chunks):
                output, policy = self.run_stream(state, chunks)
                self.assertEqual(output, self.temperature_question)
                self.assertTrue(policy.accepted_question)
                self.assertFalse(policy.needs_retry)
                self.assertEqual(policy.blocked, [])

    def test_known_iced_answer_is_not_asked_again(self):
        reply = 'Would you like the drinks hot or iced?'
        state = self.iced_state()
        for chunks in self.boundaries(reply):
            with self.subTest(chunks=chunks):
                _, policy = self.run_stream(state, chunks, remain_silent=True)
                self.assertTrue(policy.blocked)
                self.assertFalse(policy.accepted_question)
                self.assertTrue(policy.needs_retry)

    def test_size_question_follows_a_known_iced_answer(self):
        reply = 'What size would you like for the drinks?'
        state = self.iced_state()
        for chunks in self.boundaries(reply):
            with self.subTest(chunks=chunks):
                output, policy = self.run_stream(state, chunks)
                self.assertEqual(output, reply)
                self.assertTrue(policy.accepted_question)
                self.assertFalse(policy.needs_retry)

    def test_acknowledgement_without_a_needed_question_stays_unpublished(self):
        reply = 'Sure, I can help with that.'
        output, policy = self.run_stream(self.order_state(), list(reply), remain_silent=True)
        self.assertEqual(output, '')
        self.assertFalse(policy.accepted_question)
        self.assertTrue(policy.needs_retry)
        self.assertTrue(any(item['reason'] == 'missing_next_detail' for item in policy.blocked))

    def test_barista_does_not_speak_as_the_customer(self):
        reply = 'Both drinks are iced, please.'
        state = self.order_state()
        for chunks in self.boundaries(reply):
            with self.subTest(chunks=chunks):
                _, policy = self.run_stream(state, chunks, remain_silent=True)
                self.assertTrue(policy.blocked)
                self.assertTrue(policy.needs_retry)
                self.assertFalse(policy.accepted_question)

    def test_safe_acknowledgement_waits_for_a_grounded_question(self):
        emitted = []
        policy = GroundedReply(self.order_state(), emitted.append)
        policy.push('Certainly. ')
        self.assertEqual(emitted, [])
        policy.push(self.temperature_question[:-1])
        self.assertEqual(emitted, [])
        policy.push('?')
        self.assertEqual(''.join(emitted), 'Certainly. ' + self.temperature_question)
        policy.finish()
        self.assertFalse(policy.needs_retry)

    def test_bad_question_in_the_middle_never_leaks_before_valid_continuation(self):
        emitted = []
        policy = GroundedReply(self.order_state(), emitted.append)
        for chunk in 'Certainly. What would you like to order?':
            policy.push(chunk)
            self.assertEqual(emitted, [])
        policy.push(' ' + self.temperature_question)
        policy.finish()
        self.assertEqual(''.join(emitted), 'Certainly. ' + self.temperature_question)
        self.assertTrue(policy.blocked)
        self.assertTrue(policy.accepted_question)
        self.assertFalse(policy.needs_retry)

    def test_non_cafe_chunks_pass_through_immediately_unchanged(self):
        state = cafe_state({'topic_id': 'travel', 'history': []}, 'I would love to visit Japan.')
        self.assertIsNone(state)
        emitted = []
        policy = GroundedReply(state, emitted.append)
        chunks = ['Japan ', 'sounds exciting. ', 'What ', 'would you do there?']
        for index, chunk in enumerate(chunks):
            policy.push(chunk)
            self.assertEqual(emitted, chunks[:index + 1])
        policy.finish()
        self.assertEqual(emitted, chunks)
        self.assertEqual(policy.blocked, [])
        self.assertFalse(policy.needs_retry)

    def test_greeting_only_after_a_real_utterance_never_reaches_ui_or_voice(self):
        for reply in ('Hi!', 'Hello there.', 'Hi! That sounds nice.', 'Okay.'):
            for chunks in self.boundaries(reply):
                with self.subTest(reply=reply, chunks=chunks):
                    emitted = []
                    policy = GroundedReply(None, emitted.append, transcript='I work as a software developer.')
                    for chunk in chunks:
                        policy.push(chunk)
                        self.assertEqual(emitted, [])
                    policy.finish()
                    self.assertEqual(emitted, [])
                    self.assertTrue(policy.needs_retry)
                    self.assertTrue(any(item['reason'] == 'no_conversation_contribution' for item in policy.blocked))

    def test_greeting_waits_for_actual_new_content_instead_of_disappearing(self):
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript='I fix software bugs.')
        policy.push('Hi! ')
        self.assertEqual(emitted, [])
        policy.push('What usually helps you find the cause?')
        policy.finish()
        self.assertEqual(''.join(emitted), 'Hi! What usually helps you find the cause?')
        self.assertFalse(policy.needs_retry)

    def test_substantive_statement_is_valid_without_a_forced_question(self):
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript='I finally fixed the bug.')
        policy.push('That must be a relief after all that effort.')
        policy.finish()
        self.assertEqual(''.join(emitted), 'That must be a relief after all that effort.')
        self.assertFalse(policy.needs_retry)

    def test_social_user_turns_keep_normal_short_replies(self):
        for user, reply in [('Hi', 'Hi!'), ('Thanks, that helps.', "You're welcome!"), ('Goodbye', 'Take care!')]:
            with self.subTest(user=user):
                emitted = []
                policy = GroundedReply(None, emitted.append, transcript=user)
                for letter in reply:
                    policy.push(letter)
                policy.finish()
                self.assertEqual(''.join(emitted), reply)
                self.assertFalse(policy.needs_retry)

    def test_echo_filter_leaving_only_hi_still_requires_a_real_retry(self):
        from scripts.conversation_reply import ReplyWithoutEcho
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript='I had coffee with my sister.')
        stream = ReplyWithoutEcho(['What do you usually enjoy doing in the morning?'], policy.push)
        for letter in 'Hi! What do you usually enjoy doing in the morning?':
            stream.push(letter)
        stream.finish(); policy.finish()
        self.assertEqual(emitted, [])
        self.assertTrue(stream.removed)
        self.assertTrue(policy.needs_retry)

    def test_actual_stock_phrasing_praise_for_content_turn_stays_private_and_requires_retry(self):
        reply = 'That’s a good way to say it.'
        for chunks in self.boundaries(reply):
            emitted = []
            policy = GroundedReply(None, emitted.append,
                                   transcript='Their forces were already moving into the north country')
            for chunk in chunks:
                policy.push(chunk)
                self.assertEqual(emitted, [])
            policy.finish()
            self.assertEqual(emitted, [])
            self.assertTrue(policy.needs_retry)
            self.assertEqual(policy.blocked, [{'text': reply, 'reason': 'off_topic_language_praise'}])

    def test_bounded_stock_language_praise_variants_are_not_substantive_answers(self):
        for reply in ("That's a nice way to phrase it.", 'Your sentence is grammatically correct.',
                      'Your wording sounds natural.', 'Well said.', 'You put that very well.'):
            with self.subTest(reply=reply):
                emitted = []
                policy = GroundedReply(None, emitted.append, transcript='I had trouble getting here.')
                for character in reply: policy.push(character)
                policy.finish()
                self.assertEqual(emitted, [])
                self.assertTrue(policy.needs_retry)

    def test_stock_praise_followed_by_real_contextual_content_can_pass_unchanged(self):
        reply = 'That’s a good way to say it. What happened when they reached the north?'
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript='Their forces were moving north.')
        policy.push('That’s a good way to say it.')
        self.assertEqual(emitted, [])
        for character in ' What happened when they reached the north?': policy.push(character)
        policy.finish()
        self.assertEqual(''.join(emitted), reply)
        self.assertFalse(policy.needs_retry)
        self.assertEqual(policy.blocked, [])

    def test_explicit_language_questions_can_receive_phrasing_praise(self):
        for user in ('Is that a good way to say it?', 'Does my sentence sound natural?',
                     'Please check my grammar.', 'How can I phrase this?', 'Is my pronunciation correct?'):
            with self.subTest(user=user):
                emitted = []
                policy = GroundedReply(None, emitted.append, transcript=user)
                for character in 'That’s a good way to say it.': policy.push(character)
                policy.finish()
                self.assertEqual(''.join(emitted), 'That’s a good way to say it.')
                self.assertFalse(policy.needs_retry)

    def test_normal_reactions_language_subjects_and_social_answers_are_preserved(self):
        for user, reply in (
                ('I took the bus.', 'That is a good way to save money.'),
                ('I teach English.', 'Do you enjoy working with your students?'),
                ('The journey was difficult.', 'That sounds like a tiring journey.'),
                ('Thank you.', 'You are welcome!')):
            with self.subTest(user=user, reply=reply):
                emitted = []
                policy = GroundedReply(None, emitted.append, transcript=user)
                for character in reply: policy.push(character)
                policy.finish()
                self.assertEqual(''.join(emitted), reply)
                self.assertFalse(policy.needs_retry)


class UnresolvedTranscriptReplyTests(unittest.TestCase):
    raw = 'It was the divicote true getesir'

    def check(self, reply):
        emitted = []
        policy = GroundedReply(None, emitted.append, transcript=self.raw, needs_clarification=True)
        for character in reply: policy.push(character)
        policy.finish()
        return ''.join(emitted), policy

    def test_reported_unrelated_reply_is_blocked_before_text_or_voice(self):
        output, policy = self.check('It was a great day! How does it feel for you today?')
        self.assertEqual(output, '')
        self.assertTrue(policy.needs_retry)
        self.assertTrue(all(item['reason'] == 'unresolved_transcript_requires_clarification' for item in policy.blocked))

    def test_clarification_is_kept_without_unrelated_leading_praise(self):
        output, policy = self.check('That sounds nice. Could you repeat the sentence?')
        self.assertEqual(output.strip(), 'Could you repeat the sentence?')
        self.assertFalse(policy.needs_retry)
        self.assertEqual(len(policy.blocked), 1)

    def test_known_word_meaning_question_is_valid_and_unclear_names_are_not_invented(self):
        for reply in ('Could you clarify what "divicote" refers to in this context?',
                      'What did you mean by the last part?', 'Could you type the sentence you said?'):
            with self.subTest(reply=reply):
                output, policy = self.check(reply)
                self.assertEqual(output, reply)
                self.assertFalse(policy.needs_retry)


if __name__ == '__main__':
    unittest.main()
