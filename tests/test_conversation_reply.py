"""Full-utterance restatements must not reach live text or speech callbacks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conversation_reply import ReplyWithoutEcho, REFERENCE_APOLOGY, MODEL_APOLOGIES


class ConversationReplyTests(unittest.TestCase):
    original = 'Yesterday I go to the park with my friend'

    def filtered(self, reply, *, candidates=None, chunks=None, omit_phrases=(), embedded_echoes=False):
        emitted = []
        guard = ReplyWithoutEcho(candidates or [self.original], emitted.append, omit_phrases=omit_phrases,
                                 embedded_echoes=embedded_echoes)
        for chunk in chunks if chunks is not None else [reply]:
            guard.push(chunk)
        result = guard.finish()
        self.assertEqual(guard.raw_text, reply)
        self.assertEqual(result, ''.join(emitted))
        return result, guard

    def test_full_sentence_echo_removed_before_callback(self):
        reply = self.original + '. What did you do there?'
        value, guard = self.filtered(reply)
        self.assertEqual(value, 'What did you do there?')
        self.assertEqual(len(guard.removed), 1)

    def test_every_chunk_boundary_preserves_same_visible_reply(self):
        reply = 'You said: “' + self.original + '.” How was your walk?'
        for index in range(len(reply) + 1):
            with self.subTest(index=index):
                value, guard = self.filtered(reply, chunks=[reply[:index], reply[index:]])
                self.assertEqual(value, 'How was your walk?')
                self.assertEqual(len(guard.removed), 1)

    def test_character_chunks_never_emit_the_echo(self):
        reply = 'Your sentence is: "' + self.original + '." What happened next?'
        emitted = []
        guard = ReplyWithoutEcho([self.original], emitted.append)
        prefix = reply.index('What')
        for chunk in reply[:prefix]:
            guard.push(chunk)
            self.assertEqual(emitted, [])
        for chunk in reply[prefix:]:
            guard.push(chunk)
        self.assertEqual(guard.finish(), 'What happened next?')

    def test_word_chunks_and_multiple_leading_echoes(self):
        reply = self.original + '! ' + self.original + '? Who went with you?'
        chunks = [word + ' ' for word in reply.split(' ')[:-1]] + [reply.split(' ')[-1]]
        value, guard = self.filtered(reply, chunks=chunks)
        self.assertEqual(value, 'Who went with you?')
        self.assertEqual(len(guard.removed), 2)

    def test_accepted_completion_is_a_separate_candidate(self):
        completion = self.original + ' yesterday'
        value, _ = self.filtered(completion + '. How did it go?', candidates=[self.original, completion])
        self.assertEqual(value, 'How did it go?')

    def test_short_context_prefix_is_buffered_until_new_content_is_clear(self):
        emitted = []
        guard = ReplyWithoutEcho(['Japan', 'I like it'], emitted.append)
        guard.push('Japan')
        self.assertEqual(emitted, [])
        guard.push(' sounds exciting!')
        self.assertEqual(guard.finish(), 'Japan sounds exciting!')
        self.assertEqual(guard.removed, [])

    def test_shared_prefix_continuing_into_substantive_sentence_is_preserved(self):
        for continuation in (' too. That could be fun.', ', and I saw a concert.', ' because I enjoy walking.'):
            reply = self.original + continuation
            value, guard = self.filtered(reply, chunks=list(reply))
            self.assertEqual(value, reply)
            self.assertEqual(guard.removed, [])

    def test_partial_word_cannot_be_mistaken_for_a_matching_word(self):
        original = 'I like coffee a lot'
        reply = 'I like coffee a lottery could never replace. Why do you enjoy it?'
        value, guard = self.filtered(reply, candidates=[original], chunks=list(reply))
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_ordinary_reply_is_forwarded_as_soon_as_prefix_differs(self):
        emitted = []
        guard = ReplyWithoutEcho([self.original], emitted.append)
        guard.push('That')
        self.assertEqual(emitted, ['That'])
        guard.push(' sounds fun!')
        self.assertEqual(emitted, ['That', ' sounds fun!'])

    def test_ordinary_you_statement_is_forwarded_once_marker_is_impossible(self):
        reply = 'You sound excited about the walk. What stood out?'
        value, guard = self.filtered(reply, chunks=list(reply))
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_marker_alone_does_not_trigger_rewriting(self):
        reply = 'You said: it was a rainy day. What happened?'
        value, guard = self.filtered(reply, chunks=list(reply))
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_case_apostrophes_and_whitespace_are_not_meaningful_changes(self):
        original = "I don't like cold coffee"
        reply = '  YOU   SAID : “I DON’T like cold coffee!”\nWhat do you prefer?'
        value, _ = self.filtered(reply, candidates=[original], chunks=list(reply))
        self.assertEqual(value, 'What do you prefer?')

    def test_newline_is_a_complete_echo_boundary(self):
        value, _ = self.filtered(self.original + '\nHow was the weather?')
        self.assertEqual(value, 'How was the weather?')

    def test_truncated_candidate_is_preserved_at_finish(self):
        for reply in ('Yesterday I go', 'You said: "Yesterday I go'):
            value, guard = self.filtered(reply, chunks=list(reply))
            self.assertEqual(value, reply)
            self.assertEqual(guard.removed, [])

    def test_end_of_stream_completes_a_bare_full_echo(self):
        for reply in (self.original, 'You said: "' + self.original + '"',
                      'Your sentence is: ' + self.original + ' ', '“' + self.original + '”'):
            with self.subTest(reply=reply):
                value, guard = self.filtered(reply, chunks=list(reply))
                self.assertEqual(value, '')
                self.assertEqual(len(guard.removed), 1)

    def test_echo_after_ordinary_reply_is_removed(self):
        reply = 'That is interesting. ' + self.original + '. Tell me more.'
        value, guard = self.filtered(reply, chunks=list(reply))
        self.assertEqual(value, 'That is interesting. Tell me more.')
        self.assertEqual(len(guard.removed), 1)

    def assert_chunk_invariant(self, reply, expected, candidates, *, removed=1):
        for index in range(len(reply) + 1):
            with self.subTest(reply=reply, boundary=index):
                result, guard = self.filtered(reply, candidates=candidates, chunks=[reply[:index], reply[index:]])
                self.assertEqual(result, expected)
                self.assertEqual(len(guard.removed), removed)
        result, guard = self.filtered(reply, candidates=candidates, chunks=list(reply))
        self.assertEqual(result, expected)
        self.assertEqual(len(guard.removed), removed)

    def test_complete_short_echoes_are_removed_with_or_without_wrappers(self):
        for source in ('Japan', 'My friend', 'I like pizza'):
            for wrapper in ('', 'So, ', 'Okay, ', 'I see, ', 'You said: ', 'You mentioned: ', 'Okay, so, you said: “'):
                with self.subTest(source=source, wrapper=wrapper):
                    reply = wrapper + source + '. What do you enjoy about it?'
                    self.assert_chunk_invariant(reply, 'What do you enjoy about it?', [source])

    def test_ordinary_intro_and_later_echo_never_publish_echo_words(self):
        emitted = []
        guard = ReplyWithoutEcho([self.original], emitted.append)
        for chunk in 'That is interesting. ':
            guard.push(chunk)
        self.assertEqual(guard.text.rstrip(), 'That is interesting.')
        for chunk in 'Okay, you said: “' + self.original + '.” ':
            guard.push(chunk)
            self.assertNotIn(self.original, ''.join(emitted))
        for chunk in 'What happened next?':
            guard.push(chunk)
        self.assertEqual(guard.finish(), 'That is interesting. What happened next?')
        self.assertEqual(len(guard.removed), 1)

    def test_middle_and_final_echo_removal_is_chunk_invariant(self):
        self.assert_chunk_invariant('That sounds nice. Japan. What do you enjoy there?',
            'That sounds nice. What do you enjoy there?', ['Japan'])
        self.assert_chunk_invariant('Tell me more. I see, I like pizza.', 'Tell me more.', ['I like pizza'])
        self.assert_chunk_invariant('Nice choice. You said: Japan. Okay, Japan! Which city?',
            'Nice choice. Which city?', ['Japan'], removed=2)

    def test_short_bare_questions_are_clarifications_but_labelled_questions_are_echoes(self):
        for source in ('Japan', 'My friend', 'I like pizza'):
            for wrapper in ('', 'So, ', 'Okay, '):
                reply = wrapper + source + '? Could you explain?'
                self.assert_chunk_invariant(reply, reply, [source], removed=0)
            self.assert_chunk_invariant('You said: ' + source + '? Tell me more.', 'Tell me more.', [source])

    def test_source_context_with_additional_meaningful_content_is_preserved(self):
        for reply in ('Japan sounds exciting. Which city?', 'So, Japan sounds exciting!',
                      'Japan, especially Kyoto, is beautiful.', 'I like pizza too. Which toppings?',
                      'I see, I like pizza because it is simple.'):
            self.assert_chunk_invariant(reply, reply, ['Japan', 'I like pizza'], removed=0)

    def test_small_social_greetings_stay_unless_explicitly_labelled_as_echoes(self):
        for greeting in ('Hi', 'Hello', 'Hey', 'Thanks', 'Thank you'):
            self.assert_chunk_invariant(greeting + '. Nice to meet you.', greeting + '. Nice to meet you.', [greeting], removed=0)
            self.assert_chunk_invariant('You said: ' + greeting + '. Nice to meet you.', 'Nice to meet you.', [greeting])

    def test_exact_person_shift_restatements_are_removed_without_changing_negation(self):
        for source, echo in (
                ('I like pizza', 'You like pizza'),
                ('I do not like pizza', 'You do not like pizza'),
                ("I don't like pizza", "You don't like pizza"),
                ('My friend helps me', 'Your friend helps you'),
                ('I am tired', 'You are tired'),
                ("I'm from Japan", "You're from Japan"),
                ("I've visited Japan", "You've visited Japan"),
                ("I'll visit Japan", "You'll visit Japan"),
                ("I'd like coffee", "You'd like coffee")):
            self.assert_chunk_invariant('Okay, ' + echo + '. What would you choose next?', 'What would you choose next?', [source])

    def test_negation_changed_content_and_new_continuations_are_never_guessed_as_echoes(self):
        for source, reply in (
                ('I do not like pizza', 'You like pizza. Really?'),
                ('I like pizza', 'You do not like pizza. Is that right?'),
                ('I like pizza', 'You like pasta. Is that right?'),
                ('I like pizza', 'You like pizza because it is convenient. Me too.'),
                ('I like pizza', 'You sound like a pizza enthusiast.'),
                ('I visited Japan', 'You travelled to Japan. How was it?')):
            self.assert_chunk_invariant(reply, reply, [source], removed=0)

    def test_quoted_marker_without_colon_is_removed_only_for_actual_matching_content(self):
        self.assert_chunk_invariant('You said “I like pizza.” What toppings?', 'What toppings?', ['I like pizza'])
        reply = 'You said “I prefer pasta.” What sauce?'
        self.assert_chunk_invariant(reply, reply, ['I like pizza'], removed=0)

    def test_matching_words_in_separate_sentences_are_not_combined(self):
        reply = 'I like. Pizza is delicious.'
        self.assert_chunk_invariant(reply, reply, ['I like pizza'], removed=0)

    def test_bare_short_echo_at_end_is_removed_but_truncated_longer_source_stays(self):
        self.assert_chunk_invariant('Japan', '', ['Japan'])
        self.assert_chunk_invariant('I like pizza', '', ['I like pizza'])
        self.assert_chunk_invariant('I like', 'I like', ['I like pizza'], removed=0)

    def test_no_candidates_means_exact_immediate_passthrough(self):
        emitted = []
        guard = ReplyWithoutEcho([None, '', '...'], emitted.append)
        for chunk in ('Hello.', ' ', 'Japan?'):
            guard.push(chunk)
        self.assertEqual(emitted, ['Hello.', ' ', 'Japan?'])
        self.assertEqual(guard.finish(), 'Hello. Japan?')

    def test_removed_audit_and_unicode_spacing_are_chunk_invariant(self):
        reply = 'Nice choice.\u00a0You said: “Japan.”\u2003\nWhat do you like there?'
        expected, reference = self.filtered(reply, candidates=['Japan'])
        self.assertEqual(expected, 'Nice choice. What do you like there?')
        for index in range(len(reply) + 1):
            with self.subTest(index=index):
                actual, guard = self.filtered(reply, candidates=['Japan'], chunks=[reply[:index], reply[index:]])
                self.assertEqual(actual, expected)
                self.assertEqual(guard.removed, reference.removed)
        actual, guard = self.filtered(reply, candidates=['Japan'], chunks=list(reply))
        self.assertEqual(actual, expected)
        self.assertEqual(guard.removed, reference.removed)

    def test_actual_compound_question_echo_ignores_only_favorite_spelling_variant(self):
        previous = 'What would you like to order, and why is it your favourite?'
        repeated = previous.replace('favourite', 'favorite')
        for candidate, reply in ((previous, repeated), (repeated, previous)):
            self.assert_chunk_invariant(reply + ' Is it usually served hot?', 'Is it usually served hot?', [candidate])
            emitted = []
            guard = ReplyWithoutEcho([candidate], emitted.append)
            guard.push('Great!')
            for character in ' Okay, ' + reply:
                guard.push(character)
                self.assertEqual(''.join(emitted), 'Great!')
            self.assertEqual(guard.finish(), 'Great!')
            self.assertEqual(len(guard.removed), 1)

    def test_spelling_alias_does_not_remove_changed_question_or_new_content(self):
        previous = 'What would you like to order, and why is it your favourite?'
        for reply in ('What would you like to order, and why is it your favorite food today?',
                      'Which tea is your favorite?', 'What is your favorite flavor?'):
            self.assert_chunk_invariant(reply, reply, [previous], removed=0)

    def test_actual_topic_shift_sentence_contraction_never_repeats_before_callback(self):
        source = 'Also, I am visiting Da Nang next week.'
        contracted = "Also, I'm visiting Da Nang next week."
        for candidate, echo in ((source, contracted), (contracted, source)):
            self.assert_chunk_invariant(echo + ' Do you already have a plan?', 'Do you already have a plan?', [candidate])
            emitted = []
            guard = ReplyWithoutEcho([candidate], emitted.append)
            for character in echo:
                guard.push(character)
                self.assertEqual(emitted, [])
            self.assertEqual(guard.finish(), '')

    def test_bounded_contractions_keep_content_and_negation_unchanged(self):
        for source, echo in (("You are visiting Japan", "You're visiting Japan"),
                             ("I have visited Japan", "I've visited Japan"),
                             ("I will visit Japan", "I'll visit Japan"),
                             ("I do not like coffee", "I don't like coffee")):
            self.assert_chunk_invariant(echo + '. What happens next?', 'What happens next?', [source])
        for source, reply in (("I am visiting Da Nang", "I'm visiting Hanoi."),
                              ("I do not like coffee", 'I like coffee.'),
                              ("I like coffee", "I don't like coffee.")):
            self.assert_chunk_invariant(reply, reply, [source], removed=0)

    def test_only_echo_can_result_in_empty_visible_reply(self):
        value, guard = self.filtered('"' + self.original + '."', chunks=list('"' + self.original + '."'))
        self.assertEqual(value, '')
        self.assertEqual(len(guard.removed), 1)

    def test_finish_is_idempotent_and_rejects_later_chunks(self):
        guard = ReplyWithoutEcho([self.original], lambda text: None)
        guard.push('Yesterday')
        self.assertEqual(guard.finish(), 'Yesterday')
        self.assertEqual(guard.finish(), 'Yesterday')
        with self.assertRaises(RuntimeError):
            guard.push(' extra')

    def test_supplied_apology_and_wrapped_echo_never_reach_model_contribution_callback(self):
        target = 'It was difficult to get here'
        for prefix in (REFERENCE_APOLOGY + ' ', 'Sorry for the misunderstanding, ',
                       'Sorry for the misunderstanding — ', "I'm sorry for the misunderstanding. "):
            for wrapper in ('', 'You said: ', 'You said ', 'You meant that ', 'Your corrected sentence is: '):
                reply = prefix + wrapper + target + '. Was the traffic heavy?'
                for chunks in ([reply], list(reply), [reply[:len(prefix)+3], reply[len(prefix)+3:]]):
                    with self.subTest(prefix=prefix, wrapper=wrapper, chunks=len(chunks)):
                        value, guard = self.filtered(reply, candidates=[target], chunks=chunks,
                                                     omit_phrases=MODEL_APOLOGIES)
                        self.assertEqual(value, 'Was the traffic heavy?')
                        self.assertTrue(guard.removed)

    def test_every_split_of_apology_embedded_echo_preserves_new_question(self):
        target = 'I would like one coffee and one milk tea please'
        reply = 'Sorry for the misunderstanding, you said ' + target + '. Hot or iced?'
        for split in range(len(reply)+1):
            with self.subTest(split=split):
                value, _ = self.filtered(reply, candidates=[target], chunks=[reply[:split], reply[split:]],
                                         omit_phrases=MODEL_APOLOGIES)
                self.assertEqual(value, 'Hot or iced?')

    def test_apology_plus_echo_alone_is_empty_and_can_trigger_retry(self):
        for reply in (REFERENCE_APOLOGY, REFERENCE_APOLOGY + ' ' + self.original + '.',
                      'Sorry for the misunderstanding, you said ' + self.original + '.'):
            value, guard = self.filtered(reply, chunks=list(reply), omit_phrases=MODEL_APOLOGIES)
            self.assertEqual(value, '')
            self.assertTrue(guard.removed)

    def test_model_apology_is_removed_after_useful_content_too(self):
        reply = 'Was the traffic heavy? ' + REFERENCE_APOLOGY + ' You said ' + self.original + '.'
        value, _ = self.filtered(reply, chunks=list(reply), omit_phrases=MODEL_APOLOGIES)
        self.assertEqual(value.rstrip(), 'Was the traffic heavy?')

    def test_ordinary_apology_and_new_content_are_preserved_without_supplied_phrase_policy(self):
        reply = REFERENCE_APOLOGY + ' Was the traffic heavy?'
        value, guard = self.filtered(reply, chunks=list(reply))
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_apology_word_prefix_with_new_meaning_is_not_suppressed(self):
        reply = 'Sorry for the misunderstanding that delayed your journey. Did you take a taxi?'
        value, guard = self.filtered(reply, chunks=list(reply), omit_phrases=MODEL_APOLOGIES)
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_strict_regeneration_removes_even_short_question_or_social_copies(self):
        for source in ('Japan', 'I like pizza', 'Hello'):
            reply = source + '? What interests you most?'
            for chunks in ([reply], list(reply)):
                emitted = []
                guard = ReplyWithoutEcho([source], emitted.append, preserve_short_exchanges=False)
                for chunk in chunks: guard.push(chunk)
                self.assertEqual(guard.finish(), 'What interests you most?')
                self.assertEqual(''.join(emitted), guard.text)

    def test_embedded_complete_statement_after_unfamiliar_wrapper_never_leaks(self):
        source = 'I would like one coffee please'
        reply = 'I now understand that you said "' + source + '". Is it hot or iced?'
        for split in range(len(reply)+1):
            value, guard = self.filtered(reply, candidates=[source], chunks=[reply[:split], reply[split:]],
                                         embedded_echoes=True)
            self.assertEqual(value, 'Is it hot or iced?')
            self.assertEqual(len(guard.removed), 1)
        emitted = []
        guard = ReplyWithoutEcho([source], emitted.append, embedded_echoes=True)
        start = reply.index('Is it')
        for character in reply[:start]:
            guard.push(character)
            self.assertEqual(emitted, [])
        for character in reply[start:]: guard.push(character)
        self.assertEqual(guard.finish(), 'Is it hot or iced?')

    def test_embedded_copy_followed_by_new_question_in_same_sentence_preserves_question(self):
        source = 'I would like one coffee please'
        for wrapper in ('You just said, ', 'My understanding of your words is ', 'Let me check: '):
            reply = wrapper + '"' + source + '", and what size would you prefer?'
            for split in range(len(reply)+1):
                value, guard = self.filtered(reply, candidates=[source], chunks=[reply[:split], reply[split:]],
                                             embedded_echoes=True)
                self.assertEqual(value, 'what size would you prefer?')
                self.assertTrue(guard.removed)

    def test_strict_embedded_copy_after_new_information_is_removed_without_replaying_it(self):
        reply = 'Did you take the bus? I understand your message as "' + self.original + '".'
        value, _ = self.filtered(reply, chunks=list(reply), embedded_echoes=True)
        self.assertEqual(value, 'Did you take the bus?')

    def test_unsafe_mixed_echo_sentence_is_discarded_instead_of_becoming_a_fake_answer(self):
        reply = 'You just said ' + self.original + ', and that sounds nice.'
        value, guard = self.filtered(reply, chunks=list(reply), embedded_echoes=True)
        self.assertEqual(value, '')
        self.assertTrue(guard.removed)

    def test_strict_embedded_guard_preserves_new_words_negation_and_semantic_paraphrases(self):
        source = 'I would like one coffee please'
        for reply in ('Would you prefer a large cup?', 'I would not like one coffee please.',
                      'You wanted a cup of coffee. Would you like milk?',
                      'Coffee is a popular choice. Would you prefer it iced?'):
            value, guard = self.filtered(reply, candidates=[source], chunks=list(reply), embedded_echoes=True)
            self.assertEqual(value, reply)
            self.assertEqual(guard.removed, [])

    def test_strict_long_copy_detection_keeps_short_topic_reuse_with_new_information(self):
        reply = 'Japan has many interesting cities. Which region interests you?'
        value, guard = self.filtered(reply, candidates=['Japan'], chunks=list(reply), embedded_echoes=True)
        self.assertEqual(value, reply)
        self.assertEqual(guard.removed, [])

    def test_strict_copy_matches_person_and_contraction_variants_inside_unknown_wrapper(self):
        source = 'I am visiting Da Nang next week'
        reply = 'As I heard it, "you\'re visiting Da Nang next week". What do you plan to visit?'
        value, _ = self.filtered(reply, candidates=[source], chunks=list(reply), embedded_echoes=True)
        self.assertEqual(value, 'What do you plan to visit?')


if __name__ == '__main__':
    unittest.main()
