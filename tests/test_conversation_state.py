"""Cafe facts and reask decisions must follow explicit, current user evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.conversation_state import cafe_state, question_violation, state_note


def request(*messages, **changes):
    return {'topic_id': 'food', 'history': [{'role': role, 'content': text} for role, text in messages], **changes}


class ConversationStateTests(unittest.TestCase):
    def ordered(self, text='I would like a coffee and a milk tea, please.'):
        result = cafe_state(request(), text)
        self.assertIsNotNone(result)
        return result

    def test_explicit_items_are_grounded_and_longest_drink_name_wins(self):
        raw = 'I would like a coffee and a milk tea, please.'
        state = self.ordered(raw)
        self.assertEqual(state['items'], ['coffee', 'milk tea'])
        self.assertEqual(state['facts'], {'items': {'value': ['coffee', 'milk tea'], 'evidence': raw, 'turn': 1}})
        self.assertEqual(state['next_action'], 'ASK_TEMPERATURE')

    def test_direct_polite_order_question_is_an_explicit_choice(self):
        for text in ('Can I have a coffee?', 'Could I have a latte, please?', 'May I have two espressos?'):
            self.assertIsNotNone(cafe_state(request(), text))

    def test_exact_reported_order_and_contracted_to_order_prefix_are_grounded(self):
        for raw in ('I would like to order one coffee and one milk tea please',
                    "I'd like to order one coffee and one milk tea please",
                    'I want to order one coffee and one milk tea please'):
            with self.subTest(raw=raw):
                state = cafe_state(request(), raw)
                self.assertIsNotNone(state)
                self.assertEqual(state['items'], ['coffee', 'milk tea'])
                self.assertEqual(state['facts']['items']['evidence'], raw)
                self.assertEqual(state['next_action'], 'ASK_TEMPERATURE')

    def test_short_selection_needs_an_actual_selection_question(self):
        state = cafe_state(request(('assistant', 'What would you like to order?')), 'Coffee and milk tea, please.')
        self.assertEqual(state['items'], ['coffee', 'milk tea'])
        self.assertIsNone(cafe_state(request(('assistant', 'What did you drink yesterday?')), 'Coffee and milk tea.'))

    def test_ambiguous_negative_hypothetical_or_recommendation_is_not_a_choice(self):
        for text in ('Coffee or milk tea?', 'Should I get coffee?', 'What coffee do you recommend?',
                     'I do not want coffee.', "I don't want coffee.", 'Maybe a coffee.',
                     'If I order coffee, what would you get?', 'I drank coffee yesterday.',
                     'My favourite is coffee.', 'I would like a coffee or milk tea.', 'I want no coffee.'):
            with self.subTest(text=text):
                self.assertIsNone(cafe_state(request(), text))

    def test_known_details_in_initial_order_are_not_inferred_from_other_words(self):
        state = self.ordered('I would like two large iced coffees to go, please.')
        self.assertEqual(state['items'], ['coffee'])
        self.assertEqual(state['facts']['temperature']['value'], 'iced')
        self.assertEqual(state['facts']['size']['value'], 'large')
        self.assertEqual(state['facts']['service']['value'], 'to go')
        self.assertEqual(state['next_action'], 'NEXT_CONTRIBUTION')

    def test_consecutive_short_answers_advance_only_the_new_detail(self):
        first = 'I want a coffee and a milk tea.'
        second = 'Both iced, please.'
        third = 'Medium, please.'
        history = request(('user', first), ('assistant', 'Would you like both drinks hot or iced?'))
        state = cafe_state(history, second)
        self.assertEqual(state['next_action'], 'ASK_SIZE')
        self.assertEqual(state['facts']['temperature'], {'value': 'iced', 'evidence': second, 'turn': 2})
        history['history'] += [{'role': 'user', 'content': second}, {'role': 'assistant', 'content': 'What size would you like?'}]
        state = cafe_state(history, third)
        self.assertEqual(state['next_action'], 'ASK_SERVICE')
        self.assertEqual(state['facts']['size'], {'value': 'medium', 'evidence': third, 'turn': 3})
        history['history'] += [{'role': 'user', 'content': third}, {'role': 'assistant', 'content': 'Is that for here or to go?'}]
        state = cafe_state(history, 'For here, please.')
        self.assertEqual(state['facts']['service']['value'], 'for here')
        self.assertEqual(state['next_action'], 'NEXT_CONTRIBUTION')
        self.assertEqual(state['facts']['items']['evidence'], first)

    def test_unrelated_or_mixed_topic_continuation_deactivates_frame(self):
        history = request(('user', 'I want coffee.'), ('assistant', 'Would you like it hot or iced?'),
                          ('user', 'Iced, please.'), ('assistant', 'What size would you like?'))
        for text in ("Medium, please. Also, I'm visiting Da Nang next week.", 'I am visiting Da Nang next week.',
                     'What do you think about my trip?', 'Never mind, let us discuss movies.', 'Thank you.'):
            with self.subTest(text=text):
                self.assertIsNone(cafe_state(history, text))

    def test_unrecognized_unicode_topic_content_is_not_silently_discarded(self):
        self.assertIsNone(cafe_state(request(), 'I want coffee 旅行'))

    def test_old_order_does_not_reactivate_after_topic_change(self):
        history = request(('user', 'I want coffee.'), ('assistant', 'Hot or iced?'),
                          ('user', 'I am visiting Da Nang.'), ('assistant', 'What will you do there?'))
        self.assertIsNone(cafe_state(history, 'Medium, please.'))

    def test_new_order_or_replacement_does_not_reuse_old_details(self):
        history = request(('user', 'I want a large iced coffee to go.'), ('assistant', 'Anything else?'))
        for text in ('Actually, make it tea instead.', 'I would like a latte, please.', 'Change my order to a mocha.'):
            with self.subTest(text=text):
                state = cafe_state(history, text)
                self.assertEqual(state['next_action'], 'ASK_TEMPERATURE')
                self.assertEqual(set(state['facts']), {'items'})
                self.assertNotIn('coffee', state['items'])

    def test_cancellation_removes_old_frame_and_new_explicit_order_can_start(self):
        history = request(('user', 'I want coffee.'), ('assistant', 'Hot or iced?'))
        for text in ('Cancel my order.', 'Never mind.', 'No thanks.'):
            self.assertIsNone(cafe_state(history, text))
        history['history'] += [{'role': 'user', 'content': 'Cancel my order.'}, {'role': 'assistant', 'content': 'Okay.'}]
        self.assertIsNone(cafe_state(history, 'Medium.'))
        state = cafe_state(history, 'I want tea.')
        self.assertEqual(state['items'], ['tea'])

    def test_short_detail_requires_matching_pending_question(self):
        history = request(('user', 'I want coffee.'), ('assistant', 'Why do you like coffee?'))
        self.assertIsNone(cafe_state(history, 'Medium.'))
        self.assertIsNone(cafe_state(history, 'Iced.'))

    def test_actual_model_size_wording_keeps_medium_answer_in_the_order_frame(self):
        question = "Could you please specify the size of the coffee and milk tea you'd like?"
        history = request(('user', 'I would like to order one coffee and one milk tea please'),
                          ('assistant', 'Would you like both drinks hot or iced?'),
                          ('user', 'Both drinks are iced, please.'), ('assistant', question))
        state = cafe_state(history, 'Medium, please.')
        self.assertIsNotNone(state)
        self.assertEqual(state['facts']['size']['value'], 'medium')
        self.assertEqual(state['next_action'], 'ASK_SERVICE')
        self.assertEqual(question_violation(question, state), 'repeated_size')

    def test_conflicting_or_partial_per_item_details_stay_unknown_without_losing_selection(self):
        for text in ('I want a hot coffee and an iced milk tea.', 'I want a small coffee and a large tea.'):
            state = cafe_state(request(), text)
            self.assertIsNotNone(state)
            self.assertEqual(set(state['facts']), {'items'})
        state = self.ordered('I want a large iced coffee and a milk tea.')
        self.assertEqual(state['items'], ['coffee', 'milk tea'])
        self.assertEqual(set(state['facts']), {'items'})
        state = self.ordered('I want an iced coffee and an iced milk tea.')
        self.assertEqual(state['facts']['temperature']['value'], 'iced')
        state = self.ordered('I want a coffee and a milk tea, both iced please.')
        self.assertEqual(state['facts']['temperature']['value'], 'iced')

    def test_other_topics_reading_and_invalid_requests_have_no_order_frame(self):
        for payload in ({'topic_id': 'travel'}, {'topic_id': None}, {'topic_id': 'food', 'reference_text': 'I want coffee.'},
                        {'topic_id': 'food', 'history': None}, None):
            self.assertIsNone(cafe_state(payload, 'I want coffee.'))

    def test_state_note_is_compact_and_requires_one_new_detail_without_scripted_wording(self):
        state = self.ordered()
        note = state_note(state)
        self.assertIn('You are the barista', note)
        self.assertIn('ASK_TEMPERATURE', note)
        self.assertIn('MUST ask ONE', note)
        self.assertIn('coffee, milk tea', note)
        self.assertLess(len(note.split()), 100)
        self.assertNotIn('Would you like', note)
        self.assertEqual(state_note(None), '')

    def test_repeated_selection_is_rejected_but_new_facets_are_preserved(self):
        state = self.ordered()
        for text in ('What would you like to order?', 'What can I get you to drink?', 'What is your order?',
                     'What would you like to order, and why is it your favorite?'):
            with self.subTest(text=text):
                self.assertEqual(question_violation(text, state), 'repeated_selection')
        for text in ('Hot or iced?', 'What size would you like?', 'Is that for here or to go?',
                     'Anything else?', 'Why do you like coffee?', 'What do you like about milk tea?',
                     'Are both drinks for you?', 'Would you like coffee with milk?', 'How many cups?', 'Sure!'):
            with self.subTest(text=text):
                self.assertIsNone(question_violation(text, state))

    def test_only_already_known_temperature_size_and_service_questions_are_rejected(self):
        state = self.ordered('I want a medium iced coffee to go.')
        for text, reason in (('Hot or iced?', 'repeated_temperature'),
                             ('What size would you like?', 'repeated_size'),
                             ('Is that for here or to go?', 'repeated_service')):
            self.assertEqual(question_violation(text, state), reason)
            self.assertIsNone(question_violation(text, self.ordered()))

    def test_welcome_restart_and_obvious_customer_role_are_blocked(self):
        state = self.ordered()
        self.assertEqual(question_violation('Welcome to our cafe!', state), 'cafe_restart')
        self.assertEqual(question_violation("I'd like a coffee and a milk tea, please.", state), 'customer_order_recap')
        self.assertEqual(question_violation('Both drinks are iced, please.', state), 'customer_order_recap')
        for text in ("You're welcome!", 'I want you to enjoy your visit.', 'Would you like both drinks iced?'):
            self.assertIsNone(question_violation(text, state))

    def test_no_state_never_rewrites_normal_conversation(self):
        self.assertIsNone(question_violation('Welcome!', None))
        self.assertIsNone(question_violation("I'd like coffee.", None))

    def test_actual_known_order_recaps_are_blocked_without_guessing_paraphrases(self):
        state = self.ordered('I want one iced coffee and one iced milk tea.')
        for text in ('I see you want one coffee and one milk tea.', 'I see you want iced drinks.',
                     'You ordered one coffee and one milk tea.', 'Your order is coffee and milk tea.'):
            with self.subTest(text=text):
                self.assertEqual(question_violation(text, state), 'cafe_order_recap')
        for text in ('Your order will take a few minutes.', 'Your order is ready.',
                     'I see you want hot drinks.', 'I see you want a latte.',
                     'I see you want coffee because you are tired.', 'I see your point.'):
            with self.subTest(text=text):
                self.assertIsNone(question_violation(text, state))

    def test_unprovided_business_claims_are_blocked_with_no_general_semantic_guess(self):
        state = self.ordered()
        for text, reason in (
                ('I can only provide one item at a time.', 'unsupported_item_limit'),
                ('We can only serve one drink per customer.', 'unsupported_item_limit'),
                ('The coffee costs five dollars.', 'unsupported_price'),
                ('That is $5.', 'unsupported_price'),
                ('We are out of milk tea.', 'unsupported_stock'),
                ("We're out of milk tea.", 'unsupported_stock'),
                ("We don't have coffee.", 'unsupported_stock'),
                ('Milk tea is unavailable.', 'unsupported_stock')):
            with self.subTest(text=text):
                self.assertEqual(question_violation(text, state), reason)
        for text in ('Would you like one or two cups?', 'Would you like to know the price?',
                     'That sounds delicious.', 'That tastes out of this world.',
                     'We can serve both drinks.', 'Your order will take a few minutes.'):
            self.assertIsNone(question_violation(text, state))

    def test_request_history_and_prior_facts_are_never_mutated(self):
        payload = request(('user', 'I want coffee.'), ('assistant', 'Hot or iced?'))
        before = deepcopy(payload)
        state = cafe_state(payload, 'Iced, please.')
        self.assertEqual(payload, before)
        state['facts']['items']['value'].append('latte')
        self.assertEqual(payload, before)


if __name__ == '__main__':
    unittest.main()
