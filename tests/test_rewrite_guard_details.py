"""Protect grammatical contexts and diagnostics at the delivery boundary."""
import json
import unittest

from scripts.sentence_rewrite import parse_rewrite, validate_rewrite


class RewriteGuardDetailsTests(unittest.TestCase):
    def test_singular_other_determiner_has_a_bounded_noun_context(self):
        self.assertTrue(validate_rewrite('Other important aspect is teamwork',
                                        'Another important aspect is teamwork.')['accepted'])
        self.assertFalse(validate_rewrite('Other people work here', 'Another people work here.')['accepted'])
        self.assertFalse(validate_rewrite('Other important aspects matter', 'Another important aspects matter.')['accepted'])

    def test_participles_are_not_mistaken_for_independent_past_assertions(self):
        for source, target in (
            ('I studied in Al Qassim University and my major Microbiology.',
             'I studied at Al Qassim University and my major was Microbiology.'),
            ('They were drove home', 'They were driven home.'),
            ('This fact is important compare to the other facts',
             'This fact is important compared to the other facts.'),
            ('People drive in narrowed streets', 'People drive in narrow streets.'),
            ('They did not expected the bus', 'They did not expect the bus.'),
        ):
            with self.subTest(source=source):
                self.assertTrue(validate_rewrite(source, target)['accepted'])

    def test_degree_word_cannot_move_to_another_clause_or_object_quantity(self):
        self.assertTrue(validate_rewrite('I can more enjoy the trip',
                                        'I can enjoy the trip more.')['accepted'])
        for source, target in (
            ('I more like coffee and hate tea', 'I like coffee and hate tea more.'),
            ('I more like coffee with milk', 'I like coffee with milk more.'),
            ('I like more coffee', 'I like coffee more.'),
            ('I more like bitter coffee', 'I like more bitter coffee.'),
        ):
            with self.subTest(source=source):
                self.assertFalse(validate_rewrite(source, target)['accepted'])

    def test_search_location_and_study_location_are_not_complement_repairs(self):
        for source, target in (
            ('I search the room', 'I search for the room.'),
            ('I studied in London', 'I studied at London.'),
            ('I studied in a university', 'I studied at a different university.'),
        ):
            with self.subTest(source=source):
                self.assertFalse(validate_rewrite(source, target)['accepted'])

    def test_percent_currency_sign_and_progressive_aspect_are_preserved(self):
        for source, target in (
            ('It costs $15', 'It costs 15.'),
            ('It costs $15', 'It costs €15.'),
            ('I scored 56%', 'I scored 56.'),
            ('The temperature is -2 degrees', 'The temperature is 2 degrees.'),
            ('I am reading a book', 'I read a book.'),
            ('Yesterday I went to work', 'Yesterday I go to work.'),
            ('I want one hundred books', 'I wanted one hundred books.'),
            ('I feel tired', 'I felt tired.'),
            ('I feel very tired', 'I felt very tired.'),
        ):
            with self.subTest(source=source):
                self.assertFalse(validate_rewrite(source, target)['accepted'])

    def test_surface_forms_do_not_license_ambiguous_word_splits(self):
        self.assertTrue(validate_rewrite('Our life styles changed', 'Our lifestyles changed.')['accepted'])
        self.assertFalse(validate_rewrite('I enjoy everyday life', 'I enjoy every day life.')['accepted'])
        self.assertFalse(validate_rewrite('I go nowhere', 'I go now here.')['accepted'])
        self.assertFalse(validate_rewrite('I went home. Work was hard', 'I went homework was hard.')['accepted'])

    def test_public_result_has_per_field_decisions_without_replacing_input(self):
        source = 'I am able do two things'
        raw = json.dumps({'grammar_corrected': 'I am able to do two things.',
                          'natural': 'I am able to do three things.', 'explanation': 'A draft.'})
        result = parse_rewrite(raw, True, source)
        self.assertEqual(result['original'], source)
        self.assertEqual(result['natural'], '')
        self.assertEqual(result['guard_validation']['natural']['reason'], 'protected_fact_changed')
        self.assertIn('missing_infinitive_to', result['guard_validation']['grammar_corrected']['licensed_edit_types'])
        self.assertNotIn('reference_text', result)


if __name__ == '__main__':
    unittest.main()
