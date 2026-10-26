"""Edit-aware writing suggestions preserve facts while allowing bounded repairs.

These checks exercise the public parser without loading a model. Acceptance is
permission to show a separate draft, never evidence that it is an audio target
or a complete grammar judgment.
"""
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sentence_rewrite import parse_rewrite


class RewriteGuardRegressionTests(unittest.TestCase):
    def parse(self, source, grammar, natural=None):
        payload = {
            'grammar_corrected': grammar,
            'natural': grammar if natural is None else natural,
            'explanation': 'A separate writing suggestion.',
        }
        return parse_rewrite(json.dumps(payload), True, source)

    def assert_allowed(self, source, target):
        result = self.parse(source, target)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['grammar_status'], 'ready')
        self.assertEqual(result['natural_status'], 'ready')
        self.assertEqual(result['grammar_corrected'], target)
        self.assertEqual(result['natural'], target)
        self.assertEqual(result['original'], source)
        self.assertNotIn('reference_text', result)
        self.assertNotIn('target_words', result)
        return result

    def assert_withheld(self, source, target):
        result = self.parse(source, target)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['grammar_status'], 'unavailable')
        self.assertEqual(result['natural_status'], 'unavailable')
        self.assertEqual(result['grammar_corrected'], '')
        self.assertEqual(result['natural'], '')
        self.assertEqual(result['original'], source)

    def test_missing_infinitive_to_after_able_preserves_action_and_quantity(self):
        self.assert_allowed('I am able do two things', 'I am able to do two things.')

    def test_missing_for_in_search_complement_is_allowed(self):
        self.assert_allowed('I search a job', 'I search for a job.')

    def test_specialize_complement_repairs_preposition_without_new_content(self):
        self.assert_allowed('I specialize on software', 'I specialize in software.')

    def test_goal_infinitive_and_gerund_complements_preserve_intended_goal(self):
        self.assert_allowed('I have a goal to lose weight', 'I have a goal of losing weight.')

    def test_dummy_it_repairs_impersonal_clause_without_new_actor(self):
        self.assert_allowed('I think is important to read', 'I think it is important to read.')

    def test_temporal_everyday_can_be_split_into_every_day(self):
        self.assert_allowed('I go to work everyday', 'I go to work every day.')

    def test_agreement_and_user_routine_remain_accepted(self):
        for source, target in (
            ('She have two books', 'She has two books.'),
            ('He go to work', 'He goes to work.'),
            ('i usually wakes up at seven am then i brush my teeth and go to work by bus',
             'I usually wake up at seven a.m., then brush my teeth and go to work by bus.'),
        ):
            with self.subTest(source=source):
                self.assert_allowed(source, target)

    def test_unchanged_sentence_is_not_reported_as_a_wording_correction(self):
        result = self.assert_allowed('I usually wake up at seven.', 'I usually wake up at seven.')
        self.assertFalse(result['changed'])
        self.assertFalse(result['grammar_changed'])
        self.assertFalse(result['natural_changed'])

    def test_names_and_quantities_cannot_be_changed_or_reassigned(self):
        for source, target in (
            ('I work with Sarah', 'I work with Sara.'),
            ('I work with Sarah', 'I work with Sarahs.'),
            ('I order one coffee', 'I order two coffees.'),
            ('I pay 15 dollars', 'I pay 50 dollars.'),
            ('I order two coffees and three teas', 'I order three coffees and two teas.'),
        ):
            with self.subTest(target=target):
                self.assert_withheld(source, target)

    def test_negation_modal_degree_and_intensity_remain_protected(self):
        for source, target in (
            ('I do not like tea', 'I like tea.'),
            ('I do not like tea', 'I like not tea.'),
            ('I can work tomorrow', 'I will work tomorrow.'),
            ('I might work tomorrow', 'I work tomorrow.'),
            ('I like coffee', 'I really like coffee.'),
            ('I like coffee', 'I love coffee.'),
            ('I am happy', 'I am very happy.'),
            ('I am very happy', 'I am happy.'),
            ('I sometimes drink tea', 'I always drink tea.'),
        ):
            with self.subTest(target=target):
                self.assert_withheld(source, target)

    def test_lexical_have_and_tense_aspect_cannot_be_erased_as_auxiliaries(self):
        for source, target in (
            ('I have work', 'I work.'),
            ('I was happy', 'I am happy.'),
            ('I had read the book', 'I read the book.'),
        ):
            with self.subTest(source=source):
                self.assert_withheld(source, target)

    def test_new_facts_and_substituted_content_are_withheld(self):
        for source, target in (
            ('I like coffee', 'I like tea.'),
            ('I like coffee', 'I like fresh coffee.'),
            ('I like coffee', 'I like coffee because it tastes good.'),
            ('I search a job', 'I search for a good job.'),
            ('I specialize on software', 'I specialize in software for banks.'),
            ('I am able do two things', 'I am able to do three things.'),
        ):
            with self.subTest(target=target):
                self.assert_withheld(source, target)

    def test_actors_action_order_and_direction_cannot_be_reversed(self):
        for source, target in (
            ('John likes Mary', 'Mary likes John.'),
            ('The dog sees the man', 'The man sees the dog.'),
            ('I eat then I sleep', 'I sleep then I eat.'),
            ('I travel from London to Paris', 'I travel to London from Paris.'),
            ('I work from home', 'I work to home.'),
            ('I get a letter from Sarah', 'I get a letter to Sarah.'),
        ):
            with self.subTest(target=target):
                self.assert_withheld(source, target)

    def test_unclear_asr_words_are_not_replaced_by_plausible_guesses(self):
        for source, target in (
            ('My fravorit is coffee', 'My favorite is coffee.'),
            ('I am qning a car', 'I am owning a car.'),
            ('I want cupf coffee', 'I want a cup of coffee.'),
        ):
            with self.subTest(source=source):
                self.assert_withheld(source, target)

    def test_bounded_repairs_do_not_license_adding_an_arbitrary_subject(self):
        self.assert_withheld('I think is important to read', 'I think he is important to read.')

    def test_grammar_and_natural_drafts_are_decided_independently(self):
        source = 'I am able do two things'
        valid = 'I am able to do two things.'
        invented = 'I am able to do three things.'
        for grammar, natural, ready_field, blocked_field in (
            (valid, invented, 'grammar_corrected', 'natural'),
            (invented, valid, 'natural', 'grammar_corrected'),
        ):
            with self.subTest(ready_field=ready_field):
                result = self.parse(source, grammar, natural)
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(result[ready_field], valid)
                self.assertEqual(result[blocked_field], '')
                self.assertEqual(result['grammar_status'], 'ready' if ready_field == 'grammar_corrected' else 'unavailable')
                self.assertEqual(result['natural_status'], 'ready' if ready_field == 'natural' else 'unavailable')
                self.assertEqual(result['original'], source)


if __name__ == '__main__':
    unittest.main()
