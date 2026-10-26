"""Meaning-preservation checks for the experimental phrase decoder."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_bert_phrase_recovery import eligible, finish_result, spans


class PhraseGuardsTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = dict.fromkeys(('i will would like to order a cup of coffee one two '
                                        'ice ta tea please allowed him not no try yes yesterday '
                                        'go work sarah').split())

    def test_phrase_repair_is_not_a_sentence_lookup(self):
        self.assertTrue(eligible('bottl', 'bottle', {'bottle': []}))
        self.assertTrue(eligible('cupf', 'cup of', self.dictionary))

    def test_new_pronoun_cannot_change_who_orders(self):
        self.assertFalse(eligible('will wi', 'allowed him', self.dictionary))

    def test_quantities_and_negation_are_not_reconstructed(self):
        self.assertFalse(eligible('one ta', 'two tea', self.dictionary))
        self.assertFalse(eligible('not ta', 'no tea', self.dictionary))

    def test_clear_function_word_is_not_expanded(self):
        self.assertFalse(eligible('to', 'sort of', self.dictionary | {'sort': []}))
        self.assertNotIn('to', ['I will wi to order a cupf coffee'[a:b]
                               for a, b in spans('I will wi to order a cupf coffee', self.dictionary)])

    def test_clear_and_learner_grammar_sentences_are_not_rewritten(self):
        for text in ('I would like tea please', 'Yesterday I go to work', 'Yes'):
            with self.subTest(text=text):
                self.assertEqual(spans(text, self.dictionary), [])

    def test_capitalized_unknown_name_is_preserved(self):
        self.assertEqual(spans('I like Soo', self.dictionary), [])

    def test_short_unknown_is_not_greedily_replaced_in_isolation(self):
        text = 'I will wi to order'
        self.assertEqual([text[a:b] for a, b in spans(text, self.dictionary)], ['will wi'])

    def test_incomplete_recovery_cannot_publish_hallucinated_partial_sentence(self):
        result = finish_result('Flerm narg zibble', 'From the a der zibble', [],
                               {'from': [], 'the': [], 'a': [], 'der': []})
        self.assertEqual(result['text'], 'Flerm narg zibble')
        self.assertEqual(result['status'], 'ambiguous')
        self.assertTrue(result['needs_clarification'])
        self.assertFalse(result['reference_confirmed'])


if __name__ == '__main__':
    unittest.main()
