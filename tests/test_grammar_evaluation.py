"""Evaluation never rewards invalid output or silently changes the cohort."""
import unittest

from scripts.evaluate_grammar_model import parse_output, score, selected_rows


class GrammarEvaluationTests(unittest.TestCase):
    def test_strict_schema_budget_and_no_fallback(self):
        self.assertEqual(parse_output('{"natural":"I like it."}', 'natural'), ('I like it.', None))
        for text in ['Hi!', '{"grammar_corrected":"I like it."}',
                     '{"natural":""}', '{"natural":"OK","explanation":"good"}', '["OK"]']:
            with self.subTest(text=text):
                self.assertIsNone(parse_output(text, 'natural')[0])
        self.assertEqual(parse_output('{"natural":"OK"}', 'natural', False),
                         (None, 'generation_budget_exhausted'))

    def test_development_limit_is_stable_and_includes_both_tasks(self):
        rows = [{'id': str(i), 'task': 'grammar_corrected' if i < 10 else 'natural'} for i in range(12)]
        a = selected_rows(rows, 4)
        self.assertEqual(a, selected_rows(list(reversed(rows)), 4))
        self.assertEqual({r['task'] for r in a}, {'grammar_corrected', 'natural'})
        with self.assertRaises(ValueError):
            selected_rows(rows, 0)

    def test_invalid_correct_control_stays_failure_multi_reference_accepted(self):
        observations = [
            {'task': 'grammar_corrected', 'source': 'I go.', 'references': ['I go.'],
             'prediction': None, 'seconds': 1.},
            {'task': 'grammar_corrected', 'source': 'She go.', 'references': ['She goes.', 'She went.'],
             'prediction': 'She went.', 'seconds': 1.},
        ]
        result = score(observations)['tasks']['grammar_corrected']
        self.assertEqual(result['valid_json_rate'], .5)
        self.assertEqual(result['exact_reference_match'], .5)
        self.assertEqual(result['unchanged_preservation_rate'], 0.)
        self.assertEqual(result['reference_supported_unchanged_controls'], 1)


if __name__ == '__main__':
    unittest.main()
