"""Boundary checks for the frozen editing comparison, without model inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts import compare_grammar_models as comparison


class GrammarComparisonTests(unittest.TestCase):
    def test_canonical_does_not_drop_negation_or_merge_numbers(self):
        self.assertEqual(comparison.canonical('I do NOT like it!'), 'i do not like it')
        self.assertNotEqual(comparison.canonical('three'), comparison.canonical('thirteen'))

    def test_native_coedit_prompts_keep_tasks_distinct(self):
        self.assertNotEqual(comparison.PROMPTS['grammar_corrected'], comparison.PROMPTS['natural'])
        self.assertIn('Fix grammatical errors', comparison.PROMPTS['grammar_corrected'])

    def test_json_failure_is_not_a_grammar_score(self):
        from scripts.evaluate_grammar_model import parse_output
        self.assertEqual(parse_output('{"transcript":"I wake up."}', 'grammar_corrected'),
                         (None, 'invalid_schema'))
        self.assertEqual(parse_output('{"grammar_corrected":"I wake up."}', 'grammar_corrected'),
                         ('I wake up.', None))

    def test_hash_is_content_not_path(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory) / name for name in ('a', 'b')]
            first.write_bytes(b'one')
            second.write_bytes(b'one')
            self.assertEqual(comparison.digest(first), comparison.digest(second))
            second.write_bytes(b'two')
            self.assertNotEqual(comparison.digest(first), comparison.digest(second))


if __name__ == '__main__':
    unittest.main()
