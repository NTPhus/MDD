"""Keep the secondary content audit transparent and free of word repair."""
import unittest

from scripts.audit_grammar_content import extract_content


class GrammarContentAuditTests(unittest.TestCase):
    def row(self, raw, **changes):
        return {"task": "grammar_corrected", "raw": raw, "prediction": None,
                "error": "invalid_schema", "finished": True, **changes}

    def test_wrong_schema_retains_exact_string_without_word_repair(self):
        text, source = extract_content(self.row('{"task":"grammar_corrected","transcript":"I wakes up."}'))
        self.assertEqual(text, "I wakes up.")
        self.assertEqual(source, "posthoc_transcript_field")

    def test_native_contract_has_priority(self):
        text, source = extract_content(self.row("irrelevant", prediction="I wake up."))
        self.assertEqual((text, source), ("I wake up.", "native_contract"))

    def test_mismatching_task_or_extra_fields_are_not_recovered(self):
        for raw in ('{"task":"natural","transcript":"I wake up."}',
                    '{"task":"grammar_corrected","transcript":"I wake up.","extra":1}',
                    '{"transcript":"I wake up."}',
                    '```json\n{"task":"grammar_corrected","transcript":"I wake up."}\n```'):
            self.assertEqual(extract_content(self.row(raw)), (None, "unavailable"))

    def test_unfinished_or_malformed_generation_remains_unavailable(self):
        raw = '{"task":"grammar_corrected","transcript":"I wake up."}'
        self.assertEqual(extract_content(self.row(raw, finished=False)), (None, "unavailable"))
        self.assertEqual(extract_content(self.row(raw, error="invalid_json")), (None, "unavailable"))

    def test_duplicates_and_empty_text_are_rejected(self):
        for raw in ('{"task":"grammar_corrected","transcript":"good","transcript":"bad"}',
                    '{"task":"grammar_corrected","transcript":" "}'):
            self.assertEqual(extract_content(self.row(raw)), (None, "unavailable"))


if __name__ == "__main__":
    unittest.main()
