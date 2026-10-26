"""Check that guard auditing does not turn unavailable/copy output into a fix."""
import unittest

from scripts.audit_grammar_delivery import inspect_prediction, summarize


class GrammarDeliveryAuditTests(unittest.TestCase):
    def row(self, prediction, task="grammar_corrected"):
        return {"id": "example", "source": "I like coffee.", "task": task,
                "cohort": "user_development", "prediction": prediction, "error": None}

    def test_native_failure_stays_unavailable(self):
        value = inspect_prediction(self.row(None))
        self.assertFalse(value["delivered"])
        self.assertEqual(value["reason"], "native_contract_unavailable")
        self.assertIsNone(value["delivered_prediction"])

    def test_copy_is_not_counted_as_lexical_improvement(self):
        value = inspect_prediction(self.row("I like coffee."))
        self.assertTrue(value["delivered"])
        self.assertFalse(value["changed"])
        self.assertFalse(value["format_changed"])

    def test_fact_invention_is_withheld(self):
        value = inspect_prediction(self.row("My sister works in London."))
        self.assertEqual(value["reason"], "lexical_guard_withheld")
        self.assertIsNone(value["delivered_prediction"])

    def test_other_field_does_not_hide_failed_natural_suggestion(self):
        value = inspect_prediction(self.row("My sister works in London.", "natural"))
        self.assertFalse(value["delivered"])

    def test_summary_preserves_each_failure_and_copy(self):
        values = [inspect_prediction(self.row(p)) for p in (None, "I like coffee.", "My sister works in London.")]
        counts = summarize(values)["user_development"]["grammar_corrected"]
        self.assertEqual(counts["rows"], 3)
        self.assertEqual(counts["native_available"], 2)
        self.assertEqual(counts["delivered"], 1)
        self.assertEqual(counts["lexical_changes_delivered"], 0)

    def test_reference_match_does_not_restore_unavailable_output(self):
        row = self.row(None)
        row["references"] = ["I like coffee."]
        unavailable = inspect_prediction(row)
        self.assertFalse(unavailable["matches_published_reference"])
        self.assertFalse(unavailable["delivered"])
        row["prediction"] = "I  like coffee."
        matched = inspect_prediction(row)
        self.assertTrue(matched["matches_published_reference"])
        self.assertEqual(summarize([matched])["user_development"]["grammar_corrected"]["reference_matching_native_outputs"], 1)


if __name__ == "__main__":
    unittest.main()
