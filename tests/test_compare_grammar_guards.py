"""Saved-output guard comparisons must not invent model quality evidence."""
import json
import unittest

from scripts.compare_grammar_guards import compare_row, inspect, report_html, summarize, verify_baseline


class GrammarGuardComparisonTests(unittest.TestCase):
    def row(self, task="grammar_corrected", prediction="I like coffee.", cohort="published_heldout"):
        return {"id": "example-" + task + "-" + cohort, "source": "I like coffee", "task": task,
                "cohort": cohort, "prediction": prediction, "references": ["I like coffee."]}

    def parser(self, accepted):
        def parse(raw, finished, original):
            value = json.loads(raw)
            self.assertTrue(finished)
            self.assertEqual(original, "I like coffee")
            return {**value, "grammar_status": "ready" if accepted else "unavailable",
                    "natural_status": "ready" if accepted else "unavailable",
                    "grammar_changed": False, "natural_changed": False,
                    "grammar_format_changed": accepted, "natural_format_changed": accepted}
        return parse

    def test_prediction_passes_unchanged_to_both_parsers_and_field_independently(self):
        calls = []
        def parse(raw, finished, original):
            value = json.loads(raw)
            calls.append(value)
            self.assertEqual(value["grammar_corrected"], original)
            self.assertEqual(value["natural"], "I like coffee.")
            return {**value, "natural_status": "ready", "natural_changed": False, "natural_format_changed": True}
        row = compare_row(self.row("natural"), parse, parse)
        self.assertEqual(row["transition"], "retained")
        self.assertEqual(len(calls), 2)
        self.assertTrue(row["matches_published_reference"])
        self.assertFalse(row["current"]["changed"])

    def test_unavailable_native_output_is_not_replaced_by_source_or_reference(self):
        def forbidden(*args):
            self.fail("A missing prediction must never be sent to a guard")
        row = compare_row(self.row(prediction=None), forbidden, forbidden)
        self.assertEqual(row["transition"], "still_withheld")
        self.assertIsNone(row["current"]["delivered_prediction"])
        self.assertFalse(row["matches_published_reference"])

    def test_newly_accepted_reference_is_a_delivery_count_not_a_grammar_verdict(self):
        row = compare_row(self.row(), self.parser(False), self.parser(True))
        summary = summarize([row])["published_heldout"]["grammar_corrected"]
        self.assertEqual(summary["reference_matches_recovered"], 1)
        self.assertEqual(summary["newly_accepted"], 1)
        self.assertEqual(summary["current"]["lexical_changes_delivered"], 0)
        self.assertEqual(summary["current"]["format_only_delivered"], 1)
        self.assertNotIn("accuracy", summary)

    def test_development_cases_and_tasks_do_not_change_published_counts(self):
        rows = [compare_row(self.row(), self.parser(False), self.parser(True)),
                compare_row(self.row("natural", cohort="user_development"), self.parser(True), self.parser(False))]
        summary = summarize(rows)
        self.assertEqual(summary["published_heldout"]["grammar_corrected"]["rows"], 1)
        self.assertEqual(summary["published_heldout"]["natural"]["rows"], 0)
        self.assertEqual(summary["user_development"]["natural"]["newly_withheld"], 1)

    def test_baseline_must_reproduce_original_decisions(self):
        row = compare_row(self.row(), self.parser(False), self.parser(True))
        previous = {"models": {"qwen-grammar": {"rows": [{"id": row["id"], **row["baseline"]}]}}}
        verify_baseline([row], previous)
        previous["models"]["qwen-grammar"]["rows"][0]["delivered"] = True
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            verify_baseline([row], previous)

    def test_html_escapes_saved_model_output(self):
        row = compare_row(self.row(prediction="<script>alert('x')</script>"), self.parser(False), self.parser(True))
        rendered = report_html({"summary": summarize([row]), "rows": [row], "limitations": ["No semantic proof."]})
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("No semantic proof.", rendered)


if __name__ == "__main__":
    unittest.main()
