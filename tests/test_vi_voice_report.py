"""Frozen report contracts: no inference, model download or voice API."""
from copy import deepcopy
import unittest

from scripts.report_vi_voice_pipeline import (
    baseline_comparison, case_html, metric_table, per_case_table, recovery_table,
)


class ViVoiceReportTests(unittest.TestCase):
    def row(self, **changes):
        value = {"id": "clip", "corpus": "L2-ARCTIC", "speaker_id": "TLV",
                 "audio": "clip.wav", "audio_sha256": "a" * 64,
                 "reference_text": "Today I work.", "status": "ready",
                 "transcript": "To day I work", "normalized": "To day I work.",
                 "conversation_text": "Today I work.", "reply": "What do you do?",
                 "interpretation": {"status": "accepted", "reason": None,
                                    "requires_confirmation": True, "text": "Today I work."}}
        value.update(changes)
        return value

    def results(self, *rows):
        return {"identity": {"manifest_sha256": "manifest"}, "records": list(rows)}

    def test_recovery_is_a_separate_text_stage_and_failures_not_scored_as_silence(self):
        records = [self.row(), self.row(id="failed", status="error", transcript="", conversation_text="")]
        metrics = metric_table(records)
        all_clips = metrics[metrics["corpus"] == "All clips"].set_index("text_stage")
        self.assertEqual(set(all_clips.index), {"Raw ASR", "Normalized", "Conversation interpretation"})
        self.assertEqual(int(all_clips.loc["Raw ASR", "word_errors"]), 2)
        self.assertEqual(int(all_clips.loc["Conversation interpretation", "word_errors"]), 0)
        self.assertEqual(int(all_clips.loc["Conversation interpretation", "reference_words"]), 3)
        self.assertEqual(int(all_clips.loc["Conversation interpretation", "failed_clips"]), 1)
        self.assertNotIn("semantic_accuracy", metrics.columns)

    def test_legacy_rows_keep_two_metric_stages_and_missing_recovery_visible(self):
        row = self.row(); row.pop("interpretation")
        self.assertEqual(set(metric_table([row])["text_stage"]), {"Raw ASR", "Normalized"})
        self.assertEqual(recovery_table([row]).iloc[0]["interpretation_status"], "not_recorded")

    def test_status_table_and_export_keep_rejections_and_request_failures(self):
        records = [self.row(), self.row(id="bad", status="error", interpretation={
            "status": "unavailable", "reason": "model_unavailable", "requires_confirmation": False})]
        statuses = recovery_table(records)
        self.assertEqual(int(statuses["clips"].sum()), 2)
        self.assertEqual(set(statuses["request_status"]), {"ready", "error"})
        cases = per_case_table(records)
        self.assertTrue(cases.iloc[0]["interpretation_requires_confirmation"])
        self.assertEqual(cases.iloc[1]["interpretation_reason"], "model_unavailable")
        self.assertTrue(cases.iloc[1]["conversation_word_errors"] != cases.iloc[1]["conversation_word_errors"])

    def test_baseline_pairing_requires_same_identity_audio_and_prompt(self):
        current = self.results(self.row())
        baseline = deepcopy(current)
        paired = baseline_comparison(current, baseline)
        self.assertEqual(len(paired), 1)
        self.assertTrue(paired.iloc[0]["raw_asr_tokens_equal"])
        for field in ["audio_sha256", "audio", "reference_text"]:
            changed = deepcopy(baseline); changed["records"][0][field] = "different"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                baseline_comparison(current, changed)
        changed = deepcopy(baseline); changed["records"].append(deepcopy(changed["records"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            baseline_comparison(current, changed)
        changed = deepcopy(baseline); changed["identity"]["manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "manifest"):
            baseline_comparison(current, changed)

    def test_failed_baseline_pair_remains_present_without_error_score(self):
        current = self.results(self.row())
        baseline = self.results(self.row(status="error"))
        paired = baseline_comparison(current, baseline)
        self.assertEqual(len(paired), 1)
        self.assertFalse(paired.iloc[0]["paired_ready"])
        self.assertIsNone(paired.iloc[0]["baseline_word_errors"])
        self.assertIsNone(paired.iloc[0]["current_word_errors"])

    def test_case_html_displays_recovery_without_executing_markup(self):
        row = self.row(reply="<script>no</script>")
        page = case_html(row, {"duration_sec": 1, "source_url": "https://example.org"},
                         ".", include_audio=False)
        self.assertIn("Interpretation status", page)
        self.assertIn("confirmation required: True", page)
        self.assertIn("&lt;script&gt;no&lt;/script&gt;", page)
        self.assertNotIn("<script>", page)


if __name__ == "__main__":
    unittest.main()
