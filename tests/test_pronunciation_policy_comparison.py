"""Pure scalar-policy and denominator tests; no model/audio/cache required."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts.compare_pronunciation_policies import metrics, predict, failures, select_policies, word_features


def row(gold, score, *, eligible=True, group="fit", speaker="HJK", baseline=0):
    return {"gold": gold, "score": score, "score_eligible": eligible, "group": group,
            "speaker_id": speaker, "baseline_prediction": baseline,
            "reason": "reference_unconfirmed" if eligible else "insertion_or_deletion_not_supported"}


def policy():
    return {"selection_restrictions": {"grid_points": 31}, "families": {
        "error-cutoff-f05": {"constraint_FPR_all_expert_correct_max": .1,
                              "constraint_recall_all_expert_errors_min": .4, "min_flagged_words": 20},
        "selective-two-threshold": {"max_FPR_decided": .1, "max_FNR_decided": .2,
                                     "min_coverage": .25, "min_correct_decisions": 10, "min_error_decisions": 5}}}


class PronunciationPolicyTests(unittest.TestCase):
    def test_error_only_never_accepts_unflagged_or_ineligible_words(self):
        rows = [row(1, 4), row(0, 4), row(0, -4), row(1, 9, eligible=False)]
        result = metrics(rows, "error-cutoff-f05", {"error_threshold": 1.})
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 0, 0))
        self.assertEqual(result["abstained_errors"], 1)
        self.assertEqual(result["abstained_correct"], 1)
        self.assertEqual(result["recall_all_errors"], .5)
        self.assertEqual(result["false_positive_all_correct"], .5)
        self.assertEqual(result["false_positive_rate_decided"], 1.)
        self.assertEqual(result["flag_all_fn"], 1)
        self.assertEqual(result["flag_all_tn"], 1)

    def test_selective_matrix_excludes_abstained_error_from_fn(self):
        rows = [row(1, 4), row(0, 4), row(0, -4), row(1, -4), row(1, 0), row(0, 0)]
        result = metrics(rows, "selective-two-threshold", {"accept_threshold": -1., "error_threshold": 1.})
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["decided_words"], 4)
        self.assertEqual(result["abstained_errors"], 1)
        self.assertEqual(result["recall_all_errors"], 1/3)
        self.assertEqual(result["recall_decided"], .5)

    def test_exact_cutoff_is_a_flag_and_missing_threshold_is_unknown(self):
        self.assertEqual(predict(row(1, 2), "error-cutoff-f05", {"error_threshold": 2}), 1)
        self.assertIsNone(predict(row(1, 2), "error-cutoff-f05", None))

    def test_selection_cannot_read_heldout_or_test_rows(self):
        for group, speaker in (("heldout", "PNV"), ("test", "TLV")):
            with self.assertRaisesRegex(ValueError, "Only HJK/LXC"):
                select_policies([row(1, 5, group=group, speaker=speaker)], policy())

    def test_no_eligible_threshold_remains_failure_not_relaxed(self):
        selected, _ = select_policies([row(0, 0) for _ in range(40)], policy())
        for choice in selected.values():
            self.assertIsNone(choice["thresholds"])
            self.assertFalse(choice["fit_eligible"])
            self.assertIn("no_threshold_meets_fit_constraints", choice["fit_failures"])

    def test_error_only_floor_uses_all_errors_including_unknown(self):
        rows = [row(1, 5) for _ in range(20)] + [row(1, 5, eligible=False) for _ in range(80)] + [row(0, -5) for _ in range(100)]
        result = metrics(rows, "error-cutoff-f05", {"error_threshold": 0})
        self.assertEqual(result["recall_all_errors"], .2)
        self.assertIn("recall_all_errors", failures(result, "error-cutoff-f05", policy()["families"]["error-cutoff-f05"]))

    def test_perfect_separation_selects_error_cutoff_with_zero_false_alarm(self):
        rows = [row(1, 4) for _ in range(30)] + [row(0, -4) for _ in range(30)]
        selected, _ = select_policies(rows, policy())
        choice = selected["error-cutoff-f05"]
        self.assertTrue(choice["fit_eligible"])
        self.assertEqual(choice["fit_metrics"]["false_positive_all_correct"], 0.)
        self.assertEqual(choice["fit_metrics"]["recall_all_errors"], 1.)

    def test_explicit_word_gold_ignores_only_unmapped_event_not_entire_sentence(self):
        annotation = {"utterance_id": "HJK/example", "speaker_id": "HJK", "prompt_text": "hello world",
                      "word_intervals": [{"text": "hello"}, {"text": "world"}],
                      "gold_error_events": [{"word_index": 1}, {"word_index": None}]}
        scored = {"words": [{"error_score": -3., "score_eligible": True}, {"error_score": 2., "score_eligible": True}]}
        with patch("scripts.compare_pronunciation_policies.variant_review", return_value={"status": "available", "hint_word_indices": [1]}):
            result = word_features(annotation, scored, {"phonemes": [{"phone": "HH"}]}, group="fit", evidence_sha="hash")
        self.assertEqual([r["gold"] for r in result], [0, 1])
        self.assertTrue(all(r["score_eligible"] for r in result))

    def test_ineligible_reasons_are_preserved(self):
        result = metrics([row(1, 0, eligible=False)], "error-cutoff-f05", {"error_threshold": 0})
        self.assertEqual(result["score_ineligible_reasons"], {"insertion_or_deletion_not_supported": 1})


if __name__ == "__main__":
    unittest.main()
