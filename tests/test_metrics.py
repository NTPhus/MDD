"""Auditable phone/MDD scorer examples; no models, network, or training."""

import json
import unittest

from src.mdd.metrics import aggregate_scores, align_phones, score_utterance


class MetricsTests(unittest.TestCase):
    def test_alignment_substitution_deletion_and_addition(self):
        self.assertEqual([x["kind"] for x in align_phones(["TH", "IH", "NG", "K"], ["T", "IH", "NG"])],
                         ["substitution", "match", "match", "deletion"])
        insertion = align_phones(["AA"], ["T", "AA", "K"])
        self.assertEqual([(x["kind"], x["position"]) for x in insertion],
                         [("addition", 0), ("match", 0), ("addition", 1)])

    def test_ambiguous_repeat_alignment_is_deterministic(self):
        alignment = align_phones(["AA", "AA"], ["AA"])
        self.assertEqual([(x["kind"], x["position"]) for x in alignment],
                         [("deletion", 0), ("match", 1)])
        for _ in range(3):
            self.assertEqual(align_phones(["AA", "AA"], ["AA"]), alignment)

    def test_no_errors_has_no_invented_insertion_true_negatives(self):
        score = score_utterance(["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"])
        self.assertEqual(score["counts"]["TA"], 4)
        self.assertEqual(score["metrics"]["per"], 0)
        self.assertEqual(score["metrics"]["frr"], 0)
        for key in ("precision", "recall", "f1", "far", "der"):
            self.assertIsNone(score["metrics"][key])

    def test_correct_error_detection_and_diagnosis(self):
        score = score_utterance(["TH", "IH", "NG", "K"], ["T", "IH", "NG"], ["T", "IH", "NG"])
        self.assertEqual({k: score["counts"][k] for k in ("TA", "TR", "FA", "FR", "CD", "DE")},
                         {"TA": 2, "TR": 2, "FA": 0, "FR": 0, "CD": 2, "DE": 0})
        self.assertEqual(score["metrics"]["f1"], 1)
        self.assertEqual(score["metrics"]["der"], 0)

    def test_detected_but_wrong_diagnosis_and_false_rejection(self):
        score = score_utterance(["TH", "IH", "NG"], ["T", "IH", "NG"], ["S", "IY", "NG"])
        self.assertEqual({k: score["counts"][k] for k in ("TA", "TR", "FR", "DE")},
                         {"TA": 1, "TR": 1, "FR": 1, "DE": 1})
        self.assertEqual(score["metrics"]["precision"], 0.5)
        self.assertEqual(score["metrics"]["recall"], 1)
        self.assertEqual(score["metrics"]["frr"], 0.5)
        self.assertEqual(score["metrics"]["der"], 1)
        self.assertEqual(score["counts"]["PER_S"], 2)

    def test_error_accepted_means_false_acceptance(self):
        score = score_utterance(["TH", "IH"], ["T", "IH"], ["TH", "IH"])
        self.assertEqual(score["counts"]["FA"], 1)
        self.assertEqual(score["metrics"]["far"], 1)
        self.assertEqual(score["metrics"]["f1"], 0)
        self.assertIsNone(score["metrics"]["precision"])

    def test_wrong_insertion_gap_does_not_count_as_detected(self):
        score = score_utterance(["AA", "B"], ["T", "AA", "B"], ["AA", "B", "T"])
        self.assertEqual({k: score["counts"][k] for k in ("TA", "TR", "FA", "FR")},
                         {"TA": 2, "TR": 0, "FA": 1, "FR": 1})
        self.assertEqual(score["metrics"]["f1"], 0)

    def test_multiple_insertions_paired_by_order_in_gap(self):
        score = score_utterance(["AA"], ["T", "K", "AA"], ["T", "S", "N", "AA"])
        self.assertEqual({k: score["counts"][k] for k in ("TA", "TR", "FR", "CD", "DE")},
                         {"TA": 1, "TR": 2, "FR": 1, "CD": 1, "DE": 1})
        self.assertEqual(score["metrics"]["f1"], 4 / 5)
        self.assertEqual(score["metrics"]["der"], 0.5)

    def test_err_and_unknown_excluded_from_diagnosis_not_detection(self):
        for unknown in ("err", "[UNK]"):
            with self.subTest(unknown=unknown):
                score = score_utterance(["TH", "IH"], [unknown, "IH"], ["T", "IH"])
                self.assertEqual(score["counts"]["TR"], 1)
                self.assertEqual(score["counts"]["DIAG_EXCLUDED"], 1)
                self.assertEqual(score["counts"]["CD"] + score["counts"]["DE"], 0)
                self.assertEqual(score["counts"]["PER_S"], 1)
                self.assertIsNone(score["metrics"]["der"])

    def test_distortion_marker_is_not_removed(self):
        detected = score_utterance(["R"], ["R*"], ["R*"])
        self.assertEqual(detected["counts"]["TR"], 1)
        self.assertEqual(detected["counts"]["CD"], 1)
        missed = score_utterance(["R"], ["R*"], ["R"])
        self.assertEqual(missed["counts"]["FA"], 1)
        self.assertEqual(missed["counts"]["PER_S"], 1)

    def test_explicit_gold_retains_errors_erased_by_normalization(self):
        event = {"kind": "substitution", "position": 0, "expected_phone": "AH", "observed_phone": "AH"}
        score = score_utterance(["AH"], ["AH"], ["AH"], [event])
        self.assertEqual(score["counts"]["FA"], 1)
        self.assertEqual(score["gold_event_source"], "expert_annotation")
        self.assertEqual(score["metrics"]["per"], 0)

    def test_gold_annotation_cannot_change_predicted_alignment(self):
        a = score_utterance(["AA", "AA"], ["AA"], ["AA"])
        b = score_utterance(["AA", "AA"], ["T", "AA", "AA"], ["AA"])
        self.assertEqual(a["predicted_events"], b["predicted_events"])
        self.assertEqual(a["alignments"]["canonical_to_predicted"], b["alignments"]["canonical_to_predicted"])

    def test_per_counts_all_edit_types_and_empty_denominator(self):
        cases = [(["AA", "B"], ["AA"], (0, 1, 0, 2)),
                 (["AA"], ["AA", "B"], (0, 0, 1, 1)),
                 (["AA"], ["B"], (1, 0, 0, 1))]
        for perceived, predicted, expected in cases:
            score = score_utterance(perceived, perceived, predicted)
            self.assertEqual(tuple(score["counts"][f"PER_{key}"] for key in "SDIN"), expected)
        score = score_utterance([], [], ["AA"])
        self.assertEqual(score["counts"]["PER_I"], 1)
        self.assertIsNone(score["metrics"]["per"])
        json.dumps(score, allow_nan=False)

    def test_word_flags_count_once_and_boundary_insertions_map_right(self):
        score = score_utterance(["TH", "IH", "S", "IH", "Z"],
                                ["T", "IY", "S", "IH", "Z"],
                                ["T", "IH", "S", "T", "IH", "Z"],
                                canonical_word_indices=[0, 0, 0, 1, 1])
        self.assertEqual({k: score["counts"][k] for k in ("WORD_TP", "WORD_FP", "WORD_FN", "WORD_TN")},
                         {"WORD_TP": 1, "WORD_FP": 1, "WORD_FN": 0, "WORD_TN": 0})
        self.assertEqual(score["metrics"]["word_f1"], 2 / 3)

    def test_unknown_word_mapping_reports_coverage(self):
        score = score_utterance(["TH", "IH"], ["T", "IH"], ["T", "IH"],
                                canonical_word_indices=[None, 0])
        self.assertEqual(score["counts"]["WORD_UNMAPPED_EVENTS"], 2)
        self.assertEqual(score["counts"]["WORD_TN"], 1)
        self.assertEqual(score["counts"]["WORD_SCORED_UTTERANCES"], 1)
        self.assertIsNone(score["metrics"]["word_f1"])

    def test_aggregate_uses_micro_counters_not_mean_f1(self):
        good = score_utterance(["TH"], ["T"], ["T"])
        bad = score_utterance(["TH", "TH", "TH"], ["T", "T", "T"], ["TH", "TH", "TH"])
        combined = aggregate_scores([good, bad])
        self.assertEqual(combined["counts"]["UTTERANCES"], 2)
        self.assertEqual(combined["metrics"]["f1"], 2 / 5)
        self.assertEqual(combined["metrics"]["recall"], 1 / 4)
        self.assertIsNone(aggregate_scores([])["metrics"]["f1"])

    def test_rejects_invalid_inputs(self):
        invalid_events = [
            [{"kind": "substitution", "position": 1, "observed_phone": "T"}],
            [{"kind": "addition", "position": -1, "observed_phone": "T"}],
            [{"kind": "addition", "position": 0, "expected_phone": "AA", "observed_phone": "T"}],
            [{"kind": "deletion", "position": 0, "observed_phone": "T"}],
            [{"kind": "substitution", "position": 0, "expected_phone": "B", "observed_phone": "T"}],
            [{"kind": "substitution", "position": 0, "observed_phone": "T"}] * 2,
        ]
        for events in invalid_events:
            with self.subTest(events=events), self.assertRaises(ValueError):
                score_utterance(["AA"], ["T"], ["T"], events)
        with self.assertRaises(ValueError):
            align_phones("AA", ["AA"])
        with self.assertRaises(ValueError):
            score_utterance(["AA"], ["AA"], ["AA"], canonical_word_indices=[])
        with self.assertRaises(ValueError):
            aggregate_scores([{"counts": {"TA": -1}}])


if __name__ == "__main__":
    unittest.main()
