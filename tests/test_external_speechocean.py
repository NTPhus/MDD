"""Hand-checked external-label fixtures; no model, dataset, or provider calls."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import wave

from scripts.evaluate_external_speechocean import (
    derive_metrics, normalize_phone, parse_expert_word, parse_human_labels, read_verified_audio, score_prediction,
)


def labels_for_words(words):
    """Each fixture word is (text, canonical phones, five expert strings)."""
    text = " ".join(word for word, _phones, _experts in words)
    score = {"text": text, "words": [
        {"text": word, "phones": phones, "phones-accuracy": [2] * len(phones)}
        for word, phones, _experts in words
    ]}
    detail = {"text": text, "words": [
        {"text": word, "ref-phones": " ".join(phones), "phones": experts}
        for word, phones, experts in words
    ]}
    return parse_human_labels(score, detail, text)


def mixed_human_labels():
    # TH has only two error votes; AH and R have three. AX is acceptable
    # accented pronunciation (1) for every expert, while K is always correct.
    return labels_for_words([
        ("alpha", ["TH", "AH"], ["(TH AH)", "(TH AH)", "TH (AH)", "TH AH", "TH AH"]),
        ("beta", ["AX", "K"], ["{AX} K"] * 5),
        ("gamma", ["R"], ["(R)"] * 3 + ["R"] * 2),
    ])


def empty_counts():
    return dict.fromkeys((
        "position_tp", "position_fp", "position_fn", "position_tn",
        "unanimous_correct_positions", "unanimous_correct_rejections",
        "unanimous_acceptable_positions", "unanimous_acceptable_rejections",
        "predicted_insertions", "reference_phones", "reference_words",
        "word_tp", "word_fp", "word_fn", "word_tn",
        "strict_perfect_utterances", "strict_perfect_sentence_flags",
        "strict_perfect_reference_phones", "strict_perfect_phone_rejections",
        "strict_perfect_reference_words", "strict_perfect_word_flags",
    ), 0)


class ExternalPhoneParsingTests(unittest.TestCase):
    def test_stress_is_removed_only_at_the_end_and_ax_and_distortion_stay_distinct(self):
        for source, expected in (("ah0", "AH"), ("AH12", "AH"), ("AX1", "AX"), ("R*2", "R*"), (" TH ", "TH")):
            with self.subTest(source=source):
                self.assertEqual(normalize_phone(source), expected)
        self.assertNotEqual(normalize_phone("AH0"), normalize_phone("AX0"))
        self.assertNotEqual(normalize_phone("R"), normalize_phone("R*"))
        for source in ("A1H", "AH0*", "TH3", "UNKNOWN", "[UNK]", ""):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    normalize_phone(source)

    def test_groups_rate_every_canonical_phone_and_insertions_do_not_consume_positions(self):
        result = parse_expert_word("[Y Z] TH {IH NG} (K) [S]", ["TH", "IH", "NG", "K"])
        self.assertEqual(result["ratings"], [2, 1, 1, 0])
        self.assertEqual(result["insertions"], [
            {"position": 0, "phone": "Y"}, {"position": 0, "phone": "Z"}, {"position": 4, "phone": "S"},
        ])

    def test_expert_reference_comparison_normalizes_stress_without_collapsing_ax(self):
        result = parse_expert_word("AH0 {AX1} (R*2)", ["AH", "AX", "R*"])
        self.assertEqual(result["ratings"], [2, 1, 0])
        for annotation, phones in (("AH AH R*", ["AH", "AX", "R*"]), ("TH IH", ["TH", "IH", "NG"]),
                                   ("TH IH NG K", ["TH", "IH", "NG"]), ("TH UNKNOWN", ["TH", "IH"])):
            with self.subTest(annotation=annotation, phones=phones):
                with self.assertRaises(ValueError):
                    parse_expert_word(annotation, phones)

    def test_two_vs_three_experts_and_acceptable_accent_votes_are_kept_separate(self):
        human = mixed_human_labels()
        self.assertEqual(human["canonical_phones"], ["TH", "AH", "AX", "K", "R"])
        self.assertEqual(human["canonical_word_indices"], [0, 0, 1, 1, 2])
        self.assertEqual(human["phone_ratings"], [
            [0, 0, 2, 2, 2], [0, 0, 0, 2, 2], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [0, 0, 0, 2, 2],
        ])
        self.assertEqual(human["majority_phone_error"], [False, True, False, False, True])
        self.assertEqual(human["word_error_votes"], [3, 0, 3])
        self.assertEqual(human["majority_word_error"], [True, False, True])
        self.assertFalse(human["strict_perfect"])

    def test_ratings_come_from_five_experts_not_the_aggregate_score(self):
        score = {"text": "CAT", "words": [{"text": "CAT", "phones": ["K", "AE1", "T"], "phones-accuracy": [0, 0, 0]}]}
        detail = {"text": "CAT", "words": [{"text": "CAT", "ref-phones": "K AE1 T", "phones": ["K AE1 T"] * 5}]}
        human = parse_human_labels(score, detail, "CAT")
        self.assertEqual(human["phone_ratings"], [[2] * 5] * 3)
        self.assertEqual(human["majority_phone_error"], [False] * 3)
        self.assertTrue(human["strict_perfect"])

    def test_sentence_punctuation_can_differ_but_nested_word_identity_and_boundaries_cannot(self):
        score = {"text": "THE SAME.I.", "words": [
            {"text": "THE", "phones": ["DH", "AH0"], "phones-accuracy": [2, 2]},
            {"text": "SAME", "phones": ["S", "EY1", "M"], "phones-accuracy": [2, 2, 2]},
            {"text": "I", "phones": ["AY1"], "phones-accuracy": [2]},
        ]}
        detail = {"text": "THE SAME.I.", "words": [
            {"text": "THE", "ref-phones": "DH AH0", "phones": ["DH AH0"] * 5},
            {"text": "SAME", "ref-phones": "S EY1 M", "phones": ["S EY1 M"] * 5},
            {"text": "I", "ref-phones": "AY1", "phones": ["AY1"] * 5},
        ]}
        human = parse_human_labels(score, detail, "THE SAME I")
        self.assertEqual(human["canonical_phones"], ["DH", "AH", "S", "EY", "M", "AY"])
        self.assertEqual(human["canonical_word_indices"], [0, 0, 1, 1, 1, 2])
        self.assertTrue(human["strict_perfect"])

        for changed_source in ("score", "detail"):
            changed_score, changed_detail = deepcopy(score), deepcopy(detail)
            target = changed_score if changed_source == "score" else changed_detail
            target["words"][1]["text"] = "OTHER"
            with self.subTest(changed_source=changed_source):
                with self.assertRaises(ValueError):
                    parse_human_labels(changed_score, changed_detail, "THE SAME I")

        for changed_boundary in ("shifted", "merged"):
            changed_score, changed_detail = deepcopy(score), deepcopy(detail)
            for target in (changed_score, changed_detail):
                if changed_boundary == "shifted":
                    target["words"][1]["text"], target["words"][2]["text"] = "SAM", "EI"
                else:
                    target["words"][1]["text"] = "SAMEI"
                    target["words"].pop()
            with self.subTest(changed_boundary=changed_boundary):
                with self.assertRaises(ValueError):
                    parse_human_labels(changed_score, changed_detail, "THE SAME I")

    def test_word_phone_and_expert_alignment_mismatches_fail_instead_of_shifting_labels(self):
        score = {"text": "CAT DOG", "words": [
            {"text": "CAT", "phones": ["K", "AE1", "T"]}, {"text": "DOG", "phones": ["D", "AO1", "G"]},
        ]}
        detail = {"text": "CAT DOG", "words": [
            {"text": "CAT", "ref-phones": "K AE1 T", "phones": ["K AE1 T"] * 5},
            {"text": "DOG", "ref-phones": "D AO1 G", "phones": ["D AO1 G"] * 5},
        ]}
        bad_score = deepcopy(score)
        bad_score["words"][0]["phones"] = ["T"]
        bad_detail = deepcopy(detail)
        bad_detail["words"][1]["phones"] = ["D AO1 G"] * 4
        for current_score, current_detail, text in ((bad_score, detail, "CAT DOG"), (score, bad_detail, "CAT DOG"),
                                                    (score, detail, "CAT OTHER")):
            with self.subTest(text=text, score=current_score, detail=current_detail):
                with self.assertRaises(ValueError):
                    parse_human_labels(current_score, current_detail, text)


class ExternalDetectionTests(unittest.TestCase):
    def test_hand_checked_detection_counts_include_false_alarms_and_misses(self):
        result = score_prediction(mixed_human_labels(), ["T", "AH", "AX", "G", "L"])
        counts = result["counts"]
        expected = {
            "position_tp": 1, "position_fp": 2, "position_fn": 1, "position_tn": 1,
            "unanimous_correct_positions": 1, "unanimous_correct_rejections": 1,
            "unanimous_acceptable_positions": 2, "unanimous_acceptable_rejections": 1,
            "word_tp": 2, "word_fp": 1, "word_fn": 0, "word_tn": 0,
            "predicted_insertions": 0, "reference_phones": 5, "reference_words": 3,
            "strict_perfect_utterances": 0,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(counts[key], value)
        for phone, key in (("TH", "position_fp"), ("AH", "position_fn"), ("AX", "position_tn"),
                           ("K", "position_fp"), ("R", "position_tp")):
            with self.subTest(phone=phone):
                self.assertEqual(result["per_phone"][phone][key], 1)
        metrics = derive_metrics(counts)
        self.assertAlmostEqual(metrics["canonical_position_detection"]["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["canonical_position_detection"]["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["canonical_position_detection"]["f1"], 0.4)

    def test_expert_insertions_vote_for_word_error_without_creating_canonical_phone_labels(self):
        human = labels_for_words([
            ("A", ["AA"], ["AA"] * 5),
            ("B", ["B"], ["[Z] B"] * 3 + ["B"] * 2),
        ])
        self.assertEqual(human["phone_ratings"], [[2] * 5, [2] * 5])
        self.assertEqual(human["majority_phone_error"], [False, False])
        self.assertEqual(human["word_error_votes"], [0, 3])
        self.assertEqual(human["majority_word_error"], [False, True])
        self.assertFalse(human["strict_perfect"])
        result = score_prediction(human, ["AA", "Z", "B"])
        self.assertEqual(result["counts"]["position_tn"], 2)
        self.assertEqual(result["counts"]["position_fp"], 0)
        self.assertEqual(result["counts"]["predicted_insertions"], 1)
        self.assertEqual(result["counts"]["word_tp"], 1)
        self.assertEqual(result["counts"]["word_tn"], 1)
        addition = [event for event in result["events"] if event["kind"] == "addition"]
        self.assertEqual(len(addition), 1)
        self.assertEqual(addition[0]["position"], 1)
        self.assertEqual(addition[0]["word_index"], 1, "boundary additions belong to the following word")

    def test_strict_perfect_sentence_counts_end_insertion_as_a_flag_without_phone_rejection(self):
        human = labels_for_words([("A", ["AA"], ["AA"] * 5), ("B", ["B"], ["B"] * 5)])
        self.assertTrue(human["strict_perfect"])
        result = score_prediction(human, ["AA", "B", "Z"])
        counts = result["counts"]
        for key, expected in {
            "strict_perfect_utterances": 1, "strict_perfect_sentence_flags": 1,
            "strict_perfect_reference_phones": 2, "strict_perfect_phone_rejections": 0,
            "strict_perfect_reference_words": 2, "strict_perfect_word_flags": 1,
            "word_fp": 1, "word_tn": 1, "predicted_insertions": 1,
        }.items():
            with self.subTest(key=key):
                self.assertEqual(counts[key], expected)
        addition = [event for event in result["events"] if event["kind"] == "addition"]
        self.assertEqual(addition[0]["word_index"], 1, "utterance-end additions belong to the final word")

    def test_two_expert_insertions_do_not_form_a_word_error_majority(self):
        human = labels_for_words([("A", ["AA"], ["[Z] AA"] * 2 + ["AA"] * 3)])
        self.assertEqual(human["word_error_votes"], [2])
        self.assertEqual(human["majority_word_error"], [False])
        self.assertFalse(human["strict_perfect"], "strict perfect excludes any expert insertion")
        counts = score_prediction(human, ["Z", "AA"])["counts"]
        self.assertEqual(counts["word_fp"], 1)
        self.assertEqual(counts["word_tp"], 0)
        self.assertEqual(counts["position_tn"], 1)

    def test_all_acceptable_accented_phones_are_not_strict_perfect_or_majority_errors(self):
        human = labels_for_words([("A", ["AX"], ["{AX}"] * 5)])
        self.assertFalse(human["strict_perfect"])
        self.assertEqual(human["majority_phone_error"], [False])
        counts = score_prediction(human, ["AH"])["counts"]
        self.assertEqual(counts["position_fp"], 1)
        self.assertEqual(counts["unanimous_correct_positions"], 0)
        self.assertEqual(counts["unanimous_acceptable_positions"], 1)
        self.assertEqual(counts["unanimous_acceptable_rejections"], 1)
        self.assertEqual(counts["strict_perfect_utterances"], 0)

    def test_deleting_a_human_error_is_detected_but_an_accepted_deletion_is_a_false_alarm(self):
        human = labels_for_words([("A", ["AA", "B"], ["(AA) B"] * 5)])
        detected = score_prediction(human, ["B"])["counts"]
        self.assertEqual(detected["position_tp"], 1)
        self.assertEqual(detected["position_tn"], 1)
        false_alarm = score_prediction(human, ["AA"])["counts"]
        self.assertEqual(false_alarm["position_fn"], 1)
        self.assertEqual(false_alarm["position_fp"], 1)


class ExternalMetricTests(unittest.TestCase):
    def test_rates_use_their_named_denominators_and_hand_checked_wilson_interval(self):
        counts = empty_counts()
        counts.update(position_tp=3, position_fp=1, position_fn=2, position_tn=4,
                      unanimous_correct_positions=10, unanimous_correct_rejections=2,
                      unanimous_acceptable_positions=12, unanimous_acceptable_rejections=3,
                      word_fp=2, word_tn=8, strict_perfect_utterances=4, strict_perfect_sentence_flags=1,
                      predicted_insertions=3, reference_phones=40, reference_words=10)
        metrics = derive_metrics(counts)
        detection = metrics["canonical_position_detection"]
        self.assertAlmostEqual(detection["precision"], 0.75)
        self.assertAlmostEqual(detection["recall"], 0.6)
        self.assertAlmostEqual(detection["f1"], 2 / 3)
        for name, numerator, denominator in (
            ("unanimous_correct_position_rejection_rate", 2, 10),
            ("unanimous_acceptable_position_rejection_rate", 3, 12),
            ("majority_accepted_word_false_positive_rate", 2, 10),
            ("strict_perfect_sentence_any_flag_rate", 1, 4),
        ):
            with self.subTest(name=name):
                self.assertEqual(metrics[name]["numerator"], numerator)
                self.assertEqual(metrics[name]["denominator"], denominator)
                self.assertAlmostEqual(metrics[name]["rate"], numerator / denominator)
        interval = metrics["unanimous_correct_position_rejection_rate"]["wilson95"]
        self.assertEqual(len(interval), 2)
        self.assertAlmostEqual(interval[0], 0.0566821515, places=5)
        self.assertAlmostEqual(interval[1], 0.5098375285, places=5)
        self.assertAlmostEqual(metrics["predicted_insertions_per_100_reference_phones"], 7.5)
        self.assertNotIn("per", metrics, "accuracy ratings do not provide a perceived-phone PER reference")
        self.assertNotIn("der", metrics, "accuracy ratings do not provide substitution diagnosis truth")

    def test_empty_denominators_remain_undefined_instead_of_becoming_perfect_scores(self):
        metrics = derive_metrics(empty_counts())
        self.assertEqual(metrics["canonical_position_detection"], {"precision": None, "recall": None, "f1": None})
        for name in ("unanimous_correct_position_rejection_rate", "unanimous_acceptable_position_rejection_rate",
                     "majority_accepted_word_false_positive_rate", "strict_perfect_sentence_any_flag_rate"):
            with self.subTest(name=name):
                self.assertIsNone(metrics[name]["rate"])
                self.assertIsNone(metrics[name]["wilson95"])
                self.assertEqual(metrics[name]["denominator"], 0)
        self.assertIsNone(metrics["predicted_insertions_per_100_reference_phones"])


class ExternalAudioIntegrityTests(unittest.TestCase):
    def test_exact_audio_bytes_are_pinned_and_valid_wave_changes_cannot_bypass_verification(self):
        def wav_bytes(samples):
            stream = io.BytesIO()
            with wave.open(stream, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(samples)
            return stream.getvalue()

        original = wav_bytes(b"\x00\x00" * 16)
        changed = wav_bytes(b"\x00\x00" * 15 + b"\x01\x00")
        longer = wav_bytes(b"\x00\x00" * 17)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fixture:
            path = Path(fixture.name)
            fixture.write(original)
        self.addCleanup(path.unlink, missing_ok=True)
        row = {"audio_path": str(path), "expected_audio_bytes": len(original),
               "expected_audio_sha256": hashlib.sha256(original).hexdigest()}
        content, digest = read_verified_audio(row)
        self.assertEqual(content, original)
        self.assertEqual(digest, row["expected_audio_sha256"])

        self.assertEqual(len(changed), len(original), "the altered PCM remains a valid WAV of identical length")
        path.write_bytes(changed)
        with self.assertRaises(ValueError):
            read_verified_audio(row)

        path.write_bytes(longer)
        # Match the new digest to isolate the independent byte-length check.
        wrong_length_row = {**row, "expected_audio_sha256": hashlib.sha256(longer).hexdigest()}
        with self.assertRaises(ValueError):
            read_verified_audio(wrong_length_row)


if __name__ == "__main__":
    unittest.main()
