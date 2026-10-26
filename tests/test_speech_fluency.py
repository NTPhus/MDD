"""Measured timing must not invent grades, silence, or precise missing word times."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.speech_fluency import assess_fluency


def audio(seconds=4):
    return np.full(round(seconds * 16000), .01, dtype=np.float32)


def intervals(*values):
    return [{"index": index, "text": text, "start_sec": start, "end_sec": end}
            for index, (text, start, end) in enumerate(values)]


class SpeechFluencyTests(unittest.TestCase):
    def test_recording_window_and_articulation_denominators_stay_distinct(self):
        result = assess_fluency(audio(4), "I like tea.", intervals(
            ("I", .5, .7), ("like", .8, 1.2), ("tea.", 1.7, 2.5)))
        metrics = result["metrics"]
        self.assertEqual(result["status"], "ready")
        self.assertEqual(metrics["word_count"], 3)
        self.assertEqual(metrics["recording_rate_wpm"], 45)
        self.assertEqual(metrics["speech_rate_wpm"], 90)
        self.assertEqual(metrics["articulation_rate_estimate_wpm"], 120)
        self.assertEqual(metrics["speech_window_duration_sec"], 2)
        self.assertEqual(metrics["articulation_duration_estimate_sec"], 1.5)
        self.assertEqual(metrics["internal_pause_duration_sec"], .5)
        self.assertEqual(metrics["leading_gap_sec"], .5)
        self.assertEqual(metrics["trailing_gap_sec"], 1.5)
        self.assertFalse(result["calibrated"])
        self.assertNotIn("score", result)

    def test_boundary_gaps_are_never_counted_as_internal_pauses(self):
        result = assess_fluency(audio(10), "Hello there", intervals(
            ("Hello", 4, 4.5), ("there", 4.6, 5)))
        metrics = result["metrics"]
        self.assertEqual(metrics["internal_pause_count"], 0)
        self.assertEqual(metrics["longest_internal_pause_sec"], 0)
        self.assertEqual(metrics["leading_gap_sec"], 4)
        self.assertEqual(metrics["trailing_gap_sec"], 5)

    def test_punctuation_distinguishes_sentence_and_mid_sentence_gap_heuristics(self):
        result = assess_fluency(audio(4), "Hello. I like tea!", intervals(
            ("Hello.", 0, .4), ("I", 1, 1.2), ("like", 1.6, 2), ("tea!", 2.1, 2.5)))
        self.assertEqual(result["metrics"]["internal_pause_count"], 2)
        self.assertEqual(result["metrics"]["mid_sentence_pause_count"], 1)
        self.assertEqual(result["metrics"]["sentence_boundary_pause_count"], 1)
        self.assertEqual([row["sentence_boundary"] for row in result["pauses"]], [True, False])

    def test_threshold_is_explicit_and_not_a_good_bad_fluency_rule(self):
        words = intervals(("I", 0, .5), ("agree", .75, 1))
        default = assess_fluency(audio(2), "I agree", words)
        changed = assess_fluency(audio(2), "I agree", words, pause_threshold_sec=.3)
        self.assertEqual(default["metrics"]["internal_pause_count"], 1)
        self.assertEqual(changed["metrics"]["internal_pause_count"], 0)
        self.assertNotIn("rating", default)
        self.assertNotIn("slow", json.dumps(default))

    def test_contractions_and_number_tokens_count_once_and_native_clitics_merge(self):
        native = [{"word": word, "start": start, "end": end}
                  for word, start, end in [(" I", 0, .2), (" don", .2, .4), ("’t", .4, .5),
                                           (" want", .5, .8), (" 2", .8, 1), (" teas.", 1, 1.3)]]
        result = assess_fluency(audio(2), "I don't want 2 teas.", native)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["metrics"]["word_count"], 5)
        self.assertEqual(result["metrics"]["internal_pause_count"], 0)

    def test_missing_or_mismatched_word_times_keep_only_recording_metrics(self):
        for words in (None, [], intervals(("wrong", 0, 1)), "not word intervals"):
            with self.subTest(words=words):
                result = assess_fluency(audio(2), "Hello world", words)
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["metrics"]["recording_rate_wpm"], 60)
                self.assertNotIn("internal_pause_count", result["metrics"])
                self.assertNotIn("speech_rate_wpm", result["metrics"])

    def test_one_native_piece_cannot_fabricate_two_word_boundaries(self):
        result = assess_fluency(audio(2), "ice tea", [{"word": " ice tea", "start": 0, "end": 1}])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reason"], "word_timing_spans_multiple_words")
        self.assertEqual(result["pauses"], [])

    def test_invalid_overlapping_outside_or_nonfinite_word_times_abstain(self):
        bad = [intervals(("Hello", 0, .8), ("world", .7, 1)),
               intervals(("Hello", -.1, .4), ("world", .5, 1)),
               intervals(("Hello", 0, .4), ("world", .5, 2.1)),
               intervals(("Hello", 0, .4), ("world", float("nan"), 1)),
               intervals(("Hello", 0, .4), ("world", True, 1))]
        for words in bad:
            with self.subTest(words=words):
                result = assess_fluency(audio(2), "Hello world", words)
                self.assertEqual(result["status"], "partial")
                self.assertNotIn("longest_internal_pause_sec", result["metrics"])

    def test_digital_silence_never_turns_a_hallucinated_transcript_into_speech(self):
        result = assess_fluency(np.zeros(16000), "Thank you", intervals(("Thank", 0, .2), ("you", .2, .5)))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "digital_silence")
        self.assertNotIn("recording_rate_wpm", result["metrics"])

    def test_invalid_audio_rate_or_transcript_has_no_fabricated_measurements(self):
        cases = [(np.zeros((2, 2)), "hi", 16000), ([], "hi", 16000), ([float("inf")], "hi", 16000),
                 ([.1], "hi", 0), ([.1], "hi", True), ([.1], None, 16000), ([.1], "...", 16000)]
        for waveform, text, rate in cases:
            with self.subTest(text=text, rate=rate):
                result = assess_fluency(waveform, text, [], sample_rate=rate)
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("recording_rate_wpm", result["metrics"])
                json.dumps(result, allow_nan=False)

    def test_zero_timing_window_has_no_infinite_rate(self):
        result = assess_fluency(audio(2), "Hello", intervals(("Hello", .5, .5)))
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("speech_rate_wpm", result["metrics"])
        json.dumps(result, allow_nan=False)

    def test_text_observations_never_claim_stuttering_or_grammar_errors(self):
        result = assess_fluency(audio(4), "Um I I said that that is fine. Fine!", [])
        lexical = result["lexical_heuristics"]
        self.assertEqual(lexical["observed_filler_token_count"], 1)
        self.assertEqual(lexical["adjacent_duplicate_pair_count"], 2)
        self.assertFalse(lexical["repetitions_are_errors"])
        self.assertEqual(lexical["source"], "transcript_only")

    def test_very_quiet_nonzero_audio_is_not_arbitrarily_rejected(self):
        result = assess_fluency(np.full(16000, 1e-6), "Hello", intervals(("Hello", .1, .9)))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["metrics"]["word_count"], 1)


if __name__ == "__main__":
    unittest.main()
