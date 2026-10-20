"""Synthetic fixtures check leakage guards and the insertion denominator."""

from copy import deepcopy
import unittest

from scripts.audit_data_split import assert_group_disjoint, compare_groups, index_rows, normalize_prompt, read_speaker_metadata, select_rows, summarize


def fixture(speaker="A", prompt="arctic_a0003", audio="a" * 64):
    return {"utterance_id": f"{speaker}/{prompt}", "speaker_id": speaker,
            "audio_sha256": audio, "prompt_text": "A known sentence.",
            "native_language": "Vietnamese", "gender": "F", "duration_sec": 2.0,
            "canonical_phones": ["TH", "IH"], "perceived_phones": ["T", "IH", "S"],
            "gold_error_events": [{"kind": "substitution", "canonical_index": 0, "position": 0},
                                  {"kind": "addition", "canonical_index": None, "position": 2}]}


class SplitAuditTests(unittest.TestCase):
    def test_same_audio_under_different_names_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "audio_sha256"):
            assert_group_disjoint([fixture()], [fixture(speaker="B")])

    def test_speaker_reuse_is_rejected_without_shared_files(self):
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            assert_group_disjoint([fixture()], [fixture(prompt="other", audio="b" * 64)])

    def test_familiar_prompt_is_allowed_only_for_declared_speaker_scope(self):
        left, right = [fixture()], [fixture(speaker="B", audio="b" * 64)]
        assert_group_disjoint(left, right)
        with self.assertRaisesRegex(ValueError, "prompt_id"):
            assert_group_disjoint(left, right, ("speaker_id", "audio_sha256", "prompt_id"))

    def test_wording_variant_does_not_hide_seen_prompt_identity(self):
        left, right = fixture(), fixture(speaker="B", audio="b" * 64)
        right["prompt_text"] = "A differently worded known sentence."
        report = compare_groups([left], [right])
        self.assertEqual(report["prompt_id"]["shared_distinct"], 1)
        self.assertEqual(report["prompt_text"]["shared_distinct"], 0)

    def test_normalization_does_not_remove_punctuation(self):
        self.assertEqual(normalize_prompt(" A   sentence.\n"), "a sentence.")
        self.assertNotEqual(normalize_prompt("a sentence"), normalize_prompt("a sentence."))

    def test_insertion_is_not_counted_as_a_canonical_classification_error(self):
        report = summarize([fixture()])
        self.assertEqual(report["canonical_error_fraction"], 0.5)
        self.assertEqual(report["additions_per_canonical_phone"], 0.5)
        self.assertEqual(report["gold_error_counts"]["addition"], 1)

    def test_repeated_canonical_error_position_is_rejected(self):
        row = fixture()
        row["gold_error_events"].append(deepcopy(row["gold_error_events"][0]))
        with self.assertRaisesRegex(ValueError, "Repeated error"):
            summarize([row])

    def test_missing_or_repeated_split_ids_are_rejected(self):
        row = fixture()
        indexed = index_rows([row], "fixture")
        with self.assertRaisesRegex(ValueError, "absent"):
            select_rows(indexed, ["missing"])
        with self.assertRaisesRegex(ValueError, "repeated"):
            select_rows(indexed, [row["utterance_id"]] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_rows([row, row], "fixture")

    def test_speaker_table_ignores_the_readme_phoneme_table(self):
        readme = "|HQTV|M|Vietnamese|1132|150|\n|15|F|fee|F IY|fricative|\n"
        self.assertEqual(read_speaker_metadata(readme), {
            "HQTV": {"gender": "M", "native_language": "Vietnamese",
                     "full_wav_count": 1132, "manual_count": 150}})


if __name__ == "__main__":
    unittest.main()
