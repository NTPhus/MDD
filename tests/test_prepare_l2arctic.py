"""Annotation semantics and source-integrity tests; no model or GPU required."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_l2arctic import (
    ARPABET, ROOT, VOCAB, normalize_phone, parse_annotation, parse_textgrid, prepare,
)


def textgrid(phones, words=None):
    words = words if words is not None else [(0, len(phones), "word")]
    tiers = []
    for index, (name, items) in enumerate((("words", words), ("phones", phones)), 1):
        intervals = "\n".join(
            f'        intervals [{number}]:\n            xmin = {start}\n'
            f'            xmax = {end}\n            text = "{label.replace(chr(34), chr(34)*2)}"'
            for number, (start, end, label) in enumerate(items, 1)
        )
        tiers.append(f'    item [{index}]:\n        class = "IntervalTier"\n'
                     f'        name = "{name}"\n        xmin = 0\n        xmax = {len(phones)}\n'
                     f'        intervals: size = {len(items)}\n{intervals}')
    return ('File type = "ooTextFile"\nObject class = "TextGrid"\n'
            f'xmin = 0\nxmax = {len(phones)}\ntiers? <exists>\nsize = 2\nitem []:\n'
            + "\n".join(tiers) + "\n")


def entry():
    return {"utterance_id": "EXAMPLE/utterance", "speaker_id": "EXAMPLE",
            "native_language": "Vietnamese", "sample_rate": 44100, "duration_sec": 20,
            "audio": {"path": "example.wav"}, "annotation": {"sha256": "example-hash"}}


class AnnotationTests(unittest.TestCase):
    def test_vocabulary_is_predeclared_and_has_ctc_blank(self):
        self.assertEqual(len(ARPABET), 40)
        self.assertEqual(len(VOCAB), 83)
        self.assertEqual(VOCAB["[PAD]"], 0)
        self.assertEqual(VOCAB["[UNK]"], 1)
        self.assertEqual(set(VOCAB.values()), set(range(len(VOCAB))))
        self.assertTrue(all(phone + "*" in VOCAB for phone in ARPABET))

    def test_normalization_preserves_deviation_and_unknowns(self):
        for raw in ("AA*1", "AA1*", " a a 0 * "):
            self.assertEqual(normalize_phone(raw), "AA*")
        self.assertEqual(normalize_phone("Ah1"), "AH")
        self.assertEqual(normalize_phone("ERR"), "err")
        for raw in ("sil", "SIL", "sp", "spn", "", "   "):
            self.assertIsNone(normalize_phone(raw))
        for raw in ("V8", "ER)", "D_", "W`", "invented-phone"):
            self.assertEqual(normalize_phone(raw), "[UNK]")

    def test_substitution_deletion_addition_build_actual_target(self):
        phones = [(0, 1, "sil"), (1, 2, "TH,T,s"), (2, 3, "IH1"),
                  (3, 4, "NG,sil,D"), (4, 5, "sil,K,a"), (5, 6, "R,R*,S")]
        result = parse_annotation(entry(), parse_textgrid(textgrid(phones)), "think")
        self.assertEqual(result["canonical_phones"], ["TH", "IH", "NG", "R"])
        self.assertEqual(result["perceived_phones"], ["T", "IH", "K", "R*"])
        self.assertEqual(result["target_phones"], result["perceived_phones"])
        self.assertEqual([error["kind"] for error in result["gold_error_events"]],
                         ["substitution", "deletion", "addition", "substitution"])
        self.assertEqual([error["position"] for error in result["gold_error_events"]], [0, 2, 3, 3])
        self.assertIsNone(result["gold_error_events"][1]["perceived_index"])
        self.assertIsNone(result["gold_error_events"][2]["canonical_index"])
        self.assertEqual(result["gold_error_events"][2]["perceived_index"], 2)

    def test_additions_at_both_edges_and_repeated_gap_preserve_order(self):
        phones = [(0, 1, "sil,K,a"), (1, 2, "sil,AH,a"), (2, 3, "T"), (3, 4, "sil,S,a")]
        result = parse_annotation(entry(), parse_textgrid(textgrid(phones)), "t")
        self.assertEqual(result["canonical_phones"], ["T"])
        self.assertEqual(result["perceived_phones"], ["K", "AH", "T", "S"])
        self.assertEqual([error["position"] for error in result["gold_error_events"]], [0, 0, 1])

    def test_human_error_event_is_kept_when_normalization_equals_phones(self):
        result = parse_annotation(entry(), parse_textgrid(textgrid([(0, 1, "AH0,AH1,s")])), "a")
        self.assertEqual(result["canonical_phones"], result["perceived_phones"])
        self.assertEqual(len(result["gold_error_events"]), 1)
        self.assertEqual(result["gold_error_events"][0]["raw_label"], "AH0,AH1,s")

    def test_err_and_oov_keep_auditable_raw_labels(self):
        result = parse_annotation(entry(), parse_textgrid(textgrid([(0, 1, "TH,err,s"),
                                                                  (1, 2, "V8")])), "the")
        self.assertEqual(result["perceived_phones"], ["err", "[UNK]"])
        self.assertEqual(result["unknown_phone_labels"], {"V8": 1})
        self.assertEqual(result["phone_intervals"][1]["raw_label"], "V8")

    def test_word_overlap_is_explicit_and_interword_addition_is_unassigned(self):
        phones = [(0, 1, "AH1"), (1, 2, "sil,S,a"), (2, 3, "T")]
        words = [(0, 1, 'a "word"'), (1, 2, ""), (2, 3, "two")]
        result = parse_annotation(entry(), parse_textgrid(textgrid(phones, words)), "a two")
        self.assertEqual(result["canonical_word_indices"], [0, 1])
        self.assertIsNone(result["gold_error_events"][0]["word_index"])
        self.assertEqual(result["word_intervals"][0]["text"], 'a "word"')

    def test_invalid_error_codes_and_semantics_fail(self):
        for label in ("T,D,wrong", "T,D", "T,D,d", "sil,D,s", "T,sil,a"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_annotation(entry(), parse_textgrid(textgrid([(0, 1, label)])), "test")

    def test_malformed_tiers_and_overlapping_intervals_fail(self):
        valid = textgrid([(0, 1, "T")])
        for broken in (valid.replace('name = "phones"', 'name = "other"'),
                       valid.replace("intervals: size = 1", "intervals: size = 2"),
                       textgrid([(0, 1, "T"), (0.5, 2, "D")])):
            with self.assertRaises(ValueError):
                parse_textgrid(broken)

    def test_source_hash_mismatch_fails_and_outputs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {"annotation": ("item.TextGrid", textgrid([(0, 1, "T,D,s")])),
                     "transcript": ("item.txt", "test"), "audio": ("item.wav", "fake audio")}
            record = entry()
            for key, (name, contents) in files.items():
                path = root / name
                path.write_text(contents, encoding="utf-8")
                record[key] = {"path": name, "sha256": hashlib.sha256(contents.encode()).hexdigest()}
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            first = prepare(manifest, root / "prepared", root=root, verify_audio=True)
            second = prepare(manifest, root / "prepared", root=root, verify_audio=True)
            self.assertEqual(first, second)
            self.assertTrue(first["audio_sha256_verified"])
            (root / "item.TextGrid").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                prepare(manifest, root / "prepared-again", root=root)

    def test_real_manual_annotation_semantics(self):
        manifest = ROOT / "data/protocol/l2arctic-v1/manifest.jsonl"
        if not manifest.exists():
            self.skipTest("Frozen corpus manifest is not installed")
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        record = next(record for record in records if record["utterance_id"] == "ABA/arctic_a0003")
        path = ROOT / record["annotation"]["path"]
        if not path.exists():
            self.skipTest("Raw corpus is not installed")
        result = parse_annotation(record, parse_textgrid(path.read_text()), "For the twentieth time")
        self.assertEqual(Counter(event["kind"] for event in result["gold_error_events"]),
                         {"addition": 1, "substitution": 2})
        self.assertEqual(result["canonical_phones"][-2:], ["D", "Z"])
        self.assertEqual(result["perceived_phones"][-2:], ["S", "S"])
        addition = result["gold_error_events"][0]
        self.assertEqual(addition["observed_phone"], "K")
        self.assertEqual(addition["word_index"], 5)


if __name__ == "__main__":
    unittest.main()
