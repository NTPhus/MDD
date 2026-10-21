"""Synthetic fixtures check leakage guards and the insertion denominator."""

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import uuid

from scripts import audit_data_split
from scripts.audit_data_split import assert_group_disjoint, compare_groups, index_rows, normalize_prompt, read_speaker_metadata, select_rows, summarize


def fixture(speaker="A", prompt="arctic_a0003", audio="a" * 64):
    return {"utterance_id": f"{speaker}/{prompt}", "speaker_id": speaker,
            "audio_sha256": audio, "prompt_text": "A known sentence.",
            "native_language": "Vietnamese", "gender": "F", "duration_sec": 2.0,
            "canonical_phones": ["TH", "IH"], "perceived_phones": ["T", "IH", "S"],
            "gold_error_events": [{"kind": "substitution", "canonical_index": 0, "position": 0},
                                  {"kind": "addition", "canonical_index": None, "position": 2}]}


@contextmanager
def utf8_audit_fixture():
    parent = Path(tempfile.gettempdir()).resolve()
    root = parent / ("mdd-audit-utf8-" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        resolved = root.resolve()
        if resolved.parent != parent or not resolved.name.startswith("mdd-audit-utf8-"):
            raise RuntimeError("Refusing cleanup outside the audit fixture")
        shutil.rmtree(resolved)


class SplitAuditTests(unittest.TestCase):
    def test_utf8_frozen_fixture_is_audited_when_locale_defaults_to_cp1252(self):
        with utf8_audit_fixture() as directory:
            root = Path(directory)

            def write_text(relative, text):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                return path

            def write_json(relative, value):
                return write_text(relative, json.dumps(value, ensure_ascii=False) + "\n")

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def source_file(relative, text):
                path = write_text(relative, text)
                return {"path": relative, "sha256": digest(path), "size_bytes": path.stat().st_size}

            # Closing curly quote has UTF-8 byte 0x9d, undefined in CP1252.
            # Vietnamese text also checks exact decoding rather than mojibake.
            prompt = "Xin chào người học.”"
            manifest, annotations = [], []
            speakers = {"SRC_A": "Arabic", "SRC_B": "Arabic",
                        "VN_A": "Vietnamese", "VN_B": "Vietnamese", "VN_C": "Vietnamese"}
            for speaker, language in speakers.items():
                stem = f"l2arctic_release_v5.0/{speaker}/arctic_a0003"
                raw = {name: source_file(stem + suffix, text) for name, suffix, text in (
                    ("audio", ".wav", f"synthetic audio bytes {speaker}"),
                    ("annotation", ".TextGrid", f"manual annotation {speaker}”"),
                    ("transcript", ".txt", prompt + "\n"))}
                row = fixture(speaker=speaker)
                row.update(native_language=language, sample_rate=16000,
                           audio_path=raw["audio"]["path"], prompt_text=prompt,
                           source_annotation_sha256=raw["annotation"]["sha256"])
                annotations.append(row)
                manifest.append({"utterance_id": row["utterance_id"], "speaker_id": speaker,
                                 "native_language": language, "sample_rate": 16000,
                                 "duration_sec": row["duration_sec"], "annotation_is_manual": True, **raw})

            protocol_dir = "data/protocol/l2arctic-v1"
            processed_dir = "data/processed/l2arctic-v1"
            manifest_path = write_text(f"{protocol_dir}/manifest.jsonl", "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
            write_text(f"{processed_dir}/annotations.jsonl", "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in annotations))
            splits = {"source_train_ids": ["SRC_A/arctic_a0003"],
                      "source_validation_ids": ["SRC_B/arctic_a0003"],
                      "source_train_speakers": ["SRC_A"], "source_validation_speakers": ["SRC_B"],
                      "folds": [{"fold": 1, "train_ids": ["VN_A/arctic_a0003"],
                                 "validation_ids": ["VN_B/arctic_a0003"], "test_ids": ["VN_C/arctic_a0003"],
                                 "train_speakers": ["VN_A"], "validation_speakers": ["VN_B"],
                                 "test_speakers": ["VN_C"]}]}
            splits_path = write_json(f"{protocol_dir}/splits.json", splits)
            write_json(f"{protocol_dir}/protocol.json", {
                "protocol_id": "synthetic-utf8-regression", "utterance_count": len(manifest),
                "counts_by_speaker": dict.fromkeys(speakers, 1),
                "manifest_sha256": digest(manifest_path), "splits_sha256": digest(splits_path)})
            for name in ("audit", "normalization", "vocab"):
                write_json(f"{processed_dir}/{name}.json", {})
            for base, names in ((protocol_dir, ("manifest.jsonl", "splits.json", "protocol.json")),
                                (processed_dir, ("annotations.jsonl", "audit.json", "normalization.json", "vocab.json"))):
                write_text(f"{base}/SHA256SUMS", "".join(
                    f"{digest(root / base / name)}  {name}\n" for name in names))
            write_text("l2arctic_release_v5.0/README.md", "Dữ liệu minh họa ”\n" + "".join(
                f"|{speaker}|F|{language}|1|1|\n" for speaker, language in speakers.items()))
            script = write_text("scripts/audit_data_split.py", "# synthetic fixture\n")

            original_text_encoding = io.text_encoding

            def cp1252_locale_default(encoding, stacklevel=2):
                return "cp1252" if encoding is None else original_text_encoding(encoding, stacklevel)

            with patch.object(audit_data_split, "__file__", str(script)), \
                    patch("io.text_encoding", side_effect=cp1252_locale_default):
                report = audit_data_split.build_audit(root)

            self.assertEqual(report["global"]["utterances"], 5)
            self.assertEqual(report["global"]["distinct_normalized_prompt_texts"], 1)
            self.assertEqual(report["raw_files_verified_against_manifest"],
                             {"audio": 5, "annotation": 5, "transcript": 5})
            self.assertTrue(report["all_speaker_utterance_audio_hash_disjointness_checks_passed"])

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
