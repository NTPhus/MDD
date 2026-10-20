"""Synthetic checks for the inactive metadata-only split proposal."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.propose_balanced_split import TARGET_FOLDS, export_proposal, validate_design


def fixture_design():
    metadata = {"HQTV": {"gender": "M"}, "PNV": {"gender": "F"},
                "THV": {"gender": "F"}, "TLV": {"gender": "M"},
                "SOURCE_A": {"gender": "F"}, "SOURCE_B": {"gender": "M"}}
    manifest = {}
    for speaker in metadata:
        for index in range(150 if speaker in {"HQTV", "PNV", "THV", "TLV"} else 2):
            key = f"{speaker}/arctic_{index:04d}"
            manifest[key] = {"utterance_id": key, "speaker_id": speaker,
                             "native_language": "Arabic" if speaker.startswith("SOURCE") else "Vietnamese",
                             "audio_sha256": key}
    parent = {"source_train_speakers": ["SOURCE_A"], "source_validation_speakers": ["SOURCE_B"],
              "source_train_ids": sorted(key for key in manifest if key.startswith("SOURCE_A/")),
              "source_validation_ids": sorted(key for key in manifest if key.startswith("SOURCE_B/"))}
    proposed = {**deepcopy(parent), "folds": []}
    for index, design in enumerate(TARGET_FOLDS, 1):
        fold = {"fold": index}
        for role, speakers in zip(("train", "validation", "test"), design):
            fold[role + "_speakers"] = list(speakers)
            fold[role + "_ids"] = sorted(key for key, row in manifest.items() if row["speaker_id"] in speakers)
        proposed["folds"].append(fold)
    return manifest, parent, proposed, metadata


class BalancedProposalTests(unittest.TestCase):
    def test_valid_design_preserves_sources_and_balances_roles(self):
        manifest, parent, proposed, metadata = fixture_design()
        report = validate_design(manifest, parent, proposed, metadata)
        self.assertTrue(report["source_partitions_identical_to_parent"])
        self.assertEqual(report["role_counts"]["HQTV"], {"train": 2, "validation": 1, "test": 1})
        self.assertTrue(all(fold["counts"] == {"train": 300, "validation": 150, "test": 150}
                            for fold in report["folds"]))

    def test_source_changes_are_rejected_even_with_unchanged_counts(self):
        manifest, parent, proposed, metadata = fixture_design()
        proposed["source_train_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "changed frozen source"):
            validate_design(manifest, parent, proposed, metadata)

    def test_single_gender_train_pair_is_rejected(self):
        manifest, parent, proposed, metadata = fixture_design()
        metadata["THV"]["gender"] = "M"
        with self.assertRaisesRegex(ValueError, "one M and one F"):
            validate_design(manifest, parent, proposed, metadata)

    def test_reused_audio_is_rejected_across_source_and_target(self):
        manifest, parent, proposed, metadata = fixture_design()
        manifest["HQTV/arctic_0000"]["audio_sha256"] = "SOURCE_A/arctic_0000"
        with self.assertRaisesRegex(ValueError, "audio_sha256"):
            validate_design(manifest, parent, proposed, metadata)

    def test_unbalanced_outer_roles_are_rejected(self):
        manifest, parent, proposed, metadata = fixture_design()
        proposed["folds"][3] = {**deepcopy(proposed["folds"][0]), "fold": 4}
        with self.assertRaisesRegex(ValueError, "role assignment"):
            validate_design(manifest, parent, proposed, metadata)

    def test_duplicate_utterance_in_a_partition_is_rejected(self):
        manifest, parent, proposed, metadata = fixture_design()
        proposed["folds"][0]["train_ids"][1] = proposed["folds"][0]["train_ids"][0]
        with self.assertRaisesRegex(ValueError, "repeated"):
            validate_design(manifest, parent, proposed, metadata)

    def test_export_refuses_overwrite_and_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "proposal"
            with patch("scripts.propose_balanced_split.build_proposal", return_value=(
                    {"folds": []}, {"status": "proposed_not_active"})):
                export_proposal(temporary, output)
                export_proposal(temporary, output, verify_only=True)
                with self.assertRaises(FileExistsError):
                    export_proposal(temporary, output)
                (output / "splits.json").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    export_proposal(temporary, output, verify_only=True)


if __name__ == "__main__":
    unittest.main()
