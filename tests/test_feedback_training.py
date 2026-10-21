"""Review gates, grouped held-out data and assistant-only training supervision."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.llm_client import ROOT
from scripts.prepare_feedback_data import load_partitions, prepare, read_rows, split_rows, write_rows
from scripts.train_feedback_model import collate_pairs, encode_pair


class FeedbackTrainingTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "data/feedback/review.example.jsonl").read_text())
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def row(self, index):
        row = deepcopy(self.example)
        identifier = f"sample-{index}"
        row["assessment"]["utterance_id"] = row["feedback"]["utterance_id"] = identifier
        row["group_id"] = f"family-{index}"
        row["assessment"]["reference_text"] += f" {index}"
        for segment in row["feedback"]["tts_segments"]:
            if segment["purpose"] == "reference":
                segment["text"] = row["assessment"]["reference_text"]
        row["review"] = {"approved": True, "reviewer": "unit-test-only", "reviewed_at": "2026-10-21"}
        return row

    def test_unreviewed_drafts_cannot_become_training_labels(self):
        path = self.root / "drafts.jsonl"
        write_rows(path, [self.example])
        with self.assertRaisesRegex(ValueError, "teacher review"):
            read_rows(path)

    def test_frozen_grouped_split_has_no_prompt_leakage_and_detects_changes(self):
        rows = [self.row(index) for index in range(12)]
        rows[1]["assessment"]["reference_text"] = rows[0]["assessment"]["reference_text"]
        for segment in rows[1]["feedback"]["tts_segments"]:
            if segment["purpose"] == "reference":
                segment["text"] = rows[1]["assessment"]["reference_text"]
        source, destination = self.root / "reviewed.jsonl", self.root / "dataset"
        write_rows(source, rows)
        prepare(source, destination)
        parts, _ = load_partitions(destination)
        locations = {row["assessment"]["utterance_id"]: name for name, values in parts.items() for row in values}
        self.assertEqual(locations["sample-0"], locations["sample-1"])
        with (destination / "train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_partitions(destination)

    def test_duplicate_ids_and_too_few_groups_are_rejected(self):
        rows = [self.row(index) for index in range(3)]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_rows(rows + [deepcopy(rows[0])])
        with self.assertRaisesRegex(ValueError, "three independent"):
            split_rows(rows[:2])

    def test_training_masks_prompt_and_padding_without_truncating_targets(self):
        class Tokenizer:
            eos_token = "!"

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is False
                return "ABC"

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(char) for char in text]}

        row = self.row(0)
        encoded = encode_pair(Tokenizer(), row, 4096)
        self.assertEqual(encoded["labels"][:3], [-100] * 3)
        self.assertEqual(encoded["labels"][3:], encoded["input_ids"][3:])
        self.assertEqual(encoded["labels"][-1], ord("!"))
        short = {key: value[:-3] for key, value in encoded.items()}
        batch = collate_pairs([encoded, short], 0)
        self.assertEqual(batch["labels"][1, -3:].tolist(), [-100] * 3)
        self.assertEqual(batch["attention_mask"][1, -3:].tolist(), [0] * 3)
        with self.assertRaisesRegex(ValueError, "do not truncate"):
            encode_pair(Tokenizer(), row, 4)


if __name__ == "__main__":
    unittest.main()
