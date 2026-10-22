"""Drafts stay unapproved; only explicit validated human edits can be saved."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.llm_client import ROOT, validate_local_feedback
from scripts.create_feedback_review_drafts import template_feedback
from scripts.prepare_feedback_data import read_rows
from scripts.review_feedback_drafts import approve_one
from scripts.validate_contracts import validate_pair


class FakeTokenizer:
    eos_token = "!"
    def apply_chat_template(self, messages, **kwargs): return "PROMPT"
    def __call__(self, text, **kwargs): return {"input_ids": [ord(character) for character in text]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT / "data/feedback/review.example.jsonl").read_text(encoding="utf-8"))
        self.original["feedback"] = template_feedback(self.original["assessment"])
        temporary_root = ROOT / "data/generated"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "reviewed.jsonl"

    def save(self, original=None, **changes):
        original = self.original if original is None else original
        kwargs = dict(original=original, assessment=deepcopy(original["assessment"]), feedback=deepcopy(original["feedback"]),
                      reviewer="Unit test fixture only", destination=self.destination, tokenizer=FakeTokenizer())
        kwargs.update(changes)
        return approve_one(**kwargs)

    def test_template_is_valid_grounded_and_does_not_approve_source(self):
        validate_pair(self.original["assessment"], self.original["feedback"])
        validate_local_feedback(self.original["assessment"], self.original["feedback"])
        self.assertFalse(self.original["review"]["approved"])
        self.assertEqual({tip["error_id"] for tip in self.original["feedback"]["tips"]}, {"e1"})
        self.assertEqual([s["purpose"] for s in self.original["feedback"]["tts_segments"]], ["reference", "feedback"])

    def test_blank_reviewer_invalid_error_or_changed_identity_never_writes(self):
        bad_feedback = deepcopy(self.original["feedback"])
        bad_feedback["tips"][0]["error_id"] = "invented-error"
        bad_assessment = deepcopy(self.original["assessment"])
        bad_assessment["utterance_id"] = "changed"
        bad_words = deepcopy(self.original["assessment"])
        bad_words["words"][0]["text"] = "WRONG"
        for kwargs in ({"reviewer":" "}, {"feedback":bad_feedback}, {"assessment":bad_assessment},
                       {"assessment":bad_words}, {"assessment":[]}, {"max_length":4}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): self.save(**kwargs)
                self.assertFalse(self.destination.exists())

    def test_explicit_save_keeps_original_unapproved_and_updates_only_matching_id(self):
        before = deepcopy(self.original)
        result = self.save(notes="Fixture, not a real teacher review")
        self.assertTrue(result["review"]["approved"])
        self.assertEqual(self.original, before)
        rows = read_rows(self.destination)
        self.assertEqual(len(rows),1)
        self.save(reviewer="Updated fixture reviewer")
        self.assertEqual(len(read_rows(self.destination)),1)
        self.assertEqual(read_rows(self.destination)[0]["review"]["reviewer"],"Updated fixture reviewer")
        second = deepcopy(self.original)
        second["assessment"]["utterance_id"] = second["feedback"]["utterance_id"] = "example-002"
        self.save(original=second)
        self.assertEqual({r["assessment"]["utterance_id"] for r in read_rows(self.destination)}, {"example-001","example-002"})


if __name__ == "__main__": unittest.main()
