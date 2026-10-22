"""Tentative decoding must not grade an unread suffix or call paid providers."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.mdd.demo import DemoApplication, DemoError
from src.mdd.demo_inference import canonical_comparison
from src.mdd.demo_realtime import practice_reference, prefix_comparison
from src.mdd.demo_reference import COMMIT, dictionary_reference, _load


ROW = {"prompt_text": "Think well", "canonical_phones": ["TH", "IH", "NG", "K", "W", "EH", "L"],
       "canonical_word_indices": [0, 0, 0, 0, 1, 1, 1]}


class PrefixTests(unittest.TestCase):
    def test_unread_words_and_unfinished_tail_are_not_deletions(self):
        reference = deepcopy(ROW)
        result = prefix_comparison("Think well", reference, ["TH", "IH"])
        self.assertEqual([w["status"] for w in result["words"]], ["listening", "pending"])
        self.assertEqual(result["counts"]["deletion"], 0)
        self.assertEqual(result["consumed_reference_phones"], 2)
        self.assertEqual(reference, ROW)

    def test_completed_word_errors_remain_visible_when_next_word_arrives(self):
        result = prefix_comparison("Think well", ROW, ["T", "IH", "K", "W", "EH"])
        self.assertEqual([w["status"] for w in result["words"]], ["difference", "listening"])
        self.assertEqual(result["counts"]["deletion"], 1)  # NG, not the unread L
        self.assertTrue(all(e["word_index"] == 0 for e in result["events"]))

    def test_empty_decode_does_not_invent_progress_or_errors(self):
        result = prefix_comparison("Think well", ROW, [])
        self.assertEqual(result["consumed_reference_phones"], 0)
        self.assertEqual(result["events"], [])

    def test_repeated_words_choose_earliest_equal_cost_prefix(self):
        reference = {"prompt_text": "go go", "canonical_phones": ["G", "OW", "G", "OW"], "canonical_word_indices": [0, 0, 1, 1]}
        result = prefix_comparison("go go", reference, ["G", "OW"])
        self.assertEqual(result["consumed_reference_phones"], 2)
        self.assertEqual(result["words"][1]["status"], "pending")

    def test_word_reference_contains_only_selected_canonical_phones(self):
        row = {**ROW, "perceived_phones": ["FORBIDDEN"], "gold_error_events": [{"phone": "FORBIDDEN"}]}
        text, reference = practice_reference(row, "Think well", "word", 1)
        self.assertEqual(text, "well")
        self.assertEqual(reference, {"prompt_text": "well", "canonical_phones": ["W", "EH", "L"], "canonical_word_indices": [0, 0, 0]})
        for index in (-1, 2, True):
            with self.assertRaises(ValueError):
                practice_reference(row, "Think well", "word", index)

    def test_broken_word_mapping_and_interval_labels_are_rejected(self):
        for changes in ({"canonical_word_indices": [0] * 7}, {"canonical_word_indices": [1, 0, 0, 0, 1, 1, 1]},
                        {"word_intervals": [{"index": 0, "text": "WRONG"}, {"index": 1, "text": "well"}]}):
            with self.assertRaises(ValueError):
                practice_reference({**ROW, **changes}, "Think well", "sentence")


class LiveApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        run = self.root / "runs/test-B"
        checkpoint = run / "best_checkpoint"
        checkpoint.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"stage": "B"}))
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "metadata.json").write_text(json.dumps({"stage": "B", "purpose": "scientific_training"}))
        (checkpoint / "model.safetensors").write_bytes(b"fixture-weights")
        self.feedback = Mock()
        self.feedback.register.return_value = {"analysis_id": "final-only", "feedback_available": True}
        self.engine = Mock()
        self.engine.analyze.side_effect = self.infer
        self.app = DemoApplication(self.root, self.engine, feedback_service=self.feedback)
        self.app.samples = {"sample": deepcopy(ROW)}
        self.predicted = ["TH", "IH"]
        self.body = {"model_id": "run:test-B:best", "reference_text": "Think well", "audio_base64": base64.b64encode(b"fixture").decode(),
                     "mode": "sentence", "final": False, "word_index": 0}

    def infer(self, model, audio, text, reference=None):
        return {"model": {"kind": "phoneme", "experimental": False}, "reference_text": text,
                "phonemes": [{"phone": phone} for phone in self.predicted],
                "comparison": canonical_comparison(text, reference, self.predicted)}

    def test_previews_never_register_feedback_and_final_sentence_grades_missing_tail(self):
        preview = self.app.analyze(self.body, live=True)
        self.assertFalse(preview["feedback_available"])
        self.assertIsNone(preview["analysis_id"])
        self.assertEqual(preview["assessment"]["status"], "uncertain")
        self.assertEqual(preview["comparison"]["words"][1]["status"], "pending")
        self.feedback.register.assert_not_called()
        final = self.app.analyze({**self.body, "final": True}, live=True)
        self.assertEqual(final["analysis_id"], "final-only")
        self.assertFalse(final["live"]["tentative"])
        self.assertEqual(final["comparison"]["words"][1]["status"], "difference")
        self.feedback.register.assert_called_once()

    def test_final_word_scopes_text_mapping_and_feedback_to_that_word(self):
        self.predicted = ["W", "EH", "L"]
        result = self.app.analyze({**self.body, "mode": "word", "word_index": 1, "final": True}, live=True)
        self.assertEqual(result["reference_text"], "well")
        self.assertEqual(result["live"]["parent_reference_text"], "Think well")
        self.assertEqual(result["comparison"]["words"], [{"text": "well", "status": "match"}])
        reference = self.engine.analyze.call_args.kwargs["reference"]
        self.assertEqual(reference["canonical_word_indices"], [0, 0, 0])
        self.assertNotIn("perceived_phones", reference)

    def test_live_rejects_client_labels_and_bad_modes_before_inference(self):
        for changes in ({"canonical_phones": ["L"]}, {"mode": []}, {"mode": "bad"}, {"final": "true"}, {"word_index": True}):
            with self.assertRaises(DemoError) as caught:
                self.app.analyze({**self.body, **changes}, live=True)
            self.assertEqual(caught.exception.status, 400)
        self.engine.analyze.assert_not_called()
        self.feedback.register.assert_not_called()

    def test_missing_reference_or_busy_model_cannot_issue_feedback(self):
        with self.assertRaises(DemoError) as caught:
            self.app.analyze({**self.body, "reference_text": "Unknown example"}, live=True)
        self.assertEqual(caught.exception.code, "reference_unavailable")
        self.app.inference_lock.acquire()
        try:
            with self.assertRaises(DemoError) as caught:
                self.app.analyze(self.body, live=True)
            self.assertEqual(caught.exception.code, "busy")
        finally:
            self.app.inference_lock.release()
        self.feedback.register.assert_not_called()

    def test_reference_preflight_does_not_run_model(self):
        self.assertTrue(self.app.reference_info({"reference_text": "Think well"})["available"])
        self.assertFalse(self.app.reference_info({"reference_text": "Unknown example"})["available"])
        self.engine.analyze.assert_not_called()


class DictionaryTests(unittest.TestCase):
    def test_complete_dictionary_sentence_preserves_tokens_and_word_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/external/cmudict" / COMMIT / "cmudict.dict"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
            with patch("src.mdd.demo_reference._load", return_value={"i": ["AY"], "think": ["TH", "IH", "NG", "K"]}):
                reference, missing = dictionary_reference(root, "I think!")
                self.assertEqual(missing, [])
                self.assertEqual(reference["canonical_word_indices"], [0, 1, 1, 1, 1])
                self.assertEqual(reference["prompt_text"], "I think!")
                reference, missing = dictionary_reference(root, "I qzxword")
                self.assertIsNone(reference)
                self.assertEqual(missing, ["qzxword"])
            self.assertEqual(_load(str(path), 0, 7), {})  # corrupted lexicon is not used


if __name__ == "__main__":
    unittest.main()
