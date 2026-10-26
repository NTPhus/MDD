"""Offline BERT measurement/provenance contracts; no model or API access."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.evaluate_vi_voice_pipeline import (
    EXPECTED_MODELS, aggregate, frozen_config, record_case, runtime_source_hashes,
    verify_job_models, verify_job_recovery,
)
from scripts.report_vi_voice_pipeline import (
    baseline_comparison, latency_table, per_case_table, recovery_table,
)


class BertEvaluationTests(unittest.TestCase):
    revision = "a" * 40

    def config(self):
        return {**EXPECTED_MODELS, "speech_ready": True,
                "transcript_recovery": "bert_masked_recovery_v1",
                "transcript_recovery_provider": "bert",
                "transcript_recovery_model": "google-bert/bert-base-uncased",
                "transcript_recovery_revision": self.revision}

    def job(self):
        raw = "To day I work"
        return {"id": "a" * 32, "status": "ready", "text": "What do you do?",
                "speech": {"asr_model": EXPECTED_MODELS["asr_model_id"],
                           "rewrite": {"model_id": EXPECTED_MODELS["grammar_model_id"]},
                           "transcript": raw, "reference_text": raw, "reference_source": "asr",
                           "conversation_text": "Today I work.", "conversation_source": "asr_interpreted",
                           "interpretation": {"version": "bert_masked_recovery_v1", "provider": "bert",
                                              "model_id": "google-bert/bert-base-uncased",
                                              "revision": self.revision, "device": "cpu",
                                              "original": raw, "text": "Today I work.",
                                              "status": "accepted", "requires_confirmation": True,
                                              "reference_confirmed": False, "latency_sec": 0.123}},
                "metadata": {"reply_model_id": EXPECTED_MODELS["reply_model_id"]},
                "sentence_voice": {"status": "not_requested"}}

    def case(self):
        return {"id": "one", "corpus": "test", "speaker_id": "v1", "audio": "clip.wav",
                "audio_sha256": "a" * 64, "reference_text": "Today I work.",
                "reference_kind": "published_reading_prompt", "source_url": "https://example.org",
                "split_notes": "synthetic contract fixture, not an evaluation observation"}

    def reference_config(self):
        return {**self.config(), "pronunciation_reference_policy": "bert_output", "show_raw_asr": False}

    def reference_job(self):
        job = self.job()
        speech = job["speech"]
        speech.update(reference_text=speech["interpretation"]["text"],
                      reference_source="bert_inferred", pronunciation_reference_ready=True)
        speech["interpretation"]["applied_as_pronunciation_reference"] = True
        return job

    def test_frozen_config_and_actual_job_agree(self):
        config = frozen_config(self.config())
        self.assertEqual(config["transcript_recovery_revision"], self.revision)
        verify_job_models(self.job())
        verify_job_recovery(self.job(), config)

    def test_authorized_bert_output_can_be_used_for_pronunciation_without_claiming_confirmation(self):
        config = frozen_config(self.reference_config())
        self.assertEqual(config["pronunciation_reference_policy"], "bert_output")
        self.assertIs(config["show_raw_asr"], False)
        for status in ("accepted", "unchanged"):
            with self.subTest(status=status):
                job = self.reference_job()
                job["speech"]["interpretation"]["status"] = status
                if status == "unchanged":
                    job["speech"]["interpretation"]["text"] = job["speech"]["transcript"]
                    job["speech"]["reference_text"] = job["speech"]["transcript"]
                verify_job_recovery(job, config)
                self.assertIs(job["speech"]["interpretation"]["reference_confirmed"], False)

    def test_pronunciation_policy_requires_an_explicit_pinned_bert_provider(self):
        for key, value in (("transcript_recovery_provider", "qwen"),
                           ("transcript_recovery_provider", None),
                           ("transcript_recovery", None),
                           ("pronunciation_reference_policy", "any_model"),
                           ("show_raw_asr", "false")):
            with self.subTest(key=key, value=value):
                config = self.reference_config(); config[key] = value
                with self.assertRaises(ValueError):
                    frozen_config(config)
        config = self.reference_config(); del config["transcript_recovery"]
        with self.assertRaisesRegex(ValueError, "requires BERT recovery"):
            verify_job_recovery(self.reference_job(), config)

    def test_bert_pronunciation_reference_must_match_recovery_and_be_ready(self):
        for key, value in (("reference_text", "This is not the recovered sentence."),
                           ("reference_source", "user_confirmed"),
                           ("reference_source", "asr"),
                           ("pronunciation_reference_ready", False),
                           ("pronunciation_reference_ready", None)):
            with self.subTest(key=key, value=value):
                job = self.reference_job(); job["speech"][key] = value
                with self.assertRaisesRegex(ValueError, "BERT pronunciation reference"):
                    verify_job_recovery(job, self.reference_config())
        for key, value in (("text", ""), ("text", None),
                           ("applied_as_pronunciation_reference", False),
                           ("applied_as_pronunciation_reference", None)):
            with self.subTest(recovery_key=key, value=value):
                job = self.reference_job(); job["speech"]["interpretation"][key] = value
                if key == "text":
                    job["speech"]["reference_text"] = value
                with self.assertRaisesRegex(ValueError, "BERT pronunciation reference"):
                    verify_job_recovery(job, self.reference_config())

    def test_bert_reference_policy_does_not_allow_unbound_or_confirmed_recovery(self):
        for key, value in (("original", "Another audio transcript"),
                           ("reference_confirmed", True), ("reference_confirmed", None)):
            with self.subTest(key=key):
                job = self.reference_job(); job["speech"]["interpretation"][key] = value
                with self.assertRaises(ValueError):
                    verify_job_recovery(job, self.reference_config())

    def test_ambiguous_and_unavailable_recoveries_keep_the_raw_reference(self):
        for status in ("ambiguous", "unavailable"):
            for applied in (None, False):
                with self.subTest(status=status, applied=applied):
                    job = self.job()
                    job["speech"]["interpretation"].update(
                        status=status, applied_as_pronunciation_reference=applied)
                    verify_job_recovery(job, self.reference_config())
            job = self.reference_job(); job["speech"]["interpretation"]["status"] = status
            with self.subTest(status=status, unsafe_reference=True):
                with self.assertRaisesRegex(ValueError, "ambiguous recovery replaced"):
                    verify_job_recovery(job, self.reference_config())

    def test_legacy_policy_cannot_silently_apply_a_bert_pronunciation_reference(self):
        with self.assertRaisesRegex(ValueError, "acoustic reference"):
            verify_job_recovery(self.reference_job(), self.config())
        job = self.job()
        job["speech"]["interpretation"]["applied_as_pronunciation_reference"] = True
        with self.assertRaisesRegex(ValueError, "acoustic reference"):
            verify_job_recovery(job, self.config())

    def test_bert_reference_policy_records_the_exact_assessed_text(self):
        row = record_case(self.case(), self.reference_job())
        self.assertEqual(row["transcript"], "To day I work")
        self.assertEqual(row["pronunciation_reference_text"], "Today I work.")
        self.assertIs(row["pronunciation_reference_ready"], True)
        self.assertEqual(row["assessment"]["reference_source"], "bert_inferred")

    def test_config_rejects_unpinned_or_different_bert(self):
        for key, value in [("transcript_recovery_revision", "main"),
                           ("transcript_recovery_revision", None),
                           ("transcript_recovery_model", "unverified/model")]:
            with self.subTest(key=key, value=value):
                config = self.config(); config[key] = value
                with self.assertRaisesRegex(ValueError, "BERT recovery"):
                    frozen_config(config)

    def test_completed_job_cannot_claim_another_provider_model_revision_or_device(self):
        for key, value in [("provider", "qwen"), ("model_id", "other/model"),
                           ("revision", "b" * 40), ("device", "cuda"),
                           ("version", "different"), ("original", "Different recording")]:
            with self.subTest(key=key):
                job = self.job(); job["speech"]["interpretation"][key] = value
                with self.assertRaises(ValueError):
                    verify_job_recovery(job, self.config())

    def test_missing_model_identity_is_not_silently_accepted(self):
        for key in ("provider", "model_id", "revision", "device"):
            with self.subTest(key=key):
                job = self.job(); del job["speech"]["interpretation"][key]
                with self.assertRaises(ValueError):
                    verify_job_recovery(job, self.config())

    def test_inference_cannot_replace_acoustic_reference_or_claim_confirmation(self):
        for key, value in [("reference_text", "Today I work."), ("reference_source", "user_confirmed")]:
            with self.subTest(key=key):
                job = self.job(); job["speech"][key] = value
                with self.assertRaisesRegex(ValueError, "acoustic reference"):
                    verify_job_recovery(job, self.config())
        for value in (True, None):
            with self.subTest(reference_confirmed=value):
                job = self.job(); job["speech"]["interpretation"]["reference_confirmed"] = value
                with self.assertRaisesRegex(ValueError, "cannot claim"):
                    verify_job_recovery(job, self.config())

    def test_no_sentence_tts_can_appear_in_an_evaluation_job(self):
        job = self.job(); job["sentence_voice"]["status"] = "ready"
        with self.assertRaisesRegex(ValueError, "Unexpected sentence TTS"):
            verify_job_recovery(job, self.config())

    def test_old_qwen_results_remain_valid_without_bert_identity_fields(self):
        job = self.job()
        job["speech"]["interpretation"] = {"version": "guarded_intent_recovery_v1",
                                             "original": job["speech"]["transcript"]}
        config = {**EXPECTED_MODELS, "speech_ready": True,
                  "transcript_recovery": "guarded_intent_recovery_v1"}
        verify_job_recovery(job, frozen_config(config))

    def test_bert_module_bytes_are_part_of_frozen_runtime_identity(self):
        with patch("scripts.evaluate_vi_voice_pipeline.source_hashes", return_value={"base.py": "base"}), \
                patch("scripts.evaluate_vi_voice_pipeline.digest", side_effect=lambda path: str(path)), \
                patch.object(Path, "is_file", return_value=True):
            hashes = runtime_source_hashes(self.config())
        self.assertIn("scripts/bert_transcript_recovery.py", hashes)
        self.assertIn("src/mdd/phonetic_candidates.py", hashes)
        self.assertTrue(hashes["scripts/bert_transcript_recovery.py"].endswith("bert_transcript_recovery.py"))

    def test_configured_bert_cannot_omit_a_missing_runtime_module(self):
        def digest_or_fail(path):
            if path.name == "bert_transcript_recovery.py":
                raise FileNotFoundError(path)
            return "hash"
        with patch("scripts.evaluate_vi_voice_pipeline.source_hashes", return_value={}), \
                patch("scripts.evaluate_vi_voice_pipeline.digest", side_effect=digest_or_fail), \
                patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                runtime_source_hashes(self.config())
            self.assertNotIn("scripts/bert_transcript_recovery.py", runtime_source_hashes({}))

    def test_records_and_report_preserve_measured_timings_and_provider(self):
        timing = {"reply_ready": {"elapsed_sec": 2.25}, "full_elapsed_sec": 3.5,
                  "submission_wait_sec": 0.025}
        row = record_case(self.case(), self.job(), timing)
        self.assertEqual(row["latencies"], timing)
        self.assertEqual(row["transcript"], "To day I work")
        self.assertEqual(row["conversation_text"], "Today I work.")
        self.assertEqual(aggregate([row])["interpretation_providers"], {"bert": 1})
        latency = latency_table([row]).iloc[0]
        self.assertEqual(latency["Reply ready (s)"], 2.25)
        self.assertEqual(latency["All feedback ready (s)"], 3.5)
        self.assertEqual(latency["Interpretation (s)"], 0.123)
        self.assertEqual(latency["Recovery device"], "cpu")
        self.assertEqual(per_case_table([row]).iloc[0]["recovery_revision"], self.revision)
        self.assertEqual(recovery_table([row]).iloc[0]["provider"], "bert")

    def test_baseline_keeps_distinct_recorded_provider_and_latency(self):
        row = record_case(self.case(), self.job())
        current = {"identity": {"manifest_sha256": "manifest"}, "records": [row]}
        baseline = deepcopy(current)
        baseline["records"][0]["interpretation"].update(provider="qwen", model_id="Qwen3-0.6B", latency_sec=1.5)
        comparison = baseline_comparison(current, baseline).iloc[0]
        self.assertEqual(comparison["baseline_recovery_provider"], "qwen")
        self.assertEqual(comparison["current_recovery_provider"], "bert")
        self.assertEqual(comparison["baseline_recovery_sec"], 1.5)
        self.assertEqual(comparison["current_recovery_sec"], 0.123)


if __name__ == "__main__":
    unittest.main()
