"""Temporary synthetic fixtures exercise comparison guards; no scientific results."""

from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_mdd_runs import ROOT, compare_runs, sha256
from src.mdd.metrics import SCORING_PROTOCOL, aggregate_scores, score_utterance


def fixture_row(utterance_id, canonical, perceived, predicted, mapping=None):
    mapping = mapping if mapping is not None else [0] * len(canonical)
    score = score_utterance(canonical, perceived, predicted, canonical_word_indices=mapping)
    return {"utterance_id": utterance_id, "speaker_id": "FIXTURE",
            "status": "ok", "audio_sha256": "a" * 64,
            "canonical_phones": canonical, "perceived_phones": perceived,
            "predicted_phones": predicted, "canonical_word_indices": mapping,
            "counts": score["counts"], "metrics": score["metrics"],
            "gold_events": score["gold_events"], "predicted_events": score["predicted_events"]}


def write_fixture(directory, stage, rows):
    directory.mkdir()
    path = directory / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    aggregate = aggregate_scores(rows)
    per_speaker = defaultdict(list)
    for row in rows:
        per_speaker[row["speaker_id"]].append(row)
    by_speaker = {speaker: aggregate_scores(items) for speaker, items in per_speaker.items()}
    f1s = [row["metrics"]["f1"] for row in by_speaker.values() if row["metrics"]["f1"] is not None]
    files = {name: {"path": "/fixture-only/" + name, "sha256": "d" * 64}
             for name in ("annotations", "vocab", "normalization", "manifest", "splits", "protocol", "scorer", "base_model")}
    files["scorer"]["sha256"] = sha256(ROOT / "src/mdd/metrics.py")
    config = {
        "stage": stage, "purpose": "scientific_training", "fold": 1,
        "max_steps": 10 if stage == "A" else 4, "batch_size": 1,
        "gradient_accumulation_steps": 4, "eval_every": 2,
        "backbone_learning_rate": 1e-5, "head_learning_rate": 1e-4,
        "seed": 42, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "freeze_feature_encoder": True, "freeze_backbone": False,
        "selection_metric": "f1", "mixed_precision": "fp16",
        "max_train_samples": None, "max_eval_samples": None, "max_audio_seconds": None,
    }
    initialization = {"weight_sha256": "1" * 64,
                      "mode": "weights_only_new_optimizer" if stage in {"B", "C"} else "pretrained_backbone_new_random_phone_head"}
    if stage in {"B", "C"}:
        initialization["source_metadata"] = {"stage": "A", "purpose": "scientific_training"}
    training = {"run_id": "fixture-" + stage, "stage": stage, "purpose": "scientific_training",
                "effective_config": config, "initialization": initialization,
                "completed_training": {"optimizer_steps": config["max_steps"]}}
    report = {**aggregate, "stage": stage, "effective_config": config, "training_metadata": training,
              "by_speaker": by_speaker, "speaker_macro_f1": sum(f1s) / len(f1s) if f1s else None,
              "fold": 1, "split": "test", "purpose": "scientific_training", "limited_technical_subset": False,
              "checkpoint": "/fixture-only/" + stage, "checkpoint_sha256": {"A": "1", "B": "2", "C": "3"}[stage] * 64,
              "prediction_sha256": sha256(path), "protocol": {"scoring_protocol": SCORING_PROTOCOL, "files": files},
              "selection": {"selected_ids": [row["utterance_id"] for row in rows], "selected_count": len(rows),
                            "requested_count": len(rows), "excluded": []}}
    (directory / "metrics.json").write_text(json.dumps(report), encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = {stage: self.root / stage for stage in ("A", "B", "C")}
        self.rows = {}
        for stage in self.directories:
            first = ["TH", "IH"] if stage == "A" else ["T", "IH"]
            second = ["R*"] if stage == "B" else ["R"]
            self.rows[stage] = [fixture_row("FIXTURE/one", ["TH", "IH"], ["T", "IH"], first),
                                fixture_row("FIXTURE/two", ["R"], ["R*"], second)]
            write_fixture(self.directories[stage], stage, self.rows[stage])
        self.output = self.root / "comparison"

    def report(self, stage):
        return json.loads((self.directories[stage] / "metrics.json").read_text())

    def update_report(self, stage, report):
        (self.directories[stage] / "metrics.json").write_text(json.dumps(report), encoding="utf-8")

    def update_rows(self, stage, rows):
        path = self.directories[stage] / "predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = self.report(stage)
        report["prediction_sha256"] = sha256(path)
        self.update_report(stage, report)

    def compare(self):
        return compare_runs(self.directories["A"], self.directories["B"], self.directories["C"], self.output)

    def test_valid_comparison_recomputes_counts_and_writes_auditable_deltas(self):
        result = self.compare()
        self.assertEqual(result["utterance_count"], 2)
        self.assertEqual(result["runs"]["B"]["counts"]["TR"], 2)
        self.assertEqual(result["differences"]["f1"]["B_minus_A_percentage_points"], 100)
        self.assertAlmostEqual(result["differences"]["f1"]["B_minus_C_percentage_points"], 100 / 3)
        self.assertLess(result["differences"]["per"]["B_minus_A_percentage_points"], 0)
        self.assertEqual(result["differences"]["per"]["better"], "lower")
        saved = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(saved, result)
        with (self.output / "comparison.csv").open(newline="") as stream:
            csv_rows = {row["metric"]: row for row in csv.DictReader(stream)}
        self.assertEqual(float(csv_rows["f1"]["B_minus_A_percentage_points"]), 100)
        self.assertEqual(csv_rows["f1"]["A_prediction_sha256"], result["runs"]["A"]["prediction_sha256"])
        self.assertEqual(csv_rows["f1"]["B_directory"], str(self.directories["B"]))
        self.assertEqual(csv_rows["f1"]["protocol_scorer_sha256"], result["protocol_hashes"]["scorer"])

    def test_row_order_and_provenance_path_changes_do_not_break_pairing(self):
        self.update_rows("B", list(reversed(self.rows["B"])))
        report = self.report("C")
        report["protocol"]["files"]["annotations"]["path"] = "/different-vm/same-content.jsonl"
        self.update_report("C", report)
        self.assertEqual(self.compare()["utterance_count"], 2)

    def test_duplicate_utterance_ids_are_rejected(self):
        self.update_rows("B", self.rows["B"] + [self.rows["B"][0]])
        with self.assertRaisesRegex(ValueError, "Duplicate utterance ID"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_mismatched_id_sets_are_rejected(self):
        rows = deepcopy(self.rows["B"])
        rows[0]["utterance_id"] = "FIXTURE/different"
        self.update_rows("B", rows)
        report = self.report("B")
        report["selection"]["selected_ids"] = [row["utterance_id"] for row in rows]
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "Utterance ID sets differ"):
            self.compare()

    def test_paired_audio_and_gold_changes_are_rejected(self):
        for field, value in (("audio_sha256", "b" * 64), ("speaker_id", "OTHER")):
            with self.subTest(field=field):
                rows = deepcopy(self.rows["B"])
                rows[0][field] = value
                self.update_rows("B", rows)
                with self.assertRaises(ValueError):
                    self.compare()
                self.assertFalse(self.output.exists())
        self.update_rows("B", self.rows["B"])
        rows = deepcopy(self.rows["B"])
        rows[0] = fixture_row("FIXTURE/one", ["TH", "IH"], ["S", "IH"], ["T", "IH"])
        self.update_rows("B", rows)
        with self.assertRaises(ValueError):
            self.compare()

    def test_changed_protocol_and_scorer_hashes_are_rejected(self):
        original = self.report("B")
        for name in ("annotations", "base_model", "scorer"):
            with self.subTest(name=name):
                report = deepcopy(original)
                report["protocol"]["files"][name]["sha256"] = "e" * 64
                self.update_report("B", report)
                with self.assertRaisesRegex(ValueError, "hash"):
                    self.compare()
        self.update_report("B", original)

    def test_validation_smoke_subset_and_wrong_fold_cannot_be_compared(self):
        original = self.report("B")
        for field, value in (("split", "validation"), ("purpose", "technical_smoke_only"),
                             ("limited_technical_subset", True), ("limited_technical_subset", None), ("fold", 2)):
            with self.subTest(field=field, value=value):
                report = deepcopy(original)
                report[field] = value
                self.update_report("B", report)
                with self.assertRaises(ValueError):
                    self.compare()
        self.update_report("B", original)

    def test_corrupted_prediction_file_hash_is_rejected(self):
        path = self.directories["B"] / "predictions.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Prediction file checksum mismatch"):
            self.compare()

    def test_row_and_summary_counters_and_rates_are_verified(self):
        for field, key, value in (("counts", "TR", 99), ("metrics", "f1", 0.123)):
            with self.subTest(field=field):
                rows = deepcopy(self.rows["B"])
                rows[0][field][key] = value
                self.update_rows("B", rows)
                with self.assertRaisesRegex(ValueError, "Recomputed"):
                    self.compare()
        self.update_rows("B", self.rows["B"])
        report = self.report("B")
        report["counts"]["TR"] = 99
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "Recomputed counter mismatch"):
            self.compare()

    def test_explicit_word_mapping_and_successful_prediction_are_required(self):
        for field, value in (("canonical_word_indices", None), ("status", "failed")):
            with self.subTest(field=field):
                rows = deepcopy(self.rows["B"])
                rows[0][field] = value
                self.update_rows("B", rows)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_incomplete_selection_is_rejected_even_if_subset_flag_is_false(self):
        report = self.report("B")
        report["selection"]["requested_count"] = 150
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "full saved test set"):
            self.compare()

    def test_same_A_initialization_and_role_are_verified(self):
        original = self.report("B")
        for field, value in (("weight_sha256", "9" * 64), ("mode", "resume_optimizer")):
            report = deepcopy(original)
            report["training_metadata"]["initialization"][field] = value
            self.update_report("B", report)
            with self.assertRaisesRegex(ValueError, "exact evaluated stage-A weights"):
                self.compare()
        self.update_report("B", original)
        with self.assertRaisesRegex(ValueError, "stage-A"):
            compare_runs(self.directories["B"], self.directories["A"], self.directories["C"], self.output)

    def test_B_C_budget_and_completed_training_are_verified(self):
        original = self.report("B")
        report = deepcopy(original)
        report["effective_config"]["batch_size"] = 2
        report["training_metadata"]["effective_config"]["batch_size"] = 2
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "B/C training budgets differ: batch_size"):
            self.compare()
        report = deepcopy(original)
        report["training_metadata"]["completed_training"]["optimizer_steps"] = 3
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "completed.*optimizer-step budget"):
            self.compare()
        report = deepcopy(original)
        report["effective_config"]["max_train_samples"] = 10
        report["training_metadata"]["effective_config"]["max_train_samples"] = 10
        self.update_report("B", report)
        with self.assertRaisesRegex(ValueError, "sample filters"):
            self.compare()

    def test_undefined_metrics_remain_null_and_csv_empty(self):
        for stage, directory in self.directories.items():
            rows = [fixture_row("FIXTURE/one", ["TH"], ["TH"], ["TH"])]
            path = directory / "predictions.jsonl"
            path.unlink()
            (directory / "metrics.json").unlink()
            directory.rmdir()
            write_fixture(directory, stage, rows)
        result = self.compare()
        self.assertIsNone(result["differences"]["f1"]["B_minus_A_percentage_points"])
        self.assertEqual(result["runs"]["A"]["metrics"]["per"], 0)
        with (self.output / "comparison.csv").open(newline="") as stream:
            rows = {row["metric"]: row for row in csv.DictReader(stream)}
        self.assertEqual(rows["f1"]["B_minus_A_percentage_points"], "")

    def test_existing_output_is_never_replaced(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.compare()
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
