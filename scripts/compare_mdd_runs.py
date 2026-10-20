#!/usr/bin/env python3
"""Compare paired, full held-out A/B/C MDD evaluations without running models.

All counters are recomputed from saved phone predictions and human gold events.
The command rejects incompatible inputs and never replaces an existing output.
Rates in JSON/CSV remain fractions; differences are reported in percentage points.
"""

from collections import defaultdict
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdd.metrics import COUNT_KEYS, SCORING_PROTOCOL, aggregate_scores, score_utterance


REQUIRED_PROTOCOL_FILES = {"annotations", "vocab", "normalization", "manifest", "splits", "protocol", "scorer"}
PAIRED_FIELDS = ("speaker_id", "audio_sha256", "canonical_phones", "perceived_phones",
                 "gold_events", "canonical_word_indices")
LOWER_IS_BETTER = {"far", "frr", "der", "per"}
BUDGET_KEYS = (
    "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every",
    "backbone_learning_rate", "head_learning_rate", "seed", "weight_decay",
    "max_grad_norm", "freeze_feature_encoder", "freeze_backbone",
    "selection_metric", "mixed_precision",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError(f"Invalid SHA256 for {name}")
    return value.lower()


def _check_counts(actual, expected, context):
    if not isinstance(actual, dict) or set(actual) != set(COUNT_KEYS):
        raise ValueError(f"Missing/unexpected counters in {context}")
    for key, expected_value in expected.items():
        value = actual[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value != expected_value:
            raise ValueError(f"Recomputed counter mismatch in {context}: {key}")


def _check_metrics(actual, expected, context):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"Missing/unexpected metric in {context}")
    for key, expected_value in expected.items():
        value = actual[key]
        if expected_value is None:
            if value is not None:
                raise ValueError(f"Undefined metric must remain null in {context}: {key}")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or not math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"Recomputed metric mismatch in {context}: {key}")


def load_evaluation(directory):
    directory = Path(directory).resolve()
    metrics_path, predictions_path = directory / "metrics.json", directory / "predictions.jsonl"
    report = _load_json(metrics_path)
    if report.get("split") != "test":
        raise ValueError(f"Only held-out test evaluations may be compared: {directory}")
    if report.get("purpose") != "scientific_training":
        raise ValueError(f"Technical/non-scientific run cannot support comparison: {directory}")
    if report.get("limited_technical_subset") is not False:
        raise ValueError(f"Evaluation must explicitly use the full test set: {directory}")
    fold = report.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 5):
        raise ValueError(f"Invalid held-out fold: {directory}")
    if report.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError(f"Scoring protocol mismatch: {directory}")
    protocol = report.get("protocol", {})
    if protocol.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError(f"Provenance scoring protocol mismatch: {directory}")
    files = protocol.get("files", {})
    if not isinstance(files, dict) or not REQUIRED_PROTOCOL_FILES <= files.keys():
        raise ValueError(f"Incomplete protocol file provenance: {directory}")
    protocol_hashes = {}
    for name, metadata in files.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Malformed protocol file provenance: {name}")
        protocol_hashes[name] = _hash(metadata.get("sha256"), f"protocol/{name}")
    if protocol_hashes["scorer"] != sha256(ROOT / "src/mdd/metrics.py"):
        raise ValueError("Saved scorer hash differs from the scorer used for this comparison")
    prediction_hash = sha256(predictions_path)
    if _hash(report.get("prediction_sha256"), "predictions.jsonl") != prediction_hash:
        raise ValueError(f"Prediction file checksum mismatch: {directory}")
    checkpoint_hash = _hash(report.get("checkpoint_sha256"), "checkpoint")
    training = report.get("training_metadata")
    config = report.get("effective_config")
    stage = report.get("stage")
    if not isinstance(training, dict) or not isinstance(config, dict) or stage not in {"A", "B", "C"}:
        raise ValueError(f"Missing completed training provenance: {directory}")
    if (training.get("stage") != stage or training.get("purpose") != "scientific_training"
            or training.get("effective_config") != config or config.get("stage") != stage
            or config.get("purpose") != "scientific_training"):
        raise ValueError(f"Inconsistent training stage/configuration provenance: {directory}")
    completed_steps = training.get("completed_training", {}).get("optimizer_steps")
    expected_steps = config.get("max_steps")
    if (isinstance(expected_steps, bool) or not isinstance(expected_steps, int) or expected_steps <= 0
            or isinstance(completed_steps, bool) or not isinstance(completed_steps, int)
            or completed_steps != expected_steps):
        raise ValueError(f"Training must have completed its configured optimizer-step budget: {directory}")
    if any(config.get(key) is not None for key in ("max_train_samples", "max_eval_samples", "max_audio_seconds")):
        raise ValueError(f"Training/evaluation sample filters cannot support this scientific comparison: {directory}")
    if not set(BUDGET_KEYS) <= config.keys():
        raise ValueError(f"Incomplete training budget configuration: {directory}")
    if stage in {"B", "C"} and config.get("fold") != fold:
        raise ValueError(f"Adapted/control checkpoint evaluated on a different fold: {directory}")
    initialization = training.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError(f"Missing checkpoint initialization provenance: {directory}")
    _hash(initialization.get("weight_sha256"), "initialization weights")
    rows, scores, by_speaker = {}, [], defaultdict(list)
    for number, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Empty prediction row at line {number}: {directory}")
        row = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row must be an object: line {number}")
        utterance_id = row.get("utterance_id")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError(f"Missing utterance ID at prediction line {number}")
        if utterance_id in rows:
            raise ValueError(f"Duplicate utterance ID: {utterance_id}")
        if row.get("status") != "ok":
            raise ValueError(f"Prediction must have successful inference status: {utterance_id}")
        if not isinstance(row.get("speaker_id"), str) or not row["speaker_id"]:
            raise ValueError(f"Missing speaker ID: {utterance_id}")
        _hash(row.get("audio_sha256"), f"audio/{utterance_id}")
        required = {"canonical_phones", "perceived_phones", "predicted_phones", "gold_events",
                    "canonical_word_indices", "predicted_events", "counts", "metrics"}
        if not required <= row.keys():
            raise ValueError(f"Incomplete prediction/gold record: {utterance_id}")
        if not isinstance(row["gold_events"], list) or not isinstance(row["canonical_word_indices"], list):
            raise ValueError(f"Explicit gold events and word mapping are required: {utterance_id}")
        score = score_utterance(row["canonical_phones"], row["perceived_phones"], row["predicted_phones"],
                                gold_error_events=row["gold_events"],
                                canonical_word_indices=row["canonical_word_indices"])
        _check_counts(row["counts"], score["counts"], utterance_id)
        _check_metrics(row["metrics"], score["metrics"], utterance_id)
        if row["predicted_events"] != score["predicted_events"]:
            raise ValueError(f"Saved predicted events disagree with decoded phones: {utterance_id}")
        rows[utterance_id] = row
        scores.append(score)
        by_speaker[row["speaker_id"]].append(score)
    if not rows:
        raise ValueError(f"Cannot compare an empty evaluation: {directory}")
    selection = report.get("selection", {})
    selected_ids = selection.get("selected_ids")
    if (not isinstance(selected_ids, list) or len(selected_ids) != len(set(selected_ids))
            or set(selected_ids) != set(rows)
            or selection.get("requested_count") != len(rows)
            or selection.get("selected_count") != len(rows) or selection.get("excluded") != []):
        raise ValueError(f"Selection metadata does not describe the full saved test set: {directory}")
    aggregate = aggregate_scores(scores)
    _check_counts(report.get("counts"), aggregate["counts"], f"summary/{directory}")
    _check_metrics(report.get("metrics"), aggregate["metrics"], f"summary/{directory}")
    speaker_scores = {speaker: aggregate_scores(items) for speaker, items in sorted(by_speaker.items())}
    saved_speaker_scores = report.get("by_speaker")
    if not isinstance(saved_speaker_scores, dict) or set(saved_speaker_scores) != set(speaker_scores):
        raise ValueError(f"Saved per-speaker results disagree with predictions: {directory}")
    for speaker, expected in speaker_scores.items():
        _check_counts(saved_speaker_scores[speaker].get("counts"), expected["counts"], f"speaker/{speaker}")
        _check_metrics(saved_speaker_scores[speaker].get("metrics"), expected["metrics"], f"speaker/{speaker}")
    f1s = [score["metrics"]["f1"] for score in speaker_scores.values() if score["metrics"]["f1"] is not None]
    macro = sum(f1s) / len(f1s) if f1s else None
    _check_metrics({"speaker_macro_f1": report.get("speaker_macro_f1")}, {"speaker_macro_f1": macro}, "speaker macro")
    return {
        "directory": str(directory), "checkpoint": report.get("checkpoint"),
        "model_step": report.get("model_step"),
        "checkpoint_sha256": checkpoint_hash, "metrics_sha256": sha256(metrics_path),
        "prediction_sha256": prediction_hash, "protocol_hashes": protocol_hashes,
        "fold": fold, "split": report["split"], "purpose": report["purpose"],
        "stage": stage, "effective_config": config, "training_metadata": training,
        "counts": aggregate["counts"], "metrics": {**aggregate["metrics"], "speaker_macro_f1": macro},
        "by_speaker": speaker_scores, "rows": rows,
    }


def compare_runs(a, b, c, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    evaluations = {name: load_evaluation(path) for name, path in (("A", a), ("B", b), ("C", c))}
    reference = evaluations["A"]
    for name, evaluation in evaluations.items():
        if evaluation["stage"] != name:
            raise ValueError(f"--{name.lower()} must refer to a completed stage-{name} evaluation")
    for name in ("B", "C"):
        initialization = evaluations[name]["training_metadata"]["initialization"]
        source = initialization.get("source_metadata", {})
        if (initialization["weight_sha256"].lower() != reference["checkpoint_sha256"]
                or initialization.get("mode") != "weights_only_new_optimizer"
                or source.get("stage") != "A" or source.get("purpose") != "scientific_training"):
            raise ValueError(f"{name} must initialize from the exact evaluated stage-A weights with a new optimizer")
    for key in BUDGET_KEYS:
        if evaluations["B"]["effective_config"][key] != evaluations["C"]["effective_config"][key]:
            raise ValueError(f"B/C training budgets differ: {key}")
    for name in ("B", "C"):
        evaluation = evaluations[name]
        if evaluation["fold"] != reference["fold"]:
            raise ValueError(f"A/{name} must evaluate the same held-out fold")
        if evaluation["protocol_hashes"] != reference["protocol_hashes"]:
            raise ValueError(f"Protocol file hashes differ between A and {name}")
        if evaluation["rows"].keys() != reference["rows"].keys():
            raise ValueError(f"Utterance ID sets differ between A and {name}")
        for utterance_id, original in reference["rows"].items():
            other = evaluation["rows"][utterance_id]
            for field in PAIRED_FIELDS:
                if other[field] != original[field]:
                    raise ValueError(f"Paired input/gold mismatch A/{name}: {utterance_id}, {field}")
    deltas = {}
    for metric in reference["metrics"]:
        values = {name: item["metrics"][metric] for name, item in evaluations.items()}
        deltas[metric] = {
            "better": "lower" if metric in LOWER_IS_BETTER else "higher",
            "B_minus_A_percentage_points": None if values["B"] is None or values["A"] is None else 100 * (values["B"] - values["A"]),
            "B_minus_C_percentage_points": None if values["B"] is None or values["C"] is None else 100 * (values["B"] - values["C"]),
        }
    summary = {
        "schema_version": "1.0", "scoring_protocol": SCORING_PROTOCOL,
        "fold": reference["fold"], "split": "test", "utterance_count": len(reference["rows"]),
        "utterance_ids": sorted(reference["rows"]), "protocol_hashes": reference["protocol_hashes"],
        "verification": {"paired_ids_audio_and_gold": True, "full_test_evaluations": True,
                         "all_phone_and_word_counts_recomputed": True,
                         "prediction_file_hashes_verified": True,
                         "same_stage_A_initialization_weights": True,
                         "matched_B_C_completed_optimizer_step_budget_and_configuration": True},
        "runs": {name: {key: value for key, value in item.items() if key != "rows"}
                 for name, item in evaluations.items()},
        "differences": deltas,
        "units": "Metrics are fractions; signed B-A and B-C differences are percentage points. Null remains undefined.",
        "scope": "Paired results for one held-out fold. Matching optimizer-step budgets/configurations does not imply equal training wall time or audio duration. This comparison alone does not establish statistical significance or generalize to all Vietnamese speakers.",
    }
    csv_stream = io.StringIO(newline="")
    provenance_columns = [f"{name}_{key}" for name in evaluations
                          for key in ("directory", "checkpoint_sha256", "prediction_sha256", "metrics_sha256")]
    shared_hash_columns = [f"protocol_{name}_sha256" for name in sorted(reference["protocol_hashes"])]
    writer = csv.DictWriter(csv_stream, fieldnames=["metric", "better", "A", "B", "C",
                            "B_minus_A_percentage_points", "B_minus_C_percentage_points"] + provenance_columns + shared_hash_columns)
    writer.writeheader()
    for metric, difference in deltas.items():
        row = {"metric": metric, **difference, **{name: item["metrics"][metric] for name, item in evaluations.items()}}
        row.update({f"{name}_{key}": item[key] for name, item in evaluations.items()
                    for key in ("directory", "checkpoint_sha256", "prediction_sha256", "metrics_sha256")})
        row.update({f"protocol_{name}_sha256": digest for name, digest in reference["protocol_hashes"].items()})
        writer.writerow(row)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "comparison.csv").write_text(csv_stream.getvalue(), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="A full test evaluation directory")
    parser.add_argument("--b", type=Path, required=True, help="B full test evaluation directory")
    parser.add_argument("--c", type=Path, required=True, help="C full test evaluation directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        summary = compare_runs(args.a, args.b, args.c, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError) as error:
        print(f"Comparison stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {summary['utterance_count']} paired utterances for fold {summary['fold']}")
    print(f"Comparison saved: {args.output}")


if __name__ == "__main__":
    main()
