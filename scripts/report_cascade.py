"""Paired four-case cascade evidence, separate oracle scores and real plots."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mdd.metrics import metrics_from_counts
from scripts.prepare_cascade_data import CONFIG, check_common_split, save
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest
from scripts.published_feedback_metrics import paired_bootstrap, summarize

CASES = {"A_Q0": "A + original Qwen", "A_Q1": "A + trained Qwen", "B_Q0": "B + original Qwen", "B_Q1": "B + trained Qwen"}
TEXT_METRICS = ["json_valid_percent", "grounded_json_percent", "feedback_word_f1_percent", "suggestion_bleu2",
                "suggestion_rougeL_on_reference_error_rows", "target_token_nll", "target_perplexity", "latency_median_sec"]


def read_feedback(path):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or metadata["identity"]["purpose"] != "held_out_benchmark":
        raise ValueError("Not a completed full benchmark: " + str(path))
    if digest(path / "predictions.jsonl") != metadata["predictions_sha256"]:
        raise ValueError("Feedback predictions changed: " + str(path))
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != metadata["identity"]["full_split_rows"] or len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate feedback evaluation")
    return metadata, rows


def verify_pair(before, after):
    first, second = before[0]["identity"], after[0]["identity"]
    for key in ("task", "split", "rows", "full_split_rows", "dataset_split_sha256", "prompt_sha256", "tokenizer_sha256",
                "base_weight_sha256", "base_revision", "protocol_sha256", "ordered_rows_sha256", "gpu", "packages", "input_condition"):
        if first[key] != second[key]:
            raise ValueError("Before/after condition mismatch: " + key)
    if first["adapter_sha256"] is not None or not second["adapter_sha256"]:
        raise ValueError("Expected original model before and an adapter after")
    for left, right in zip(before[1], after[1]):
        if left["utterance_id"] != right["utterance_id"] or left["reference_feedback"] != right["reference_feedback"]:
            raise ValueError("Paired examples or references differ")


def audio_bootstrap(before, after, repetitions, seed):
    if [row["utterance_id"] for row in before] != [row["utterance_id"] for row in after]:
        raise ValueError("Audio examples are not paired")
    speakers = sorted({row["speaker_id"] for row in before})
    def counts(rows, speaker):
        return np.array([sum(row["counts"][key] for row in rows if row["speaker_id"] == speaker) for key in ("TR", "FR", "FA")], dtype=float)
    a, b = np.array([counts(before, s) for s in speakers]), np.array([counts(after, s) for s in speakers])
    def f1(value):
        denominator = 2 * value[0] + value[1] + value[2]
        return 200 * value[0] / denominator if denominator else 0.
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(speakers), size=len(speakers))
        delta.append(f1(b[selected].sum(axis=0)) - f1(a[selected].sum(axis=0)))
    return {"metric": "phone_detection_f1", "delta_percentage_points": f1(b.sum(axis=0)) - f1(a.sum(axis=0)),
            "ci95": np.percentile(delta, [2.5, 97.5]).tolist(), "resampling_unit": "speaker", "clusters": len(speakers)}


def render_training(config, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    any_data = False
    for stage in ("A", "B"):
        folder = rooted(config["run_root"]) / stage
        path = folder / "training_log.jsonl"
        if path.is_file():
            frame = pd.read_json(path, lines=True)
            axes[0].plot(frame["step"], frame["loss"].rolling(20, min_periods=1).mean(), label=stage)
            any_data = True
        path = folder / "validation_log.jsonl"
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            axes[1].plot([row["step"] for row in rows], [100 * row["metrics"]["f1"] for row in rows], marker="o", label=stage)
    path = rooted(config["run_root"]) / "Q/training_history.jsonl"
    if path.is_file():
        logs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for key in ("loss", "eval_loss"):
            values = [row for row in logs if key in row]
            if values:
                axes[2].plot([row["step"] for row in values], [row[key] for row in values], label=key)
                any_data = True
    for axis, title in zip(axes, ("Audio training loss (rolling 20)", "Audio validation F1 (%)", "Qwen training / validation loss")):
        axis.set(title=title, xlabel="Optimizer step")
        if axis.lines:
            axis.legend()
        else:
            axis.text(.5, .5, "Awaiting real logs", ha="center", va="center", transform=axis.transAxes)
    fig.tight_layout()
    if any_data:
        fig.savefig(output / "training_curves.png", dpi=160)
        fig.savefig(output / "training_curves.svg")
    plt.close(fig)
    return any_data


def report(config):
    protocol, splits, manifest = check_common_split(config)
    folder = rooted(config["report_root"])
    output = folder / "report"
    output.mkdir(parents=True, exist_ok=True)
    protocol_settings = json.loads(rooted(config["feedback_eval_config"]).read_text(encoding="utf-8"))
    paths = {case: folder / "feedback" / case for case in (*CASES, "oracle_Q0", "oracle_Q1")}
    missing = [case for case, path in paths.items() if not (path / "metadata.json").is_file() or
               json.loads((path / "metadata.json").read_text(encoding="utf-8")).get("status") != "completed"]
    audio_missing = [stage for stage in ("A", "B") if not (folder / "audio" / stage / "metrics.json").is_file()]
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest(rooted(config["protocol_dir"]) / "protocol.json"), "counts": protocol["joint_counts"],
        "status": "awaiting_evaluations" if missing or audio_missing else "completed",
        "missing_feedback": missing, "missing_audio": audio_missing,
        "scope": "One training seed; lexical/reference compliance and acoustic MDD measured separately; no claim of learner improvement",
        "shared_prompts": "Speaker holdout, not unseen-prompt testing",
        "training_label_provenance": "Authors' GPT-4o training feedback; authors' human-checked test references; English originals",
        "metrics": {}, "paired_intervals": {}}
    if not missing and not audio_missing:
        evaluations = {case: read_feedback(path) for case, path in paths.items()}
        for left, right in (("A_Q0", "A_Q1"), ("B_Q0", "B_Q1"), ("oracle_Q0", "oracle_Q1")):
            verify_pair(evaluations[left], evaluations[right])
        reference_identity = evaluations["A_Q0"][0]["identity"]
        for case, (metadata, rows) in evaluations.items():
            for key in ("dataset_split_sha256", "ordered_rows_sha256", "base_weight_sha256", "prompt_sha256", "protocol_sha256"):
                if metadata["identity"][key] != reference_identity[key]:
                    raise ValueError("Four-case comparison fingerprint mismatch: " + key)
            expected_adapter = evaluations["A_Q1" if case.endswith("Q1") else "A_Q0"][0]["identity"]["adapter_sha256"]
            if metadata["identity"]["adapter_sha256"] != expected_adapter:
                raise ValueError("Different Qwen weights across audio conditions")
            if {row["utterance_id"] for row in rows} != set(splits["cascade_test_ids"]):
                raise ValueError("Feedback does not cover the common 900 test examples")
        scores = {}
        for case, (_, rows) in evaluations.items():
            result["metrics"][case], scores[case] = summarize(rows)
        for before, after in (("A_Q0", "A_Q1"), ("B_Q0", "B_Q1"), ("A_Q0", "B_Q0"), ("A_Q1", "B_Q1"), ("A_Q0", "B_Q1"), ("oracle_Q0", "oracle_Q1")):
            result["paired_intervals"][before + "_to_" + after] = paired_bootstrap(scores[before], scores[after], repetitions=protocol_settings["bootstrap_repetitions"], seed=protocol_settings["seed"])
        audio_rows, audio_table = {}, {}
        for stage in ("A", "B"):
            path = folder / "audio" / stage
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            if metrics["purpose"] != "scientific_training" or metrics["limited_technical_subset"] or metrics["selection"]["selected_count"] != 900:
                raise ValueError("Audio scores are not the complete scientific holdout")
            if metrics["protocol"]["files"]["protocol"]["sha256"] != result["protocol_sha256"] or digest(path / "predictions.jsonl") != metrics["prediction_sha256"]:
                raise ValueError("Audio results do not match this protocol or checksum")
            audio_rows[stage] = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for case in (stage + "_Q0", stage + "_Q1"):
                if evaluations[case][0]["identity"]["input_condition"]["audio_checkpoint_sha256"] != metrics["checkpoint_sha256"]:
                    raise ValueError("Cascade input and audio metric checkpoints differ")
            audio_table[stage] = {name: 100 * metrics["metrics"][name] if metrics["metrics"][name] is not None else None for name in ("f1", "precision", "recall", "per", "frr", "word_f1")}
        result["audio_metrics_percent"] = audio_table
        result["audio_f1_paired_interval"] = audio_bootstrap(audio_rows["A"], audio_rows["B"], protocol_settings["bootstrap_repetitions"], protocol_settings["seed"])
        table = pd.DataFrame({case: {key: values.get(key) for key in TEXT_METRICS} for case, values in result["metrics"].items()}).T
        table.to_csv(output / "feedback_comparison.csv", index_label="case")
        pd.DataFrame(audio_table).T.to_csv(output / "audio_comparison.csv", index_label="audio_model")
        per_speaker = []
        for case in CASES:
            for speaker in sorted({row["speaker_id"] for row in evaluations[case][1]}):
                summary, _ = summarize([row for row in evaluations[case][1] if row["speaker_id"] == speaker])
                per_speaker.append({"case": case, "speaker_id": speaker, **summary})
        pd.DataFrame(per_speaker).to_csv(output / "feedback_by_speaker.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        table.loc[list(CASES), ["suggestion_rougeL_on_reference_error_rows", "suggestion_bleu2", "grounded_json_percent"]].plot.bar(ax=axes[0], rot=15)
        axes[0].set(title="Four real cascade conditions", ylabel="Score (0–100)")
        pd.DataFrame(audio_table).T[["f1", "per", "frr"]].plot.bar(ax=axes[1], rot=0)
        axes[1].set(title="Acoustic model: F1 up; PER / FRR down", ylabel="Percent")
        fig.tight_layout()
        fig.savefig(output / "before_after.png", dpi=160)
        fig.savefig(output / "before_after.svg")
        plt.close(fig)
        # Fixed examples: first utterance of each speaker, not selected for improvement.
        fixed = [min(row["utterance_id"] for row in evaluations["A_Q0"][1] if row["speaker_id"] == speaker)
                 for speaker in sorted({row["speaker_id"] for row in evaluations["A_Q0"][1]})]
        save(output / "fixed_examples.json", {case: [row for row in evaluations[case][1] if row["utterance_id"] in fixed] for case in CASES})
    render_training(config, output)
    save(output / "comparison.json", result)
    sections = ["<h1>Wav2vec2 + Qwen3-0.6B: measured comparison</h1>",
                "<p>Status: " + html.escape(result["status"]) + "</p>",
                "<p>" + html.escape(result["scope"]) + "</p>"]
    if result["status"] != "completed":
        sections.append("<p>Awaiting real training/evaluations. Missing: " + html.escape(", ".join(missing + audio_missing)) + ". No after-training scores have been filled in.</p>")
    else:
        sections.append(pd.read_csv(output / "feedback_comparison.csv").round(4).to_html(index=False))
        sections.append(pd.read_csv(output / "audio_comparison.csv").round(4).to_html(index=False))
        sections.append("<pre>" + html.escape(json.dumps(result["paired_intervals"], indent=2)) + "</pre>")
    for name in ("before_after.png", "training_curves.png"):
        if (output / name).is_file():
            sections.append('<img style="max-width:100%" src="data:image/png;base64,' + base64.b64encode((output / name).read_bytes()).decode() + '">')
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1300px;margin:32px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>' + "\n".join(sections), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = report(json.loads(args.config.read_text(encoding="utf-8")))
    print(json.dumps({"status": result["status"], "missing_feedback": result["missing_feedback"], "missing_audio": result["missing_audio"]}, indent=2))


if __name__ == "__main__":
    main()
