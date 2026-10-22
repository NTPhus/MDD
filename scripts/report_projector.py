"""Honest pending/completed dashboard and paired projector/cascade comparison."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import CONFIG, load, read_json, read_rows, save
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest
from scripts.published_feedback_metrics import summarize, paired_bootstrap


def read_evaluation(folder):
    metadata = read_json(folder / "metadata.json")
    rows = read_rows(folder / "predictions.jsonl")
    identity = metadata["identity"]
    if (metadata["status"] != "completed" or identity["purpose"] != "held_out_benchmark"
            or identity["split"] != "test" or len(rows) != identity["full_split_rows"] or len(rows) != 900
            or len({row["utterance_id"] for row in rows}) != len(rows)
            or digest(folder / "predictions.jsonl") != metadata["predictions_sha256"]):
        raise ValueError("Scientific report requires a full verified 900-row test: " + str(folder))
    return metadata, rows


def verify_pair(left, right):
    a, b = left[0]["identity"], right[0]["identity"]
    for key in ("task", "split", "rows", "full_split_rows", "dataset_split_sha256", "protocol_sha256", "ordered_rows_sha256",
                "base_weight_sha256", "tokenizer_sha256", "input_condition", "prompt_sha256", "generation",
                "precision_profile", "seed", "gpu", "packages", "model_code_sha256", "evaluation_code_sha256", "context_length", "metrics_code_sha256"):
        if a[key] != b[key]:
            raise ValueError("Projector before/after mismatch: " + key)
    if a["case"] != "P0" or b["case"] != "P1":
        raise ValueError("Expected aligned P0 versus task-tuned P1")
    for first, second in zip(left[1], right[1]):
        if first["utterance_id"] != second["utterance_id"] or first["reference_feedback"] != second["reference_feedback"]:
            raise ValueError("Unpaired projector references")


def report(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    parts, dataset = load(config)
    folder = rooted(config["report_root"])
    output = folder / "report"
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, stage in zip(axes, ("alignment", "feedback")):
        path = rooted(config["run_root"]) / stage / "training_history.jsonl"
        if path.exists():
            rows = read_rows(path)
            # A resumed optimizer step supersedes any stale log observation for that step.
            rows = sorted({row["step"]: row for row in rows}.values(), key=lambda row: row["step"])
            axis.plot([r["step"] for r in rows], [r["loss"] for r in rows], alpha=.5, label="train")
            val = [r for r in rows if "validation_loss" in r]
            axis.plot([r["step"] for r in val], [r["validation_loss"] for r in val], marker="o", label="validation")
            axis.legend()
        else:
            axis.text(.5, .5, "Awaiting real training logs", ha="center", transform=axis.transAxes)
        axis.set(title=stage, xlabel="Optimizer step", ylabel="Response token cross entropy")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / ("training_curves." + extension), dpi=150)
    plt.close(fig)
    figures.append("training_curves.png")
    results, evaluations = [], {}
    for case in ("P0", "P1"):
        path = folder / case / "metadata.json"
        if path.exists() and read_json(path)["status"] == "completed":
            evaluations[case] = read_evaluation(path.parent)
            meta, rows = evaluations[case]
            if meta["identity"]["protocol_sha256"] != dataset["protocol_sha256"]:
                raise ValueError("Evaluation protocol differs from current frozen data")
            if [(r["utterance_id"], r["source_row_sha256"], r["reference_feedback"]) for r in rows] != [(r["utterance_id"], r["source_row_sha256"], r["feedback"]) for r in parts["test"]]:
                raise ValueError("Evaluation references/IDs differ from the frozen test")
    state = {"status": "awaiting_evaluations", "counts": dataset["counts"], "completed_cases": list(evaluations),
             "scope": "P0 has transcript alignment only; P1 adds task training. Same frozen B encoder, speaker holdout and greedy decoding. Not a reproduction of the paper's larger models/data.",
             "metric_scope": "Word-key detection and feedback lexical agreement, not phone-slot F1 or human-rated pedagogical quality."}
    if len(evaluations) == 2:
        verify_pair(evaluations["P0"], evaluations["P1"])
        scores = {}
        for case, (_, rows) in evaluations.items():
            summary, scores[case] = summarize(rows)
            results.append({"case": case, **summary})
        state.update(status="completed", before_after=paired_bootstrap(scores["P0"], scores["P1"],
            repetitions=config["bootstrap_repetitions"], seed=config["seed"]))
        cascade = read_json(rooted(config["cascade_config"]))
        path = rooted(cascade["report_root"]) / "B_Q1"
        if (path / "metadata.json").exists() and read_json(path / "metadata.json")["status"] == "completed":
            meta, rows = read_evaluation(path)
            pmeta, prows = evaluations["P1"]
            first, second = meta["identity"], pmeta["identity"]
            if first["input_condition"]["audio_checkpoint_sha256"] != second["input_condition"]["encoder"]["files"]["model.safetensors"]:
                raise ValueError("Cascade/Projector encoder weights differ")
            for key in ("base_weight_sha256", "tokenizer_sha256", "gpu", "packages", "precision_profile"):
                if first[key] != second[key]:
                    raise ValueError("Cascade/Projector mismatch: " + key)
            if first["protocol"]["generation"] != second["generation"]:
                raise ValueError("Cascade/Projector decoding differs")
            if [(r["utterance_id"], r["reference_feedback"]) for r in rows] != [(r["utterance_id"], r["reference_feedback"]) for r in prows]:
                raise ValueError("Cascade/Projector held-out references differ")
            summary, cascade_scores = summarize(rows)
            results.append({"case": "B_Q1 cascade", **summary})
            state["projector_vs_cascade"] = paired_bootstrap(cascade_scores, scores["P1"],
                repetitions=config["bootstrap_repetitions"], seed=config["seed"])
            state["comparison_scope"] = "Different architectures/prompts with identical supervision and test references; projector has an additional transcript alignment stage. Not an equal-compute ablation."
        pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
        pd.DataFrame(results).set_index("case")[["json_valid_percent", "feedback_word_f1_percent", "suggestion_rougeL_on_reference_error_rows"]].plot.bar(figsize=(11, 4), rot=0)
        plt.ylabel("Percent / 0–100 lexical score")
        plt.tight_layout()
        for ext in ("png", "svg"):
            plt.savefig(output / ("comparison." + ext), dpi=150)
        plt.close()
        figures.append("comparison.png")
        pd.DataFrame([{"case": case, "speaker": speaker, **summarize([r for r in rows if r["speaker_id"] == speaker])[0]}
                      for case, (_, rows) in evaluations.items() for speaker in sorted({r["speaker_id"] for r in rows})]).to_csv(output / "per_speaker.csv", index=False)
        # Fixed first sample per speaker, not chosen by whether a model wins.
        samples = [next(r for r in parts["test"] if r["speaker_id"] == s) for s in sorted({r["speaker_id"] for r in parts["test"]})]
        save(output / "examples.json", [{"utterance_id": sample["utterance_id"], **{case: next(r for r in rows if r["utterance_id"] == sample["utterance_id"])
             for case, (_, rows) in evaluations.items()}} for sample in samples])
    save(output / "summary.json", state)
    images = "".join('<img style="max-width:100%" src="data:image/png;base64,' + base64.b64encode((output / name).read_bytes()).decode() + '">' for name in figures)
    table = pd.DataFrame(results).to_html(index=False) if results else "<p>Chưa có benchmark hoàn tất; chưa có điểm trước/sau để kết luận.</p>"
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Projector / cascade</title><h1>Projector + Qwen3-0.6B</h1><pre>' + html.escape(json.dumps(state, ensure_ascii=False, indent=2)) + '</pre>' + table + images, encoding="utf-8")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(report(read_json(args.config)), ensure_ascii=False, indent=2))
