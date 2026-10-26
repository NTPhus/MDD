"""Post-hoc format-only analysis of frozen editing-model outputs.

The native comparison remains unchanged. This audit can extract an existing
transcript string from exactly {task, transcript} when Qwen emits the wrong
schema. It never repairs words, generates text, edits prompts or changes the
service parser. The alternate schema was noticed after inspecting initial test
outputs, so these are explicitly secondary diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_grammar_models as comparison


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate field")
        value[key] = item
    return value


def extract_content(row):
    if row.get("prediction") is not None:
        return row["prediction"], "native_contract"
    if row.get("error") != "invalid_schema" or not row.get("finished"):
        return None, "unavailable"
    try:
        value = json.loads(row["raw"], object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None, "unavailable"
    if (not isinstance(value, dict) or set(value) != {"task", "transcript"}
            or value["task"] != row["task"] or not isinstance(value["transcript"], str)
            or not value["transcript"].strip()):
        return None, "unavailable"
    return value["transcript"].strip(), "posthoc_transcript_field"


def quality(rows, hypotheses, destination):
    if not rows:
        return None
    from sacrebleu.metrics import CHRF
    reference_count = max(len(row["references"]) for row in rows)
    references = [[row["references"][min(index, len(row["references"]) - 1)] for row in rows]
                  for index in range(reference_count)]
    metric = CHRF()
    chrf = metric.corpus_score(hypotheses, references)
    normalize = lambda text: " ".join(text.split())
    return {"rows": len(rows), "chrf": chrf.score, "chrf_signature": str(metric.get_signature()),
            "exact_reference_rate": sum(normalize(hypothesis) in {normalize(ref) for ref in row["references"]}
                                        for row, hypothesis in zip(rows, hypotheses)) / len(rows),
            "gleu": comparison.gleu(rows, hypotheses, destination)}


def audit(directory):
    directory = Path(directory)
    comparison.OUT = directory
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if comparison.digest(directory / "source-audit/official_jfleg_gleu.py") != manifest["gleu_sha256"]:
        raise ValueError("Frozen official scorer differs")
    result = {
        "schema": "posthoc-format-only-content-audit-v1", "manifest_sha256": comparison.digest(manifest_path),
        "audit_code_sha256": comparison.digest(__file__), "no_generation": True,
        "posthoc_defined_after_initial_output_inspection": True,
        "rule": "Keep native predictions; otherwise extract only a finished raw JSON object with exactly task/transcript fields, matching task, nonempty string. No word repair or other schema heuristics.",
        "limitations": [
            "Secondary post-hoc content diagnostic, not the frozen native service result or a production parser change.",
            "Reference similarity is not human grammar accuracy, factual preservation or naturalness assessment.",
            "Unavailable content is explicitly replaced with the original source only in the labelled all-row fallback metric.",
            "The same reference selection, author GLEU scorer and Moses tokenization are used as in the frozen native comparison.",
        ], "models": {}}
    for arm in comparison.ARMS:
        predictions_path = directory / arm / "predictions.jsonl"
        metadata = json.loads((directory / arm / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "completed" or metadata["manifest_sha256"] != comparison.digest(manifest_path):
            raise ValueError("Incomplete saved predictions: " + arm)
        if metadata["predictions_sha256"] != comparison.digest(predictions_path):
            raise ValueError("Saved predictions changed: " + arm)
        observations = comparison.read_jsonl(predictions_path)
        if [row["id"] for row in observations] != [row["id"] for row in manifest["rows"]]:
            raise ValueError("Saved cases differ: " + arm)
        rows = []
        for observed in observations:
            content, source = extract_content(observed)
            rows.append({**observed, "content": content, "content_source": source})
        tasks = {}
        for task in ("grammar_corrected", "natural"):
            chosen = [row for row in rows if row["cohort"] == "published_heldout" and row["task"] == task]
            available = [row for row in chosen if row["content"] is not None]
            destination = directory / "metric-inputs-content-audit" / arm / task
            tasks[task] = {
                "rows": len(chosen), "content_available": len(available),
                "sources": dict(Counter(row["content_source"] for row in chosen)),
                "quality_on_available_only": quality(available, [row["content"] for row in available], destination / "available"),
                "quality_with_explicit_source_fallback": quality(chosen,
                    [row["content"] if row["content"] is not None else row["source"] for row in chosen], destination / "fallback"),
            }
        result["models"][arm] = {
            "predictions_sha256": comparison.digest(predictions_path), "tasks": tasks,
            "rows": [{key: row.get(key) for key in ("id", "source", "task", "cohort", "prediction", "content", "content_source", "error")}
                     for row in rows],
        }
    destination = directory / "content_audit.json"
    comparison.save(destination, result)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=comparison.OUT)
    args = parser.parse_args()
    print(audit(args.directory))


if __name__ == "__main__":
    main()
