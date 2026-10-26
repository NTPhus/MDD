"""Apply the existing demo's lexical guard to saved comparison predictions.

No generation or rewriting takes place. This is a delivery compatibility audit,
not a grammar-quality or meaning-preservation classifier. An unchanged second
field lets the existing per-field guard inspect each task independently.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sentence_rewrite import parse_rewrite


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_prediction(row):
    task = row["task"]
    prefix = {"grammar_corrected": "grammar", "natural": "natural"}[task]
    prediction = row.get("prediction")
    result = {key: row.get(key) for key in ("id", "source", "task", "cohort", "prediction", "error")}
    references = row.get("references", [])
    result["matches_published_reference"] = (
        prediction is not None and " ".join(prediction.split()) in {" ".join(ref.split()) for ref in references}
    ) if references else None
    if prediction is None:
        return {**result, "delivered": False, "reason": "native_contract_unavailable",
                "changed": False, "format_changed": False, "delivered_prediction": None}
    candidate = {"grammar_corrected": row["source"], "natural": row["source"],
                 "explanation": "Saved model output; lexical delivery audit."}
    candidate[task] = prediction
    guarded = parse_rewrite(json.dumps(candidate, ensure_ascii=False), True, row["source"])
    accepted = guarded.get(prefix + "_status") == "ready" and bool(guarded.get(task))
    return {**result, "delivered": accepted, "reason": "accepted" if accepted else "lexical_guard_withheld",
            "changed": bool(accepted and guarded.get(prefix + "_changed")),
            "format_changed": bool(accepted and guarded.get(prefix + "_format_changed")),
            "delivered_prediction": guarded.get(task) if accepted else None}


def summarize(rows):
    output = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        output[cohort] = {}
        for task in ("grammar_corrected", "natural"):
            chosen = [row for row in rows if row["cohort"] == cohort and row["task"] == task]
            counts = Counter(row["reason"] for row in chosen)
            output[cohort][task] = {
                "rows": len(chosen), "native_available": sum(row["prediction"] is not None for row in chosen),
                "delivered": sum(row["delivered"] for row in chosen),
                "lexical_changes_delivered": sum(row["changed"] for row in chosen),
                "format_only_changes_delivered": sum(row["format_changed"] for row in chosen),
                "reference_matching_native_outputs": sum(row["matches_published_reference"] is True for row in chosen),
                "reference_matching_outputs_withheld": sum(
                    row["matches_published_reference"] is True and not row["delivered"] for row in chosen),
                "reasons": dict(counts),
            }
    return output


def audit(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {"schema": "saved-prediction-delivery-audit-v1", "manifest_sha256": digest(manifest_path),
              "audit_code_sha256": digest(__file__),
              "guard_sha256": digest(ROOT / "scripts/sentence_rewrite.py"),
              "production_orchestrator_sha256": digest(ROOT / "scripts/grammar_inference.py"),
              "limitations": [
                  "The benchmark's frozen generation budget differs from the production worker; this audits saved outputs, not a new live run.",
                  "The lexical guard is a conservative delivery rule, not an independent semantic or grammatical correctness metric.",
                  "Reference-matching output means exact equality after whitespace normalization with at least one published reference; this does not grade other unmatched suggestions as wrong.",
                  "CoEdIT is audited with the same guard only as a compatibility diagnostic; it is not integrated into the demo.",
                  "Each field is checked independently with the unmodified source in the other field, matching the production per-field guard.",
              ], "models": {}}
    for arm in manifest["models"]:
        predictions_path = directory / arm / "predictions.jsonl"
        metadata = json.loads((directory / arm / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "completed" or metadata["manifest_sha256"] != digest(manifest_path):
            raise ValueError("Incomplete or mismatched saved predictions: " + arm)
        if metadata["predictions_sha256"] != digest(predictions_path):
            raise ValueError("Saved predictions checksum differs: " + arm)
        observations = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [r["id"] for r in observations] != [r["id"] for r in manifest["rows"]]:
            raise ValueError("Saved prediction case order differs: " + arm)
        rows = [inspect_prediction(row) for row in observations]
        result["models"][arm] = {"predictions_sha256": digest(predictions_path), "summary": summarize(rows), "rows": rows}
    destination = directory / "delivery_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/generated/model-comparison-20261024/grammar")
    args = parser.parse_args()
    print(audit(args.directory))


if __name__ == "__main__":
    main()
