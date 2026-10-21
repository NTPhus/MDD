"""Evaluate local coaching contract compliance on frozen held-out pairs; no TTS/API calls."""

import argparse
import json
import os
from pathlib import Path

if __package__:
    from .llm_client import LLMClient, ROOT
    from .prepare_feedback_data import load_partitions
else:
    from llm_client import LLMClient, ROOT
    from prepare_feedback_data import load_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed/feedback-v1")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/pretrained/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve previous evaluations")
    parts, manifest = load_partitions(args.dataset_dir)
    os.environ.update(LLM_BACKEND="local", LLM_LOCAL_MODEL_PATH=str(args.model_path.resolve()),
                      LLM_LOCAL_ADAPTER_PATH=str(args.adapter.resolve()) if args.adapter else "")
    client = LLMClient()
    observations = []
    from transformers import set_seed
    for row in parts[args.split]:
        set_seed(args.seed)
        generated = client.generate_feedback(row["assessment"])
        observations.append({"utterance_id": row["assessment"]["utterance_id"], "group_id": row["group_id"],
                             "assessment": row["assessment"], "reference_feedback": row["feedback"],
                             "feedback": generated.feedback, "metadata": generated.metadata(row["assessment"]),
                             "teacher_quality_review": None})
    eligible = [item for item in observations if item["assessment"]["status"] == "ok"]
    report = {"split": args.split, "model": "Qwen/Qwen3-0.6B", "adapter": str(args.adapter) if args.adapter else None,
              "dataset_sha256": manifest["sha256"][args.split], "rows": observations,
              "eligible_count": len(eligible),
              "valid_model_feedback_count": sum(item["metadata"]["feedback_source"] == "llm" for item in eligible),
              "first_attempt_valid_count": sum(item["metadata"]["feedback_source"] == "llm" and not item["metadata"]["validation_errors"] for item in eligible),
              "measures": "JSON and assessment-reference compliance, not correctness of teaching advice or learner improvement"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(observations)} held-out generations for comparison and teacher review.")


if __name__ == "__main__":
    main()
