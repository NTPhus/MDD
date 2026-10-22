"""Test composing the TWO trained components, without calling it task-tuned P1."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import LatentDataset, load, read_json, save
from scripts.projector_inference import ProjectorPipeline
from scripts.l2arctic_plus_data import rooted, load as load_native
from scripts.evaluate_projector_checkpoint import select_balanced, run_case
from scripts.prepare_feedback_data import digest


def evaluate(settings):
    import torch
    from peft import PeftModel
    config = read_json(rooted(settings["projector_config"]))
    root = rooted(settings["output_dir"])
    shared = read_json(root / "benchmark_manifest.json")
    if shared["settings"] != settings:
        raise ValueError("Transfer settings differ from the original benchmark")
    parts, manifest = load(config)
    if manifest != shared["dataset"]:
        raise ValueError("Transfer dataset differs")
    rows = select_balanced(parts["test"], settings["feedback_per_speaker"], settings["seed"], "feedback")
    cascade = read_json(rooted(config["cascade_config"]))
    qconfig = read_json(rooted(cascade["feedback_config"]))
    _, native_manifest = load_native(rooted(qconfig["dataset_dir"]))
    adapter = rooted(cascade["run_root"]) / "Q/adapter"
    provenance = read_json(adapter.parent / "training_provenance.json")
    if (provenance["status"] != "completed" or provenance["purpose"] != "scientific_training"
            or provenance["dataset"] != native_manifest or provenance["base_weight_sha256"] != shared["base_weight_sha256"]
            or provenance["adapter_sha256"] != digest(adapter / "adapter_model.safetensors")):
        raise ValueError("Invalid cascade adapter provenance")
    pipeline = ProjectorPipeline(config, "alignment")
    pipeline.model.llm = PeftModel.from_pretrained(pipeline.model.llm, str(adapter), local_files_only=True,
                                                  is_trainable=False).eval().requires_grad_(False)
    dataset = LatentDataset(rows, pipeline.tokenizer, config, "feedback")
    identity = {**shared, "purpose": "balanced_generation_pilot", "task": "english_feedback",
                "generation": {"do_sample": False, "max_new_tokens": settings["feedback_max_new_tokens"]},
                "extension_evaluation_code_sha256": digest(Path(__file__)),
                "shared_evaluator_code_sha256": digest(ROOT / "scripts/evaluate_projector_checkpoint.py"),
                "adapter_sha256": provenance["adapter_sha256"], "feedback_task_trained": False,
                "training_input_condition": "Cascade adapter trained on oracle phone text, not projected audio embeddings",
                "scope": "Zero-shot composition of aligned projector and cascade LoRA; NOT P1 task-specific projector tuning"}
    torch.cuda.reset_peak_memory_stats()
    def predict(index, row):
        batch = {k: v.cuda() for k, v in dataset[index].items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=pipeline.dtype):
            loss = float(pipeline.model(**batch))
        count = batch["target_ids"].numel()
        return {"reference_feedback": row["feedback"], "word_order": row["word_order"],
                "nll_sum": loss * count, "target_tokens": count,
                **pipeline.generate(batch["features"], row, "feedback", max_new_tokens=settings["feedback_max_new_tokens"])}
    run_case(root, "feedback_projector_cascade_adapter_transfer", identity, rows, predict)
    save(root / "transfer_execution.json", {"status": "completed", "rows": len(rows), "training_started": False,
         "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scope": identity["scope"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/projector_validation.json")
    args = parser.parse_args()
    evaluate(read_json(args.config))
