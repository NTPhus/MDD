"""Raw audio through aligned projector + existing cascade Qwen LoRA, experimental transfer."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import CONFIG, read_json, save
from scripts.projector_inference import ProjectorPipeline
from scripts.l2arctic_plus_data import rooted, load as load_native
from scripts.prepare_feedback_data import digest


def compose(config):
    from peft import PeftModel
    cascade = read_json(rooted(config["cascade_config"]))
    qconfig = read_json(rooted(cascade["feedback_config"]))
    _, manifest = load_native(rooted(qconfig["dataset_dir"]))
    adapter = rooted(cascade["run_root"]) / "Q/adapter"
    provenance = read_json(adapter.parent / "training_provenance.json")
    pipeline = ProjectorPipeline(config, "alignment")
    if (provenance["status"] != "completed" or provenance["purpose"] != "scientific_training"
            or provenance["dataset"] != manifest
            or provenance["base_weight_sha256"] != pipeline.training["identity"]["base_weight_sha256"]
            or provenance["adapter_sha256"] != digest(adapter / "adapter_model.safetensors")):
        raise ValueError("Invalid cascade adapter provenance")
    pipeline.model.llm = PeftModel.from_pretrained(pipeline.model.llm, str(adapter), is_trainable=False,
                                                  local_files_only=True).eval().requires_grad_(False)
    return pipeline, provenance["adapter_sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from filelock import FileLock
    with FileLock(str(ROOT / "data/generated/cascade_gpu.lock"), timeout=1):
        pipeline, adapter_sha = compose(read_json(args.config))
        result = pipeline.predict(args.audio, args.reference, max_new_tokens=args.max_new_tokens)
        result.update(composition="aligned_projector_plus_cascade_adapter_zero_shot_transfer",
                      cascade_adapter_sha256=adapter_sha,
                      status_note="Existing cascade LoRA is trained on phone text, not audio embeddings; composition is NOT task-tuned P1")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
