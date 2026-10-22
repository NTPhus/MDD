"""Local raw-audio -> frozen wav2vec2 B -> projector -> Qwen inference, no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import CONFIG, encoder_identity, prompt_parts, read_json, save
from scripts.l2arctic_plus_data import rooted
from scripts.train_projector import scientific_stage
from scripts.published_feedback_metrics import parse_prediction
from scripts.prepare_feedback_data import digest


class ProjectorPipeline:
    def __init__(self, config, stage="auto"):
        import torch
        from transformers import AutoTokenizer, set_seed
        from scripts.projector_model import build_model
        if stage == "auto":
            path = rooted(config["run_root"]) / "feedback/training_provenance.json"
            stage = "feedback" if path.exists() and read_json(path).get("status") == "completed" else "alignment"
        self.weights, self.training = scientific_stage(config, stage)
        for key, path in (("base_weight_sha256", rooted(config["model_path"]) / "model.safetensors"),
                          ("tokenizer_sha256", rooted(config["model_path"]) / "tokenizer.json"),
                          ("feature_manifest_sha256", rooted(config["feature_dir"]) / "manifest.json")):
            if self.training["identity"][key] != digest(path):
                raise ValueError("Checkpoint input identity changed: " + key)
        self.encoder_identity = encoder_identity(config)
        self.config, self.stage = config, stage
        torch.set_num_threads(config["cpu_threads"])
        set_seed(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(rooted(config["model_path"])), local_files_only=True)
        self.model = build_model(config, stage, training=False, initial=self.weights).eval().requires_grad_(False)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.encoder = self.extractor = None

    def encode_audio(self, audio_path):
        import numpy as np
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        sys.path.insert(0, str(ROOT / "src"))
        from mdd.runtime import load_audio
        if self.encoder is None:
            path = str(rooted(self.config["encoder_checkpoint"]))
            self.encoder = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).wav2vec2.cpu().eval().requires_grad_(False)
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        waveform = load_audio({"audio_path": str(Path(audio_path).resolve())})
        batch = self.extractor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            values = self.encoder(**batch).last_hidden_state[0].float().numpy().astype(np.float16)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite wav2vec2 features")
        # Match the scientifically frozen float16 feature-cache representation exactly.
        return torch.from_numpy(values.astype(np.float32)).unsqueeze(0)

    def generate(self, features, row, task, *, max_new_tokens):
        import torch
        prefix, suffix = prompt_parts(self.tokenizer, row, task)
        count = len(prefix) + features.shape[1] + len(suffix)
        if max_new_tokens < 1 or count + max_new_tokens > 4096:
            raise ValueError("Input plus generation exceeds the fixed 4096-token context; audio is never truncated")
        batch = {"features": features.cuda(), "prefix_ids": torch.tensor([prefix], device="cuda"),
                 "suffix_ids": torch.tensor([suffix], device="cuda")}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            generated = self.model.generate_response(**batch, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        finished = bool(len(generated) and generated[-1].item() == self.tokenizer.eos_token_id)
        parsed, error = None, None
        if task == "feedback":
            try:
                if not finished:
                    raise ValueError("Token budget reached without EOS")
                parsed = parse_prediction(text)
            except (ValueError, TypeError) as failure:
                error = str(failure)
        return {"raw_text": text, "prediction": parsed, "valid_json": task == "feedback" and error is None,
                "validation_error": error, "finished_with_eos": finished, "generated_tokens": len(generated),
                "prompt_tokens": count, "generation_sec": elapsed}

    def predict(self, audio_path, reference_text=None, *, task="feedback", max_new_tokens=512):
        if task == "feedback" and not reference_text:
            raise ValueError("Feedback mode requires the reference sentence")
        features = self.encode_audio(audio_path)
        result = self.generate(features, {"reference_text": reference_text or ""}, task, max_new_tokens=max_new_tokens)
        return {"task": task, "checkpoint_stage": self.stage, "feedback_task_trained": self.stage == "feedback",
                "audio_frames": features.shape[1], "audio_hidden_size": features.shape[2],
                "status_note": "Task-tuned feedback checkpoint" if self.stage == "feedback" else
                    "Alignment only: feedback output is experimental and has not received feedback-task training",
                **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--stage", choices=("auto", "alignment", "feedback"), default="auto")
    parser.add_argument("--task", choices=("alignment", "feedback"), default="feedback")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from filelock import FileLock
    with FileLock(str(ROOT / "data/generated/cascade_gpu.lock"), timeout=1):
        pipeline = ProjectorPipeline(read_json(args.config), args.stage)
        result = pipeline.predict(args.audio, args.reference, task=args.task, max_new_tokens=args.max_new_tokens)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
