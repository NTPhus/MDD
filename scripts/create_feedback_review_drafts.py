"""Create UNREVIEWED coaching drafts from cached validation inference, offline.

The template is a review starting point. It is not teacher supervision and this
command cannot approve a row or write a train/validation/test dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mdd.demo_feedback import build_assessment
from src.mdd.demo_inference import canonical_comparison
from scripts.llm_client import validate_local_feedback
from scripts.prepare_feedback_data import digest, prompt_key, split_rows, write_rows
from scripts.train_feedback_model import encode_pair
from scripts.validate_contracts import validate_pair


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class UnavailableDraft(ValueError):
    pass


def template_feedback(assessment):
    tips = []
    for error in assessment["errors"][:3]:
        word = assessment["words"][error["word_index"]]["text"]
        expected = "/" + error["expected_ipa"] + "/" if error["expected_ipa"] else str(error["expected_phone"])
        observed = "/" + error["observed_ipa"] + "/" if error["observed_ipa"] else str(error["observed_phone"])
        if error["type"] == "substitution":
            explanation = f"Ở từ {word}, âm mục tiêu là {expected}; model nhận diện {observed}. Cần nghe lại bản thu để xác nhận khác biệt."
        elif error["type"] == "deletion":
            explanation = f"Model chưa nhận ra âm mục tiêu {expected} trong từ {word}. Cần nghe lại để kiểm tra âm này có thực sự bị thiếu."
        else:
            explanation = f"Model nhận thêm âm {observed} khi đối chiếu từ {word}. Cần nghe lại để kiểm tra đây có phải âm thừa."
        tips.append({"error_id": error["id"], "explanation": explanation,
                     "instruction": f"Nghe cách đọc mẫu tiếng Anh Mỹ của từ {word}, đọc chậm từng âm rồi nối lại cả từ và câu. Ghi âm và nghe lại để so sánh.",
                     "practice_words": [word]})
    summary = ("Model gợi ý một số vị trí cần nghe lại; đây chưa phải kết luận chắc chắn về lỗi phát âm."
               if assessment["errors"] else "Model chưa đánh dấu khác biệt âm vị trong bản thu này; kết quả chưa bảo đảm mọi âm đều đúng.")
    return {"schema_version": "1.0", "utterance_id": assessment["utterance_id"], "language": "vi",
            "summary": summary, "tips": tips,
            "tts_segments": [{"language": assessment["target_locale"], "purpose": "reference", "text": assessment["reference_text"]},
                             {"language": "vi-VN", "purpose": "feedback", "text": summary}]}


def build_draft(prediction, annotation, run_id, prediction_file, weight_sha):
    text = annotation["prompt_text"]
    reference = {key: annotation[key] for key in ("prompt_text", "canonical_phones", "canonical_word_indices", "word_intervals") if key in annotation}
    if prediction["canonical_phones"] != reference["canonical_phones"]:
        raise ValueError("Cached canonical phones differ from frozen annotation")
    result = {"model_id": "run:" + run_id + ":best", "reference_text": text,
              "model": {"id": "run:" + run_id + ":best", "kind": "phoneme", "experimental": False},
              "phonemes": [{"phone": phone} for phone in prediction["predicted_phones"]],
              "comparison": canonical_comparison(text, reference, prediction["predicted_phones"])}
    identifier = run_id + ":" + prediction["utterance_id"]
    assessment, notices = build_assessment(result, identifier)
    if assessment["status"] != "ok":
        raise UnavailableDraft("Word mapping/decoded phones unavailable")
    feedback = template_feedback(assessment)
    validate_pair(assessment, feedback)
    validate_local_feedback(assessment, feedback)
    return {"group_id": "prompt-" + hashlib.sha256(prompt_key(assessment).encode()).hexdigest()[:20],
            "source": "cached-validation-inference-template-draft", "assessment": assessment, "feedback": feedback,
            "review": {"approved": False, "reviewer": None, "reviewed_at": None},
            "provenance": {"mdd_utterance_id": prediction["utterance_id"], "speaker_id": prediction["speaker_id"],
                           "native_language": annotation.get("native_language"), "audio_path": prediction["audio_path"],
                           "audio_sha256": prediction["audio_sha256"], "prediction_file": prediction_file,
                           "checkpoint_weight_sha256": weight_sha, "mdd_split": "validation", "feedback_source": "offline_template"},
            "review_context": {"canonical_phones": reference["canonical_phones"], "predicted_phones": prediction["predicted_phones"],
                               "human_perceived_phones": annotation["perceived_phones"],
                               "human_annotation_events": annotation["gold_error_events"],
                               "notices": notices,
                               "needs_review": ["Nghe audio và đối chiếu nhãn thủ công; cảnh báo model có thể sai.",
                                                "Sửa assessment nếu cần; giữ error_id, từ, loại lỗi và âm liên kết đúng.",
                                                "Bổ sung hướng dẫn môi/lưỡi và bài luyện cụ thể; template hiện chỉ gợi ý chung.",
                                                "Sửa đoạn giọng đọc tiếng Việt để khớp hướng dẫn đã duyệt."]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", default=[
        ROOT / "runs/windows-notebook-seed42-B-fold1/predictions_validation_best.jsonl",
        ROOT / "runs/windows-notebook-seed42-A/predictions_validation_best.jsonl"])
    parser.add_argument("--per-prompt", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Choose a new draft file within this workspace")
    if not 1 <= args.per_prompt <= 10 or args.max_examples < 1:
        raise ValueError("Invalid draft selection size")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/pretrained/Qwen3-0.6B"), local_files_only=True, trust_remote_code=False)
    annotations_path = ROOT / "data/processed/l2arctic-v1/annotations.jsonl"
    annotations = {row["utterance_id"]: row for row in read_rows(annotations_path)}
    groups, rejected, sources = defaultdict(list), Counter(), []
    for priority, path in enumerate(args.predictions):
        path = path.resolve()
        if not path.is_relative_to(ROOT / "runs") or path.name != "predictions_validation_best.jsonl":
            raise ValueError("Use saved best-checkpoint validation inference; test predictions are excluded")
        run = path.parent
        metadata = json.loads((run / "best_checkpoint/metadata.json").read_text(encoding="utf-8"))
        if metadata.get("purpose") != "scientific_training":
            raise ValueError("Only scientific MDD checkpoints are eligible")
        weight_sha = digest(run / "best_checkpoint/model.safetensors")
        rows = read_rows(path)
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "records": len(rows), "checkpoint_weight_sha256": weight_sha})
        for prediction in rows:
            if prediction.get("status") != "ok":
                rejected["failed_inference"] += 1
                continue
            annotation = annotations[prediction["utterance_id"]]
            try:
                draft = build_draft(prediction, annotation, run.name, path.relative_to(ROOT).as_posix(), weight_sha)
                encoded = encode_pair(tokenizer, draft, args.max_length)
            except UnavailableDraft:
                rejected["unavailable_word_mapping"] += 1
                continue
            except ValueError as error:
                if "do not truncate" not in str(error):
                    raise
                rejected["too_long"] += 1
                continue
            draft["provenance"]["training_tokens"] = len(encoded["input_ids"])
            rank = hashlib.sha256((str(args.seed) + draft["assessment"]["utterance_id"]).encode()).hexdigest()
            groups[draft["group_id"]].append((priority, rank, draft))
    selected = []
    for group in sorted(groups, key=lambda value: hashlib.sha256((str(args.seed) + value).encode()).hexdigest()):
        selected.extend(row for _, _, row in sorted(groups[group], key=lambda item: item[:2])[:args.per_prompt])
        if len(selected) >= args.max_examples:
            selected = selected[:args.max_examples]
            break
    if not selected:
        raise ValueError("No eligible drafts")
    assert all(row["review"] == {"approved": False, "reviewer": None, "reviewed_at": None} for row in selected)
    write_rows(output, selected)
    summary = {"status": "unreviewed_drafts_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "drafts": len(selected), "approved": 0, "unique_prompt_groups": len({row["group_id"] for row in selected}),
               "seed": args.seed, "max_per_prompt": args.per_prompt, "max_length": args.max_length,
               "token_lengths": {"min": min(row["provenance"]["training_tokens"] for row in selected),
                                 "max": max(row["provenance"]["training_tokens"] for row in selected)},
               "candidate_rejections": dict(rejected), "sources": sources, "annotations_sha256": digest(annotations_path),
               "drafts_sha256": digest(output), "feedback_method": "offline template, no provider calls; human correction required",
               "prospective_split_if_every_draft_is_approved": {key: len(rows) for key, rows in split_rows(selected, args.seed).items()},
               "training_started": False}
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["utterance_id", "reference_text", "speaker_id", "model_id", "error_count", "training_tokens", "approved", "reviewer"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"utterance_id": row["assessment"]["utterance_id"], "reference_text": row["assessment"]["reference_text"],
                             "speaker_id": row["provenance"]["speaker_id"], "model_id": row["assessment"]["model_id"],
                             "error_count": len(row["assessment"]["errors"]), "training_tokens": row["provenance"]["training_tokens"],
                             "approved": False, "reviewer": ""})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
