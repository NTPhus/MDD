"""Validate and save a single explicit human review; no generation or training."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unicodedata

from filelock import FileLock

from .llm_client import ROOT, validate_local_feedback
from .prepare_feedback_data import read_rows
from .train_feedback_model import encode_pair
from .validate_contracts import validate_pair


def approve_one(original: dict, assessment: dict, feedback: dict, reviewer: str, destination: Path,
                *, tokenizer, max_length=3072, notes="") -> dict:
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Nhập tên người thực sự kiểm tra audio và nội dung hướng dẫn.")
    if not isinstance(assessment, dict) or not isinstance(feedback, dict):
        raise ValueError("Assessment và feedback phải là JSON object.")
    for key in ("utterance_id", "data_kind", "model_id"):
        if assessment.get(key) != original["assessment"][key]:
            raise ValueError("Giữ nguyên nguồn dữ liệu và định danh mẫu: " + key)
    validate_pair(assessment, feedback)
    def word_key(word):
        return unicodedata.normalize("NFKC", word).replace("’", "'").casefold().strip('.,!?;:()[]{}"“”')

    expected_words = [word_key(word) for word in assessment["reference_text"].split()]
    actual_words = [word_key(word["text"]) for word in assessment["words"]]
    if actual_words != expected_words or not all(expected_words):
        raise ValueError("Danh sách từ phải khớp với câu tham chiếu (bỏ qua dấu câu ở hai đầu từ).")
    validate_local_feedback(assessment, feedback)
    if feedback["language"] != "vi":
        raise ValueError("Hướng dẫn cần bằng tiếng Việt.")
    result = deepcopy(original)
    result.update(assessment=deepcopy(assessment), feedback=deepcopy(feedback))
    result["review"] = {"approved": True, "reviewer": reviewer.strip(),
                        "reviewed_at": datetime.now(timezone.utc).isoformat(), "notes": str(notes)}
    result.setdefault("review_context", {})["assessment_edited"] = assessment != original["assessment"]
    result["review_context"]["feedback_edited"] = feedback != original["feedback"]
    tokens = len(encode_pair(tokenizer, result, max_length)["input_ids"])
    result["review_context"]["reviewed_training_tokens"] = tokens
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Lưu dữ liệu đã duyệt trong workspace.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=5):
        existing = read_rows(destination) if destination.is_file() else []
        identifier = assessment["utterance_id"]
        existing = [row for row in existing if row["assessment"]["utterance_id"] != identifier]
        existing.append(result)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing), encoding="utf-8")
        temporary.replace(destination)
    return result
