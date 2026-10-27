"""Canonical-only, tentative prefix alignment for the bidirectional demo model.

The acoustic model still decodes complete snapshots. No phone is committed
across requests; unread suffixes are free rather than treated as deletions.
"""
from __future__ import annotations

from .demo_inference import canonical_comparison, normalize_text
from .demo_feedback import build_assessment


def practice_reference(row: dict, text: str, mode: str, word_index: int = 0) -> tuple[str, dict]:
    words = text.strip().split()
    canonical = row.get("canonical_phones", [])
    mapping = row.get("canonical_word_indices", [])
    if (normalize_text(row.get("prompt_text", "")) != normalize_text(text)
            or not isinstance(canonical, list) or not canonical
            or any(not isinstance(phone, str) or not phone for phone in canonical)
            or not isinstance(mapping, list) or len(mapping) != len(canonical)
            or any(type(index) is not int or not 0 <= index < len(words) for index in mapping)
            or mapping != sorted(mapping) or set(mapping) != set(range(len(words)))):
        raise ValueError("Câu này chưa có ánh xạ âm chuẩn đầy đủ theo từ. Hãy chọn câu mẫu khác.")
    intervals = row.get("word_intervals")
    if intervals is not None:
        key = lambda value: "".join(c for c in normalize_text(value) if c.isalnum() or c == "'")
        if (not isinstance(intervals, list) or len(intervals) != len(words)
                or any(item.get("index") != index or key(item.get("text", "")) != key(words[index])
                       for index, item in enumerate(intervals))):
            raise ValueError("Câu này chưa có ranh giới từ phù hợp; hãy chọn câu mẫu khác.")
    has_variants = "word_all_variants" in row
    all_variants = row.get("word_all_variants", [])
    if mode == "word":
        if type(word_index) is not int or not 0 <= word_index < len(words):
            raise ValueError("Vị trí từ không hợp lệ.")
        phones = [phone for phone, index in zip(canonical, mapping) if index == word_index]
        res = {"prompt_text": words[word_index], "canonical_phones": phones,
               "canonical_word_indices": [0] * len(phones)}
        if has_variants:
            res["word_all_variants"] = [all_variants[word_index]] if word_index < len(all_variants) else []
        return words[word_index], res
    res = {"prompt_text": text.strip(), "canonical_phones": list(canonical),
           "canonical_word_indices": list(mapping)}
    if has_variants:
        res["word_all_variants"] = list(all_variants)
    return text.strip(), res


def prefix_comparison(text: str, reference: dict, phones: list[str]) -> dict:
    canonical, mapping = reference["canonical_phones"], reference["canonical_word_indices"]
    # Levenshtein against each canonical prefix. Earliest equal-cost endpoint
    # avoids inventing progress through repeated words or a still-unread suffix.
    previous = list(range(len(phones) + 1))
    terminal = [previous[-1]]
    for i, expected in enumerate(canonical, 1):
        current = [i]
        for j, observed in enumerate(phones, 1):
            current.append(min(previous[j - 1] + (expected != observed),
                               previous[j] + 1, current[j - 1] + 1))
        terminal.append(current[-1])
        previous = current
    consumed = min(range(len(terminal)), key=lambda index: (terminal[index], index))
    words = text.split()
    result = {"words": [{"text": word, "status": "pending"} for word in words],
              "counts": dict.fromkeys(("match", "substitution", "deletion", "insertion"), 0), "events": []}
    if consumed:
        prefix = {"prompt_text": text, "canonical_phones": canonical[:consumed],
                  "canonical_word_indices": mapping[:consumed],
                  "word_all_variants": reference.get("word_all_variants", [])}
        compared = canonical_comparison(text, prefix, phones)
        result["counts"], result["events"] = compared["counts"], compared["events"]
        active = mapping[consumed - 1]
        for index in range(active):
            result["words"][index]["status"] = compared["words"][index]["status"]
        # The current word's ending needs more context. Its events remain
        # visible as tentative; no missing tail phones are invented.
        result["words"][active]["status"] = "listening"
    elif words:
        result["words"][0]["status"] = "listening"
    result.update(consumed_reference_phones=consumed, total_reference_phones=len(canonical), tentative=True)
    return result


def preview_assessment(result: dict) -> dict:
    # Reuse the existing phone/IPA translation without registering a feedback
    # cache entry. Pending words are not evidence of errors or completed speech.
    comparison = result["comparison"]
    temporary = {**result, "comparison": {**comparison, "words": [
        {**word, "status": "difference" if word["status"] == "difference" else "match"}
        for word in comparison["words"]]}}
    assessment, _ = build_assessment(temporary, "live-preview")
    assessment["status"] = "uncertain"
    return assessment
