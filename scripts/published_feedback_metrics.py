"""Native-label metrics. Invalid generations stay in every relevant denominator."""
from __future__ import annotations

import json
import math
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from scripts.l2arctic_plus_data import validate_feedback


def parse_prediction(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    validate_feedback(value)
    return value


def flatten(value, key, word_order):
    words = [word for word in word_order if word in value] + sorted(set(value) - set(word_order))
    return " ".join(f"{word}: {pair[key]}" for word in words for pair in value[word])


def score_row(row):
    gold = row["reference_feedback"]
    valid = row["valid_json"]
    prediction = row["prediction"] if valid else {}
    order = row["word_order"]
    target, actual = set(gold), set(prediction)
    reference = flatten(gold, "suggestion", order)
    hypothesis = flatten(prediction, "suggestion", order)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return {"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"],
            "valid_json": int(valid), "grounded_json": int(valid and not (actual - set(order))),
            "tp": len(target & actual), "fp": len(actual - target), "fn": len(target - actual),
            "reference_suggestions": reference, "predicted_suggestions": hypothesis,
            "has_reference_errors": bool(target),
            "suggestion_rougeL": rouge.score(reference, hypothesis)["rougeL"].fmeasure if target and valid else 0.,
            "correct_empty": int(valid and not actual) if not target else None,
            "nll_sum": row.get("nll_sum"), "target_tokens": row.get("target_tokens", 0),
            "generation_sec": row["generation_sec"], "generated_tokens": row.get("generated_tokens", 0)}


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    scores = [score_row(row) for row in rows]
    errors = [row for row in scores if row["has_reference_errors"]]
    tp, fp, fn = (sum(row[key] for row in scores) for key in ("tp", "fp", "fn"))
    bleu = BLEU(max_ngram_order=2, effective_order=True, tokenize="13a")
    bleu_value = bleu.corpus_score([row["predicted_suggestions"] for row in errors],
                                   [[row["reference_suggestions"] for row in errors]]).score if errors else None
    token_count = sum(row["target_tokens"] for row in scores if row["nll_sum"] is not None)
    nll = sum(row["nll_sum"] for row in scores if row["nll_sum"] is not None) / token_count if token_count else None
    clean = [row for row in scores if not row["has_reference_errors"]]
    total_time = sum(row["generation_sec"] for row in scores)
    return {"rows": len(scores), "reference_error_rows": len(errors),
        "json_valid_percent": 100 * sum(row["valid_json"] for row in scores) / len(scores),
        "grounded_json_percent": 100 * sum(row["grounded_json"] for row in scores) / len(scores),
        "feedback_word_precision_percent": 100 * tp / (tp + fp) if tp + fp else 0.,
        "feedback_word_recall_percent": 100 * tp / (tp + fn) if tp + fn else 0.,
        "feedback_word_f1_percent": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "suggestion_bleu2": bleu_value,
        "bleu_signature": str(bleu.get_signature()) if errors else None,
        "suggestion_rougeL_on_reference_error_rows": 100 * np.mean([row["suggestion_rougeL"] for row in errors]) if errors else None,
        "correct_empty_feedback_percent": 100 * np.mean([row["correct_empty"] for row in clean]) if clean else None,
        "target_token_nll": nll, "target_perplexity": math.exp(nll) if nll is not None and nll < 700 else None,
        "nll_rows": sum(row["nll_sum"] is not None for row in scores), "nll_target_tokens": token_count,
        "latency_median_sec": float(np.median([row["generation_sec"] for row in scores])),
        "latency_p95_sec": float(np.percentile([row["generation_sec"] for row in scores], 95)),
        "generation_tokens_per_sec": sum(row["generated_tokens"] for row in scores) / total_time if total_time else None}, scores


def paired_bootstrap(before, after, *, repetitions=2000, seed=42):
    if [row["utterance_id"] for row in before] != [row["utterance_id"] for row in after]:
        raise ValueError("Paired bootstrap requires identical ordered utterances")
    if [row["speaker_id"] for row in before] != [row["speaker_id"] for row in after]:
        raise ValueError("Paired speaker IDs differ")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    mask = np.array([row["has_reference_errors"] for row in before], dtype=bool)
    difference = 100 * np.array([b["suggestion_rougeL"] - a["suggestion_rougeL"] for a, b in zip(before, after)])
    speakers = sorted({row["speaker_id"] for row in before})
    groups = {speaker: np.array([i for i, row in enumerate(before) if row["speaker_id"] == speaker and mask[i]])
              for speaker in speakers}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        indices = np.concatenate([groups[speaker] for speaker in rng.choice(speakers, size=len(speakers), replace=True)])
        if len(indices):
            samples.append(float(np.mean(difference[indices])))
    if not samples:
        raise ValueError("No reference-error rows for the primary metric")
    return {"metric": "suggestion_rougeL_on_reference_error_rows", "unit": "percentage_points_after_minus_before",
            "delta": float(np.mean(difference[mask])), "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "resampling_unit": "speaker", "clusters": len(speakers), "repetitions": repetitions, "seed": seed,
            "scope": "Paired held-out sample uncertainty for this training seed; not variation across training seeds"}
