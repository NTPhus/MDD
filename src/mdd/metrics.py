"""Deterministic phone alignment and explicit MDD counters (stdlib only).

Detection units are canonical phone positions, plus insertion slots. Insertions
are paired by their order *within the same gap*, using max(gold, predicted)
slots at that gap. Absent insertion gaps do not create true negatives. This is
a project scoring protocol, not a claim to reproduce a paper's scoring script.

TA: correct phone accepted; FR: correct phone rejected; FA: error accepted;
TR: error rejected. CD/DE diagnose only detected errors with a known gold phone.
``err`` and ``[UNK]`` gold phones are excluded from diagnosis, but not detection.
PER uses the entire perceived sequence, preserving distortion markers (R*).
All rates are fractions; an undefined denominator produces None, never NaN.

Only canonical phones are used to align predictions. Gold perceived phones and
annotations are used to score an already decoded prediction, never to decode it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


COUNT_KEYS = (
    "UTTERANCES", "TA", "FR", "FA", "TR", "CD", "DE", "DIAG_EXCLUDED",
    "PER_S", "PER_D", "PER_I", "PER_N", "WORD_TP", "WORD_FP", "WORD_FN",
    "WORD_TN", "WORD_SCORED_UTTERANCES", "WORD_UNMAPPED_EVENTS",
)
SCORING_PROTOCOL = "canonical-phone-and-ordered-insertion-slots-v1"
UNKNOWN_PHONES = frozenset({"err", "[UNK]"})


def _phones(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of phone strings")
    result = list(values)
    if any(not isinstance(phone, str) or not phone or phone.strip() != phone for phone in result):
        raise ValueError(f"{name} contains an empty, non-string, or untrimmed phone")
    return result


def align_phones(reference: Sequence[str], hypothesis: Sequence[str]) -> list[dict[str, Any]]:
    """Unit-cost Levenshtein alignment, returned in sequence order.

    Backtracking runs right to left and resolves ties diagonal, deletion, then
    insertion. Thus [AA, AA] -> [AA] deletes the first AA, consistently. Insertion
    ``position`` is a canonical gap, 0 before the first phone and N after the last.
    """
    ref, hyp = _phones(reference, "reference"), _phones(hypothesis, "hypothesis")
    rows, columns = len(ref), len(hyp)
    costs = [list(range(columns + 1))]
    for i, expected in enumerate(ref, 1):
        row = [i]
        for j, observed in enumerate(hyp, 1):
            row.append(min(
                costs[i - 1][j - 1] + (expected != observed),
                costs[i - 1][j] + 1,
                row[j - 1] + 1,
            ))
        costs.append(row)
    result: list[dict[str, Any]] = []
    i, j = rows, columns
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            kind = "match" if ref[i - 1] == hyp[j - 1] else "substitution"
            result.append({"kind": kind, "position": i - 1,
                           "reference_index": i - 1, "hypothesis_index": j - 1,
                           "expected_phone": ref[i - 1], "observed_phone": hyp[j - 1]})
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            result.append({"kind": "deletion", "position": i - 1,
                           "reference_index": i - 1, "hypothesis_index": None,
                           "expected_phone": ref[i - 1], "observed_phone": None})
            i -= 1
        else:
            result.append({"kind": "addition", "position": i,
                           "reference_index": None, "hypothesis_index": j - 1,
                           "expected_phone": None, "observed_phone": hyp[j - 1]})
            j -= 1
    result.reverse()
    return result


def phone_error_events(canonical_phones: Sequence[str], perceived_phones: Sequence[str]) -> list[dict[str, Any]]:
    """Get S/D/I events via alignment when explicit annotation is unavailable."""
    return [step for step in align_phones(canonical_phones, perceived_phones) if step["kind"] != "match"]


def _validate_events(events: Sequence[Mapping[str, Any]], canonical: list[str]) -> list[dict[str, Any]]:
    result, occupied = [], set()
    for original in events:
        event = dict(original)
        kind = event.get("kind", event.get("type"))
        position = event.get("position")
        if kind not in {"substitution", "deletion", "addition"}:
            raise ValueError("gold event must have substitution/deletion/addition kind")
        upper_bound = len(canonical) + (kind == "addition")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < upper_bound:
            raise ValueError("gold event position is outside the canonical sequence")
        if kind != "addition":
            if position in occupied:
                raise ValueError("multiple gold errors occupy the same canonical phone")
            occupied.add(position)
            expected = event.get("expected_phone", canonical[position])
            if expected != canonical[position]:
                raise ValueError("gold expected phone does not match its canonical position")
        else:
            expected = event.get("expected_phone")
            if expected is not None:
                raise ValueError("addition must have null expected_phone")
        observed = event.get("observed_phone")
        if kind == "deletion":
            if observed is not None:
                raise ValueError("deletion must have null observed_phone")
        elif not isinstance(observed, str) or not observed:
            raise ValueError("substitution/addition must have an observed phone")
        event.update(kind=kind, position=position, expected_phone=expected, observed_phone=observed)
        result.append(event)
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | None]:
    """Recompute all rates from auditable micro counters; no averaging of F1s."""
    c = {key: counts.get(key, 0) for key in COUNT_KEYS}
    return {
        "precision": _ratio(c["TR"], c["TR"] + c["FR"]),
        "recall": _ratio(c["TR"], c["TR"] + c["FA"]),
        "f1": _ratio(2 * c["TR"], 2 * c["TR"] + c["FR"] + c["FA"]),
        "far": _ratio(c["FA"], c["FA"] + c["TR"]),
        "frr": _ratio(c["FR"], c["TA"] + c["FR"]),
        "der": _ratio(c["DE"], c["CD"] + c["DE"]),
        "per": _ratio(c["PER_S"] + c["PER_D"] + c["PER_I"], c["PER_N"]),
        "word_precision": _ratio(c["WORD_TP"], c["WORD_TP"] + c["WORD_FP"]),
        "word_recall": _ratio(c["WORD_TP"], c["WORD_TP"] + c["WORD_FN"]),
        "word_f1": _ratio(2 * c["WORD_TP"], 2 * c["WORD_TP"] + c["WORD_FP"] + c["WORD_FN"]),
    }


def _diagnose(gold: Mapping[str, Any], predicted: Mapping[str, Any], counts: dict[str, int]) -> None:
    if gold["observed_phone"] in UNKNOWN_PHONES or gold["expected_phone"] in UNKNOWN_PHONES:
        counts["DIAG_EXCLUDED"] += 1
        return
    correct = all(gold[key] == predicted[key] for key in ("kind", "expected_phone", "observed_phone"))
    counts["CD" if correct else "DE"] += 1


def _score_words(
    canonical: list[str], word_indices: Sequence[int | None],
    gold: list[dict[str, Any]], predicted: list[dict[str, Any]], counts: dict[str, int],
) -> None:
    if len(word_indices) != len(canonical):
        raise ValueError("canonical_word_indices must align one-to-one with canonical phones")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
           for value in word_indices):
        raise ValueError("word indices must be nonnegative integers or null")
    words = {value for value in word_indices if value is not None}
    if not words:
        counts["WORD_UNMAPPED_EVENTS"] += len(gold) + len(predicted)
        return
    counts["WORD_SCORED_UTTERANCES"] += 1

    def flagged(events: list[dict[str, Any]]) -> set[int]:
        result: set[int] = set()
        for event in events:
            # At a word boundary, assign additions to the following phone's word;
            # at utterance end, assign them to the last phone's word. A null word
            # mapping is reported as unmapped rather than guessed from neighbors.
            index = min(event["position"], len(word_indices) - 1)
            word = word_indices[index] if index >= 0 else None
            if word is None:
                counts["WORD_UNMAPPED_EVENTS"] += 1
            else:
                result.add(word)
        return result

    gold_words, predicted_words = flagged(gold), flagged(predicted)
    counts["WORD_TP"] += len(gold_words & predicted_words)
    counts["WORD_FP"] += len(predicted_words - gold_words)
    counts["WORD_FN"] += len(gold_words - predicted_words)
    counts["WORD_TN"] += len(words - gold_words - predicted_words)


def score_utterance(
    canonical_phones: Sequence[str], perceived_phones: Sequence[str], predicted_phones: Sequence[str],
    gold_error_events: Sequence[Mapping[str, Any]] | None = None,
    canonical_word_indices: Sequence[int | None] | None = None,
) -> dict[str, Any]:
    """Score decoded phones against manual perception and canonical text.

    Pass explicit ``gold_error_events`` from TextGrid annotations when available.
    This preserves expert errors even when normalization makes two labels equal.
    If omitted, events are inferred using the same deterministic aligner.
    ``canonical_word_indices`` optionally enables word-level detection counts;
    unknown mappings may be null. Phone indices and insertion gaps are zero-based.
    """
    canonical = _phones(canonical_phones, "canonical_phones")
    perceived = _phones(perceived_phones, "perceived_phones")
    predicted = _phones(predicted_phones, "predicted_phones")
    gold_alignment = align_phones(canonical, perceived)
    predicted_alignment = align_phones(canonical, predicted)
    reference_alignment = align_phones(perceived, predicted)
    gold = _validate_events(
        [step for step in gold_alignment if step["kind"] != "match"]
        if gold_error_events is None else gold_error_events, canonical,
    )
    predicted_events = [step for step in predicted_alignment if step["kind"] != "match"]
    counts = dict.fromkeys(COUNT_KEYS, 0)
    counts.update(UTTERANCES=1, PER_N=len(perceived))
    for step in reference_alignment:
        key = {"substitution": "PER_S", "deletion": "PER_D", "addition": "PER_I"}.get(step["kind"])
        if key:
            counts[key] += 1
    gold_positions = {event["position"]: event for event in gold if event["kind"] != "addition"}
    predicted_positions = {event["position"]: event for event in predicted_events if event["kind"] != "addition"}
    for index in range(len(canonical)):
        gold_event, predicted_event = gold_positions.get(index), predicted_positions.get(index)
        if gold_event is not None and predicted_event is not None:
            counts["TR"] += 1
            _diagnose(gold_event, predicted_event, counts)
        elif gold_event is not None:
            counts["FA"] += 1
        elif predicted_event is not None:
            counts["FR"] += 1
        else:
            counts["TA"] += 1
    gold_insertions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predicted_insertions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in gold:
        if event["kind"] == "addition":
            gold_insertions[event["position"]].append(event)
    for event in predicted_events:
        if event["kind"] == "addition":
            predicted_insertions[event["position"]].append(event)
    for gap in sorted(gold_insertions.keys() | predicted_insertions.keys()):
        gold_at_gap, predicted_at_gap = gold_insertions[gap], predicted_insertions[gap]
        paired = min(len(gold_at_gap), len(predicted_at_gap))
        counts["TR"] += paired
        counts["FA"] += len(gold_at_gap) - paired
        counts["FR"] += len(predicted_at_gap) - paired
        for expected, observed in zip(gold_at_gap, predicted_at_gap):
            _diagnose(expected, observed, counts)
    if canonical_word_indices is not None:
        _score_words(canonical, canonical_word_indices, gold, predicted_events, counts)
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "gold_event_source": "expert_annotation" if gold_error_events is not None else "inferred_alignment",
        "counts": counts, "metrics": metrics_from_counts(counts),
        "gold_events": gold, "predicted_events": predicted_events,
        "alignments": {"canonical_to_perceived": gold_alignment,
                       "canonical_to_predicted": predicted_alignment,
                       "perceived_to_predicted": reference_alignment},
    }


def aggregate_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum counters before computing micro metrics; accept an empty iterable."""
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for score in scores:
        if score.get("scoring_protocol", SCORING_PROTOCOL) != SCORING_PROTOCOL:
            raise ValueError("cannot aggregate results from a different scoring protocol")
        for key in COUNT_KEYS:
            value = score["counts"].get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"counter {key} must be a nonnegative integer")
            counts[key] += value
    return {"scoring_protocol": SCORING_PROTOCOL, "counts": counts, "metrics": metrics_from_counts(counts)}
