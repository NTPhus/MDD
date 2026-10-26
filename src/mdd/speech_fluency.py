"""Descriptive speech timing, not a calibrated fluency or pronunciation grade.

Rates use whitespace-delimited English word/number tokens; contractions count
once. Word timestamps are ASR predictions, not phonation or silence labels.
The articulation estimate excludes inter-word gaps of at least 250 ms, while
the speech-window rate retains those gaps. Shorter gaps remain in both rates.
This cutoff describes a measurement convention, never a good/bad boundary.

Background on separating rate, articulation, and pauses:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11119743/
Our ASR-gap estimates are not that paper's manually annotated measurements.
"""
from __future__ import annotations

import math
from numbers import Real
import re
from collections.abc import Mapping, Sequence

METHOD = "word_timing_fluency_v1"
PAUSE_THRESHOLD_SEC = .25
_EPS = 1e-6
_FILLERS = frozenset({"um", "uh", "erm", "hmm"})
_LIMITATIONS = [
    "Word times and recognized words can be inaccurate or omit hesitations.",
    "Inter-word gaps are timing estimates, not verified acoustic silence.",
    "Punctuation only approximates sentence boundaries; clause boundaries are not assessed.",
    "No expert-calibrated fluency, pronunciation, stress, intonation, or overall score is produced.",
]


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _lexical(value):
    return "".join(re.findall(r"[a-z0-9']", value.lower().replace("’", "'")))


def _tokens(text):
    return [match for match in re.finditer(r"\S+", text)
            if re.search(r"[A-Za-z0-9]", match.group())]


def _boundary(text, before, after):
    # This is explicitly a punctuation heuristic, not a grammar/clause parser.
    return bool(re.search(r'''[.!?]["'’”\)\]]*\s*$''', text[before.start():after.start()]))


def _lexical_observations(text, tokens):
    words = [_lexical(token.group()) for token in tokens]
    fillers = [{"index": index, "text": token.group()}
               for index, token in enumerate(tokens) if words[index] in _FILLERS]
    duplicates = [{"first_index": index - 1, "second_index": index,
                   "text": tokens[index].group()}
                  for index in range(1, len(tokens))
                  if words[index] == words[index - 1]
                  and not _boundary(text, tokens[index - 1], tokens[index])]
    return {
        "source": "transcript_only",
        "filler_token_set": sorted(_FILLERS),
        "observed_filler_token_count": len(fillers),
        "observed_filler_tokens": fillers,
        "adjacent_duplicate_pair_count": len(duplicates),
        "adjacent_duplicate_pairs": duplicates,
        "repetitions_are_errors": False,
        "note": "Literal transcript observations; intentional repetitions and ASR omissions are possible.",
    }


def _timed_words(tokens, rows, duration):
    """Associate exact word text; never spread one timestamp across new words.

    Accept either the demo's word_intervals or Whisper native_words. Native
    clitic pieces may be joined when their characters form one transcript word.
    """
    if rows is None or (isinstance(rows, Sequence) and not rows):
        return None, "word_timing_missing"
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, "word_timing_invalid"
    pieces, previous_start = [], -1.
    for row in rows:
        if not isinstance(row, Mapping):
            return None, "word_timing_invalid"
        value = row.get("text", row.get("word"))
        start = row.get("start_sec", row.get("start"))
        end = row.get("end_sec", row.get("end"))
        if not isinstance(value, str) or not _number(start) or not _number(end):
            return None, "word_timing_invalid"
        if start < 0 or end < start or end > duration + _EPS or start < previous_start - _EPS:
            return None, "word_timing_invalid"
        previous_start = start
        value = _lexical(value)
        if value:
            pieces.append((value, float(start), min(float(end), duration)))
    values = [_lexical(token.group()) for token in tokens]
    if not pieces or "".join(values) != "".join(piece[0] for piece in pieces):
        return None, "word_timing_text_mismatch"
    timed, piece_index, offset = [], 0, 0
    for index, token in enumerate(values):
        word_end = offset + len(token)
        selected, cursor = [], offset
        while piece_index < len(pieces) and cursor < word_end:
            value, start, end = pieces[piece_index]
            if cursor + len(value) > word_end:
                return None, "word_timing_spans_multiple_words"
            selected.append((start, end))
            cursor += len(value)
            piece_index += 1
        if not selected or cursor != word_end:
            return None, "word_timing_text_mismatch"
        start, end = selected[0][0], max(item[1] for item in selected)
        if start > end or (timed and start < timed[-1]["end_sec"] - _EPS):
            return None, "word_timing_overlaps"
        timed.append({"index": index, "text": tokens[index].group(), "start_sec": start, "end_sec": end})
        offset = word_end
    return timed, None


def assess_fluency(waveform, transcript, word_intervals, sample_rate=16000, *,
                   pause_threshold_sec=PAUSE_THRESHOLD_SEC):
    """Return JSON-safe descriptive timing without external services or models.

    ``recording_rate_wpm`` uses the full recording, including boundary gaps.
    ``speech_rate_wpm`` uses the first-to-last timed word window. Articulation
    time/rate are estimates after subtracting only gaps >= pause_threshold_sec.
    Missing or mismatched timestamps retain recording-rate measurements only.
    """
    result = {
        "status": "unavailable", "source": "whisper_word_timestamps", "method": METHOD,
        "reason": None, "calibrated": False, "timing_is_approximate": True,
        "word_count_policy": "whitespace lexical tokens; contractions and number tokens count once",
        "metrics": {}, "pauses": [], "feedback": [], "limitations": list(_LIMITATIONS),
    }
    if not _number(sample_rate) or sample_rate <= 0:
        result["reason"] = "invalid_sample_rate"
        return result
    if not _number(pause_threshold_sec) or pause_threshold_sec <= 0:
        result["reason"] = "invalid_pause_threshold"
        return result
    try:
        import numpy as np
        audio = np.asarray(waveform, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "invalid_audio"
        return result
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        result["reason"] = "invalid_audio"
        return result
    duration = audio.size / float(sample_rate)
    result["metrics"]["audio_duration_sec"] = round(duration, 4)
    # Only exact digital silence is classified here. Quiet speech is not VAD-
    # gated, and nonzero noise is not automatically claimed to contain speech.
    if not np.any(audio):
        result["reason"] = "digital_silence"
        return result
    if not isinstance(transcript, str) or not (tokens := _tokens(transcript)):
        result["reason"] = "transcript_missing"
        return result
    word_count = len(tokens)
    result["lexical_heuristics"] = _lexical_observations(transcript, tokens)
    metrics = result["metrics"]
    metrics.update(word_count=word_count, recording_rate_wpm=round(word_count * 60 / duration, 2))
    timed, reason = _timed_words(tokens, word_intervals, duration)
    if timed is None:
        result.update(status="partial", reason=reason)
        return result
    window = timed[-1]["end_sec"] - timed[0]["start_sec"]
    if window <= 0:
        result.update(status="partial", reason="word_timing_window_empty")
        return result
    pauses = []
    for index in range(1, len(timed)):
        start, end = timed[index - 1]["end_sec"], timed[index]["start_sec"]
        if end - start + _EPS >= pause_threshold_sec:
            pauses.append({"after_word_index": index - 1, "before_word_index": index,
                           "start_sec": round(start, 4), "end_sec": round(end, 4),
                           "duration_sec": round(end - start, 4),
                           "sentence_boundary": _boundary(transcript, tokens[index - 1], tokens[index])})
    pause_duration = sum(row["duration_sec"] for row in pauses)
    articulation = max(0., window - pause_duration)
    mid_sentence = sum(not row["sentence_boundary"] for row in pauses)
    metrics.update(
        speech_window_duration_sec=round(window, 4),
        speech_rate_wpm=round(word_count * 60 / window, 2),
        articulation_duration_estimate_sec=round(articulation, 4),
        articulation_rate_estimate_wpm=round(word_count * 60 / articulation, 2) if articulation > _EPS else None,
        pause_threshold_sec=float(pause_threshold_sec),
        internal_pause_count=len(pauses), internal_pause_duration_sec=round(pause_duration, 4),
        mid_sentence_pause_count=mid_sentence,
        sentence_boundary_pause_count=len(pauses) - mid_sentence,
        longest_internal_pause_sec=max((row["duration_sec"] for row in pauses), default=0.),
        leading_gap_sec=round(timed[0]["start_sec"], 4),
        trailing_gap_sec=round(max(0., duration - timed[-1]["end_sec"]), 4),
    )
    result.update(status="ready" if articulation > _EPS else "partial",
                  reason=None if articulation > _EPS else "articulation_window_empty", pauses=pauses)
    result["feedback"] = [{"kind": "timing", "text":
        f"{word_count} words across a {window:.2f}-second spoken window."}]
    if pauses:
        result["feedback"].append({"kind": "pause_locations", "text":
            f"{mid_sentence} timestamp gaps of at least {pause_threshold_sec:g} seconds occurred within sentences; "
            f"{len(pauses) - mid_sentence} occurred at punctuation boundaries."})
    return result
