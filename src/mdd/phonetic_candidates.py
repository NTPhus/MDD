"""Experimental sound-neighbour retrieval, separate from pronunciation labels.

Words are ranked by edit distance between predicted phones and CMUdict phones,
not by spelling. ASR word intervals and phoneme-B output can both be wrong;
these candidates never become a confirmed transcript or a sound-error label.
"""
from difflib import SequenceMatcher
from functools import lru_cache
import math
from pathlib import Path
import re

from .chat_reference import _distance
from .demo_reference import COMMIT, DICTIONARY_SHA256
from .pronunciation_assessment import _read_dictionary

METHOD = 'phone_neighbours_v1'
MAX_WORDS = 24


def word_intervals(frame_ids, id_to_token, blank_id, delimiter, duration_sec):
    """Greedy CTC word timing; approximate, never a forced reference alignment."""
    if not frame_ids or not math.isfinite(duration_sec) or duration_sec <= 0:
        return []
    characters, words, start, end = [], [], None, None
    def flush():
        nonlocal characters, start, end
        if characters:
            words.append({'index': len(words), 'text': ''.join(characters),
                          'start_sec': start * duration_sec / len(frame_ids),
                          'end_sec': end * duration_sec / len(frame_ids)})
        characters, start, end = [], None, None
    index = 0
    while index < len(frame_ids):
        token_id, stop = frame_ids[index], index + 1
        while stop < len(frame_ids) and frame_ids[stop] == token_id: stop += 1
        if token_id != blank_id:
            token = id_to_token.get(token_id, '')
            if token == delimiter: flush()
            elif re.fullmatch("[A-Za-z']", token):
                if start is None: start = index
                characters.append(token); end = stop
        index = stop
    flush()
    if words:
        words[0]['start_sec'] = 0.
        words[-1]['end_sec'] = duration_sec
    # Include intervening blanks, using the midpoint of adjacent CTC peaks.
    for index in range(len(words) - 1):
        boundary = (words[index]['end_sec'] + words[index + 1]['start_sec']) / 2
        words[index]['end_sec'] = words[index + 1]['start_sec'] = boundary
    return words


@lru_cache(maxsize=4)
def _phone_buckets(path, modified, size):
    dictionary = _read_dictionary(path, modified, size, DICTIONARY_SHA256)
    result = {}
    for word, variants in dictionary.items():
        if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word) or len(word) > 24:
            continue
        for phones, _ in variants:
            result.setdefault(len(phones), []).append((word, phones))
    return result


@lru_cache(maxsize=512)
def _nearest(path, modified, size, observed, asr_word):
    if not 2 <= len(observed) <= 20: return ()
    limit = min(4, max(1, len(observed) // 3))
    best = {}
    buckets = _phone_buckets(path, modified, size)
    for length in range(max(1, len(observed) - limit), len(observed) + limit + 1):
        for word, phones in buckets.get(length, ()):
            distance = _distance(observed, phones, limit)
            if distance > limit: continue
            ratio = distance / max(len(observed), len(phones))
            spelling = SequenceMatcher(None, asr_word.lower(), word, autojunk=False).ratio()
            # Minimize actual sound edits first. Dividing by word length first
            # favours longer, unrelated words; spelling is only a tie-breaker.
            score = (distance, -spelling, ratio, word, phones)
            if word not in best or score < best[word]: best[word] = score
    return tuple(sorted(best.values())[:5])


def sound_candidates(root, intervals, phonemes, transcript):
    result = {'method': METHOD, 'status': 'unavailable', 'diagnostic_only': True,
              'requires_confirmation': True, 'timing': 'approximate_asr_ctc',
              'dictionary_sha256': DICTIONARY_SHA256, 'words': []}
    tokens = transcript.split()
    if (len(intervals) != len(tokens) or not intervals
            or any(re.sub(r"[^a-z']", '', row['text'].lower()) != re.sub(r"[^a-z']", '', token.lower())
                   for row, token in zip(intervals, tokens))):
        result['reason'] = 'asr_word_timing_unavailable'; return result
    root = Path(root).resolve()
    path = root / 'data/external/cmudict' / COMMIT / 'cmudict.dict'
    try:
        if not path.resolve().is_relative_to(root): raise ValueError('dictionary_path')
        stat = path.stat()
        if not _phone_buckets(str(path), stat.st_mtime_ns, stat.st_size):
            raise ValueError('dictionary_unavailable')
    except (OSError, ValueError):
        result['reason'] = 'verified_dictionary_unavailable'; return result
    for index, interval in enumerate(intervals[:MAX_WORDS]):
        # Both streams are model predictions. A phone is assigned by its centre,
        # rather than pretending the ASR word boundaries are human annotations.
        observed = tuple(row['phone'] for row in phonemes
            if interval['start_sec'] <= (row['start_sec'] + row['end_sec']) / 2 < interval['end_sec'])
        nearest = _nearest(str(path), stat.st_mtime_ns, stat.st_size, observed, tokens[index])
        result['words'].append({'index': index, 'asr_word': tokens[index], 'heard_phones': list(observed),
            'start_sec': round(interval['start_sec'], 3), 'end_sec': round(interval['end_sec'], 3),
            'candidates': [{'word': word, 'phones': list(phones), 'edit_distance': distance,
                            'normalized_distance': round(ratio, 4)}
                           for distance, _, ratio, word, phones in nearest]})
    result['status'] = 'available'
    return result
