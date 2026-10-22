"""Experimental CTC variant alignment and calibrated word-level evidence.

This chat scorer is separate from the frozen research edit-distance scorer.
It accepts verified CMUdict variants and compares reference-phone posterior
evidence with competing nonblank phones along a constrained CTC Viterbi path.
It is not an implementation of a published CTC-GOP formula. Neither a forced
path nor an exact decoder match establishes pronunciation correctness.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from . import demo_reference
from .metrics import align_phones


METHOD = 'ctc_variant_posterior_margin_v1'
SCHEMA = 'mdd-pronunciation-calibration-v1'
MAX_FRAMES = 2000
MAX_VOCABULARY = 512
MAX_REFERENCE_PHONES = 300
MAX_LATTICE_STATES = 4096


def _word(text):
    value = unicodedata.normalize('NFKC', text).replace('’', "'").lower().strip('.,!?;:()[]{}"“”')
    return value[1:-1] if value.startswith("'") and value.endswith("'") and len(value) > 2 else value


@lru_cache(maxsize=4)
def _read_dictionary(path, modified, size, expected_hash):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        return {}
    result = {}
    for line in content.decode('utf-8').splitlines():
        fields = line.partition('#')[0].split()
        if len(fields) < 2:
            continue
        key = re.sub(r'\(\d+\)$', '', fields[0])
        raw = tuple(fields[1:])
        phones = tuple(re.sub(r'[012]$', '', phone) for phone in raw)
        if not all(re.fullmatch('[A-Z]+', phone) for phone in phones):
            continue
        variants = result.setdefault(key, [])
        if not any(existing[0] == phones for existing in variants):
            variants.append((phones, raw))
    return {key: tuple(variants) for key, variants in result.items()}


def _dictionary(root):
    try:
        root = Path(root).resolve()
        path = (root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict').resolve()
        if not path.is_relative_to(root):
            return {}
        stat = path.stat()
        return _read_dictionary(str(path), stat.st_mtime_ns, stat.st_size, demo_reference.DICTIONARY_SHA256)
    except (OSError, ValueError, UnicodeError):
        return {}


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _evidence(evidence):
    if not isinstance(evidence, dict):
        raise ValueError('acoustic_evidence_unavailable')
    try:
        probabilities = _array(evidence['log_probs'])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError('invalid_log_probabilities') from error
    if (probabilities.ndim != 2 or not 0 < probabilities.shape[0] <= MAX_FRAMES
            or not 1 < probabilities.shape[1] <= MAX_VOCABULARY
            or np.isnan(probabilities).any() or np.isposinf(probabilities).any()):
        raise ValueError('invalid_log_probabilities')
    peaks = probabilities.max(axis=1)
    if not np.isfinite(peaks).all():
        raise ValueError('invalid_log_probabilities')
    normalizers = peaks + np.log(np.exp(probabilities - peaks[:, None]).sum(axis=1))
    if np.max(np.abs(normalizers)) > .01:
        raise ValueError('expected_normalized_log_probabilities')
    try:
        raw_mapping = evidence['id_to_phone']
        if not isinstance(raw_mapping, dict):
            raise ValueError
        mapping = {int(index): phone for index, phone in raw_mapping.items()}
        if (set(mapping) != set(range(probabilities.shape[1]))
                or any(not isinstance(phone, str) or not phone for phone in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError
        blank = evidence['blank_id']
        if isinstance(blank, bool) or not isinstance(blank, (int, np.integer)) or int(blank) not in mapping:
            raise ValueError
        duration = float(evidence['duration_sec'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid_acoustic_vocabulary') from error
    return probabilities, mapping, int(blank), duration


def _lattice(choices, phone_to_id, blank):
    # Phone arcs form a DAG: all variants of a word share boundary nodes but
    # have private internal nodes, so paths cannot splice variant fragments.
    nodes, start = 1, 0
    arcs = []
    for word_index, variants in enumerate(choices):
        end = nodes
        nodes += 1
        for variant_index, (phones, raw) in enumerate(variants):
            position = start
            for phone_index, phone in enumerate(phones):
                following = end if phone_index == len(phones) - 1 else nodes
                if following == nodes:
                    nodes += 1
                arcs.append((position, following, phone_to_id[phone], word_index, variant_index, phone_index))
                position = following
        start = end
    if nodes + len(arcs) > MAX_LATTICE_STATES:
        raise ValueError('reference_lattice_budget_exceeded')
    outgoing, incoming = {}, {}
    for index, arc in enumerate(arcs):
        outgoing.setdefault(arc[0], []).append(index)
        incoming.setdefault(arc[1], []).append(index)
    labels = np.array([blank] * nodes + [arc[2] for arc in arcs], dtype=np.int32)
    edges = [(index, index) for index in range(len(labels))]
    for node, indices in outgoing.items():
        for index in indices:
            edges.append((node, nodes + index))
    for node, indices in incoming.items():
        for index in indices:
            edges.append((nodes + index, node))
            for following in outgoing.get(node, ()):
                # Repeated labels need an intervening blank. Different labels
                # may transition directly, exactly as standard CTC permits.
                if arcs[index][2] != arcs[following][2]:
                    edges.append((nodes + index, nodes + following))
    edges = np.asarray(edges, dtype=np.int32)
    finals = [start, *(nodes + index for index in incoming.get(start, ()))]
    return nodes, arcs, labels, edges[:, 0], edges[:, 1], np.asarray(finals, dtype=np.int32)


def _viterbi(log_probs, graph):
    nodes, arcs, labels, sources, destinations, finals = graph
    state_count = len(labels)
    previous = np.full(state_count, -np.inf)
    previous[0] = 0.
    back = np.empty((len(log_probs), state_count), dtype=np.int32)
    edge_indices = np.arange(len(sources), dtype=np.int32)
    sentinel = len(sources)
    for frame, values in enumerate(log_probs):
        scores = previous[sources]
        best = np.full(state_count, -np.inf)
        np.maximum.at(best, destinations, scores)
        choices = np.where(scores == best[destinations], edge_indices, sentinel)
        winner = np.full(state_count, sentinel, dtype=np.int32)
        np.minimum.at(winner, destinations, choices)
        back[frame] = sources[np.minimum(winner, sentinel - 1)]
        previous = best + values[labels]
    end = int(finals[np.argmax(previous[finals])])
    score = float(previous[end])
    if not math.isfinite(score):
        raise ValueError('ctc_alignment_impossible')
    path = np.empty(len(log_probs), dtype=np.int32)
    for frame in range(len(log_probs) - 1, -1, -1):
        path[frame] = end
        end = int(back[frame, end])
    return path, score


def _greedy(log_probs, mapping, blank):
    sequence, previous = [], None
    for token in np.argmax(log_probs, axis=1).tolist():
        if token != previous and token != blank:
            sequence.append(mapping[token])
        previous = token
    return sequence


def _policy(calibration, evidence):
    if not isinstance(calibration, dict):
        return None, 'calibration_unavailable'
    if (calibration.get('schema') != SCHEMA or calibration.get('schema_version') != 1
            or calibration.get('method') != METHOD or calibration.get('verified') is not True
            or calibration.get('eligible') is not True or calibration.get('status') != 'eligible'
            or not isinstance(calibration.get('validation'), dict)
            or calibration['validation'].get('status') != 'verified'
            or calibration['validation'].get('split') != 'calibration'):
        return None, 'calibration_not_verified_or_eligible'
    for key in ('checkpoint_sha256', 'vocabulary_sha256'):
        value = evidence.get(key)
        if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value) or calibration.get(key) != value:
            return None, 'calibration_identity_mismatch'
    if calibration.get('dictionary_sha256') != demo_reference.DICTIONARY_SHA256:
        return None, 'calibration_identity_mismatch'
    try:
        accepted, rejected = calibration['accept_threshold'], calibration['error_threshold']
        if isinstance(accepted, bool) or isinstance(rejected, bool):
            raise ValueError
        accepted, rejected = float(accepted), float(rejected)
        if not math.isfinite(accepted) or not math.isfinite(rejected) or accepted > rejected:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, 'invalid_calibration_thresholds'
    return (accepted, rejected), 'verified'


def _base(reference_text, evidence):
    tokens = reference_text.split() if isinstance(reference_text, str) else []
    return {'method': METHOD, 'status': 'unavailable', 'dictionary_sha256': demo_reference.DICTIONARY_SHA256,
            'checkpoint_sha256': evidence.get('checkpoint_sha256') if isinstance(evidence, dict) else None,
            'vocabulary_sha256': evidence.get('vocabulary_sha256') if isinstance(evidence, dict) else None,
            'alignment': {'status': 'unavailable', 'score_eligible': False, 'approximate_timestamps': True},
            'calibration': {'applied': False, 'reason': 'calibration_unavailable'},
            'target_words': [], 'raw_events': [],
            'words': [{'index': index, 'text': token, 'status': 'unavailable', 'error_score': None,
                       'score_eligible': False, 'reason': 'assessment_unavailable'} for index, token in enumerate(tokens)],
            'comparison': {'words': [{'text': token, 'status': 'unavailable'} for token in tokens],
                           'events': [], 'counts': dict.fromkeys(('match', 'substitution', 'deletion', 'insertion'), 0)}}


def assess(root, reference_text, evidence, calibration=None, reference_confirmed=False):
    """Assess server-owned acoustic evidence against a known target sentence.

    ``words[].error_score`` is the negative 20th percentile of reference-phone
    mean log-posterior margins against the strongest competing nonblank phone.
    Values are clipped to [-30, 30] only for numerical stability. It is an
    experimental continuous feature, not an accuracy/probability/GOP score.
    Score consumers MUST check ``score_eligible``; insertions, deletions and
    structurally unusable alignments are abstentions, not inferred diagnoses.

    The root application must verify the persisted calibration artifact and
    its evaluation identities before setting its internal ``verified=True``.
    Unconfirmed ASR targets and absent/ineligible calibration never produce
    match/difference decisions, even if a dictionary variant is compatible.
    """
    result = _base(reference_text, evidence)
    try:
        if not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 1000:
            raise ValueError('invalid_reference_text')
        tokens = reference_text.split()
        if len(tokens) > 150:
            raise ValueError('reference_phone_budget_exceeded')
        dictionary = _dictionary(root)
        if not dictionary:
            raise ValueError('verified_dictionary_unavailable')
        missing = [token for token in tokens if _word(token) not in dictionary]
        if missing:
            result['missing_words'] = missing
            raise ValueError('reference_contains_unknown_words')
        log_probs, mapping, blank, duration = _evidence(evidence)
        phone_to_id = {phone: index for index, phone in mapping.items() if index != blank}
        all_choices = [dictionary[_word(token)] for token in tokens]
        if sum(max(len(phones) for phones, raw in variants) for variants in all_choices) > MAX_REFERENCE_PHONES:
            raise ValueError('reference_phone_budget_exceeded')
        choices = [tuple((phones, raw) for phones, raw in variants if all(phone in phone_to_id for phone in phones))
                   for variants in all_choices]
        if any(not variants for variants in choices):
            raise ValueError('reference_phones_absent_from_vocabulary')
        graph = _lattice(choices, phone_to_id, blank)
        path, path_score = _viterbi(log_probs, graph)
        nodes, arcs = graph[:2]
        selected_arcs = list(dict.fromkeys(int(state) - nodes for state in path if state >= nodes))
        selected = {}
        phone_scores = {index: [] for index in range(len(tokens))}
        for arc_index in selected_arcs:
            arc = arcs[arc_index]
            _, _, token, word_index, variant_index, phone_index = arc
            selected[word_index] = variant_index
            frames = np.flatnonzero(path == nodes + arc_index)
            competitors = [index for index in mapping if index not in (blank, token)]
            if not competitors:
                raise ValueError('no_competing_phone_evidence')
            competitor_values = log_probs[np.ix_(frames, competitors)]
            margins = log_probs[frames, token] - competitor_values.max(axis=1)
            margin = float(np.mean(np.clip(margins, -30., 30.)))
            dominant = int(np.argmax(competitor_values.mean(axis=0)))
            phone_scores[word_index].append({'phone': mapping[token], 'phone_index': phone_index,
                'margin': margin, 'competitor_phone': mapping[competitors[dominant]],
                'frame_count': len(frames), 'start_sec': round(float(frames[0]) / len(path) * duration, 4),
                'end_sec': round(float(frames[-1] + 1) / len(path) * duration, 4),
                'mean_nonblank_posterior': float(np.mean(1. - np.exp(log_probs[frames, blank])))})
        if len(selected) != len(tokens):
            raise ValueError('incomplete_reference_alignment')
        target_words, canonical, mapping_words = [], [], []
        for index, token in enumerate(tokens):
            phones, raw = choices[index][selected[index]]
            canonical.extend(phones)
            mapping_words.extend([index] * len(phones))
            target_words.append({'text': token, 'phones': list(phones), 'pronunciation_phones': list(raw),
                                 'dictionary_variant_index': all_choices[index].index((phones, raw)),
                                 'dictionary_variant_count': len(all_choices[index])})
        decoded = _greedy(log_probs, mapping, blank)
        raw_events = []
        unsupported = set()
        for event in align_phones(canonical, decoded):
            if event['kind'] == 'match':
                continue
            word_index = mapping_words[min(event['position'], len(mapping_words) - 1)]
            kind = 'insertion' if event['kind'] == 'addition' else event['kind']
            raw_events.append({'kind': kind, 'expected_phone': event['expected_phone'],
                               'observed_phone': event['observed_phone'], 'word_index': word_index,
                               'source': 'unvalidated_greedy_edit', 'tentative': True})
            if kind in ('insertion', 'deletion'):
                unsupported.add(word_index)
        thresholds, calibration_reason = _policy(calibration, evidence)
        applied = thresholds is not None and reference_confirmed is True
        if thresholds is not None and not applied:
            calibration_reason = 'reference_unconfirmed'
        result.update(status='ready', target_words=target_words, raw_events=raw_events,
            calibration={'applied': applied, 'reason': calibration_reason},
            alignment={'status': 'aligned', 'score_eligible': bool(decoded), 'approximate_timestamps': True,
                       'ctc_viterbi_log_prob': path_score,
                       'greedy_gap_per_frame': float((log_probs.max(axis=1).sum() - path_score) / len(path)),
                       'greedy_phone_count': len(decoded), 'reference_phone_count': len(canonical),
                       'variant_compatible': not raw_events})
        words = []
        for index, token in enumerate(tokens):
            scores = phone_scores[index]
            score = float(-np.quantile([item['margin'] for item in scores], .2))
            eligible = (bool(decoded) and index not in unsupported and len(choices[index]) == len(all_choices[index])
                        and all(item['mean_nonblank_posterior'] > 1e-6 for item in scores))
            reason = ('no_decoded_phone_evidence' if not decoded else
                      'insertion_or_deletion_not_supported' if index in unsupported else
                      'incomplete_dictionary_variant_coverage' if len(choices[index]) != len(all_choices[index]) else
                      'insufficient_nonblank_evidence' if not eligible else
                      'reference_unconfirmed' if not reference_confirmed else calibration_reason)
            status = 'uncertain'
            if applied and eligible:
                if score <= thresholds[0]:
                    status, reason = 'match', 'calibrated_acceptance'
                elif score >= thresholds[1]:
                    status, reason = 'difference', 'calibrated_difference'
                else:
                    reason = 'calibration_abstention_band'
            words.append({'index': index, 'text': token, 'status': status, 'error_score': score,
                          'score_eligible': eligible, 'reason': reason, 'phone_scores': scores,
                          'variant_compatible': not any(event['word_index'] == index for event in raw_events),
                          'start_sec': min(item['start_sec'] for item in scores),
                          'end_sec': max(item['end_sec'] for item in scores)})
        events = [dict(event, source='calibrated_word_detection_with_tentative_edit') for event in raw_events
                  if words[event['word_index']]['status'] == 'difference' and event['kind'] == 'substitution']
        counts = dict.fromkeys(('match', 'substitution', 'deletion', 'insertion'), 0)
        for event in events:
            counts[event['kind']] += 1
        counts['match'] = sum(len(target_words[index]['phones']) for index, word in enumerate(words) if word['status'] == 'match')
        result.update(words=words, comparison={'words': [{'text': word['text'], 'status': word['status']} for word in words],
                                              'events': events, 'counts': counts})
    except (ValueError, TypeError, IndexError, KeyError, OverflowError) as error:
        reason = str(error) if isinstance(error, ValueError) else 'invalid_assessment_input'
        result['reason'] = reason
        result['alignment']['reason'] = reason
        for word in result['words']:
            word['reason'] = reason
    return result


def score_words(root, reference_text, evidence):
    """Return the same raw feature/eligibility report without calibrated decisions."""
    return assess(root, reference_text, evidence)


def accepted_variants(root, reference_text):
    """Look up verified reference pronunciations without audio or guessing OOVs."""
    if not isinstance(reference_text, str) or len(reference_text) > 1000:
        return []
    dictionary = _dictionary(root)
    return [{'text': token, 'available': bool(variants := dictionary.get(_word(token), ())),
             'variants': [{'phones': list(phones), 'pronunciation_phones': list(raw)} for phones, raw in variants]}
            for token in reference_text.split()]
