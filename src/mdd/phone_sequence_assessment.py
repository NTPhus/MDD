"""Dictionary-variant review of a model's perceived phones, without a pass/fail score.

CROTTC-IF predicts sounds rather than a calibrated probability of learner error.
This scorer aligns that unchanged sequence to a dictionary pronunciation lattice.
It does not reuse the CTC posterior calibration fitted for the old B checkpoint.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .demo_reference import DICTIONARY_SHA256

METHOD = 'crottc_if_variant_sequence_review_v1'
PHONE = re.compile(r'[A-Z]{1,4}')


def validate_phones(evidence):
    phones = evidence.get('phonemes') if isinstance(evidence, dict) else None
    if not isinstance(phones, list) or not 1 <= len(phones) <= 1500:
        raise ValueError('missing_perceived_phone_sequence')
    duration = evidence.get('duration_sec')
    if type(duration) not in (int, float) or not math.isfinite(duration) or not .25 <= duration <= 30:
        raise ValueError('invalid_recording_duration')
    result = []
    previous = 0.
    for item in phones:
        if not isinstance(item, dict) or not isinstance(item.get('phone'), str) or not PHONE.fullmatch(item['phone']):
            raise ValueError('unsupported_perceived_phone')
        value = {'phone': item['phone']}
        start, end = item.get('start_sec'), item.get('end_sec')
        if start is not None or end is not None:
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start <= end <= duration + .001 or start < previous - .001):
                raise ValueError('invalid_perceived_phone_timing')
            value.update(start_sec=start, end_sec=end)
            previous = start
        result.append(value)
    return result


def align_variants(choices, observed):
    """Exact unit-cost edit alignment against all dictionary variants in a DAG.

    Epsilon joins merge variants between words, so alternatives are considered
    jointly without enumerating exponentially many sentence pronunciations.
    Reference/observed sequences and empty alternatives are deliberately bounded.
    """
    if not choices or len(choices) > 200 or not observed or len(observed) > 1500:
        raise ValueError('phone_alignment_budget_exceeded')
    incoming = [[]]
    arcs = []
    start = 0
    def edge(source, target, phone, word, variant, position):
        arcs.append((source, target, phone, word, variant, position))
        incoming[target].append(len(arcs) - 1)
    for word, variants in enumerate(choices):
        if not variants or len(variants) > 16:
            raise ValueError('dictionary_variant_budget_exceeded')
        endings = []
        for variant, phones in enumerate(variants):
            if not phones or len(phones) > 64:
                raise ValueError('invalid_dictionary_variant')
            cursor = start
            for position, phone in enumerate(phones):
                incoming.append([])
                target = len(incoming) - 1
                edge(cursor, target, phone, word, variant, position)
                cursor = target
            endings.append((cursor, variant))
        incoming.append([])
        target = len(incoming) - 1
        for cursor, variant in endings:
            edge(cursor, target, None, word, variant, -1)
        start = target
    if len(incoming) * (len(observed) + 1) > 2_000_000:
        raise ValueError('phone_alignment_budget_exceeded')
    cost = np.full((len(incoming), len(observed) + 1), 30000, dtype=np.int16)
    # Positive trace = arc index + 1. Negative = insertion at this node.
    trace = np.zeros(cost.shape, dtype=np.int32)
    cost[0] = np.arange(len(observed) + 1)
    trace[0, 1:] = -1
    for node in range(1, len(incoming)):
        for position in range(len(observed) + 1):
            best, selected = 30000, 0
            for arc_index in incoming[node]:
                source, _, phone, *_ = arcs[arc_index]
                if phone is None:
                    candidate = int(cost[source, position])
                    if candidate < best:
                        best, selected = candidate, arc_index + 1
                    continue
                # Prefer matched/substituted evidence over deletion on ties.
                if position:
                    candidate = int(cost[source, position - 1]) + (phone != observed[position - 1])
                    if candidate < best:
                        best, selected = candidate, arc_index + 1
                candidate = int(cost[source, position]) + 1
                if candidate < best:
                    best, selected = candidate, -(arc_index + 2)
            if position and int(cost[node, position - 1]) + 1 < best:
                best, selected = int(cost[node, position - 1]) + 1, -1
            cost[node, position], trace[node, position] = best, selected
    node, position = len(incoming) - 1, len(observed)
    path = []
    while node or position:
        selected = int(trace[node, position])
        if selected == -1:
            path.append({'kind': 'insertion', 'observed_index': position - 1,
                         'expected_phone': None, 'observed_phone': observed[position - 1]})
            position -= 1
            continue
        if not selected:
            raise ValueError('incomplete_phone_alignment')
        arc_index = selected - 1 if selected > 0 else -selected - 2
        source, _, phone, word, variant, phone_index = arcs[arc_index]
        if phone is not None:
            consumes = selected > 0
            actual = observed[position - 1] if consumes else None
            path.append({'kind': 'match' if actual == phone else 'substitution' if consumes else 'deletion',
                         'word_index': word, 'variant_index': variant, 'phone_index': phone_index,
                         'expected_phone': phone, 'observed_phone': actual,
                         'observed_index': position - 1 if consumes else None})
            position -= int(consumes)
        node = source
    path.reverse()
    # An insertion between words follows the preceding reference word. Leading
    # insertions belong to the first word. This is alignment attribution only.
    word = 0
    for event in path:
        word = event.setdefault('word_index', word)
    return path, int(cost[-1, -1])


def assess_sequence(root, text, evidence, *, confirmed=False):
    from .pronunciation_assessment import accepted_variants
    from .chat_pronunciation import target_words
    from .chat_diagnostics import sound_candidates
    tokens = text.split() if isinstance(text, str) else []
    assessment = {'method': METHOD, 'status': 'unavailable', 'reason': 'assessment_unavailable',
        'dictionary_sha256': DICTIONARY_SHA256,
        'checkpoint_sha256': evidence.get('checkpoint_sha256'),
        'vocabulary_sha256': evidence.get('vocabulary_sha256'),
        'calibration': {'applied': False, 'reason': 'crottc_calibration_unavailable'},
        'words': [{'index': i, 'text': word, 'status': 'uncertain'} for i, word in enumerate(tokens)],
        'raw_events': [], 'alignment': {'status': 'unavailable', 'method': 'dictionary_variant_edit_alignment'}}
    comparison = {'words': [{'text': word, 'status': 'uncertain'} for word in tokens],
                  'events': [], 'counts': dict.fromkeys(('match', 'substitution', 'deletion', 'insertion'), 0)}
    targets = []
    try:
        if evidence.get('unresolved_labels'):
            raise ValueError('unsupported_perceived_phone')
        observed = validate_phones(evidence)
        variants = accepted_variants(root, text)
        missing = [word['text'] for word in variants if not word['available']]
        if missing:
            assessment['missing_words'] = missing
            raise ValueError('reference_contains_unknown_words')
        if not tokens or len(variants) != len(tokens):
            raise ValueError('invalid_reference')
        choices = [[entry['phones'] for entry in word['variants']] for word in variants]
        path, distance = align_variants(choices, [item['phone'] for item in observed])
        selected = {event['word_index']: event['variant_index'] for event in path if 'variant_index' in event}
        raw_events = []
        canonical, indices = [], []
        for index, token in enumerate(tokens):
            variant = variants[index]['variants'][selected[index]]
            targets.append({'text': token, **variant, 'dictionary_variant_index': selected[index],
                            'dictionary_variant_count': len(choices[index])})
            canonical.extend(variant['phones']); indices.extend([index] * len(variant['phones']))
            spans = [observed[event['observed_index']] for event in path if event['word_index'] == index
                     and event.get('observed_index') is not None and 'start_sec' in observed[event['observed_index']]]
            if spans:
                assessment['words'][index].update(start_sec=min(item['start_sec'] for item in spans),
                                                 end_sec=max(item['end_sec'] for item in spans))
        # Populate the same dictionary IPA/stress metadata used by word playback.
        display = target_words(root, text, {'canonical_phones': canonical, 'canonical_word_indices': indices})
        for target, item in zip(targets, display):
            if item.get('pronunciation_phones'):
                target['pronunciation_phones'] = item['pronunciation_phones']
        for event in path:
            if event['kind'] != 'match':
                raw_events.append({key: event[key] for key in ('kind', 'word_index', 'expected_phone', 'observed_phone')}
                                  | {'source': 'crottc_perceived_sequence_edit', 'tentative': True})
        assessment.update(status='ready', reason=None, raw_events=raw_events,
            alignment={'status': 'aligned', 'method': 'dictionary_variant_edit_alignment',
                       'edit_distance': distance, 'reference_phone_count': len(canonical),
                       'perceived_phone_count': len(observed), 'variant_compatible': not raw_events,
                       'timestamps': evidence.get('word_timing', 'unavailable')})
    except (ValueError, KeyError, TypeError, IndexError) as error:
        assessment['reason'] = str(error) if isinstance(error, ValueError) else 'invalid_phone_evidence'
    diagnostic = sound_candidates(assessment, text, reference_confirmed=confirmed)
    diagnostic['method'] = METHOD
    for word in diagnostic['words']:
        for event in word['events']:
            event['source'] = 'crottc_perceived_sequence_edit'
    return {'assessment': assessment, 'comparison': comparison, 'raw_comparison': comparison,
            'sound_candidates': diagnostic, 'target_words': targets,
            'notice': 'Review sounds with CROTTC-IF.'}
