"""Tentative acoustic differences for practice, never calibrated diagnoses.

These are selected-dictionary-variant versus greedy-CTC edit candidates. They
may reflect a recognizer/alignment mistake. Exposing them must not turn failed
calibration or an unconfirmed ASR transcript into a pronunciation pass/fail.
"""
from __future__ import annotations

from copy import deepcopy


def sound_candidates(assessment, reference_text, *, reference_confirmed=False):
    """Expose review hints immediately; an ASR-derived target is provisional.

    Confirmation changes the reference's provenance, never turns these hints
    into calibrated diagnoses. Unknown words/alignment failures stay explicit.
    """
    confirmed = reference_confirmed is True
    result = {'status': 'unavailable', 'diagnostic_only': True,
              'method': 'cmu_variant_greedy_review_v1', 'reference_confirmed': confirmed,
              'provisional': not confirmed, 'source': 'confirmed_reference' if confirmed else 'asr',
              'words': [], 'count': 0, 'reason': 'assessment_unavailable',
              'message': 'No usable reference or acoustic alignment was available. No pronunciation conclusion was made.'}
    if not isinstance(assessment, dict) or assessment.get('status') != 'ready':
        if isinstance(assessment, dict):
            reason = assessment.get('reason') or (assessment.get('alignment') or {}).get('reason')
            if isinstance(reason, str):
                result['reason'] = reason
            missing = assessment.get('missing_words')
            if isinstance(missing, list) and all(isinstance(word, str) for word in missing):
                result['missing_words'] = list(missing)
            if reason == 'reference_contains_unknown_words':
                result['message'] = (('Your confirmed words can be used for the reply. ' if confirmed else
                                      'Some recognized words may need editing if ASR misheard them. ')
                                     + 'Some words have no dictionary pronunciation; no pronunciation judgement was made for this recording.')
            elif reason == 'reference_phone_budget_exceeded':
                result['message'] = 'The reference is too long for this recording comparison. Use a shorter recording and its spoken text.'
        return result
    tokens = reference_text.split() if isinstance(reference_text, str) else []
    scored = assessment.get('words', [])
    if (not tokens or not isinstance(scored, list) or len(tokens) != len(scored)
            or any(not isinstance(word, dict) or word.get('text') != token for word, token in zip(scored, tokens))):
        result.update(reason='reference_word_mismatch',
                      message='The stored sound estimates do not match the reference words. Compare the recording again.')
        return result
    by_word = {}
    for event in assessment.get('raw_events', []):
        if not isinstance(event, dict):
            continue
        index = event.get('word_index')
        if (type(index) is not int or not 0 <= index < len(tokens)
                or event.get('kind') not in {'substitution', 'deletion', 'insertion'}):
            continue
        # Retain only reference/predicted phone information, never a gold label
        # or a claim that a posterior is the probability of a learner's error.
        expected, observed = event.get('expected_phone'), event.get('observed_phone')
        if any(phone is not None and (not isinstance(phone, str) or not 1 <= len(phone) <= 16)
               for phone in (expected, observed)):
            continue
        by_word.setdefault(index, []).append({'kind': event['kind'], 'expected_phone': expected,
            'observed_phone': observed, 'word_index': index, 'tentative': True,
            'source': 'unvalidated_greedy_edit'})
    words = []
    for index, events in sorted(by_word.items()):
        word = {'index': index, 'text': tokens[index], 'events': deepcopy(events), 'tentative': True}
        for key in ('start_sec', 'end_sec'):
            if type(scored[index].get(key)) in (int, float):
                word[key] = scored[index][key]
        words.append(word)
    result.update(status='available', reason='tentative_differences' if words else 'no_greedy_difference',
                  words=words, count=len(words),
                  message=('The model suggests reviewing these sounds. These are unvalidated candidates, not confirmed pronunciation errors.'
                           if words else 'No sound difference was suggested by this decoder. This does not certify correct pronunciation.'))
    if not confirmed:
        result['message'] += ' These suggestions use the ASR transcript, which may mishear words; edit the text only if it differs from what you said.'
    return result
