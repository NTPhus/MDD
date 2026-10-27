"""Tentative acoustic review hints for practice, never diagnoses.

Dictionary-variant versus greedy-CTC edits remain visible as review candidates.
Posterior margins annotate stronger candidates, but do not hide lower-scoring
edits while the categorical calibration remains ineligible.
"""
from __future__ import annotations

from copy import deepcopy

from .pronunciation_scoring_v2 import (
    REVIEW_ALERT_METHOD, REVIEW_ALERT_THRESHOLD, aggregate_phone_margins,
)


def sound_candidates(assessment, reference_text, *, reference_confirmed=False, posterior_review=False):
    """Expose tentative listening hints and annotate posterior evidence.

    Confirmation changes the reference's provenance, never turns these hints
    into calibrated diagnoses. A high review score is stronger evidence for
    replay, while lower scores remain tentative decoder edits. Absence of a hint
    does not mean the pronunciation is correct. Unknown words/alignment failures
    stay explicit.
    """
    confirmed = reference_confirmed is True
    result = {'status': 'unavailable', 'diagnostic_only': True,
              'method': ('posterior_margin_review_alert_v2' if posterior_review
                         else 'cmu_variant_greedy_review_v1'), 'reference_confirmed': confirmed,
              'provisional': not confirmed, 'source': 'confirmed_reference' if confirmed else 'asr',
              'words': [], 'count': 0, 'reason': 'assessment_unavailable',
              'score_method': REVIEW_ALERT_METHOD if posterior_review else None,
              'score_threshold': REVIEW_ALERT_THRESHOLD if posterior_review else None,
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
        score_word = scored[index]
        error_score = None
        strong_alert = False
        if posterior_review:
            try:
                error_score = aggregate_phone_margins(score_word.get('phone_scores'), REVIEW_ALERT_METHOD)
            except (TypeError, ValueError, OverflowError):
                error_score = None
            strong_alert = (score_word.get('score_eligible') is True and error_score is not None
                            and error_score >= REVIEW_ALERT_THRESHOLD)
        by_word.setdefault(index, []).append({'kind': event['kind'], 'expected_phone': expected,
            'observed_phone': observed, 'word_index': index, 'tentative': True,
            'source': ('posterior_margin_review_alert_v2' if strong_alert else 'unvalidated_greedy_edit'),
            **({'error_score': error_score, 'score_eligible': score_word.get('score_eligible') is True,
                'review_level': 'stronger_posterior_support' if strong_alert else 'tentative_edit'}
               if posterior_review and error_score is not None else {})})
    words = []
    for index, events in sorted(by_word.items()):
        word = {'index': index, 'text': tokens[index], 'events': deepcopy(events), 'tentative': True}
        if posterior_review and events[0].get('error_score') is not None:
            word.update(error_score=events[0]['error_score'], score_method=REVIEW_ALERT_METHOD,
                        review_level=events[0]['review_level'])
        for key in ('start_sec', 'end_sec'):
            if type(scored[index].get(key)) in (int, float):
                word[key] = scored[index][key]
        words.append(word)
    strong_count = sum(word.get('review_level') == 'stronger_posterior_support' for word in words)
    result.update(status='available', reason='tentative_differences' if words else 'no_greedy_difference',
                  words=words, count=len(words),
                  stronger_posterior_count=strong_count if posterior_review else None,
                  message=(('The decoder found tentative sound differences. Posterior evidence is stronger for some candidates; replay before drawing a conclusion.'
                            if words else 'The decoder did not suggest a sound difference. This does not certify correct pronunciation.')
                           if posterior_review else
                           ('The model suggests reviewing these sounds. These are unvalidated candidates, not confirmed pronunciation errors.'
                            if words else 'No sound difference was suggested by this decoder. This does not certify correct pronunciation.')))
    if not confirmed:
        result['message'] += (' Confirm the transcript before reviewing pronunciation.' if posterior_review else
                              ' These suggestions use the ASR transcript, which may mishear words; edit the text only if it differs from what you said.')
    return result
