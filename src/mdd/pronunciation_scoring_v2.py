"""Candidate posterior-margin aggregations for pronunciation review.

These scores rank words for review; they are not calibrated probabilities.
The aggregation is deliberately separate from the frozen v1 scorer so an
experiment cannot silently change the production or research baseline.
"""
from __future__ import annotations

import math


METHODS = ('q20', 'mean', 'worst_phone')

# Exploratory one-sided review threshold fitted on HJK/LXC in the frozen
# pronunciation-scorer-v2 comparison. This may emit a listening hint only; it
# must never be used to label non-alerted words as correct.
REVIEW_ALERT_METHOD = 'worst_phone'
REVIEW_ALERT_THRESHOLD = 0.1871480941772461


def aggregate_phone_margins(phone_scores, method):
    """Convert per-phone log-posterior margins into one word error score.

    Higher values indicate weaker support for the reference pronunciation.
    ``q20`` is the existing baseline; ``mean`` averages all phone margins;
    ``worst_phone`` uses the least-supported phone. The latter two are
    candidates, not approved diagnoses.
    """
    if method not in METHODS:
        raise ValueError('unknown_pronunciation_score_method')
    margins = []
    for item in phone_scores if isinstance(phone_scores, list) else ():
        value = item.get('margin') if isinstance(item, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('invalid_phone_margin')
        margins.append(float(value))
    if not margins:
        return None
    if method == 'q20':
        ordered = sorted(margins)
        rank = .2 * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        margin = ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)
    elif method == 'mean':
        margin = sum(margins) / len(margins)
    else:
        margin = min(margins)
    return -margin
