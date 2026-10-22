"""Experimental chat policy; the frozen research edit-distance baseline is separate."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .demo_reference import DICTIONARY_SHA256

CALIBRATION_PATH = 'data/generated/pronunciation-calibration-v1/full-validation/calibration.json'
METHOD = 'ctc_variant_posterior_margin_v1'


@lru_cache(maxsize=32)
def _digest(path, modified, size):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_identity(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Calibration source is outside the workspace')
    stat = path.stat()
    return _digest(str(path), stat.st_mtime_ns, stat.st_size)


def _validation_proof(root, path, value):
    """Reject a stale scorer, changed split, relaxed policy, or failed metrics."""
    config = json.loads((root / 'configs/pronunciation_calibration.json').read_text(encoding='utf-8'))
    validation = value['validation']
    if (config.get('method') != METHOD or config.get('no_training') is not True
            or config.get('no_llm_or_tts') is not True or validation.get('failures') != []
            or validation.get('registered_policy') != config['threshold_policy']
            or validation.get('fit_speakers') != config['fit_speakers']
            or validation.get('heldout_speakers') != config['heldout_speakers']):
        return False
    if value.get('method_sha256') != file_identity(root, 'src/mdd/pronunciation_assessment.py'):
        return False
    for key, source in (('protocol_sha256', config['protocol_dir'] + '/protocol.json'),
                        ('splits_sha256', config['protocol_dir'] + '/splits.json'),
                        ('annotations_sha256', config['annotations'])):
        if value.get(key) != config.get(key) or value.get(key) != file_identity(root, source):
            return False
    plan = json.loads((path.parent / 'selection.json').read_text(encoding='utf-8'))
    records = plan['records']
    selected_hash = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False,
                                              separators=(',', ':')).encode()).hexdigest()
    if (plan.get('mode') != 'full-validation' or validation.get('selection_mode') != plan['mode']
            or plan.get('frozen_before_inference') is not True or plan.get('test_audio_or_gold_used') is not False
            or selected_hash != plan.get('selection_sha256') or selected_hash != validation.get('selection_sha256')
            or any(value.get(key) != item for key, item in plan['identity'].items())):
        return False
    splits = json.loads((root / config['protocol_dir'] / 'splits.json').read_text(encoding='utf-8'))
    ids = {row['utterance_id'] for row in records}
    if len(ids) != len(records) or ids != set(splits['joint_validation_ids']):
        return False
    if ids & (set(splits['joint_train_ids']) | set(splits['cascade_test_ids'])):
        return False
    for role, speakers in (('fit', config['fit_speakers']), ('heldout', config['heldout_speakers'])):
        if sum(row['role'] == role for row in records) != validation[role + '_utterances']:
            return False
        if any(row['speaker_id'] not in speakers for row in records if row['role'] == role):
            return False
        result = validation[role + '_metrics']
        policy = config['threshold_policy']
        count_keys = ('tp', 'fp', 'tn', 'fn', 'gold_correct', 'gold_error', 'gold_labelled_words',
                      'decided_words', 'abstained_words', 'correct_abstained', 'error_abstained')
        if any(type(result.get(key)) is not int or result[key] < 0 for key in count_keys):
            return False
        decided = sum(result[key] for key in ('tp', 'fp', 'tn', 'fn'))
        labelled = result['gold_correct'] + result['gold_error']
        if (not labelled or result['gold_labelled_words'] != labelled or result['decided_words'] != decided
                or result['abstained_words'] != result['correct_abstained'] + result['error_abstained']
                or decided + result['abstained_words'] != labelled
                or result['tn'] + result['fp'] + result['correct_abstained'] != result['gold_correct']
                or result['tp'] + result['fn'] + result['error_abstained'] != result['gold_error']):
            return False
        coverage = result['coverage']
        if (type(coverage) not in (int, float) or not math.isfinite(coverage)
                or not 0 <= coverage <= 1 or abs(coverage - decided / labelled) > 1e-9):
            return False
        for metric, bound in (('false_positive_rate', 'max_false_positive_rate'),
                              ('false_negative_rate', 'max_false_negative_rate')):
            rate = result[metric]
            if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= policy[bound]:
                return False
            numerator, denominator = ((result['fp'], result['fp'] + result['tn']) if metric == 'false_positive_rate'
                                      else (result['fn'], result['fn'] + result['tp']))
            if not denominator or abs(rate - numerator / denominator) > 1e-9:
                return False
        if (result['coverage'] < policy['min_coverage']
                or result['tn'] + result['fp'] < policy['min_correct_decisions']
                or result['tp'] + result['fn'] < policy['min_error_decisions']):
            return False
    return True


def load_calibration(root, evidence):
    """Trust only an eligible local validation artifact matching these exact weights.

    A flag supplied by the browser or a persisted ``verified`` flag is never
    sufficient. A pilot is exploratory; only a verified full-validation artifact
    may enable decisions. Failed/missing calibration means abstention in demo.
    """
    root = Path(root).resolve()
    path = root / CALIBRATION_PATH
    try:
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 1000000:
            return None
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value.get('schema') != 'mdd-pronunciation-calibration-v1'
                or value.get('schema_version') != 1 or value.get('method') != METHOD
                or value.get('eligible') is not True or value.get('status') != 'eligible'
                or value.get('validation', {}).get('status') != 'verified'
                or value.get('validation', {}).get('split') != 'calibration'):
            return None
        if value.get('dictionary_sha256') != DICTIONARY_SHA256:
            return None
        if any(value.get(key) != evidence.get(key) or not evidence.get(key)
               for key in ('checkpoint_sha256', 'vocabulary_sha256')):
            return None
        thresholds = [value.get(key) for key in ('accept_threshold', 'error_threshold')]
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in thresholds):
            return None
        if thresholds[0] >= thresholds[1] or not _validation_proof(root, path, value):
            return None
        return {**value, 'verified': True}
    except (OSError, ValueError, AttributeError, TypeError, KeyError, IndexError, OverflowError):
        return None


def assessment_fields(root, text, evidence, *, confirmed=False, raw_comparison=None):
    from .pronunciation_assessment import assess
    from .chat_diagnostics import sound_candidates
    result = assess(root, text, evidence, calibration=load_calibration(root, evidence),
                    reference_confirmed=confirmed)
    comparison = result.get('comparison')
    if comparison is None and raw_comparison:
        # A failed/OOV alignment must not fall back to categorical edit errors.
        comparison = deepcopy(raw_comparison)
        comparison['events'] = []
        comparison['counts'] = {}
        for word in comparison.get('words', []):
            word['status'] = 'uncertain'
    return {'assessment': {key: value for key, value in result.items()
                           if key not in {'comparison', 'target_words'}},
            'comparison': comparison, 'raw_comparison': raw_comparison,
            'sound_candidates': sound_candidates(result, text, reference_confirmed=confirmed),
            'target_words': result.get('target_words', []),
            'notice': ('Sound suggestions use the ASR transcript and may reflect misheard words. Edit the text only if it differs from what you said.' if not confirmed else
                       'Sound estimates use pronunciation variants and acoustic evidence; uncertain words are not marked wrong.')}
