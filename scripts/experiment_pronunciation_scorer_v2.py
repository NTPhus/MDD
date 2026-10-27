"""Compare fixed posterior-margin word-score aggregations on cached validation.

Uses no new neural inference and never fits thresholds on the held-out speaker.
The output is an exploratory scorer study, not a production calibration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from mdd.pronunciation_scoring_v2 import METHODS, aggregate_phone_margins
from scripts.experiment_pronunciation_calibration import (
    DEFAULT_CONFIG, fit_thresholds, load_config, load_predictions, metrics, read_json,
    threshold_candidates,
)

INPUT = ROOT / 'data/generated/pronunciation-calibration-v1/full-validation'
OUTPUT = ROOT / 'data/generated/pronunciation-scorer-v2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(rows):
    labelled = [row for row in rows if row['score_eligible'] and row['gold'] in (0, 1)]
    positives = [row['error_score'] for row in labelled if row['gold'] == 1]
    negatives = [row['error_score'] for row in labelled if row['gold'] == 0]
    if not positives or not negatives:
        return None
    pairs = sum(1 if positive > negative else .5 if positive == negative else 0
                for positive in positives for negative in negatives)
    return pairs / (len(positives) * len(negatives))


def candidate_rows(predictions, method):
    output = []
    for row in predictions:
        for word, gold in zip(row['scoring']['words'], row['gold_word_errors']):
            score = aggregate_phone_margins(word.get('phone_scores'), method)
            strict_words = (row.get('strict_baseline') or {}).get('words') or []
            strict_status = strict_words[word['index']].get('status') if word['index'] < len(strict_words) else 'unavailable'
            output.append({'utterance_id': row['utterance_id'], 'speaker_id': row['speaker_id'],
                'role': row['role'], 'index': word['index'], 'text': word['text'], 'gold': gold,
                'error_score': score, 'score_eligible': word.get('score_eligible') is True and score is not None,
                'greedy_difference': strict_status == 'difference', 'reason': word.get('reason')})
    return output


def alert_metrics(rows, threshold, *, require_greedy=False):
    labelled = [row for row in rows if row['gold'] in (0, 1)]
    alerts = [row for row in labelled if row['score_eligible'] and row['error_score'] >= threshold
              and (not require_greedy or row['greedy_difference'])]
    tp = sum(row['gold'] == 1 for row in alerts)
    fp = sum(row['gold'] == 0 for row in alerts)
    errors = sum(row['gold'] == 1 for row in labelled)
    correct = sum(row['gold'] == 0 for row in labelled)
    ratio = lambda a, b: a / b if b else None
    return {'labelled_words': len(labelled), 'eligible_words': sum(row['score_eligible'] for row in labelled),
        'alert_words': len(alerts), 'tp_alerts': tp, 'fp_alerts': fp,
        'precision': ratio(tp, len(alerts)), 'recall_all_errors': ratio(tp, errors),
        'false_positive_rate_all_correct': ratio(fp, correct), 'alert_coverage': ratio(len(alerts), len(labelled)),
        'correct_words_alerted': fp, 'error_words_alerted': tp,
        'non_alerted_words_are_unclassified': True}


def fit_alert_threshold(fit, max_fpr, min_true_alerts, *, require_greedy=False):
    candidates = threshold_candidates(fit, 101)
    accepted = []
    for threshold in candidates:
        result = alert_metrics(fit, threshold, require_greedy=require_greedy)
        if (result['false_positive_rate_all_correct'] is not None
                and result['false_positive_rate_all_correct'] <= max_fpr
                and result['tp_alerts'] >= min_true_alerts):
            accepted.append((result['recall_all_errors'], result['precision'] or 0.,
                             -result['false_positive_rate_all_correct'], threshold, result))
    return max(accepted) if accepted else None


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_config(DEFAULT_CONFIG)
    plan = read_json(INPUT / 'selection.json')
    if plan.get('mode') != 'full-validation' or plan.get('frozen_before_inference') is not True:
        raise ValueError('Expected the frozen 450-utterance validation selection')
    predictions = load_predictions(INPUT, plan)
    if len(predictions) != 450:
        raise ValueError('Expected all 450 cached validation predictions')
    policy = config['threshold_policy']
    summaries, flattened, threshold_rows = [], [], []
    for method in METHODS:
        rows = candidate_rows(predictions, method)
        fit = [row for row in rows if row['role'] == 'fit']
        heldout = [row for row in rows if row['role'] == 'heldout']
        selected, search = fit_thresholds(fit, policy)
        thresholds = selected[-2:] if selected else (None, None)
        fit_metrics = metrics(fit, *thresholds)
        heldout_metrics = metrics(heldout, *thresholds)
        # Keep the original word-level strict edit baseline alongside candidate scores.
        strict_fit = [dict(row, score_eligible=True) for row in fit]
        strict_heldout = [dict(row, score_eligible=True) for row in heldout]
        for target, role in ((strict_fit, 'fit'), (strict_heldout, 'heldout')):
            source = {item['utterance_id']: item for item in predictions if item['role'] == role}
            for item in target:
                index = item['index']
                strict = source[item['utterance_id']].get('strict_baseline') or {}
                words = strict.get('words') or []
                item['strict_status'] = words[index]['status'] if index < len(words) else 'unavailable'
        baseline_fit = metrics(strict_fit, baseline=True)
        baseline_heldout = metrics(strict_heldout, baseline=True)
        failures_fit = [] if selected else ['no_threshold_meets_fit_policy']
        failures_heldout = [] if selected else ['not_evaluated_no_fit_threshold']
        if selected:
            for field, maximum in (('false_positive_rate', policy['max_false_positive_rate']),
                                   ('false_negative_rate', policy['max_false_negative_rate'])):
                if heldout_metrics[field] is None or heldout_metrics[field] > maximum:
                    failures_heldout.append(field)
            if heldout_metrics['coverage'] is None or heldout_metrics['coverage'] < policy['min_coverage']:
                failures_heldout.append('coverage')
        alert = fit_alert_threshold(fit, policy['max_false_positive_rate'], policy['min_error_decisions'],
                                    require_greedy=True)
        if alert:
            _, _, _, alert_threshold, alert_fit = alert
            alert_heldout = alert_metrics(heldout, alert_threshold, require_greedy=True)
        else:
            alert_threshold, alert_fit, alert_heldout = None, None, None
        summaries.append({'method': method, 'ranking_auc_fit': auc(fit), 'ranking_auc_heldout': auc(heldout),
            'selected_on_fit': bool(selected), 'accept_threshold': thresholds[0], 'error_threshold': thresholds[1],
            'fit_policy_failures': failures_fit, 'heldout_policy_failures': failures_heldout,
            'one_sided_review_alert': {'selected_on_fit': alert is not None,
                'requires_greedy_difference': True,
                'fit_fpr_cap': policy['max_false_positive_rate'], 'minimum_fit_true_alerts': policy['min_error_decisions'],
                'threshold': alert_threshold, 'fit': alert_fit, 'heldout': alert_heldout},
            'fit_metrics': fit_metrics, 'heldout_metrics': heldout_metrics,
            'strict_greedy_baseline_fit': baseline_fit, 'strict_greedy_baseline_heldout': baseline_heldout,
            'threshold_grid_rows': len(search)})
        for row in rows:
            flattened.append({'method': method, **row})
        for candidate in search:
            threshold_rows.append({'method': method, **candidate})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'scores.csv', flattened)
    write_csv(OUTPUT / 'thresholds.csv', threshold_rows)
    report = {'schema': 'pronunciation-scorer-comparison-v2', 'status': 'exploratory_validation',
        'experiment': 'cached_B_posterior_word_margin_aggregation', 'methods': summaries,
        'provenance': {'config_sha256': sha256(DEFAULT_CONFIG),
            'selection_sha256': plan['selection_sha256'],
            'prediction_sha256': sha256(INPUT / 'predictions.jsonl'),
            'original_scorer_sha256': read_json(INPUT / 'inference.json')['method_sha256'],
            'candidate_scorer_sha256': sha256(ROOT / 'src/mdd/pronunciation_scoring_v2.py'),
            'checkpoint_sha256': plan['identity']['checkpoint_sha256'],
            'fit_speakers': config['fit_speakers'], 'heldout_speakers': config['heldout_speakers'],
            'fit_utterances': sum(row['role'] == 'fit' for row in predictions),
            'heldout_utterances': sum(row['role'] == 'heldout' for row in predictions),
            'new_model_inference': False, 'training': False, 'thresholds_fit_on': 'HJK/LXC only'},
        'limitations': [
            'PNV is a held-out scorer speaker but the B checkpoint was selected using all 450 validation utterances, including PNV.',
            'All candidates reuse the v1 word eligibility mask; insertion/deletion and unsupported words remain abstentions.',
            'This experiment ranks manually labeled word errors from L2-ARCTIC speakers; it is not a native-speaker evaluation.',
            'No candidate is promoted to the live demo by this script.',
            'The strict greedy edit baseline is exploratory and is not a calibrated pronunciation diagnosis.'
        ]}
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'methods': summaries}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
