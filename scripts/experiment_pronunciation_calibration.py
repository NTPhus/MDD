"""Frozen-checkpoint word-error calibration, with speaker-held-out development.

Default: prepare a deterministic 40-utterance pilot, with no model inference.
--pilot --run performs inference, calibration and reports; it never trains.
--full-validation --run explicitly processes 450 development utterances.
--test --frozen-calibration PATH --run evaluates the untouched 900 test only
after a frozen eligible artifact exists; it never refits thresholds on test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIG = ROOT / "configs/pronunciation_calibration.json"


def rooted(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if config.get("no_training") is not True or config.get("no_llm_or_tts") is not True:
        raise ValueError("This experiment must remain inference/calibration only")
    return config


def identity(config):
    files = {"protocol_sha256": rooted(config["protocol_dir"]) / "protocol.json",
             "splits_sha256": rooted(config["protocol_dir"]) / "splits.json",
             "annotations_sha256": rooted(config["annotations"]),
             "checkpoint_sha256": rooted(config["checkpoint"]) / "model.safetensors",
             "vocabulary_sha256": rooted(config["checkpoint"]) / "vocab.json",
             "dictionary_sha256": rooted(config["dictionary"])}
    values = {name: digest(path) for name, path in files.items()}
    for name, actual in values.items():
        if config.get(name) != actual:
            raise ValueError("Pinned file changed: " + name)
    protocol = read_json(files["protocol_sha256"])
    manifest = rooted(config["protocol_dir"]) / "manifest.jsonl"
    if digest(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest checksum mismatch")
    metadata = read_json(rooted(config["checkpoint"]) / "metadata.json")
    if metadata.get("purpose") != "scientific_training" or metadata["provenance"]["files"]["protocol"]["sha256"] != values["protocol_sha256"]:
        raise ValueError("Checkpoint is not the frozen cascade B experiment")
    return {**values, "method": config["method"], "protocol_id": protocol["protocol_id"]}


def _rank(config, uid):
    return hashlib.sha256((config["selection_hash_salt"] + ":" + uid).encode()).hexdigest()


def selection_plan(config, mode="pilot"):
    """Select from IDs/manifest only, before loading labels or model outputs."""
    split = read_json(rooted(config["protocol_dir"]) / "splits.json")
    validation = set(split["joint_validation_ids"])
    test = set(split["cascade_test_ids"])
    train = set(split["joint_train_ids"])
    groups = {"fit": set(config["fit_speakers"]), "heldout": set(config["heldout_speakers"])}
    if groups["fit"] & groups["heldout"] or train & validation or test & (train | validation):
        raise ValueError("Split leakage")
    if {uid.split('/')[0] for uid in validation} != groups["fit"] | groups["heldout"]:
        raise ValueError("Development speaker partition changed")
    if {uid.split('/')[0] for uid in test} & (groups["fit"] | groups["heldout"]):
        raise ValueError("Test speaker leakage")
    manifest = {row["utterance_id"]: row for row in (json.loads(line) for line in
        (rooted(config["protocol_dir"]) / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    selected = []
    if mode == "test":
        roles = [(uid, "test") for uid in sorted(test, key=lambda uid: _rank(config, uid))]
    else:
        roles = []
        for role, speakers in groups.items():
            for speaker in sorted(speakers):
                ids = sorted((uid for uid in validation if uid.split('/')[0] == speaker), key=lambda uid: _rank(config, uid))
                if mode == "pilot":
                    ids = ids[:config["pilot_counts"][speaker]]
                roles.extend((uid, role) for uid in ids)
    for uid, role in roles:
        item = manifest[uid]
        selected.append({"utterance_id": uid, "speaker_id": item["speaker_id"], "role": role,
                         "audio_path": item["audio"]["path"], "audio_sha256": item["audio"]["sha256"],
                         "duration_sec": item["duration_sec"], "annotation_is_manual": item["annotation_is_manual"]})
    return {"experiment_id": config["experiment_id"], "mode": mode,
            "selection_rule": "SHA256(salt + ':' + utterance_id), stratified by pre-registered speaker counts; no label/prediction filtering",
            "identity": identity(config), "records": selected,
            "selection_sha256": json_digest(selected), "frozen_before_inference": True,
            "test_audio_or_gold_used": mode == "test", "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "limitations": config["limitations"]}


def prepare(config, output, mode="pilot"):
    output = Path(output)
    plan = selection_plan(config, mode)
    path = output / "selection.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Frozen selection changed; choose a new output directory")
    else:
        save_json(path, plan)
    return plan


def selected_annotations(config, plan):
    selected = {row["utterance_id"] for row in plan["records"]}
    result = {}
    # Only selected rows are parsed/analysed; test labels are not inspected in
    # default development runs. The shared file is still checksummed in full.
    uid_pattern = re.compile(r'"utterance_id"\s*:\s*"([^"\\]+)"')
    with rooted(config["annotations"]).open(encoding="utf-8") as stream:
        for line in stream:
            match = uid_pattern.search(line)
            if match and match[1] in selected:
                value = json.loads(line)
                result[value["utterance_id"]] = value
    if set(result) != selected:
        raise ValueError("Selected annotations are missing")
    return result


def word_labels(row):
    """Manual annotation labels only; never pass this object to the scorer."""
    words = row["prompt_text"].split()
    intervals = row.get("word_intervals", [])
    key = lambda text: ''.join(c for c in str(text).casefold() if c.isalnum() or c in "'’")
    aligned = len(words) == len(intervals) and all(item.get("index") == index and
        key(item.get("text")) == key(word) for index, (word, item) in enumerate(zip(words, intervals)))
    errors = row.get("gold_error_events", [])
    if not aligned or any(not isinstance(event.get("word_index"), int) or not 0 <= event["word_index"] < len(words) for event in errors):
        return [None] * len(words)
    flagged = {event["word_index"] for event in errors}
    return [int(index in flagged) for index in range(len(words))]


def data_audit(config, plan):
    annotations = selected_annotations(config, plan)
    rows = []
    for selected in plan["records"]:
        row = annotations[selected["utterance_id"]]
        labels = word_labels(row)
        rows.append({"utterance_id": row["utterance_id"], "role": selected["role"], "speaker_id": row["speaker_id"],
                     "duration_sec": selected["duration_sec"], "words": len(labels), "gold_correct": labels.count(0),
                     "gold_errors": labels.count(1), "gold_unavailable": labels.count(None)})
    return rows


def _cache_path(output, uid):
    return Path(output) / "evidence" / (hashlib.sha256(uid.encode()).hexdigest()[:20] + ".npz")


def save_evidence(path, evidence):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in evidence.items() if key != "log_probs"}
    np.savez_compressed(path, log_probs=np.asarray(evidence["log_probs"], dtype=np.float32),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))


def load_evidence(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        result = json.loads(str(data["metadata"].item()))
        result["log_probs"] = data["log_probs"].copy()
    result["id_to_phone"] = {int(key): value for key, value in result["id_to_phone"].items()}
    return result


def infer(config, output, plan):
    """The audio network receives bytes only; CMU reference follows inference."""
    from mdd.demo_inference import LocalInference, canonical_comparison
    from mdd.demo_reference import dictionary_reference
    from mdd.pronunciation_assessment import score_words
    annotations = selected_annotations(config, plan)
    model = {"id": "cascade-B", "label": "Frozen cascade B", "kind": "phoneme", "experimental": False,
             "path": rooted(config["checkpoint"])}
    runtime = LocalInference(ROOT, threads=config["cpu_threads"])
    rows = []
    method_hash = digest(ROOT / "src/mdd/pronunciation_assessment.py")
    then = time.monotonic()
    for index, selected in enumerate(plan["records"], 1):
        uid = selected["utterance_id"]
        source = annotations[uid]
        audio_path = rooted(selected["audio_path"])
        if digest(audio_path) != selected["audio_sha256"]:
            raise ValueError("Audio checksum changed: " + uid)
        cache = _cache_path(output, uid)
        shared_cache = _cache_path(rooted(config['output_root']) / 'pilot', uid)
        existing_cache = cache if cache.exists() else shared_cache
        if existing_cache.exists():
            evidence = load_evidence(existing_cache)
            cache_reused = True
        else:
            evidence = runtime.posterior_evidence(model, audio_path.read_bytes())
            evidence["source_audio_sha256"] = selected["audio_sha256"]
            cache_reused = False
        if (evidence.get("checkpoint_sha256") != config["checkpoint_sha256"] or
            evidence.get("vocabulary_sha256") != config["vocabulary_sha256"] or
            evidence.get("source_audio_sha256") != selected["audio_sha256"]):
            raise ValueError("Cached posterior identity differs: " + uid)
        if not cache.exists():
            save_evidence(cache, evidence)
        # No perceived phones, manual word boundaries or gold error labels go
        # into score_words. The intended scripted text is the reference.
        scored = score_words(ROOT, source["prompt_text"], evidence)
        if scored.get("method") != config["method"]:
            raise ValueError("Scorer method differs from registered calibration")
        reference, missing = dictionary_reference(ROOT, source["prompt_text"])
        strict = canonical_comparison(source["prompt_text"], reference, [item["phone"] for item in evidence["phonemes"]])
        labels = word_labels(source)
        if len(scored["words"]) != len(labels):
            raise ValueError("Scorer word order differs: " + uid)
        row = {"utterance_id": uid, "speaker_id": selected["speaker_id"], "role": selected["role"],
               "reference_text": source["prompt_text"], "source_audio_sha256": selected["audio_sha256"],
               "cache_reused": cache_reused, "method_sha256": method_hash,
               "scoring": scored, "strict_baseline": strict, "missing_dictionary_words": missing,
               "gold_word_errors": labels}
        rows.append(row)
        # Save progress after each actual utterance; reruns reuse verified raw
        # posterior caches, not cached gold-conditioned predictions.
        path = Path(output) / "predictions.jsonl"
        path.write_text(''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n' for value in rows), encoding="utf-8")
        print(f"[{index}/{len(plan['records'])}] {uid} ({selected['role']}) cache={cache_reused}", flush=True)
    save_json(Path(output) / "inference.json", {"utterances": len(rows), "elapsed_sec": round(time.monotonic() - then, 3),
        "selection_sha256": plan["selection_sha256"], "identity": plan["identity"], "method_sha256": method_hash,
        "network_inputs": "audio only", "scorer_inputs": "intended text + frozen CMU variants + audio posterior; no manual word times or perceived/gold labels",
        "training": False, "llm_or_tts_calls": False})
    return rows


def load_predictions(output, plan):
    path = Path(output) / "predictions.jsonl"
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["utterance_id"] for row in values] != [row["utterance_id"] for row in plan["records"]]:
        raise ValueError("Predictions do not cover exactly the frozen selection")
    inference = read_json(Path(output) / "inference.json")
    if inference["selection_sha256"] != plan["selection_sha256"] or inference["identity"] != plan["identity"]:
        raise ValueError("Prediction provenance differs")
    if inference["method_sha256"] != digest(ROOT / "src/mdd/pronunciation_assessment.py"):
        raise ValueError("Scorer changed; rescore the frozen posterior cache before calibration")
    return values


def flatten(rows, role=None):
    values = []
    for row in rows:
        if role is not None and row["role"] != role:
            continue
        strict = (row.get("strict_baseline") or {}).get("words", [])
        for index, (word, gold) in enumerate(zip(row["scoring"]["words"], row["gold_word_errors"])):
            score = word.get("error_score")
            eligible = word.get("score_eligible") is True and isinstance(score, (int, float)) and math.isfinite(score)
            values.append({"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"], "role": row["role"],
                "index": index, "text": word["text"], "gold": gold, "error_score": score,
                "score_eligible": eligible, "reason": word.get("reason"),
                "strict_status": strict[index]["status"] if index < len(strict) else "unavailable"})
    return values


def classify(row, accept_threshold, error_threshold):
    if not row["score_eligible"] or accept_threshold is None or error_threshold is None:
        return None
    score = row["error_score"]
    if score >= error_threshold:
        return 1
    return 0 if score <= accept_threshold else None


def metrics(words, accept_threshold=None, error_threshold=None, baseline=False):
    counts = Counter(total_words=len(words))
    counts.update({name: 0 for name in ('gold_labelled_words', 'gold_unavailable_words', 'eligible_score_words',
        'gold_correct', 'gold_error', 'decided_words', 'abstained_words', 'correct_abstained', 'error_abstained',
        'tp', 'fp', 'tn', 'fn')})
    for row in words:
        counts['eligible_score_words'] += int(row['score_eligible'])
        if row['gold'] is None:
            counts['gold_unavailable_words'] += 1
            continue
        counts['gold_labelled_words'] += 1
        gold = row['gold']
        counts['gold_error' if gold else 'gold_correct'] += 1
        if baseline:
            predicted = {'match': 0, 'difference': 1}.get(row['strict_status'])
        else:
            predicted = classify(row, accept_threshold, error_threshold)
        if predicted is None:
            counts['abstained_words'] += 1
            counts['error_abstained' if gold else 'correct_abstained'] += 1
        else:
            counts['decided_words'] += 1
            counts[{(1, 1): 'tp', (0, 1): 'fp', (0, 0): 'tn', (1, 0): 'fn'}[(gold, predicted)]] += 1
    ratio = lambda a, b: a / b if b else None
    result = dict(counts)
    result.update(precision=ratio(counts['tp'], counts['tp'] + counts['fp']),
        recall_decided=ratio(counts['tp'], counts['tp'] + counts['fn']),
        recall_all_errors=ratio(counts['tp'], counts['gold_error']),
        false_positive_rate=ratio(counts['fp'], counts['fp'] + counts['tn']),
        false_negative_rate=ratio(counts['fn'], counts['tp'] + counts['fn']),
        false_rejection_rate=ratio(counts['fp'], counts['fp'] + counts['tn']),
        false_acceptance_rate=ratio(counts['fn'], counts['tp'] + counts['fn']),
        false_positive_all_correct=ratio(counts['fp'], counts['gold_correct']),
        false_negative_all_errors=ratio(counts['fn'], counts['gold_error']),
        coverage=ratio(counts['decided_words'], counts['gold_labelled_words']),
        correct_coverage=ratio(counts['tn'] + counts['fp'], counts['gold_correct']),
        error_coverage=ratio(counts['tp'] + counts['fn'], counts['gold_error']),
        f1_decided=ratio(2 * counts['tp'], 2 * counts['tp'] + counts['fp'] + counts['fn']))
    return result


def criteria(result, policy):
    reasons = []
    for field, maximum in [('false_positive_rate', policy['max_false_positive_rate']),
                           ('false_negative_rate', policy['max_false_negative_rate'])]:
        if result[field] is None or result[field] > maximum:
            reasons.append(field)
    if result['coverage'] is None or result['coverage'] < policy['min_coverage']:
        reasons.append('coverage')
    if result['tn'] + result['fp'] < policy['min_correct_decisions']:
        reasons.append('too_few_correct_decisions')
    if result['tp'] + result['fn'] < policy['min_error_decisions']:
        reasons.append('too_few_error_decisions')
    return reasons


def threshold_candidates(words, points):
    scores = sorted({row['error_score'] for row in words if row['score_eligible'] and row['gold'] is not None})
    if not scores:
        return []
    selected = sorted({scores[int(index * (len(scores) - 1) / max(points - 1, 1))] for index in range(points)})
    epsilon = max(1e-6, (selected[-1] - selected[0]) * 1e-6)
    return [selected[0] - epsilon, *[(a + b) / 2 for a, b in zip(selected, selected[1:])], selected[-1] + epsilon]


def fit_thresholds(fit, policy):
    """Pure threshold selection. Held-out rows cannot enter this interface."""
    candidates = threshold_candidates(fit, policy['grid_points'])
    search, accepted = [], []
    for accept in candidates:
        for error in candidates:
            if accept >= error:
                continue
            result = metrics(fit, accept, error)
            failures = criteria(result, policy)
            search.append({'accept_threshold': accept, 'error_threshold': error,
                           **result, 'satisfies_fit_policy': not failures})
            if not failures:
                accepted.append((result['coverage'], result['f1_decided'] or 0., -result['false_positive_rate'],
                                 -result['false_negative_rate'], -(error - accept), accept, error))
    selected = max(accepted) if accepted else None
    return selected, search


def calibrate(config, output, plan, rows):
    if plan['mode'] == 'test':
        raise ValueError('Test labels must never choose thresholds')
    fit = flatten(rows, 'fit')
    heldout = flatten(rows, 'heldout')
    selected, search = fit_thresholds(fit, config['threshold_policy'])
    accept, error = selected[-2:] if selected else (None, None)
    fitted, audited = metrics(fit, accept, error), metrics(heldout, accept, error)
    failures = criteria(audited, config['threshold_policy']) if selected else ['no_threshold_meets_fit_policy']
    eligible = selected is not None and not failures
    artifact = {'schema': 'mdd-pronunciation-calibration-v1', 'schema_version': 1, 'method': config['method'],
        **plan['identity'], 'method_sha256': digest(ROOT / 'src/mdd/pronunciation_assessment.py'),
        'accept_threshold': accept, 'error_threshold': error, 'eligible': eligible,
        'status': 'eligible' if eligible else 'ineligible', 'experimental': True,
        'verified': False, 'verified_field_note': 'Runtime verifier must validate identities and eligibility before setting verified=True in memory.',
        'validation': {'status': 'verified' if eligible else 'failed', 'split': 'calibration',
            'fit_speakers': config['fit_speakers'], 'heldout_speakers': config['heldout_speakers'],
            'selection_mode': plan['mode'], 'selection_sha256': plan['selection_sha256'],
            'fit_utterances': sum(row['role'] == 'fit' for row in rows),
            'heldout_utterances': sum(row['role'] == 'heldout' for row in rows),
            'registered_policy': config['threshold_policy'], 'fit_metrics': fitted, 'heldout_metrics': audited,
            'failures': failures, 'checkpoint_selection_used_all_450_validation': True},
        'created_at': datetime.now(timezone.utc).isoformat(), 'limitations': config['limitations']}
    save_json(Path(output) / 'calibration.json', artifact)
    write_csv(Path(output) / 'threshold_search.csv', search)
    return artifact


def validate_frozen(config, artifact):
    actual = identity(config)
    if not artifact.get('eligible') or artifact.get('status') != 'eligible' or artifact.get('validation', {}).get('status') != 'verified':
        raise ValueError('An ineligible/failed calibration cannot be used for test or trusted decisions')
    if artifact.get('schema') != 'mdd-pronunciation-calibration-v1' or artifact.get('method') != config['method']:
        raise ValueError('Calibration schema/method mismatch')
    for key, expected in actual.items():
        if artifact.get(key) != expected:
            raise ValueError('Calibration identity mismatch: ' + key)
    if artifact.get('method_sha256') != digest(ROOT / 'src/mdd/pronunciation_assessment.py'):
        raise ValueError('Calibration scorer code changed')
    a, e = artifact.get('accept_threshold'), artifact.get('error_threshold')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, e)) or a > e:
        raise ValueError('Calibration thresholds invalid')
    return {**deepcopy(artifact), 'verified': True}


def report(config, output, plan, rows, artifact):
    output = Path(output)
    a, e = artifact.get('accept_threshold'), artifact.get('error_threshold')
    metric_rows = []
    roles = ['test'] if plan['mode'] == 'test' else ['fit', 'heldout']
    for role in roles:
        words = flatten(rows, role)
        metric_rows.append({'role': role, 'system': 'strict_greedy_single_CMU', **metrics(words, baseline=True)})
        metric_rows.append({'role': role, 'system': 'variant_posterior_candidate', **metrics(words, a, e)})
        # Actual safe behaviour: an ineligible artifact produces abstention,
        # even if its untrusted candidate thresholds look useful in a chart.
        metric_rows.append({'role': role, 'system': 'safe_verified_policy',
                            **metrics(words, a if artifact.get('eligible') else None, e if artifact.get('eligible') else None)})
    write_csv(output / 'metrics.csv', metric_rows)
    word_rows = flatten(rows)
    for word in word_rows:
        word['candidate_prediction'] = classify(word, a, e)
        word['safe_prediction'] = word['candidate_prediction'] if artifact.get('eligible') else None
    write_csv(output / 'words.csv', word_rows)
    write_csv(output / 'data_audit.csv', data_audit(config, plan))
    by_speaker = []
    for speaker in sorted({word['speaker_id'] for word in word_rows}):
        words = [word for word in word_rows if word['speaker_id'] == speaker]
        by_speaker.append({'speaker_id': speaker, **metrics(words, a, e)})
    write_csv(output / 'per_speaker.csv', by_speaker)
    summary = {'experiment_id': config['experiment_id'], 'selection_mode': plan['mode'],
        'utterances': len(rows), 'selection_sha256': plan['selection_sha256'], 'identity': plan['identity'],
        'artifact_status': artifact['status'], 'eligible': artifact['eligible'], 'validation': artifact['validation'],
        'thresholds': {'accept_threshold': a, 'error_threshold': e}, 'metrics': metric_rows,
        'metric_definitions': {'positive': 'manual word has a phone error; correct words are negative controls',
            'false_positive_rate': 'FP / (FP + TN), decided manual-correct words only',
            'false_negative_rate': 'FN / (TP + FN), decided manual-error words only',
            'false_rejection_rate': 'Same as FPR: a manual-correct word flagged as error, FP / (FP + TN)',
            'false_acceptance_rate': 'Same as FNR: a manual-error word accepted as correct, FN / (TP + FN)',
            'coverage': 'decided words / all gold-labelled words; OOV, unreliable alignment and the threshold gap abstain',
            'recall_all_errors': 'TP / all manual-error words, including abstained errors in the denominator',
            'counts': 'TP/FP/TN/FN exclude abstentions; correct_abstained/error_abstained remain visible'},
        'neural_weights_updated': False, 'llm_or_tts_calls': False, 'test_used': plan['mode'] == 'test',
        'limitations': config['limitations']}
    save_json(output / 'report.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, ('false_positive_rate', 'recall_all_errors', 'coverage')):
        view = [row for row in metric_rows if row['role'] == roles[-1]]
        ax.bar(range(len(view)), [row.get(metric) or 0. for row in view], color=['#b7c6be', '#7faaa0', '#255b4b'])
        ax.set_xticks(range(len(view)), ['Strict', 'Candidate', 'Safe'])
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace('_', ' '))
        for index, row in enumerate(view):
            if row.get(metric) is None:
                ax.text(index, .03, 'undefined', ha='center', fontsize=8, rotation=90)
    figure.suptitle(f"{plan['mode']} / {roles[-1]} — {artifact['status']} (same frozen B weights)")
    figure.tight_layout()
    figure.savefig(output / 'heldout_metrics.png', dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(9, 4))
    for gold, label, color in ((0, 'Manual correct', '#34816a'), (1, 'Manual error', '#c66954')):
        scores = [word['error_score'] for word in word_rows if word['role'] == roles[-1] and word['gold'] == gold and word['score_eligible']]
        if scores:
            ax.hist(scores, bins=20, alpha=.5, label=label, color=color)
    for threshold, label in ((a, 'Accept max'), (e, 'Error min')):
        if threshold is not None:
            ax.axvline(threshold, linestyle='--', label=label)
    ax.set_xlabel('CTC variant posterior margin error score (not a probability)')
    ax.set_ylabel('Words')
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    figure.tight_layout()
    figure.savefig(output / 'score_distributions.png', dpi=160)
    plt.close(figure)
    search_path = output / 'threshold_search.csv'
    if search_path.is_file() and search_path.stat().st_size:
        with search_path.open(encoding='utf-8') as stream:
            search = list(csv.DictReader(stream))
        defined = [row for row in search if row.get('false_positive_rate') and row.get('false_negative_rate')]
        figure, ax = plt.subplots(figsize=(6.5, 4.5))
        if defined:
            points = ax.scatter([float(row['false_positive_rate']) for row in defined],
                [float(row['false_negative_rate']) for row in defined],
                c=[float(row['coverage']) for row in defined], cmap='viridis', s=13, alpha=.65)
            figure.colorbar(points, ax=ax, label='Decision coverage')
        ax.axvline(config['threshold_policy']['max_false_positive_rate'], color='#bb7056', linestyle='--')
        ax.axhline(config['threshold_policy']['max_false_negative_rate'], color='#bb7056', linestyle='--')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False-positive rate on decided correct words',
               ylabel='False-negative rate on decided error words', title='Threshold search on FIT speakers only')
        figure.tight_layout()
        figure.savefig(output / 'threshold_tradeoff.png', dpi=160)
        plt.close(figure)
    table_keys = ['role', 'system', 'tp', 'fp', 'tn', 'fn', 'correct_abstained', 'error_abstained',
                  'false_positive_rate', 'false_negative_rate', 'recall_all_errors', 'coverage']
    table = '<table><thead><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in table_keys) + '</tr></thead><tbody>'
    for row in metric_rows:
        table += '<tr>' + ''.join('<td>' + html.escape(str(round(row[k], 4) if isinstance(row[k], float) else row[k])) + '</td>' for k in table_keys) + '</tr>'
    table += '</tbody></table>'
    document = ('<!doctype html><meta charset="utf-8"><title>Pronunciation calibration</title>'
        '<style>body{font:16px system-ui;max-width:1180px;margin:32px auto;color:#173c31;background:#f8f7f1}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #bbb}img{max-width:100%}</style>'
        '<h1>Frozen-B pronunciation calibration</h1><p>Mode: ' + html.escape(plan['mode']) + ' · Status: ' + html.escape(artifact['status']) +
        '</p><p>Score calibration only; no neural training. Candidate results are not trusted deployment decisions when eligibility fails.</p>' +
        '<img src="heldout_metrics.png"><img src="score_distributions.png">' +
        ('<img src="threshold_tradeoff.png">' if (output / 'threshold_tradeoff.png').exists() else '') +
        table + '<h2>Limits</h2><ul>' +
        ''.join('<li>' + html.escape(item) + '</li>' for item in config['limitations']) + '</ul>')
    (output / 'report.html').write_text(document, encoding='utf-8')
    return summary


def self_check():
    """Three bounded metric/selection edge cases; no files, models or labels."""
    def row(gold, score=0., eligible=True):
        return {'gold': gold, 'error_score': score, 'score_eligible': eligible, 'strict_status': 'unavailable'}
    empty = metrics([row(0, eligible=False), row(1, eligible=False)], -.5, .5)
    assert empty['coverage'] == 0 and empty['false_positive_rate'] is None and empty['false_negative_rate'] is None
    assert empty['correct_abstained'] == empty['error_abstained'] == 1
    mixed = metrics([row(0, -1), row(0, 1), row(1, 1), row(1, -1), row(1, eligible=False)], -.5, .5)
    assert all(mixed[name] == 1 for name in ('tp', 'fp', 'tn', 'fn'))
    assert mixed['false_positive_rate'] == mixed['false_negative_rate'] == .5
    assert mixed['recall_all_errors'] == 1 / 3 and mixed['coverage'] == .8
    policy = {'grid_points': 8, 'max_false_positive_rate': .1, 'max_false_negative_rate': .2,
              'min_coverage': .5, 'min_correct_decisions': 1, 'min_error_decisions': 1}
    selected, _ = fit_thresholds([row(0, -1), row(0, 0), row(0, 1)], policy)
    assert selected is None, 'A one-class calibration must not invent an eligible error threshold'
    print('PASS: 3 calibration edge cases (all-abstain denominators, confusion/coverage, one-class rejection); no inference.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--full-validation', action='store_true')
    modes.add_argument('--test', action='store_true')
    stages = parser.add_mutually_exclusive_group()
    for name in ('prepare', 'infer', 'calibrate', 'report', 'run', 'self-test'):
        stages.add_argument('--' + name, action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frozen-calibration', type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_check()
        return 0
    config = load_config(args.config)
    mode = 'test' if args.test else 'full-validation' if args.full_validation else 'pilot'
    output = args.output or rooted(config['output_root']) / mode
    if mode == 'test' and (not args.frozen_calibration or args.calibrate):
        parser.error('--test requires --frozen-calibration and cannot fit thresholds')
    frozen = validate_frozen(config, read_json(args.frozen_calibration)) if mode == 'test' else None
    plan = prepare(config, output, mode)
    write_csv(output / 'data_audit.csv', data_audit(config, plan))
    print(f"Frozen {mode}: {len(plan['records'])} utterances; {Counter(row['role'] for row in plan['records'])}")
    if args.infer or args.run:
        rows = infer(config, output, plan)
    elif args.calibrate or args.report:
        rows = load_predictions(output, plan)
    else:
        print('Prepared only; no inference or training. Selection:', output / 'selection.json')
        return 0
    if args.calibrate or args.run:
        artifact = frozen if mode == 'test' else calibrate(config, output, plan, rows)
        if mode == 'test':
            save_json(output / 'calibration.json', {**frozen, 'frozen_source_sha256': digest(args.frozen_calibration)})
    elif args.report:
        artifact = frozen if mode == 'test' else read_json(output / 'calibration.json')
    else:
        print('Inference complete. Next run --calibrate then --report. No weights updated.')
        return 0
    if args.report or args.run:
        summary = report(config, output, plan, rows, artifact)
        print('Calibration status:', summary['artifact_status'])
        print('Report:', output / 'report.html')
    else:
        print('Calibration status:', artifact['status'], '; report with --report')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
