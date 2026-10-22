"""Inspect frozen validation errors; optionally re-run four real audio files.

This writes a separate diagnostic report. It never trains, changes calibration,
uses the 900 test set, or calls Qwen/LLM/TTS. Gold labels are joined only after
audio inference. The examples are deliberately illustrative, not a new test set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import experiment_pronunciation_calibration as experiment

CACHE = ROOT / "data/generated/pronunciation-calibration-v1/full-validation"
OUTPUT = ROOT / "data/generated/pronunciation-error-qa"
CASES = (
    ("error", "HJK/arctic_a0028", "A labelled vowel substitution, with an additional false phone diagnosis."),
    ("error_deletion", "LXC/arctic_a0029", "A labelled deletion: posterior word decisions deliberately abstain on deletion/insertion candidates."),
    ("error_missed", "PNV/arctic_a0127", "A manually labelled error that greedy phoneme decoding misses."),
    ("correct", "HJK/arctic_b0047", "All words manually labelled correct, yet the decoder proposes false alarms."),
)
CLEAN_CONTROL = "HJK/arctic_a0102"


def ratio(a, b):
    return a / b if b else None


def candidate_metrics(rows, *, strict=False):
    counts = Counter(tp=0, fp=0, tn=0, fn=0, unlabelled_words=0)
    correct_utterances, flagged_correct = 0, 0
    for row in rows:
        if strict:
            flagged = {index for index, word in enumerate((row.get("strict_baseline") or {}).get("words", []))
                       if word.get("status") == "difference"}
        else:
            flagged = {event.get("word_index") for event in row["scoring"].get("raw_events", [])}
        for index, gold in enumerate(row["gold_word_errors"]):
            if gold is None:
                counts["unlabelled_words"] += 1
                continue
            counts[{(1, 1): "tp", (0, 1): "fp", (0, 0): "tn", (1, 0): "fn"}[(gold, int(index in flagged))]] += 1
        if row["gold_word_errors"] and all(gold == 0 for gold in row["gold_word_errors"]):
            correct_utterances += 1
            flagged_correct += bool(flagged)
    return {**dict(counts), "precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "false_positive_rate": ratio(counts["fp"], counts["fp"] + counts["tn"]),
            "fully_manual_correct_utterances": correct_utterances,
            "fully_manual_correct_utterances_with_candidates": flagged_correct,
            "candidate_flags_are_not_calibrated_decisions": True}


def word_report(row, annotation, *, scoring=None, strict=None):
    scoring = scoring or row["scoring"]
    strict = strict or row.get("strict_baseline") or {}
    interval_map = {word["index"]: word for word in annotation["word_intervals"]}
    result = []
    for index, word in enumerate(scoring.get("words", [])):
        interval = interval_map.get(index, {})
        result.append({"index": index, "text": word["text"], "manual_gold_error": row["gold_word_errors"][index],
            "manual_start_sec": interval.get("start_sec"), "manual_end_sec": interval.get("end_sec"),
            "manual_events": [event for event in annotation["gold_error_events"] if event.get("word_index") == index],
            "strict_first_variant_status": strict.get("words", [{}] * len(scoring["words"]))[index].get("status"),
            "candidate_events": [event for event in scoring.get("raw_events", []) if event.get("word_index") == index],
            "posterior_status": word.get("status"), "error_score": word.get("error_score"),
            "score_eligible": word.get("score_eligible"), "abstention_reason": word.get("reason"),
            "approx_ctc_start_sec": word.get("start_sec"), "approx_ctc_end_sec": word.get("end_sec")})
    return result


def fixture(speech, row, selected, source):
    return {"utterance_id": row["utterance_id"], "provenance": source,
            "audio_path": selected["audio_path"], "audio_sha256": selected["audio_sha256"],
            "gold_word_errors": row["gold_word_errors"], "speech": speech}


def cached_speech(row):
    from mdd.chat_assessment import assessment_fields
    from mdd.chat_reference import reference_resolution
    evidence = experiment.load_evidence(experiment._cache_path(CACHE, row["utterance_id"]))
    text = row["reference_text"]
    resolution = reference_resolution(ROOT, text)
    if not resolution["missing_words"]:
        resolution["status"] = "provided"
    return {"transcript": None, "reference_text": text, "reference_source": "reading_prompt",
            "reference_resolution": resolution,
            **assessment_fields(ROOT, text, evidence, confirmed=True, raw_comparison=row.get("strict_baseline"))}


def exhaustive_fit_only(rows, policy):
    """Check every fit-score partition, without changing or fitting on heldout."""
    import numpy as np
    words = experiment.flatten(rows, "fit")
    valid = [word for word in words if word["score_eligible"] and word["gold"] is not None]
    scores = np.array([word["error_score"] for word in valid])
    labels = np.array([word["gold"] for word in valid])
    values, inverse = np.unique(scores, return_inverse=True)
    totals = np.bincount(inverse, minlength=len(values))
    errors = np.bincount(inverse, weights=labels, minlength=len(values)).astype(int)
    prefix_error = np.r_[0, np.cumsum(errors)]
    prefix_correct = np.r_[0, np.cumsum(totals - errors)]
    labelled = sum(word["gold"] is not None for word in words)
    feasible = 0
    best_rate_coverage = None
    # k accepted score groups; j rejected tail starts after j groups. k<=j
    # keeps accept/error thresholds disjoint; equality permits a fully decided
    # partition using two different thresholds in the same open score gap.
    for k in range(len(values) + 1):
        js = np.arange(k, len(values) + 1)
        tn, fn = prefix_correct[k], prefix_error[k]
        fp, tp = prefix_correct[-1] - prefix_correct[js], prefix_error[-1] - prefix_error[js]
        correct_n, error_n = tn + fp, fn + tp
        with np.errstate(divide="ignore", invalid="ignore"):
            rates_ok = ((correct_n >= policy["min_correct_decisions"])
                        & (error_n >= policy["min_error_decisions"])
                        & (fp / correct_n <= policy["max_false_positive_rate"])
                        & (fn / error_n <= policy["max_false_negative_rate"]))
        coverage = (correct_n + error_n) / labelled
        feasible += int(np.count_nonzero(rates_ok & (coverage >= policy["min_coverage"])))
        if np.any(rates_ok):
            choice = np.flatnonzero(rates_ok)[np.argmax(coverage[rates_ok])]
            candidate = {"coverage": float(coverage[choice]), "tp": int(tp[choice]), "fp": int(fp[choice]),
                         "tn": int(tn), "fn": int(fn), "accepted_score_groups": k,
                         "rejected_tail_starts_after_group": int(js[choice])}
            if best_rate_coverage is None or candidate["coverage"] > best_rate_coverage["coverage"]:
                best_rate_coverage = candidate
    return {"exploratory_only": True, "heldout_used_to_select_thresholds": False,
            "fit_labelled_words": labelled, "distinct_eligible_fit_scores": len(values),
            "feasible_partitions_under_unchanged_registered_criteria": feasible,
            "maximum_coverage_satisfying_rates_and_minimum_counts": best_rate_coverage,
            "registered_policy": policy, "calibration_artifact_modified": False}


def phone_event(event):
    return f"{event.get('expected_phone') or '∅'} → {event.get('observed_phone') or '∅'}"


def render_report(report, output, real=None):
    esc = lambda value: html.escape(str(value))
    blocks = []
    for case in report["examples"]:
        source = next((item for item in (real or {}).get("examples", []) if item["utterance_id"] == case["utterance_id"]), None)
        table = []
        for word in (source or case)["words"]:
            gold = word["manual_gold_error"]
            events = "; ".join(phone_event(event) for event in word["candidate_events"]) or "No candidate"
            manual = "; ".join(phone_event(event) for event in word["manual_events"]) or "No labelled sound error"
            start, end = word["manual_start_sec"], word["manual_end_sec"]
            clip = f'<a href="{esc(case["audio_copy"])}#t={start},{end}">{start}–{end}s</a>'
            score = "—" if word["error_score"] is None else f'{word["error_score"]:.3f}'
            table.append(f'<tr><td>{esc(word["text"])}</td><td>{"Error" if gold == 1 else "Correct" if gold == 0 else "Unavailable"}</td>'
                         f'<td>{esc(manual)}</td><td>{esc(events)}</td><td>{esc(word["strict_first_variant_status"])}</td>'
                         f'<td>{score}</td><td>{esc(word["posterior_status"])}<br><small>{esc(word["abstention_reason"])}</small></td><td>{clip}</td></tr>')
        asr = f'<p><b>Fresh ASR:</b> {esc(source["transcript"])} · {source["elapsed_sec"]:.2f}s CPU ASR+B</p>' if source else ""
        blocks.append(f'<section><h2>{esc(case["utterance_id"])}</h2><p>{esc(case["purpose"])}</p><p><b>Published prompt:</b> {esc(case["reference_text"])}</p>'
                      f'<audio controls preload="none" src="{esc(case["audio_copy"])}"></audio>{asr}'
                      f'<p><small>Original source: {esc(case["audio_path"])}; SHA256 {esc(case["audio_sha256"])}</small></p>'
                      '<table><thead><tr><th>Word</th><th>Manual label</th><th>Manual sounds</th><th>Tentative decoder differences</th><th>Old strict</th><th>Score</th><th>Posterior decision</th><th>Manual interval</th></tr></thead>'
                      f'<tbody>{"".join(table)}</tbody></table></section>')
    metrics = report["validation_candidate_metrics"]["all"]
    pct = lambda value: f"{value:.2%}" if value is not None else "unavailable"
    summary = (f'450 validation utterances: tentative word candidates TP={metrics["tp"]}, FP={metrics["fp"]}, '
               f'TN={metrics["tn"]}, FN={metrics["fn"]}; precision {pct(metrics["precision"])}, recall {pct(metrics["recall"])}, '
               f'false-positive rate {pct(metrics["false_positive_rate"])}. These are diagnostic edit flags, not calibrated error decisions.')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pronunciation error QA</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;background:#fcfaf4;color:#243b32}}h1{{font-size:28px}}h2{{font-size:21px}}section{{margin-top:32px;padding-top:14px;border-top:1px solid #cbd8ce}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;text-align:left;border:1px solid #dce2d9;vertical-align:top}}th{{background:#e5eee5}}small{{color:#5d675e}}audio{{width:100%;max-width:550px}}.notice{{padding:16px;background:#fff1cf;border-radius:10px}}a{{color:#2b6c55}}</style>
<h1>Real pronunciation errors: what the current model does and misses</h1><p class="notice">Failed calibration means <b>uncertain</b>, never “all correct”. Amber candidates can include false alarms and omit real errors. Four examples were chosen after inspecting validation labels for illustration; they are not an independent accuracy estimate.</p>
<p>{esc(summary)}</p><p>Fresh real-audio re-inference: <b>{"completed on all four audio files" if real else "not run (frozen posterior cache only)"}</b>. No training, Qwen, TTS, external API, or 900-test access. Manual word intervals are annotation times; CTC intervals in JSON are approximate alignment times.</p>
<p><a href="report.json">Full cached evidence</a> · <a href="words.csv">Word comparison CSV</a> · <a href="real-audio.json">Fresh real-audio results (when run)</a></p>{"".join(blocks)}
<section><h2>Accepted pronunciation variants</h2><p>Examples below use a non-primary dictionary variant without a candidate on a manually correct word. An accepted variant does not certify the rest of the recording.</p><pre>{esc(json.dumps(report["variant_examples"], ensure_ascii=False, indent=2))}</pre></section></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-audio", action="store_true", help="Re-run CPU ASR+B on exactly four selected validation WAVs, with one cached engine.")
    parser.add_argument("--exhaustive-fit", action="store_true", help="Explore all fit-score partitions under unchanged criteria; never changes calibration.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if (not output.is_relative_to(ROOT / "data/generated") or output == CACHE.resolve()
            or output.is_relative_to(CACHE.resolve())):
        raise ValueError("Use a separate diagnostic directory within the workspace")
    output.mkdir(parents=True, exist_ok=True)
    config = experiment.load_config()
    plan = experiment.read_json(CACHE / "selection.json")
    if plan.get("mode") != "full-validation" or plan.get("test_audio_or_gold_used") is not False:
        raise ValueError("Only the frozen 450 validation cache may be inspected")
    if plan["identity"] != experiment.identity(config):
        raise ValueError("Validation identities changed")
    splits = experiment.read_json(ROOT / config["protocol_dir"] / "splits.json")
    selected_ids = {record["utterance_id"] for record in plan["records"]}
    if selected_ids != set(splits["joint_validation_ids"]) or selected_ids & set(splits["cascade_test_ids"]):
        raise ValueError("Validation membership changed")
    rows = experiment.load_predictions(CACHE, plan)
    by_id = {row["utterance_id"]: row for row in rows}
    records = {row["utterance_id"]: row for row in plan["records"]}
    annotations = experiment.selected_annotations(config, plan)
    for uid, row in by_id.items():
        if experiment.word_labels(annotations[uid]) != row["gold_word_errors"]:
            raise ValueError("Cached manual labels differ: " + uid)
    metrics = {role: candidate_metrics([row for row in rows if role == "all" or row["role"] == role])
               for role in ("all", "fit", "heldout")}
    variants = []
    for row in rows:
        flagged = {event["word_index"] for event in row["scoring"]["raw_events"]}
        for index, word in enumerate(row["scoring"]["target_words"]):
            if (word.get("dictionary_variant_index", 0) > 0 and index not in flagged
                    and row["gold_word_errors"][index] == 0):
                variants.append({"utterance_id": row["utterance_id"], "index": index, "text": word["text"],
                                 "selected_phones": word["phones"], "dictionary_variant_index": word["dictionary_variant_index"],
                                 "strict_first_variant_status": row["strict_baseline"]["words"][index]["status"]})
                if len(variants) >= 5:
                    break
        if len(variants) >= 5:
            break
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "schema": "mdd-pronunciation-error-qa-v1",
        "source": str(CACHE.relative_to(ROOT)), "split": "450 validation only", "identity": plan["identity"],
        "calibration": experiment.read_json(CACHE / "calibration.json"),
        "selection_note": "Four deliberately illustrative, post-hoc examples; not a new accuracy benchmark.",
        "validation_candidate_metrics": metrics, "strict_first_variant_metrics": candidate_metrics(rows, strict=True),
        "variant_examples": variants, "examples": [],
        "limitations": ["Candidate TP/FP are word-level; a true-positive word may still contain a false phone diagnosis.",
                        "No calibrated decision is made while the registered full-validation artifact is ineligible.",
                        "Heldout PNV was held out threshold fitting, but not earlier checkpoint selection.",
                        "No native/TTS accuracy claim follows from these non-native validation examples."]}
    fixtures = {}
    for name, uid, purpose in CASES:
        row, selected = by_id[uid], records[uid]
        source = ROOT / selected["audio_path"]
        if experiment.digest(source) != selected["audio_sha256"]:
            raise ValueError("Selected source audio changed: " + uid)
        relative = "audio/" + uid.replace("/", "-") + ".wav"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        report["examples"].append({"case": name, "purpose": purpose, **selected, "audio_copy": relative,
                                   "reference_text": row["reference_text"], "words": word_report(row, annotations[uid])})
        fixtures[name] = fixture(cached_speech(row), row, selected, "cached validation posterior; no fresh ASR")
    clean = by_id[CLEAN_CONTROL]
    fixtures["no_candidates"] = fixture(cached_speech(clean), clean, records[CLEAN_CONTROL], "cached validation posterior only; not an extra real-audio inference")
    if args.exhaustive_fit:
        report["exhaustive_fit_only"] = exhaustive_fit_only(rows, config["threshold_policy"])
    experiment.save_json(output / "report.json", report)
    experiment.write_csv(output / "words.csv", [{"utterance_id": case["utterance_id"], **word}
                         for case in report["examples"] for word in case["words"]])
    experiment.save_json(output / "diagnostic-cases.json", fixtures)
    render_report(report, output)
    print(json.dumps({"validation_candidates": metrics, "report": str(output / "report.html"),
                      "exhaustive_fit_only": report.get("exhaustive_fit_only")}, ensure_ascii=False), flush=True)
    if not args.real_audio:
        return
    from scripts.local_speech_engine import LocalSpeechEngine
    engine = LocalSpeechEngine()
    actual = {"created_at": datetime.now(timezone.utc).isoformat(), "schema": "mdd-real-audio-error-qa-v1",
              "mode": "fresh CPU ASR+B only", "no_training": True, "no_qwen_or_api_or_tts": True,
              "test_900_used": False, "identity": plan["identity"], "examples": []}
    for index, (name, uid, _) in enumerate(CASES, 1):
        row, selected = by_id[uid], records[uid]
        print(f"[{index}/{len(CASES)}] Real audio: {uid}", flush=True)
        audio = (ROOT / selected["audio_path"]).read_bytes()
        if hashlib.sha256(audio).hexdigest() != selected["audio_sha256"]:
            raise ValueError("Source audio changed before inference: " + uid)
        started = time.monotonic()
        # No annotation or perceived phone enters the engine. The reference is
        # the published reading prompt, not a transcript manufactured from gold.
        speech = engine.analyze(audio, reference_text=row["reference_text"])
        speech = {key: value for key, value in speech.items() if not key.startswith("_") and key != "reference"}
        scoring = speech["assessment"]
        words = word_report(row, annotations[uid], scoring=scoring, strict=speech.get("raw_comparison"))
        flagged = {word["index"] for word in speech["sound_candidates"]["words"]}
        evaluation_row = {**row, "scoring": {"raw_events": scoring.get("raw_events", [])}}
        case = {"case": name, "utterance_id": uid, "audio_path": selected["audio_path"],
                "audio_sha256": selected["audio_sha256"], "transcript": speech["transcript"],
                "reference_text": speech["reference_text"], "elapsed_sec": round(time.monotonic() - started, 3),
                "candidate_words": [row["reference_text"].split()[i] for i in sorted(flagged)],
                "candidate_word_metrics": candidate_metrics([evaluation_row]), "words": words, "speech": speech}
        actual["examples"].append(case)
        fixtures[name] = fixture(speech, row, selected, "fresh real-audio CPU ASR+B inference")
        experiment.save_json(output / "real-audio.json", actual)
        experiment.save_json(output / "diagnostic-cases.json", fixtures)
        print(json.dumps({key: case[key] for key in ("utterance_id", "transcript", "candidate_words", "elapsed_sec", "candidate_word_metrics")}, ensure_ascii=False), flush=True)
    render_report(report, output, actual)
    print("Fresh real-audio report: " + str(output / "report.html"), flush=True)


if __name__ == "__main__":
    main()
