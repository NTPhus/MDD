"""Validation-only selection of two frozen posterior policies; no neural loading.

The only acoustic inputs are existing B posterior caches. Fit HJK/LXC selects
scalar cutoffs; PNV audits them; an already observed 120-case test is scored
only after a lock exists. No thresholds or weights are changed using test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiment_pronunciation_calibration import load_evidence, _cache_path
from scripts.evaluate_pronunciation_models import variant_review

OUT = ROOT / "data/generated/pronunciation-improvement-20261024/policies"
VALIDATION = ROOT / "data/generated/pronunciation-calibration-v1/full-validation"
TEST = ROOT / "data/generated/model-comparison-20261024/pronunciation"
TEST_CACHE = ROOT / "data/generated/pronunciation-improvement-20261024/b-test-posteriors"
METHODS = ("dictionary-variant-review", "error-cutoff-f05", "selective-two-threshold")
SCHEMA = "frozen-validation-word-policies-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def selected_annotations(ids):
    """Parse only selected rows; never load test labels while fitting."""
    pattern = re.compile(r'"utterance_id"\s*:\s*"([^"\\]+)"')
    result = {}
    with (ROOT / "data/processed/l2arctic-plus-cascade-v1/annotations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if match and match[1] in ids:
                result[match[1]] = json.loads(line)
    if set(result) != set(ids):
        raise ValueError("Missing selected expert annotations")
    return result


def plan_value():
    selection = read(VALIDATION / "selection.json")
    config = read(ROOT / "configs/pronunciation_calibration.json")
    if selection["mode"] != "full-validation" or selection["test_audio_or_gold_used"] is not False:
        raise ValueError("Expected the frozen validation-only selection")
    if set(r["speaker_id"] for r in selection["records"] if r["role"] == "fit") != {"HJK", "LXC"}:
        raise ValueError("Fit speaker partition differs")
    if set(r["speaker_id"] for r in selection["records"] if r["role"] == "heldout") != {"PNV"}:
        raise ValueError("Threshold-heldout speaker partition differs")
    files = [ROOT / "configs/pronunciation_calibration.json", VALIDATION / "selection.json",
             VALIDATION / "predictions.jsonl", VALIDATION / "inference.json",
             ROOT / "src/mdd/pronunciation_assessment.py", ROOT / "src/mdd/phone_sequence_assessment.py",
             ROOT / "src/mdd/demo_reference.py", ROOT / "scripts/evaluate_pronunciation_models.py",
             ROOT / "scripts/evaluate_crottc.py", Path(__file__)]
    pinned_inputs = {
        "annotations_sha256": ROOT / config["annotations"],
        "dictionary_sha256": ROOT / config["dictionary"],
        "protocol_sha256": ROOT / config["protocol_dir"] / "protocol.json",
        "splits_sha256": ROOT / config["protocol_dir"] / "splits.json",
        "checkpoint_sha256": ROOT / config["checkpoint"] / "model.safetensors",
        "vocabulary_sha256": ROOT / config["checkpoint"] / "vocab.json",
    }
    for name, path in pinned_inputs.items():
        if sha(path) != selection["identity"][name] or config[name] != selection["identity"][name]:
            raise ValueError("Pinned validation source identity differs: " + name)
    files.extend(pinned_inputs.values())
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    inference = read(VALIDATION / "inference.json")
    if inference["method_sha256"] != sha(ROOT / "src/mdd/pronunciation_assessment.py"):
        raise ValueError("Saved validation feature method differs; rescore caches in a new experiment")
    cache = {}
    for record in selection["records"]:
        path = _cache_path(VALIDATION, record["utterance_id"])
        cache[record["utterance_id"]] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                                         "audio_sha256": record["audio_sha256"]}
    return {"schema": SCHEMA, "method": "ctc_variant_posterior_margin_v1", "source_sha256": source_hashes,
            "validation_selection": selection, "validation_cache": cache,
            "model_identity": selection["identity"],
            "selection_restrictions": {"fit_speakers": ["HJK", "LXC"], "threshold_heldout_speakers": ["PNV"],
                                       "test_threshold_tuning": False, "neural_training": False,
                                       "neural_inference": False, "grid_points": 31},
            "families": {
                "error-cutoff-f05": {"prediction": "Flag only eligible scores >= cutoff; all other words unknown, never accepted as correct.",
                    "constraint_FPR_all_expert_correct_max": .1, "constraint_recall_all_expert_errors_min": .4,
                    "min_flagged_words": 20, "objective": "max F0.5 from flag precision and all-error recall; tie by recall, lower FPR, then higher threshold"},
                "selective-two-threshold": {"prediction": "Eligible score <= accept is acceptance candidate; >= error is difference candidate; others unknown.",
                    "max_FPR_decided": .1, "max_FNR_decided": .2, "min_coverage": .25,
                    "min_correct_decisions": 10, "min_error_decisions": 5,
                    "prior_coverage_gate": .5, "coverage_gate_explicitly_relaxed": True,
                    "objective": "maximize labelled-word decision coverage; tie by decided F1, lower FPR/FNR, then narrower band"}},
            "baseline": "Identical shared full39 dictionary-variant sequence reviewer, explicit valid expert word_index gold. No-hint is not certified correct pronunciation.",
            "gold_definition": "Every aligned word_interval is a labelled word; positive iff an expert event has its explicit valid word_index. Unmapped expert insertions counted separately, not guessed. Gold never enters acoustic scoring.",
            "limitations": [
                "The B checkpoint was selected using all 450 validation recordings; PNV is held out from threshold fitting only.",
                "120 test recordings and earlier error anatomy have already been observed. This is an exploratory known-test comparison, not a fresh blind benchmark.",
                "The selective 25% coverage gate is explicitly lower than the previous 50% gate; meeting it does not satisfy the old policy.",
                "A feature threshold does not change the B encoder, phone decoder or original-inventory recognition F1.",
                "Word flag metrics include all gold errors/correct words; abstention must not be interpreted as correct pronunciation.",
                "This existing posterior-margin feature is not the published CTC-GOP formula. No neural fine-tuning or production deployment occurs."]}


def verify_plan(plan):
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for cache in plan["validation_cache"].values():
        if sha(ROOT / cache["path"]) != cache["sha256"]:
            raise ValueError("Frozen validation cache changed")


def word_features(annotation, scored, evidence, *, group, evidence_sha):
    """Use the same explicit-word gold cohort as the fixed shared-review test."""
    words = annotation["word_intervals"]
    source_tokens = annotation["prompt_text"].split()
    norm = lambda value: ''.join(c for c in str(value).casefold() if c.isalnum() or c in "'’")
    aligned = len(words) == len(source_tokens) and all(norm(w["text"]) == norm(t) for w, t in zip(words, source_tokens))
    review = variant_review(ROOT, annotation["prompt_text"], [x["phone"] for x in evidence["phonemes"]], annotation)
    hints = set(review.get("hint_word_indices", []))
    gold = {e["word_index"] for e in annotation["gold_error_events"] if type(e.get("word_index")) is int and 0 <= e["word_index"] < len(words)}
    score_words = scored.get("words", [])
    output = []
    for index, word in enumerate(words):
        value = score_words[index] if aligned and len(score_words) == len(words) else {}
        score = value.get("error_score")
        eligible = value.get("score_eligible") is True and type(score) in (int, float) and math.isfinite(score)
        output.append({"utterance_id": annotation["utterance_id"], "speaker_id": annotation["speaker_id"], "group": group,
                       "index": index, "text": word["text"], "gold": int(index in gold),
                       "score": score if type(score) in (int, float) and math.isfinite(score) else None,
                       "score_eligible": eligible, "reason": value.get("reason", "reference_word_alignment_unavailable"),
                       "baseline_prediction": int(index in hints) if review["status"] == "available" else None,
                       "baseline_reason": review.get("reason", "available"), "evidence_sha256": evidence_sha})
    return output


def validation_features(plan):
    annotations = selected_annotations(set(plan["validation_cache"]))
    previous = {row["utterance_id"]: row for row in read_lines(VALIDATION / "predictions.jsonl")}
    output = []
    identity = plan["model_identity"]
    for record in plan["validation_selection"]["records"]:
        uid = record["utterance_id"]
        cache = plan["validation_cache"][uid]
        evidence = load_evidence(ROOT / cache["path"])
        old = previous[uid]
        if old["source_audio_sha256"] != cache["audio_sha256"] or evidence["source_audio_sha256"] != cache["audio_sha256"]:
            raise ValueError("Validation audio identity differs")
        if any(evidence.get(k) != identity[k] for k in ("checkpoint_sha256", "vocabulary_sha256")):
            raise ValueError("Validation model identity differs")
        output.extend(word_features(annotations[uid], old["scoring"], evidence, group=record["role"], evidence_sha=cache["sha256"]))
    return output


def predict(row, family, thresholds):
    if family == "dictionary-variant-review":
        return row["baseline_prediction"]
    if not row["score_eligible"] or thresholds is None:
        return None
    score = row["score"]
    if family == "error-cutoff-f05":
        return 1 if score >= thresholds["error_threshold"] else None
    if score >= thresholds["error_threshold"]:
        return 1
    return 0 if score <= thresholds["accept_threshold"] else None


def metrics(rows, family, thresholds=None):
    counts = Counter(total_words=len(rows), tp=0, fp=0, tn=0, fn=0, gold_error=0, gold_correct=0,
                     abstained_errors=0, abstained_correct=0, score_eligible_words=0, dictionary_unavailable_words=0)
    for row in rows:
        gold = row["gold"]
        counts["gold_error" if gold else "gold_correct"] += 1
        counts["score_eligible_words"] += int(row["score_eligible"])
        counts["dictionary_unavailable_words"] += int(row["reason"] == "reference_contains_unknown_words")
        chosen = predict(row, family, thresholds)
        if chosen is None:
            counts["abstained_errors" if gold else "abstained_correct"] += 1
        else:
            counts[{(1, 1): "tp", (0, 1): "fp", (0, 0): "tn", (1, 0): "fn"}[(gold, chosen)]] += 1
    divide = lambda n, d: n / d if d else None
    c = dict(counts)
    c["score_ineligible_reasons"] = dict(Counter(row["reason"] for row in rows if not row["score_eligible"]))
    c["baseline_unavailable_reasons"] = dict(Counter(row.get("baseline_reason", "unavailable") for row in rows if row["baseline_prediction"] is None))
    decided = sum(c[k] for k in ("tp", "fp", "tn", "fn"))
    fn_all, tn_all = c["gold_error"] - c["tp"], c["gold_correct"] - c["fp"]
    c.update(decided_words=decided, flagged_words=c["tp"] + c["fp"],
             flag_all_tp=c["tp"], flag_all_fp=c["fp"], flag_all_fn=fn_all, flag_all_tn=tn_all,
             precision=divide(c["tp"], c["tp"] + c["fp"]),
             recall_all_errors=divide(c["tp"], c["gold_error"]), false_positive_all_correct=divide(c["fp"], c["gold_correct"]),
             recall_decided=divide(c["tp"], c["tp"] + c["fn"]),
             false_positive_rate_decided=divide(c["fp"], c["fp"] + c["tn"]),
             false_negative_rate_decided=divide(c["fn"], c["tp"] + c["fn"]),
             f1_decided=divide(2*c["tp"], 2*c["tp"] + c["fp"] + c["fn"]),
             f05_all_errors=divide(1.25*c["tp"], 1.25*c["tp"] + .25*fn_all + c["fp"]),
             coverage=divide(decided, len(rows)), correct_coverage=divide(c["tn"] + c["fp"], c["gold_correct"]),
             error_coverage=divide(c["tp"] + c["fn"], c["gold_error"]),
             decision_error_risk=divide(c["fp"] + c["fn"], decided))
    return c


def failures(value, family, policy, *, selected=True):
    if not selected:
        return ["no_threshold_meets_fit_constraints"]
    result = []
    if family == "error-cutoff-f05":
        for key, maximum in (("false_positive_all_correct", policy["constraint_FPR_all_expert_correct_max"]),):
            if value[key] is None or value[key] > maximum: result.append(key)
        if value["recall_all_errors"] is None or value["recall_all_errors"] < policy["constraint_recall_all_expert_errors_min"]: result.append("recall_all_errors")
        if value["flagged_words"] < policy["min_flagged_words"]: result.append("too_few_flagged_words")
    else:
        for key, bound in (("false_positive_rate_decided", "max_FPR_decided"), ("false_negative_rate_decided", "max_FNR_decided")):
            if value[key] is None or value[key] > policy[bound]: result.append(key)
        if value["coverage"] is None or value["coverage"] < policy["min_coverage"]: result.append("coverage")
        if value["tp"] + value["fn"] < policy["min_error_decisions"]: result.append("too_few_error_decisions")
        if value["tn"] + value["fp"] < policy["min_correct_decisions"]: result.append("too_few_correct_decisions")
    return result


def thresholds_grid(fit, points=31):
    scores = sorted({row["score"] for row in fit if row["score_eligible"]})
    if not scores: return []
    values = sorted({scores[int(i*(len(scores)-1)/max(points-1, 1))] for i in range(points)})
    epsilon = max(1e-6, (values[-1]-values[0])*1e-6)
    return [values[0]-epsilon, *[(a+b)/2 for a,b in zip(values, values[1:])], values[-1]+epsilon]


def select_policies(fit, plan):
    if not fit or any(row["group"] != "fit" or row["speaker_id"] not in {"HJK", "LXC"} for row in fit):
        raise ValueError("Only HJK/LXC fit rows may select thresholds")
    grid = thresholds_grid(fit, plan["selection_restrictions"]["grid_points"])
    choices, search = {}, []
    for family in METHODS[1:]:
        candidates = []
        pairs = ((None, error) for error in grid) if family == "error-cutoff-f05" else ((accept,error) for accept in grid for error in grid if accept < error)
        for accept, error in pairs:
            threshold = {"error_threshold": error}
            if accept is not None: threshold["accept_threshold"] = accept
            value = metrics(fit, family, threshold)
            reasons = failures(value, family, plan["families"][family])
            search.append({"family": family, **threshold, **value, "satisfies_fit_policy": not reasons, "failures": '|'.join(reasons)})
            if not reasons:
                rank = ((value["f05_all_errors"], value["recall_all_errors"], -value["false_positive_all_correct"], error)
                        if family == "error-cutoff-f05" else
                        (value["coverage"], value["f1_decided"] or 0., -value["false_positive_rate_decided"], -value["false_negative_rate_decided"], -(error-accept)))
                candidates.append((rank, threshold, value))
        selected = max(candidates, key=lambda item:item[0]) if candidates else None
        choices[family] = {"thresholds": selected[1] if selected else None,
                           "fit_metrics": selected[2] if selected else metrics(fit, family),
                           "fit_eligible": selected is not None,
                           "fit_failures": [] if selected else ["no_threshold_meets_fit_constraints"]}
    return choices, search


def grouped_results(rows, locked, plan):
    output = {}
    for group in sorted({row["group"] for row in rows}):
        chosen = [row for row in rows if row["group"] == group]
        output[group] = {"cases": len({row["utterance_id"] for row in chosen}), "words": len(chosen), "policies": {}}
        for family in METHODS:
            threshold = locked["selected"].get(family, {}).get("thresholds")
            output[group]["policies"][family] = metrics(chosen, family, threshold)
    eligibility = {}
    for family in METHODS[1:]:
        fit = locked["selected"][family]
        audit = output["heldout"]["policies"][family]
        failed = failures(audit, family, plan["families"][family], selected=fit["fit_eligible"])
        eligibility[family] = {"fit_pass": fit["fit_eligible"], "heldout_pass": not failed,
                               "eligible": fit["fit_eligible"] and not failed,
                               "failures": failed, "test_used_to_decide_eligibility": False}
    speakers = {}
    for speaker in sorted({r["speaker_id"] for r in rows}):
        chosen = [r for r in rows if r["speaker_id"] == speaker]
        speakers[speaker] = {"group": chosen[0]["group"], "cases": len({r["utterance_id"] for r in chosen}),
                             "policies": {family: metrics(chosen, family, locked["selected"].get(family, {}).get("thresholds")) for family in METHODS}}
    return output, eligibility, speakers


def save_results(output, rows, plan, locked, *, status):
    groups, eligible, speakers = grouped_results(rows, locked, plan)
    paths = [output / "validation_features.jsonl"]
    if (output / "test_features.jsonl").is_file(): paths.append(output / "test_features.jsonl")
    result = {"schema": SCHEMA, "status": status, "plan_sha256": sha(output / "plan.json"),
              "locked_policy_sha256": sha(output / "locked_policy.json"),
              "feature_file_sha256": {p.name: sha(p) for p in paths}, "groups": groups,
              "per_speaker": speakers, "eligibility": eligible, "selected": locked["selected"],
              "neural_inference": False, "neural_training": False, "production_changed": False,
              "definitions": {"decided_matrix": "tp/fp/tn/fn include only declared decisions; abstained errors/correct remain separate. For error-only policy, tn=fn=0 because it never declares acceptance.",
                  "flag_all_matrix": "flag_all_fn=gold_error-tp, flag_all_tn=gold_correct-fp; these unflagged words are not claimed correctly pronounced.",
                  "error_only_selection": "FPR uses all expert-correct words; recall/F0.5 include all expert errors, including unavailable scores.",
                  "selective_selection": "FPR/FNR constraints condition on decided words; overall/correct/error coverage and recall_all_errors must be shown alongside them.",
                  "gold": plan["gold_definition"], "baseline": plan["baseline"]},
              "limitations": plan["limitations"]}
    write(output / "results.json", result)
    word_rows = []
    for row in rows:
        word_rows.append({**row, **{family: predict(row, family, locked["selected"].get(family, {}).get("thresholds")) for family in METHODS}})
    write_csv(output / "words.csv", word_rows)
    return result


def fit(output=OUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = plan_value()
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan: raise ValueError("Preregistered plan differs; do not overwrite")
    else:
        write(output / "plan.json", plan)
    verify_plan(plan)
    rows = validation_features(plan)
    save_lines(output / "validation_features.jsonl", rows)
    selected, search = select_policies([row for row in rows if row["group"] == "fit"], plan)
    locked = {"schema": SCHEMA, "status": "locked", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha(output / "plan.json"), "validation_features_sha256": sha(output / "validation_features.jsonl"),
              "fit_speakers": ["HJK", "LXC"], "selected": selected, "test_features_read_before_lock": False,
              "test_threshold_tuning": False, "heldout_threshold_tuning": False}
    path = output / "locked_policy.json"
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k != "locked_at_utc"} != {k:v for k,v in locked.items() if k != "locked_at_utc"}:
            raise ValueError("Locked thresholds differ; cannot overwrite")
        locked = old
    else:
        write(path, locked)
    write_csv(output / "threshold_search.csv", search)
    return save_results(output, rows, plan, locked, status="awaiting_test_cache")


def test(output=OUT, cache_directory=TEST_CACHE):
    output, cache_directory = Path(output).resolve(), Path(cache_directory).resolve()
    plan, locked = read(output / "plan.json"), read(output / "locked_policy.json")
    if locked["status"] != "locked" or locked["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("A frozen fit-only policy lock is required before reading test evidence")
    if locked["validation_features_sha256"] != sha(output / "validation_features.jsonl"):
        raise ValueError("Validation features changed after the lock")
    verify_plan(plan)
    # These are the first test feature/annotation reads in this program.
    manifest = read(TEST / "manifest.json")
    cache_manifest = read(cache_directory / "manifest.json")
    if cache_manifest.get("status") != "completed": raise ValueError("Wait for all 120 B posterior caches")
    cases = [case for case in manifest["cases"] if case["group"] == "expert_test"]
    cache_cases = {case["case_id"]: case for case in cache_manifest["cases"]}
    if set(cache_cases) != {case["id"] for case in cases} or len(cache_cases) != len(cache_manifest["cases"]):
        raise ValueError("Test posterior manifest must cover the exact fixed expert cases")
    if cache_manifest.get("source_comparison_manifest_sha256") not in {sha(TEST / "manifest.json"), object_sha(manifest)}:
        raise ValueError("Test posterior source comparison manifest differs")
    if any(cache_manifest.get(key) != plan["model_identity"][key] for key in ("checkpoint_sha256", "vocabulary_sha256")):
        raise ValueError("Test posterior manifest B identity differs")
    rows = []
    source_hashes = {}
    from mdd.pronunciation_assessment import score_words
    for case in cases:
        annotation = case["annotation"]
        uid = annotation["utterance_id"]
        cache = _cache_path(cache_directory, uid)
        if not cache.is_file(): raise ValueError("Missing test cache " + uid)
        evidence_sha = sha(cache)
        expected = cache_cases[case["id"]]
        if ((cache_directory / expected["path"]).resolve() != cache.resolve()
                or expected["sha256"] != evidence_sha or expected["audio_sha256"] != case["audio_sha256"]
                or expected["utterance_id"] != uid):
            raise ValueError("Test posterior file identity differs")
        evidence = load_evidence(cache)
        if evidence.get("source_audio_sha256") != case["audio_sha256"]:
            raise ValueError("Test posterior audio differs from the locked cohort")
        if any(evidence.get(key) != plan["model_identity"][key] for key in ("checkpoint_sha256", "vocabulary_sha256")):
            raise ValueError("Test posterior model differs from validation B")
        scored = score_words(ROOT, case["reference_text"], evidence)
        rows.extend(word_features(annotation, scored, evidence, group="test", evidence_sha=evidence_sha))
        source_hashes[case["id"]] = {"path": str(cache.relative_to(ROOT)), "sha256": evidence_sha}
    save_lines(output / "test_features.jsonl", rows)
    write(output / "test_cache_identity.json", {"cache_manifest_sha256": sha(cache_directory / "manifest.json"),
         "comparison_manifest_sha256": sha(TEST / "manifest.json"), "lock_sha256_before_test": sha(output / "locked_policy.json"),
         "cases": source_hashes, "test_threshold_tuning": False})
    rows = read_lines(output / "validation_features.jsonl") + rows
    return save_results(output, rows, plan, locked, status="completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "test"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--test-cache", type=Path, default=TEST_CACHE)
    args = parser.parse_args()
    result = fit(args.output) if args.action == "fit" else test(args.output, args.test_cache)
    print(json.dumps({"status": result["status"], "selected": result["selected"], "eligibility": result["eligibility"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
