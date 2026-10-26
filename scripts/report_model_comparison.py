"""Render completed, frozen model comparisons without inference or network access.

Loads only saved JSON/JSONL evidence. Missing or incomplete matched arms are a
hard error, never a zero score. Native output contracts, text diagnostics and
pronunciation inventories remain distinct throughout the report.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mdd.metrics import aggregate_scores

DEFAULT = ROOT / "data/generated/model-comparison-20261024"
GRAMMAR = ("qwen-base", "qwen-grammar", "coedit-large")
PHONEME = ("cascade-B", "timit-xlsr-300m", "crottc-if")
TASKS = ("grammar_corrected", "natural")
PANELS = ("research_original_inventory", "research_full39", "research_common37")
NAMES = {"qwen-base": "Qwen3-0.6B gốc", "qwen-grammar": "Qwen3-0.6B + Grammar LoRA",
         "coedit-large": "CoEdIT-large", "cascade-B": "wav2vec2 B local",
         "timit-xlsr-300m": "XLS-R 300M · TIMIT", "crottc-if": "CROTTC-IF"}
TASK_NAMES = {"grammar_corrected": "Sửa ngữ pháp", "natural": "Diễn đạt tự nhiên"}
COLORS = ("#a6a0b2", "#7853bd", "#c490dd")
SOURCES = (
    ("CoEdIT · bài báo và model chính thức", "https://aclanthology.org/2023.findings-emnlp.350/"),
    ("CoEdIT-large · Hugging Face", "https://huggingface.co/grammarly/coedit-large"),
    ("Qwen3 · model chính thức", "https://huggingface.co/Qwen/Qwen3-0.6B"),
    ("JFLEG · bài báo EACL 2017", "https://aclanthology.org/E17-2037/"),
    ("JFLEG · dữ liệu và GLEU của tác giả", "https://github.com/keisks/jfleg"),
    ("W&I+LOCNESS · BEA 2019", "https://aclanthology.org/W19-4406/"),
    ("XLS-R TIMIT · model card của tác giả", "https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-timit-phoneme"),
    ("XLS-R · bài báo", "https://arxiv.org/abs/2111.09296"),
    ("CROTTC-IF · bài báo", "https://arxiv.org/abs/2604.22133"),
    ("CROTTC-IF · checkpoint chính thức", "https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic"),
    ("L2-ARCTIC · nguồn corpus", "https://psi.engr.tamu.edu/l2-arctic-corpus/"),
    ("ERRANT · cách đánh giá GEC theo edit", "https://aclanthology.org/P17-1074/"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_sha(value, *, compact=False):
    kwargs = {"separators": (",", ":")} if compact else {}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, **kwargs).encode()).hexdigest()


def close(a, b):
    return a is None and b is None or isinstance(a, (float, int)) and isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def load_bundle(directory=DEFAULT):
    """Reject partial arms and changed identities; never load a neural model."""
    directory = Path(directory).resolve()
    gpath, ppath = directory / "grammar", directory / "pronunciation"
    gm, gs = read(gpath / "manifest.json"), read(gpath / "summary.json")
    if gs.get("manifest_sha256") != digest(gpath / "manifest.json"):
        raise ValueError("Grammar summary manifest identity differs")
    wanted = [r["id"] for r in gm["rows"]]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("Empty or duplicate grammar cases")
    observations, hashes = {}, {}
    for arm in GRAMMAR:
        metadata = read(gpath / arm / "metadata.json")
        path = gpath / arm / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if metadata.get("status") != "completed" or metadata.get("completed_rows") != len(wanted):
            raise ValueError("Wait for completed grammar arm: " + arm)
        if metadata.get("manifest_sha256") != digest(gpath / "manifest.json") or metadata.get("predictions_sha256") != digest(path):
            raise ValueError("Grammar prediction identity differs: " + arm)
        if [r["id"] for r in rows] != wanted:
            raise ValueError("Grammar arms must cover the same ordered cases: " + arm)
        for source, row in zip(gm["rows"], rows):
            if row.get("source_row_sha256") != object_sha(source) or any(row.get(k) != v for k, v in source.items()):
                raise ValueError("Grammar source row differs: " + row["id"])
        if gs["models"][arm]["metadata"] != metadata:
            raise ValueError("Grammar summary metadata differs: " + arm)
        for task in TASKS:
            chosen = [r for r in rows if r["cohort"] == "published_heldout" and r["task"] == task]
            summary = gs["models"][arm]["tasks"][task]
            available = sum(r["prediction"] is not None for r in chosen)
            if summary["rows"] != len(chosen) or not close(summary["available_rate"], available / len(chosen)):
                raise ValueError("Grammar availability denominator differs")
            if summary["quality_on_valid_only"] is not None and summary["quality_on_valid_only"]["rows"] != available:
                raise ValueError("Grammar valid-only quality denominator differs")
            if summary["quality_with_explicit_source_fallback"]["rows"] != len(chosen):
                raise ValueError("Grammar all-case denominator differs")
        observations[arm] = rows
        hashes[str(path.relative_to(directory))] = digest(path)
        hashes[str((gpath / arm / "metadata.json").relative_to(directory))] = digest(gpath / arm / "metadata.json")
    pm, pr = read(ppath / "manifest.json"), read(ppath / "results.json")
    if pr.get("status") != "completed":
        raise ValueError("Wait for every pronunciation model/audio to complete")
    if pr.get("manifest_sha256") != object_sha(pm, compact=True):
        raise ValueError("Pronunciation manifest identity differs")
    ids = {r["id"] for r in pm["cases"]}
    if len(ids) != len(pm["cases"]) or not ids:
        raise ValueError("Empty or duplicate pronunciation cases")
    expert = [r for r in pm["cases"] if r["group"] == "expert_test"]
    for arm in PHONEME:
        predictions = pr["predictions"].get(arm, {})
        if set(predictions) != ids:
            raise ValueError("Pronunciation arm is incomplete: " + arm)
        for case in pm["cases"]:
            row = predictions[case["id"]]
            if row.get("status") != "completed" or row.get("audio_sha256") != case["audio_sha256"]:
                raise ValueError("Pronunciation case incomplete or changed: " + case["id"])
            if row.get("reference_supplied_to_model") is not False:
                raise ValueError("Unexpected reference-conditioned pronunciation inference")
            if case["group"] == "user_development" and row.get("expert_metrics") is not None:
                raise ValueError("User development cases must not have invented expert scores")
        for panel in PANELS:
            actual = aggregate_scores(predictions[c["id"]][panel] for c in expert)
            if actual != pr["summary"]["models"][arm][panel]:
                raise ValueError("Saved pronunciation aggregate differs from counters: " + arm + ":" + panel)
    if pr["summary"]["paired"]["completed_expert_cases"] != len(expert):
        raise ValueError("Pronunciation matched subset is incomplete")
    audit = read(gpath / "delivery_audit.json") if (gpath / "delivery_audit.json").is_file() else None
    if audit:
        if audit.get("manifest_sha256") != digest(gpath / "manifest.json"):
            raise ValueError("Delivery audit manifest differs")
        for arm in GRAMMAR:
            if audit["models"][arm]["predictions_sha256"] != hashes[str((gpath / arm / "predictions.jsonl").relative_to(directory))]:
                raise ValueError("Delivery audit prediction identity differs")
            if [r["id"] for r in audit["models"][arm]["rows"]] != wanted:
                raise ValueError("Delivery audit case order differs")
    content = read(gpath / "content_audit.json") if (gpath / "content_audit.json").is_file() else None
    if content:
        if content.get("manifest_sha256") != digest(gpath / "manifest.json") or content.get("no_generation") is not True:
            raise ValueError("Post-hoc content audit identity differs")
        if content.get("posthoc_defined_after_initial_output_inspection") is not True:
            raise ValueError("Post-hoc content scope must be declared")
        for arm in GRAMMAR:
            if content["models"][arm]["predictions_sha256"] != digest(gpath / arm / "predictions.jsonl"):
                raise ValueError("Post-hoc content prediction identity differs")
            if [r["id"] for r in content["models"][arm]["rows"]] != wanted:
                raise ValueError("Post-hoc content case order differs")
    notes_path = ppath / "execution_notes.json"
    execution_notes = read(notes_path) if notes_path.is_file() else None
    if execution_notes and (execution_notes.get("manifest_sha256") != pr["manifest_sha256"]
                            or execution_notes.get("results_sha256") != digest(ppath / "results.json")):
        raise ValueError("Pronunciation execution notes identity differs")
    for path in (gpath / "manifest.json", gpath / "summary.json", ppath / "manifest.json", ppath / "results.json", gpath / "delivery_audit.json", gpath / "content_audit.json", notes_path):
        if path.is_file():
            hashes[str(path.relative_to(directory))] = digest(path)
    return {"directory": directory, "grammar_manifest": gm, "grammar_summary": gs,
            "grammar_predictions": observations, "pronunciation_manifest": pm,
            "pronunciation_results": pr, "delivery_audit": audit, "content_audit": content,
            "pronunciation_execution_notes": execution_notes, "input_sha256": hashes}


def grammar_rows(bundle):
    output = []
    summary = bundle["grammar_summary"]
    for task in TASKS:
        baseline = summary["copy_baseline"][task]
        output.append({"task": task, "model": "copy-source", "rows": baseline["rows"], "native_available_rate": 1.,
                       "delivery_chrf": baseline["chrf"], "delivery_gleu": baseline["gleu"]["score"],
                       "delivery_exact_reference_rate": baseline["exact_reference_rate"], "valid_only_rows": baseline["rows"],
                       "valid_only_chrf": baseline["chrf"], "valid_only_gleu": baseline["gleu"]["score"],
                       "unchanged_controls": None, "controls_preserved_exact": None,
                       "controls_preserved_lexical": None, "mean_seconds": None})
        for arm in GRAMMAR:
            row = summary["models"][arm]["tasks"][task]
            valid, delivery = row["quality_on_valid_only"], row["quality_with_explicit_source_fallback"]
            output.append({"task": task, "model": arm, "rows": row["rows"], "native_available_rate": row["available_rate"],
                           "delivery_chrf": delivery["chrf"], "delivery_gleu": delivery["gleu"]["score"],
                           "delivery_exact_reference_rate": delivery["exact_reference_rate"],
                           "valid_only_rows": valid["rows"] if valid else 0,
                           "valid_only_chrf": valid["chrf"] if valid else None,
                           "valid_only_gleu": valid["gleu"]["score"] if valid else None,
                           "unchanged_controls": row["unchanged_controls"],
                           "controls_preserved_exact": row["unchanged_controls_preserved_exact"],
                           "controls_preserved_lexical": row["unchanged_controls_preserved_lexical"],
                           "mean_seconds": row["mean_seconds"]})
    return output


def pronunciation_rows(bundle, *, by_speaker=False):
    cases = [c for c in bundle["pronunciation_manifest"]["cases"] if c["group"] == "expert_test"]
    groups = sorted({c["speaker_id"] for c in cases}) if by_speaker else ["all"]
    output = []
    predictions = bundle["pronunciation_results"]["predictions"]
    for speaker in groups:
        selected = [c for c in cases if speaker == "all" or c["speaker_id"] == speaker]
        languages = sorted({c.get("annotation", {}).get("native_language", "unknown") for c in selected})
        for arm in PHONEME:
            chosen = [predictions[arm][c["id"]] for c in selected]
            for panel in PANELS:
                score = aggregate_scores(row[panel] for row in chosen)
                review_key = "variant_review_common37" if panel == "research_common37" else "variant_review_full39"
                counts = Counter()
                for row in chosen:
                    counts.update({k: v for k, v in row[review_key]["counts"].items() if isinstance(v, int)})
                ratio = lambda n, d: n / d if d else None
                total_errors = counts["gold_error_words"]
                total_correct = counts["reference_words"] - total_errors
                available_errors = counts["hint_tp"] + counts["hint_fn"]
                available_correct = counts["hint_fp"] + counts["hint_tn"]
                output.append({"speaker": speaker, "native_languages": ", ".join(languages), "model": arm, "inventory": panel,
                               "cases": len(selected), **score["counts"], **score["metrics"],
                               "review_precision": ratio(counts["hint_tp"], counts["hint_tp"] + counts["hint_fp"]),
                               "review_recall_available_only": ratio(counts["hint_tp"], available_errors),
                               "review_recall_all_expert_errors": ratio(counts["hint_tp"], total_errors),
                               "review_false_positive_rate_available_only": ratio(counts["hint_fp"], available_correct),
                               "review_coverage": ratio(counts["available_words"], counts["reference_words"]),
                               "review_reference_words": counts["reference_words"],
                               "review_available_words": counts["available_words"],
                               "review_total_expert_error_words": total_errors,
                               "review_total_expert_correct_words": total_correct,
                               "review_available_error_words": available_errors,
                               "review_available_correct_words": available_correct,
                               "review_unavailable_error_words": total_errors - available_errors,
                               "review_unavailable_correct_words": total_correct - available_correct,
                               "review_hint_tp": counts["hint_tp"], "review_hint_fp": counts["hint_fp"],
                               "review_hint_fn_available_only": counts["hint_fn"], "review_hint_tn_available_only": counts["hint_tn"],
                               "calibrated_decided_words": counts["calibrated_decided_words"],
                               "removed_expert_events": sum(len(r["research_common37"]["removed_indistinguishable_expert_events"]) for r in chosen) if panel == "research_common37" else 0,
                               "inference_sec": sum(r["inference_sec"] for r in chosen),
                               "mean_inference_sec": sum(r["inference_sec"] for r in chosen) / len(chosen),
                               "effective_devices": ", ".join(sorted({str(r.get("runtime", {}).get("device", "unknown")) for r in chosen}))})
    return output


def case_rows(bundle):
    output = []
    for arm, rows in bundle["grammar_predictions"].items():
        for row in rows:
            output.append({"domain": "grammar", "id": row["id"], "cohort": row["cohort"], "task": row["task"], "model": arm,
                           "source": row["source"], "references": json.dumps(row["references"], ensure_ascii=False),
                           "prediction": row["prediction"], "raw": row["raw"], "error": row["error"], "seconds": row["seconds"]})
    for case in bundle["pronunciation_manifest"]["cases"]:
        for arm in PHONEME:
            row = bundle["pronunciation_results"]["predictions"][arm][case["id"]]
            output.append({"domain": "pronunciation", "id": case["id"], "cohort": case["group"], "task": "phone_recognition", "model": arm,
                           "source": case["reference_text"], "references": " ".join(case.get("annotation", {}).get("perceived_phones", [])),
                           "prediction": " ".join(row["phonemes"]), "raw": json.dumps(row.get("native_decoded_labels"), ensure_ascii=False),
                           "error": None, "seconds": row["inference_sec"]})
    return output


def content_rows(bundle):
    """Secondary format-only audit; native predictions are never replaced."""
    output = []
    audit = bundle.get("content_audit")
    if audit:
        for arm in GRAMMAR:
            for task, row in audit["models"][arm]["tasks"].items():
                valid, fallback = row["quality_on_available_only"], row["quality_with_explicit_source_fallback"]
                output.append({"model": arm, "task": task, "rows": row["rows"], "content_available": row["content_available"],
                               "native_content": row["sources"].get("native_contract", 0),
                               "posthoc_transcript_field": row["sources"].get("posthoc_transcript_field", 0),
                               "fallback_gleu": fallback["gleu"]["score"], "fallback_chrf": fallback["chrf"],
                               "available_only_gleu": valid["gleu"]["score"] if valid else None,
                               "available_only_rows": valid["rows"] if valid else 0})
    return output


def key_findings(bundle):
    """Factual paired-subset findings; never turn reference scores into accuracy."""
    result = []
    grammar = grammar_rows(bundle)
    for task in TASKS:
        selected = {r["model"]: r for r in grammar if r["task"] == task}
        qwen, coedit = selected["qwen-grammar"], selected["coedit-large"]
        delta = coedit["delivery_gleu"] - qwen["delivery_gleu"]
        result.append(
            f'{TASK_NAMES[task]} · cùng {qwen["rows"]} câu: GLEU toàn bộ câu/source fallback '
            f'CoEdIT {coedit["delivery_gleu"]:.2f}, Qwen + Grammar LoRA {qwen["delivery_gleu"]:.2f}; '
            f'chênh lệch CoEdIT − LoRA {delta:+.2f} điểm. '
            f'Native output có sẵn: CoEdIT {coedit["valid_only_rows"]}/{coedit["rows"]}, '
            f'LoRA {qwen["valid_only_rows"]}/{qwen["rows"]}.')
    auxiliary = content_rows(bundle)
    if auxiliary:
        for task in TASKS:
            selected = {r["model"]: r for r in auxiliary if r["task"] == task}
            before, after = selected["qwen-base"], selected["qwen-grammar"]
            result.append(
                f'Phụ hậu kiểm định dạng · {TASK_NAMES[task].lower()}: '
                f'GLEU nội dung toàn bộ câu/source fallback Qwen gốc {before["fallback_gleu"]:.2f} '
                f'→ LoRA {after["fallback_gleu"]:.2f} '
                f'({after["fallback_gleu"] - before["fallback_gleu"]:+.2f} điểm). '
                f'Đây là trích nguyên chuỗi từ schema khác được phát hiện sau khi xem output; '
                f'không thay kết quả native và không chứng minh độ chính xác ngữ pháp.')
    guard = bundle.get("delivery_audit")
    if guard:
        tasks = guard["models"]["qwen-grammar"]["summary"]["published_heldout"]
        native = sum(r["native_available"] for r in tasks.values())
        blocked = sum(r["reasons"].get("lexical_guard_withheld", 0) for r in tasks.values())
        reference_blocked = sum(r["reference_matching_outputs_withheld"] for r in tasks.values())
        result.append(
            f'Luồng giao kết quả · Grammar LoRA: guard chặn {blocked}/{native} đầu ra native, '
            f'trong đó {reference_blocked} đầu ra khớp ít nhất một tham chiếu công bố '
            f'(chỉ chuẩn hóa khoảng trắng). Điều này cho thấy một phần gợi ý tham chiếu hợp lệ '
            f'bị chặn; không chứng minh tất cả {blocked} gợi ý bị chặn đều đúng.')
    for row in pronunciation_rows(bundle):
        if row["inventory"] != "research_original_inventory":
            continue
        result.append(
            f'{NAMES[row["model"]]} · {row["cases"]} audio: '
            f'phone F1 target gốc {number(row["f1"], percent=True)}. '
            f'Gợi ý từ theo full39: FPR trên phần khả dụng '
            f'{number(row["review_false_positive_rate_available_only"], percent=True)}, '
            f'coverage {number(row["review_coverage"], percent=True)}; '
            f'recall trên toàn bộ lỗi từ chuyên gia '
            f'{number(row["review_recall_all_expert_errors"], percent=True)}.')
    result.append(
        'Các số trên là phép thử local trên subset cố định. GLEU đo khớp tham chiếu; '
        'native availability đo contract. Gợi ý phát âm chưa calibration. '
        'Không suy ra model thắng ở mọi tình huống và chưa đổi model mặc định trong demo.')
    return result


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def figures(bundle, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})
    g = grammar_rows(bundle)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for i, task in enumerate(TASKS):
        rows = [r for r in g if r["task"] == task and r["model"] != "copy-source"]
        labels = ["Qwen gốc", "Qwen + LoRA", "CoEdIT-large"]
        axes[i, 0].bar(labels, [100*r["native_available_rate"] for r in rows], color=COLORS)
        axes[i, 0].set(title=TASK_NAMES[task] + " · đầu ra đúng định dạng", ylim=(0, 112), ylabel="% có kết quả native")
        for x, row in enumerate(rows):
            axes[i, 0].text(x, 100*row["native_available_rate"] + 2, f'{row["valid_only_rows"]}/{row["rows"]}', ha="center")
        bars = axes[i, 1].bar(labels, [r["delivery_gleu"] for r in rows], color=COLORS)
        axes[i, 1].bar_label(bars, fmt="%.1f", padding=3)
        baseline = next(r["delivery_gleu"] for r in g if r["task"] == task and r["model"] == "copy-source")
        axes[i, 1].axhline(baseline, color="#4c3d5e", ls="--", label=f"Giữ nguyên câu: {baseline:.1f}")
        axes[i, 1].set(title=TASK_NAMES[task] + " · GLEU, toàn bộ câu", ylabel="GLEU (0–100)")
        axes[i, 1].legend(loc="best", frameon=False)
    fig.suptitle("Grammar · coverage tách khỏi chất lượng văn bản\nGLEU toàn bộ câu: lỗi định dạng dùng fallback giữ nguyên, không phải model tự sửa", fontsize=13)
    fig.savefig(output / "grammar_diagnostics.png", dpi=160); plt.close(fig)
    p = pronunciation_rows(bundle)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = ["wav2vec2 B", "XLS-R TIMIT", "CROTTC-IF"]
    for i, panel in enumerate(("research_full39", "research_common37")):
        rows = [r for r in p if r["inventory"] == panel]
        title = "Full39 · giữ lỗi chuyên gia" if i == 0 else "Common37 · tác vụ giảm inventory"
        bars = axes[i, 0].bar(labels, [100*r["f1"] if r["f1"] is not None else float("nan") for r in rows], color=COLORS)
        axes[i, 0].bar_label(bars, fmt="%.1f", padding=3)
        axes[i, 0].set(title=title + " · phone F1", ylabel="% F1", ylim=(0, 100))
        bars = axes[i, 1].bar(labels, [100*r["review_false_positive_rate_available_only"] if r["review_false_positive_rate_available_only"] is not None else float("nan") for r in rows], color=COLORS)
        axes[i, 1].bar_label(bars, fmt="%.1f", padding=3)
        axes[i, 1].set(title=title + " · gợi ý nhầm từ đúng", ylabel="% FPR · chỉ phần khả dụng", ylim=(0, 100))
    fig.suptitle("Phát âm · cùng audio, cùng nhãn chuyên gia\nGợi ý xem lại từ chưa được hiệu chỉnh thành kết luận đúng/sai", fontsize=13)
    fig.savefig(output / "pronunciation_diagnostics.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    speakers = pronunciation_rows(bundle, by_speaker=True)
    names = sorted({r["speaker"] for r in speakers})
    for index, arm in enumerate(PHONEME):
        values = [next(r["f1"] for r in speakers if r["speaker"] == s and r["model"] == arm and r["inventory"] == "research_full39") for s in names]
        ax.bar(np.arange(len(names)) + (index-1)*.25, [100*v if v is not None else float("nan") for v in values], width=.24, label=NAMES[arm], color=COLORS[index])
    ax.set_xticks(range(len(names)), [s + ("\nViệt Nam" if any(r["speaker"] == s and r["native_languages"].lower() in {"vietnamese", "vi", "vietnam"} for r in speakers) else "") for s in names])
    ax.set(ylabel="Phone F1 · full39 (%)", ylim=(0, 100), title="Theo speaker · 20 audio/người; không phải độ chính xác dân số")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    fig.savefig(output / "pronunciation_by_speaker.png", dpi=160); plt.close(fig)


def e(value):
    return html.escape(str(value if value is not None else "—"), quote=True)


def number(value, *, percent=False):
    return "—" if value is None else f"{100*value:.1f}%" if percent else f"{value:.2f}"


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join('<th>' + e(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + e(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def image_tag(path):
    return '<img class="chart" alt="Biểu đồ so sánh từ dữ liệu đã lưu" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'


def case_cards(bundle, *, development=False):
    selected = [r for r in bundle["grammar_manifest"]["rows"] if (r["cohort"] == "user_development") == development]
    by_arm = {a: {r["id"]: r for r in rows} for a, rows in bundle["grammar_predictions"].items()}
    audit = bundle["delivery_audit"]
    guard = {a: {r["id"]: r for r in v["rows"]} for a, v in audit["models"].items()} if audit else {}
    content_audit = bundle.get("content_audit")
    contents = {a: {r["id"]: r for r in v["rows"]} for a, v in content_audit["models"].items()} if content_audit else {}
    output = []
    for source in selected:
        parts = ['<details class="case" data-search="' + e(source["id"] + ' ' + source["source"]) + '"><summary><b>' + e(TASK_NAMES[source["task"]]) + '</b> · ' + e(source["source"]) + '</summary><div class="case-body"><p class="mono">' + e(source["id"]) + '</p>']
        if source["references"]:
            parts.append('<p><b>Tham chiếu:</b> ' + e(' | '.join(source["references"])) + '</p>')
        else:
            parts.append('<p class="muted">Ví dụ phát triển; không có nhãn chuyên gia để tính độ chính xác.</p>')
        for arm in GRAMMAR:
            row = by_arm[arm][source["id"]]
            parts.append('<section class="output"><b>' + e(NAMES[arm]) + '</b><p>' + e(row["prediction"]) + '</p><small>' + e(row["error"] or "native output available") + ' · ' + number(row["seconds"]) + ' s</small>')
            if guard:
                parts.append('<small>Guard hiện tại: ' + e(guard[arm][source["id"]]["reason"]) + '</small>')
                if guard[arm][source["id"]].get("matches_published_reference") is True:
                    parts.append('<small>Đầu ra khớp một tham chiếu công bố sau chuẩn hóa khoảng trắng.</small>')
            if contents and contents[arm][source["id"]]["content_source"] == "posthoc_transcript_field":
                parts.append('<p><small>Nội dung trích từ schema khác · phân tích hậu kiểm, không sửa chữ:</small>' + e(contents[arm][source["id"]]["content"]) + '</p>')
            parts.append('<details><summary>Raw output</summary><pre>' + e(row["raw"]) + '</pre></details></section>')
        parts.append('</div></details>'); output.append(''.join(parts))
    return ''.join(output)


def pronunciation_cards(bundle, *, development=False):
    output = []
    for case in bundle["pronunciation_manifest"]["cases"]:
        if (case["group"] == "user_development") != development:
            continue
        parts = ['<details class="case" data-search="' + e(case["id"] + ' ' + case["reference_text"]) + '"><summary><b>' + e(case.get("speaker_id", "User")) + '</b> · ' + e(case["reference_text"]) + '</summary><div class="case-body"><p class="mono">' + e(case["id"]) + '</p>']
        source = (ROOT / case.get("audio", "")).resolve()
        if case.get("audio") and source.is_relative_to(ROOT):
            relative = os.path.relpath(source, bundle["directory"]).replace('\\', '/')
            parts.append('<p><a href="' + e(relative) + '">Audio gốc</a></p><audio controls preload="none" src="' + e(relative) + '"></audio>')
        annotation = case.get("annotation")
        if annotation:
            parts.append('<p><b>Âm chuyên gia nghe được:</b> <span class="mono">' + e(' '.join(annotation["perceived_phones"])) + '</span></p>')
        else:
            parts.append('<p class="muted">Không có nhãn phoneme chuyên gia; không tính F1/PER cho audio này.</p>')
        for arm in PHONEME:
            row = bundle["pronunciation_results"]["predictions"][arm][case["id"]]
            parts.append('<section class="output"><b>' + e(NAMES[arm]) + '</b><p class="mono">' + e(' '.join(row["phonemes"])) + '</p><small>' + number(row["inference_sec"]) + ' s · ' + e(row.get("runtime", {}).get("device", "unknown")) + '</small>')
            if annotation:
                metrics = row["research_full39"]["metrics"]
                parts.append('<small>Full39 F1 ' + number(metrics["f1"], percent=True) + ' · PER ' + number(metrics["per"], percent=True) + '</small>')
            parts.append('<details><summary>Inventory gốc / gợi ý xem lại</summary><pre>' + e(json.dumps({"native_labels": row.get("native_decoded_labels"), "unknown_tokens": row.get("unknown_tokens"), "full39": row["variant_review_full39"], "common37": row["variant_review_common37"]}, ensure_ascii=False, indent=2)) + '</pre></details></section>')
        parts.append('</div></details>'); output.append(''.join(parts))
    return ''.join(output)


def render_html(bundle, output):
    g, p = grammar_rows(bundle), pronunciation_rows(bundle)
    gm, pm = bundle["grammar_manifest"], bundle["pronunciation_manifest"]
    grammar_cases = Counter(r["cohort"] for r in gm["rows"])
    phone_cases = Counter(r["group"] for r in pm["cases"])
    chunks = ['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>So sánh model · 24/10/2026</title><style>' + CSS + '</style><body><main>']
    chunks.append('<header><span class="eyebrow">LOCAL MODEL COMPARISON · 24/10/2026</span><h1>Model mới có tốt hơn?</h1><p>Cùng dữ liệu, đầu ra được lưu đầy đủ, không train thêm. Ngữ pháp và phát âm được đánh giá bằng hai phép thử riêng.</p><div class="stats"><span><b>' + str(grammar_cases["published_heldout"]) + '</b> câu có tham chiếu × 3 model</span><span><b>' + str(phone_cases["expert_test"]) + '</b> audio có nhãn chuyên gia × 3 model</span><span><b>' + str(grammar_cases["user_development"]) + ' + ' + str(phone_cases["user_development"]) + '</b> ví dụ phát triển, không chấm</span></div></header>')
    chunks.append('<nav><a href="#grammar">Ngữ pháp</a><a href="#pronunciation">Phát âm</a><a href="#examples">Ví dụ của bạn</a><a href="#evidence">Dữ liệu & nguồn</a></nav>')
    chunks.append('<section id="findings"><h2>Kết quả lượt thử</h2>' + ''.join('<p>' + e(finding) + '</p>' for finding in key_findings(bundle)) + '</section>')
    chunks.append('<section id="grammar"><h2>1. Ngữ pháp & cách nói tự nhiên</h2><p>Qwen gốc và Grammar LoRA dùng JSON contract đã được train. CoEdIT dùng prompt native của tác giả và trả văn bản. Tỷ lệ có đầu ra hợp lệ phản ánh khả năng đáp ứng contract; nó không phải độ chính xác tiếng Anh.</p>')
    chunks.append(image_tag(output / "grammar_diagnostics.png"))
    chunks.append('<div class="notice"><b>Cách đọc GLEU:</b> Bảng toàn bộ câu dùng fallback giữ nguyên nguồn nếu model không có đầu ra hợp lệ. Vì vậy một model không sửa được câu nào vẫn có thể có điểm giống baseline. Bảng valid-only có mẫu số khác giữa các model và không dùng làm xếp hạng ngang hàng. Điểm GLEU/chrF/exact là độ khớp tham chiếu, không chứng minh giữ nghĩa hay diễn đạt tốt theo người dùng. GLEU dùng code tác giả trên subset này; độ lệch do lấy mẫu tham chiếu không phải khoảng tin cậy của tập dữ liệu.</div>')
    chunks.append(table(["Tác vụ", "Model", "Native hợp lệ", "GLEU toàn bộ / fallback", "chrF toàn bộ", "GLEU valid-only (n)", "Control: giữ exact", "Control: giữ lexical", "Giây/câu"], [[TASK_NAMES[r["task"]], NAMES.get(r["model"], "Giữ nguyên câu"), number(r["native_available_rate"], percent=True), number(r["delivery_gleu"]), number(r["delivery_chrf"]), number(r["valid_only_gleu"]) + f' ({r["valid_only_rows"]})', f'{r["controls_preserved_exact"]}/{r["unchanged_controls"]}' if r["unchanged_controls"] is not None else "baseline", f'{r["controls_preserved_lexical"]}/{r["unchanged_controls"]}' if r["unchanged_controls"] is not None else "baseline", number(r["mean_seconds"])] for r in g]))
    chunks.append('<p class="muted">Control lexical: nguồn trùng ít nhất một tham chiếu sau khi bỏ khác biệt hoa/thường và dấu câu. Cột giữ exact đòi hỏi giữ nguyên từng ký tự, nên sửa hoa/thường hay dấu câu hợp lệ cũng làm cột này giảm. Đây là diagnostic mức bảo thủ, không phải tỷ lệ giữ nguyên tất cả câu đã đúng ngữ pháp.</p>')
    chunks.append('<p class="muted">Đã loại ' + e(sum(gm["dataset"].get("excluded", {}).values())) + ' câu ứng viên trùng lexical với nguồn train/validation được kiểm tra. Chi tiết: ' + e(json.dumps(gm["dataset"].get("excluded", {}), ensure_ascii=False)) + '. Phần trùng được loại trước khi chọn mẫu. Câu tương tự nghĩa hoặc dữ liệu pretrain chưa công bố vẫn có thể trùng. W&I ở đây là test nội bộ từ public training corpus; JFLEG là subset official test.</p>')
    content = content_rows(bundle)
    if content:
        chunks.append('<h3>Học định dạng hay thực sự sửa câu tốt hơn?</h3><p>Qwen gốc có thể đã sửa câu bên trong JSON sai schema. Vì vậy 0% native không có nghĩa model không biết tiếng Anh. Bảng phụ này chỉ lấy nguyên chuỗi transcript khi raw output có đúng hai khóa task/transcript; không sửa từ hay sinh lại câu. Quy tắc được định nghĩa sau khi nhìn thấy đầu ra ban đầu, nên đây là phân tích <b>hậu kiểm</b>, không thay kết quả native đã đóng băng và không đổi parser demo.</p>')
        chunks.append(table(["Tác vụ", "Model", "Nội dung có sẵn", "Native", "Trích schema khác", "GLEU toàn bộ / fallback", "chrF toàn bộ", "GLEU có nội dung (n)"], [[TASK_NAMES[r["task"]], NAMES[r["model"]], f'{r["content_available"]}/{r["rows"]}', r["native_content"], r["posthoc_transcript_field"], number(r["fallback_gleu"]), number(r["fallback_chrf"]), number(r["available_only_gleu"]) + f' ({r["available_only_rows"]})'] for r in content]))
    audit = bundle["delivery_audit"]
    if audit:
        rows = []
        for arm in GRAMMAR:
            for task, row in audit["models"][arm]["summary"]["published_heldout"].items():
                rows.append([TASK_NAMES[task], NAMES[arm], row["native_available"], row["delivered"], row["lexical_changes_delivered"], row["reference_matching_native_outputs"], row["reference_matching_outputs_withheld"], json.dumps(row["reasons"], ensure_ascii=False)])
        chunks.append('<h3>Đầu ra có qua được guard hiện tại?</h3><p>Áp dụng guard hiện tại vào dự đoán đã lưu; đây là kiểm tra tương thích cách giao kết quả, không phải kiểm tra đúng nghĩa/ngữ pháp. Budget benchmark 256 token khác budget production 320/512 token; không phải lượt chạy live mới. Khớp tham chiếu là bằng nhau sau chuẩn hóa khoảng trắng; câu không khớp vẫn có thể là cách nói hợp lệ.</p>' + table(["Tác vụ", "Model", "Native có sẵn", "Được giao", "Sửa lexical được giao", "Khớp tham chiếu", "Khớp tham chiếu nhưng bị chặn", "Lý do"], rows))
        blocked_examples = [row for row in audit["models"]["qwen-grammar"]["rows"] if row["cohort"] == "published_heldout" and row.get("matches_published_reference") is True and not row["delivered"]]
        if blocked_examples:
            example = blocked_examples[0]
            chunks.append('<div class="notice"><b>Ví dụ đầu tiên theo thứ tự đã đóng băng, khớp tham chiếu nhưng bị chặn:</b><p>' + e(example["source"]) + '</p><p>→ ' + e(example["prediction"]) + '</p><small>' + e(example["id"]) + ' · ' + e(example["reason"]) + '</small></div>')
    chunks.append('<h3>Tra từng câu đã chấm</h3><input class="search" data-target="grammar-cases" placeholder="Tìm câu hoặc ID…" aria-label="Tìm câu ngữ pháp"><div id="grammar-cases">' + case_cards(bundle) + '</div></section>')
    chunks.append('<section id="pronunciation"><h2>2. Phát âm · nhận biết lỗi và báo nhầm</h2><p>Ba model nhận audio độc lập, không nhận transcript hay phoneme chuyên gia. Sau khi decode, cùng một scorer mới đối chiếu nhãn. wav2vec2 B đã train tác vụ L2-ARCTIC; XLS-R mới là model TIMIT tổng quát; CROTTC-IF là model MDD công bố. So sánh thực tế hệ thống, không phải ablation chỉ khác kiến trúc.</p>')
    chunks.append(image_tag(output / "pronunciation_diagnostics.png"))
    expert_events = [event for case in pm["cases"] if case["group"] == "expert_test" for event in case["annotation"]["gold_error_events"]]
    distortion_count = sum(event["kind"] == "substitution" and '*' in str(event.get("observed_phone")) for event in expert_events)
    chunks.append('<div class="notice"><b>Ba panel, không nhập làm một:</b> Original inventory giữ nguyên ký hiệu gốc, gồm distortion; đó là phép thử strict target của dự án. Full39 bỏ dấu stress/distortion nhưng vẫn giữ sự kiện lỗi chuyên gia, vì vậy không còn đồng nhất với target gốc và có thể làm một số contrast không biểu diễn được. Bộ này có ' + str(distortion_count) + ' substitution distortion trong ' + str(len(expert_events)) + ' sự kiện lỗi. Common37 gộp thêm AO→AA, ZH→SH, DX→T, AX→AH và loại các substitution chuyên gia trở thành không phân biệt; vị trí lỗi khác được giữ. Không xem điểm reduced inventory là thay thế original target hoặc độ chính xác chẩn đoán đã hiệu chỉnh.</div>')
    chunks.append('<h3>Target gốc · giữ ký hiệu chuyên gia</h3>' + table(["Model", "Audio", "Phone P", "Phone R", "Phone F1", "PER ↓", "FAR", "FRR", "DER"], [[NAMES[r["model"]], r["cases"], *[number(r[k], percent=True) for k in ("precision", "recall", "f1", "per", "far", "frr", "der")]] for r in p if r["inventory"] == "research_original_inventory"]))
    chunks.append('<h3>Inventory chuẩn hóa và gợi ý từ</h3><p>Gợi ý dùng cùng dictionary-variant chưa calibration cho mọi model, không phải chính sách posterior B nguyên bản trong demo. Recall khả dụng và FPR khả dụng chỉ tính các từ thuộc audio có thể tạo gợi ý. Recall trên toàn bộ lỗi chuyên gia tính cả lỗi trong phần không khả dụng. Không đọc một tỷ lệ cao tách khỏi coverage.</p>')
    chunks.append(table(["Inventory", "Model", "Phone P", "Phone R", "Phone F1", "PER ↓", "Gợi ý P", "Gợi ý R · khả dụng", "Gợi ý R · tất cả lỗi", "FPR · khả dụng ↓", "Coverage", "Lỗi đã gộp"], [[r["inventory"].replace("research_", ""), NAMES[r["model"]], *[number(r[k], percent=True) for k in ("precision", "recall", "f1", "per", "review_precision", "review_recall_available_only", "review_recall_all_expert_errors", "review_false_positive_rate_available_only", "review_coverage")], r["removed_expert_events"]] for r in p if r["inventory"] != "research_original_inventory"]))
    chunks.append(table(["Inventory", "Model", "Từ khả dụng / tổng", "Lỗi có gợi ý / mọi lỗi", "Lỗi không khả dụng", "Từ đúng không khả dụng", "Quyết định calibration"], [[r["inventory"].replace("research_", ""), NAMES[r["model"]], f'{r["review_available_words"]}/{r["review_reference_words"]}', f'{r["review_hint_tp"]}/{r["review_total_expert_error_words"]}', r["review_unavailable_error_words"], r["review_unavailable_correct_words"], r["calibrated_decided_words"]] for r in p if r["inventory"] != "research_original_inventory"]))
    chunks.append('<p class="muted">Counter gốc, số insertion, FAR/FRR, DER và inventory nguyên bản có trong CSV/JSON. Các phép đo dùng canonical-phone-and-ordered-insertion-slots-v1 của dự án; không tái tạo trực tiếp F1 của từng bài báo. Số quyết định đã calibration: ' + e(', '.join(NAMES[a] + ': ' + str(next(r["calibrated_decided_words"] for r in p if r["model"] == a and r["inventory"] == "research_full39")) for a in PHONEME)) + '.</p>')
    chunks.append(image_tag(output / "pronunciation_by_speaker.png"))
    speaker_rows = pronunciation_rows(bundle, by_speaker=True)
    chunks.append(table(["Speaker", "L1 theo metadata", "Model", "Audio", "Full39 F1", "PER", "FPR từ · khả dụng", "Coverage"], [[r["speaker"], r["native_languages"], NAMES[r["model"]], r["cases"], number(r["f1"], percent=True), number(r["per"], percent=True), number(r["review_false_positive_rate_available_only"], percent=True), number(r["review_coverage"], percent=True)] for r in speaker_rows if r["inventory"] == "research_full39"]))
    chunks.append('<p class="muted">TLV là subset tiếng Anh giọng Việt khi metadata L1 xác nhận Vietnamese. Chỉ 20 câu của một người Việt trong bộ này; tổng hợp 120 câu không chứng minh độ chính xác trên toàn bộ người Việt. Checkpoint CROTTC công bố đã train L2-ARCTIC; overlap với split local chưa được chứng minh không có.</p><h3>Thời gian suy luận</h3>')
    chunks.append(table(["Model", "Audio", "Giây/audio", "Thiết bị thực tế"], [[NAMES[r["model"]], r["cases"], number(r["mean_inference_sec"]), r["effective_devices"]] for r in p if r["inventory"] == "research_full39"]))
    runtime_rows = [row for predictions in bundle["pronunciation_results"]["predictions"].values() for row in predictions.values()]
    all_cuda_fp32 = all(str(row.get("runtime", {}).get("device", "")).startswith("cuda")
                        and row.get("runtime", {}).get("compute_type") == "float32" for row in runtime_rows)
    runtime_note = ('Cả ba model đều chạy CUDA FP32, không fallback CPU. ' if all_cuda_fp32 else
                    'Thiết bị thực tế được ghi riêng cho từng model/audio. ')
    chunks.append('<p class="muted">' + runtime_note + 'Chạy tuần tự; thời gian load tách riêng trong raw JSON. Kiến trúc, decoder và chi phí IPC khác nhau; đây không phải so sánh chỉ riêng kiến trúc. Không suy ra tốc độ toàn pipeline từ thời gian từng model.</p>')
    notes = bundle.get("pronunciation_execution_notes")
    if notes and notes.get("explicit_expert_events_without_word_index"):
        chunks.append('<p class="muted">Word-review chỉ dùng nhãn có word_index hợp lệ. Có ' + e(notes["explicit_expert_events_without_word_index"]) + ' sự kiện thêm âm thiếu word_index nên không tạo thêm từ lỗi trong đáp án word-review; phone scorer vẫn giữ các sự kiện này. Không trộn mẫu số của word-review với word metrics từ scorer canonical.</p>')
    if notes:
        chunks.append('<details><summary>Ghi chú thực thi và metadata runtime</summary><pre>' + e(json.dumps(notes, ensure_ascii=False, indent=2)) + '</pre></details>')
    chunks.append('<h3>Tra từng audio có nhãn</h3><input class="search" data-target="phone-cases" placeholder="Tìm speaker, câu đọc hoặc ID…" aria-label="Tìm audio phát âm"><div id="phone-cases">' + pronunciation_cards(bundle) + '</div></section>')
    chunks.append('<section id="examples"><h2>3. Ví dụ bạn đã phản ánh</h2><p>Các ví dụ phát triển để nhìn trực tiếp đầu ra. Không đưa vào điểm test công bố, không gán điểm ngữ nghĩa hay lỗi phát âm khi thiếu nhãn chuyên gia.</p>' + case_cards(bundle, development=True) + pronunciation_cards(bundle, development=True) + '</section>')
    chunks.append('<section id="evidence"><h2>4. Bằng chứng, giới hạn & tài liệu</h2><ul>' + ''.join('<li>' + e(s) + '</li>' for s in gm["limitations"]) + '<li>' + e(pm["selection"]["scope"]) + '</li></ul><h3>Model, prompt, tham số chạy</h3><details><summary>Xem cấu hình cố định</summary><pre>' + e(json.dumps({"grammar_models": gm["models"], "grammar_generation": gm["generation"], "grammar_prompts": gm["prompts"], "pronunciation_models": pm["models"], "pronunciation_protocol": pm["protocol"]}, ensure_ascii=False, indent=2)) + '</pre></details><h3>File kết quả gốc</h3><ul>')
    for name in (*bundle["input_sha256"], "grammar_metrics.csv", "pronunciation_metrics.csv", "pronunciation_by_speaker.csv", "per_case.csv", *(("grammar_content_audit.csv",) if content else ()), "report_manifest.json"):
        chunks.append('<li><a href="' + e(name.replace('\\', '/')) + '">' + e(name) + '</a></li>')
    chunks.append('</ul><h3>Nguồn sơ cấp</h3><ol>' + ''.join('<li><a href="' + e(url) + '" target="_blank" rel="noopener">' + e(title) + '</a></li>' for title, url in SOURCES) + '</ol><p class="muted">Các nguồn cung cấp kiến trúc, dữ liệu và cách đánh giá. Số trong biểu đồ là kết quả local của lượt chạy này; không nhập điểm paper làm kết quả đã tái lập.</p></section><footer>Chỉ đọc kết quả đã lưu · không gọi API · không train · không thay model trong demo.</footer></main><script>document.querySelectorAll(".search").forEach(input=>input.addEventListener("input",()=>{const q=input.value.toLocaleLowerCase();document.querySelectorAll("#"+input.dataset.target+" > .case").forEach(card=>{card.hidden=!card.textContent.toLocaleLowerCase().includes(q)})}));</script></body></html>')
    return ''.join(chunks)


CSS = """
:root{color-scheme:light;--purple:#7040a9;--ink:#2f243a;--muted:#766c81;--line:#e8e0f0}*{box-sizing:border-box}body{margin:0;background:#f7f4fb;color:var(--ink);font:15px/1.7 'Segoe UI',Arial,sans-serif}main{max-width:1190px;margin:auto;padding:34px 24px}header,section{background:white;border:1px solid var(--line);border-radius:20px;padding:30px;margin-bottom:24px}header{background:linear-gradient(125deg,#fff,#f2e9fc)}h1{font-size:38px;line-height:1.2;margin:12px 0}h2{font-size:26px;margin:0 0 18px}h3{font-size:19px;margin-top:28px}.eyebrow{color:var(--purple);font-size:12px;font-weight:700;letter-spacing:2px}.stats{display:flex;gap:15px;flex-wrap:wrap;margin-top:24px}.stats span{padding:14px 18px;background:#ffffffc9;border:1px solid var(--line);border-radius:12px}.stats b{color:var(--purple);font-size:24px}nav{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}a{color:var(--purple);text-underline-offset:3px}nav a{background:#fff;padding:8px 17px;border:1px solid var(--line);border-radius:99px;text-decoration:none}.chart{display:block;width:100%;height:auto;margin:20px 0}.notice{background:#f7f1fc;border-left:4px solid #9870be;padding:17px 20px;border-radius:7px;margin:20px 0}.muted,small,footer{color:var(--muted)}small{display:block;font-size:12px}.table-scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px}th{background:#f1eaf8;color:#563078;text-align:left;white-space:nowrap}th,td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:hover{background:#faf7fd}.case{border:1px solid var(--line);border-radius:10px;margin:10px 0;overflow:hidden}.case>summary{cursor:pointer;padding:14px 17px;background:#faf8fd;line-height:1.5}.case-body{padding:0 18px 18px}.output{padding:12px 15px;margin:12px 0;border:1px solid var(--line);border-radius:9px;box-shadow:none}.output p{margin:7px 0}.output>details{font-size:12px;margin-top:8px}.mono,pre{font-family:Consolas,monospace;font-size:12px;word-break:break-word}pre{white-space:pre-wrap;background:#f8f6fa;padding:15px;border-radius:8px;max-height:600px;overflow:auto}.search{width:100%;padding:12px 15px;border:1px solid #cab6e0;border-radius:10px;font:inherit;margin:10px 0}footer{text-align:center;font-size:12px;padding:15px}[hidden]{display:none!important}@media(max-width:700px){main{padding:14px 10px}header,section{padding:19px 15px;border-radius:14px}h1{font-size:29px}h2{font-size:23px}.stats{gap:8px}.stats span{padding:10px;font-size:12px}.table-scroll{margin-left:-5px;margin-right:-5px}}
"""


def render(directory=DEFAULT):
    bundle = load_bundle(directory)
    output = bundle["directory"]
    figures(bundle, output)
    csv_write(output / "grammar_metrics.csv", grammar_rows(bundle))
    csv_write(output / "pronunciation_metrics.csv", pronunciation_rows(bundle))
    csv_write(output / "pronunciation_by_speaker.csv", pronunciation_rows(bundle, by_speaker=True))
    csv_write(output / "per_case.csv", case_rows(bundle))
    if bundle.get("content_audit"):
        csv_write(output / "grammar_content_audit.csv", content_rows(bundle))
    report = output / "report.html"
    report.write_text(render_html(bundle, output), encoding="utf-8")
    artifacts = ("report.html", "grammar_diagnostics.png", "pronunciation_diagnostics.png", "pronunciation_by_speaker.png", "grammar_metrics.csv", "pronunciation_metrics.csv", "pronunciation_by_speaker.csv", "per_case.csv")
    if bundle.get("content_audit"):
        artifacts += ("grammar_content_audit.csv",)
    manifest = {"schema": "saved-model-comparison-report-v1", "renderer_sha256": digest(__file__),
                "input_sha256": bundle["input_sha256"], "output_sha256": {name: digest(output / name) for name in artifacts},
                "inference_performed": False, "sources": [{"title": t, "url": u} for t, u in SOURCES]}
    (output / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(render(args.directory))


if __name__ == "__main__":
    main()
