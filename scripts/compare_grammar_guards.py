"""Compare two delivery guards against immutable, already-generated Qwen outputs.

This is a post-hoc delivery regression audit, not another model benchmark. The
same prediction is presented to both validators; no words are generated,
repaired, or replaced with a reference in this script.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / "data/generated/model-comparison-20261024/grammar"
DEFAULT_OUTPUT = ROOT / "data/generated/grammar-guard-20261024"
PREFIXES = {"grammar_corrected": "grammar", "natural": "natural"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_guard(path, label):
    """A separate module prevents the baseline importing the edited validator."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("grammar_guard_" + label + "_" + digest(path)[:12], path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load guard module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_rewrite


def verified_inputs(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    if "qwen-grammar" not in manifest.get("models", {}):
        raise ValueError("The frozen Qwen grammar model is missing")
    metadata_path = directory / "qwen-grammar/metadata.json"
    predictions_path = directory / "qwen-grammar/predictions.jsonl"
    metadata = read_json(metadata_path)
    if metadata.get("status") != "completed" or metadata.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("Saved inference is incomplete or has a different manifest")
    if metadata.get("predictions_sha256") != digest(predictions_path):
        raise ValueError("Saved predictions checksum changed")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = manifest["rows"]
    if [row["id"] for row in rows] != [row["id"] for row in cases]:
        raise ValueError("Saved prediction case order changed")
    for row, case in zip(rows, cases):
        for field in ("source", "task", "cohort", "references"):
            if row.get(field) != case.get(field):
                raise ValueError("Saved prediction case differs: " + row["id"] + " / " + field)
        if row["task"] not in PREFIXES:
            raise ValueError("Unknown saved task")
    return rows, {str(path): digest(path) for path in (manifest_path, metadata_path, predictions_path)}


def inspect(row, parser):
    task, prediction = row["task"], row.get("prediction")
    if prediction is None:
        return {"delivered": False, "reason": "native_contract_unavailable", "changed": False,
                "format_changed": False, "delivered_prediction": None, "diagnostics": {}}
    payload = {"grammar_corrected": row["source"], "natural": row["source"],
               "explanation": "Saved model output; delivery guard regression audit."}
    payload[task] = prediction
    result = parser(json.dumps(payload, ensure_ascii=False), True, row["source"])
    prefix = PREFIXES[task]
    accepted = result.get(prefix + "_status") == "ready" and bool(result.get(task))
    diagnostics = {key: value for key, value in result.items()
                   if any(marker in key for marker in ("guard", "validation", "reason", "edit_type"))}
    return {"delivered": bool(accepted), "reason": "accepted" if accepted else "guard_withheld",
            "changed": bool(accepted and result.get(prefix + "_changed")),
            "format_changed": bool(accepted and result.get(prefix + "_format_changed")),
            "delivered_prediction": result.get(task) if accepted else None, "diagnostics": diagnostics}


def compare_row(row, baseline_parser, current_parser):
    before, after = inspect(row, baseline_parser), inspect(row, current_parser)
    prediction = row.get("prediction")
    references = row.get("references", [])
    normalize = lambda text: " ".join(text.split())
    matches = (prediction is not None and normalize(prediction) in {normalize(ref) for ref in references}) if references else None
    transition = ("retained" if after["delivered"] else "still_withheld") if before["delivered"] == after["delivered"] else (
        "newly_accepted" if after["delivered"] else "newly_withheld")
    return {key: row.get(key) for key in ("id", "source", "task", "cohort", "prediction", "error")} | {
        "references": references, "matches_published_reference": matches, "transition": transition,
        "baseline": before, "current": after}


def summarize(rows):
    result = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        result[cohort] = {}
        for task in PREFIXES:
            selected = [row for row in rows if row["cohort"] == cohort and row["task"] == task]
            output = {"rows": len(selected), "native_available": sum(row["prediction"] is not None for row in selected),
                      "native_reference_matches": sum(row["matches_published_reference"] is True for row in selected)}
            for side in ("baseline", "current"):
                output[side] = {
                    "delivered": sum(row[side]["delivered"] for row in selected),
                    "withheld": sum(not row[side]["delivered"] for row in selected),
                    "lexical_changes_delivered": sum(row[side]["changed"] for row in selected),
                    "format_only_delivered": sum(row[side]["format_changed"] for row in selected),
                    "unchanged_delivered": sum(row[side]["delivered"] and not row[side]["changed"]
                                               and not row[side]["format_changed"] for row in selected),
                    "reference_matches_delivered": sum(row["matches_published_reference"] is True
                                                        and row[side]["delivered"] for row in selected),
                    "reference_matches_withheld": sum(row["matches_published_reference"] is True
                                                       and not row[side]["delivered"] for row in selected),
                }
            output["newly_accepted"] = sum(row["transition"] == "newly_accepted" for row in selected)
            output["newly_withheld"] = sum(row["transition"] == "newly_withheld" for row in selected)
            output["reference_matches_recovered"] = sum(row["matches_published_reference"] is True
                and row["transition"] == "newly_accepted" for row in selected)
            output["reference_matches_newly_withheld"] = sum(row["matches_published_reference"] is True
                and row["transition"] == "newly_withheld" for row in selected)
            result[cohort][task] = output
    return result


def verify_baseline(rows, original_audit):
    previous = {row["id"]: row for row in original_audit["models"]["qwen-grammar"]["rows"]}
    if set(previous) != {row["id"] for row in rows}:
        raise ValueError("Original delivery audit has different cases")
    for row in rows:
        old = previous[row["id"]]
        for field in ("delivered", "changed", "format_changed", "delivered_prediction"):
            if row["baseline"][field] != old[field]:
                raise ValueError("Baseline snapshot does not reproduce old audit: " + row["id"] + " / " + field)


def write_csv(path, rows):
    columns = ["id", "cohort", "task", "source", "prediction", "matches_published_reference", "transition",
               "baseline_delivered", "current_delivered", "baseline_changed", "current_changed", "current_diagnostics"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            value = {field: row.get(field) for field in columns[:7]}
            for side in ("baseline", "current"):
                value[side + "_delivered"] = row[side]["delivered"]
                value[side + "_changed"] = row[side]["changed"]
            value["current_diagnostics"] = json.dumps(row["current"]["diagnostics"], ensure_ascii=False)
            writer.writerow(value)


def report_html(result):
    esc = lambda value: escape(str(value), quote=True)
    parts = ["<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Qwen grammar delivery guard comparison</title><style>body{margin:0;background:#f8f7fc;color:#262238;font:16px/1.55 system-ui}main{max-width:1150px;margin:36px auto;padding:0 22px}h1{color:#613db0}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #ddd6ed;padding:10px;text-align:left}th{background:#eee7fa}details{padding:14px;background:white;border:1px solid #ddd6ed;border-radius:10px;margin:10px 0}summary{cursor:pointer}blockquote{border-left:3px solid #9b7bce;padding:5px 14px;margin:12px 0;white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3eff9;padding:12px}small{color:#655a78}.badge{color:#613db0;font-weight:650}a{color:#613db0}</style><main>",
             "<h1>Kiểm tra bộ lọc gợi ý ngữ pháp Qwen</h1>",
             "<p>Cùng đầu ra Qwen đã lưu được đưa vào bộ kiểm tra cũ và mới. Không chạy lại model, không thay đổi câu dự đoán, không train.</p>",
             "<p><strong>Đây là kiểm tra việc hiển thị gợi ý, không phải đo độ chính xác ngữ pháp hay mức bảo toàn ý nghĩa.</strong> Bộ quy tắc được sửa sau khi đã xem các ví dụ này; kết quả là đánh giá hồi quy trên dữ liệu đã quan sát.</p>"]
    for cohort, tasks in result["summary"].items():
        parts.extend(["<h2>" + esc(cohort) + "</h2>",
                      "<table><tr><th>Tác vụ</th><th>Số câu</th><th>Hiển thị cũ → mới</th><th>Được mở thêm</th><th>Mới bị chặn</th><th>Trùng đáp án nhưng bị chặn cũ → mới</th></tr>"])
        for task, stats in tasks.items():
            parts.append("<tr>" + "".join("<td>" + esc(value) + "</td>" for value in (
                task, stats["rows"], f'{stats["baseline"]["delivered"]} → {stats["current"]["delivered"]}',
                stats["newly_accepted"], stats["newly_withheld"],
                f'{stats["baseline"]["reference_matches_withheld"]} → {stats["current"]["reference_matches_withheld"]}')) + "</tr>")
        parts.append("</table>")
    parts.append("<h2>Các quyết định đã thay đổi</h2><p>Trùng một đáp án công bố là một tín hiệu tham khảo; câu đó vẫn có thể chưa hoàn chỉnh. Những cách viết khác đáp án cũng có thể đúng.</p>")
    changed = [row for row in result["rows"] if row["transition"] in {"newly_accepted", "newly_withheld"}]
    for row in changed:
        parts.extend(["<details><summary><span class='badge'>" + esc(row["transition"]) + "</span> · " + esc(row["task"]) + " · " + esc(row["id"]) + "</summary>",
                      "<p>Câu gốc:</p><blockquote>" + esc(row["source"]) + "</blockquote>",
                      "<p>Đầu ra model:</p><blockquote>" + esc(row["prediction"]) + "</blockquote>",
                      "<p>Trùng đáp án công bố: " + esc(row["matches_published_reference"]) + "</p>",
                      "<p>Đáp án tham khảo:</p><blockquote>" + esc("\n".join(row["references"])) + "</blockquote>",
                      "<pre>" + esc(json.dumps(row["current"]["diagnostics"], ensure_ascii=False, indent=2)) + "</pre></details>"])
    if not changed:
        parts.append("<p>Không có quyết định thay đổi.</p>")
    parts.append("<h2>Giới hạn</h2><ul>" + "".join("<li>" + esc(text) + "</li>" for text in result["limitations"]) + "</ul>")
    parts.append("<p><a href='comparison.json'>Toàn bộ dữ liệu JSON</a> · <a href='comparison.csv'>CSV</a> · <a href='report_manifest.json'>Kiểm tra nguồn và checksum</a></p></main></html>")
    return "\n".join(parts)


def run(directory=DEFAULT_INPUT, output=DEFAULT_OUTPUT):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if directory == output or directory in output.parents:
        raise ValueError("Use a separate output directory; frozen inference artifacts must remain untouched")
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "baseline_sentence_rewrite.py"
    baseline_inference_path = output / "baseline_grammar_inference.py"
    snapshot_sources_path = output / "baseline_sources.json"
    source_hashes = read_json(snapshot_sources_path)
    for name, path in (("sentence_rewrite.py", baseline_path), ("grammar_inference.py", baseline_inference_path)):
        if digest(path) != source_hashes[name]:
            raise ValueError("Baseline source snapshot differs: " + name)
    rows, inputs = verified_inputs(directory)
    baseline_parser = load_guard(baseline_path, "baseline")
    current_path = ROOT / "scripts/sentence_rewrite.py"
    current_parser = load_guard(current_path, "current")
    comparisons = [compare_row(row, baseline_parser, current_parser) for row in rows]
    old_audit_path = directory / "delivery_audit.json"
    verify_baseline(comparisons, read_json(old_audit_path))
    inputs.update({str(path): digest(path) for path in (old_audit_path, baseline_path, baseline_inference_path,
                  snapshot_sources_path, current_path, ROOT / "scripts/grammar_inference.py", Path(__file__))})
    result = {"schema": "saved-qwen-grammar-guard-comparison-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "no_model_calls": True, "no_training": True, "exploratory_after_cases_observed": True,
              "inputs_sha256": inputs, "summary": summarize(comparisons), "rows": comparisons,
              "limitations": [
                  "Only the delivery validator changes; saved model predictions are identical in both arms.",
                  "These examples were already observed before the rules were changed. This is a regression audit, not an unseen blind benchmark.",
                  "Acceptance is not a grammar or semantic correctness label, and an unchanged sentence is not an improvement.",
                  "Exact published-reference equality uses whitespace normalization only. Reference-matching does not prove a sentence is complete or intended.",
                  "Nonmatching outputs may be valid alternatives. This audit cannot estimate semantic false-accept rates from reference matching alone.",
                  "Published cases and the eight user-development examples per task are reported separately.",
                  "The original frozen benchmark generation budget differs from the live worker. No new live inference latency is measured.",
                  "Grammar and natural suggestions are checked independently and never become pronunciation references.",
              ]}
    save_json(output / "comparison.json", result)
    write_csv(output / "comparison.csv", comparisons)
    (output / "report.html").write_text(report_html(result), encoding="utf-8")
    outputs = {name: digest(output / name) for name in ("comparison.json", "comparison.csv", "report.html")}
    save_json(output / "report_manifest.json", {"schema": "grammar-guard-report-manifest-v1",
              "inputs_sha256": inputs, "outputs_sha256": outputs, "baseline_reproduces_original_audit": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.directory, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("Report:", args.output / "report.html")


if __name__ == "__main__":
    main()
