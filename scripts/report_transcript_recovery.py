"""Render saved text-recovery diagnostics without running a model or an API.

The input is the results.json written by evaluate_transcript_recovery.py.
Fixtures include exposed prompt examples and development cases; these outputs
are not an audio, WER, pronunciation, or population-accuracy benchmark.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "prompt_example": "Prompt examples (exposed)",
    "known_user_example": "Known user text examples",
    "development": "Development fixtures",
    "control": "Control fixtures",
}
COLORS = {"accepted": "#7c3aed", "unchanged": "#b7a2d8", "ambiguous": "#f59e0b",
          "rejected": "#dc2626", "unavailable": "#64748b"}
LIMITS = (
    "All inputs are TEXT fixtures, including ASR-like spellings; no audio is decoded or transcribed in this experiment.",
    "Prompt examples, known user examples, development fixtures and controls are reported separately. They are exposed cases, not an independently sampled test set.",
    "Exact-target agreement compares the returned effective text, including a preserved raw fallback, with the supplied fixture target after token normalization. Text agreement does not mean recovery was accepted; status agreement is reported separately. This is not audio WER, pronunciation quality, semantic correctness, or population accuracy.",
    "An accepted recovery is an inferred interpretation, not proof of the words spoken. Preserve the raw transcript and require confirmation before using an inferred interpretation as a pronunciation reference.",
    "Latency includes this recorded recovery operation only. It excludes microphone capture, ASR, pronunciation assessment, conversation generation, TTS and playback.",
    "This report only reads recorded results. It starts no training, model inference, API call, or audio request.",
)


def load_results(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("training_started") is not False or result.get("external_api_calls") != 0
            or result.get("audio_inputs") != 0):
        raise ValueError("Expected local text-only diagnostics without training or external API calls")
    if not isinstance(result.get("cases"), list) or not result["cases"]:
        raise ValueError("At least one recorded text fixture is required")
    for row in result["cases"]:
        if not isinstance(row.get("result"), dict) or not isinstance(row.get("transcript"), str):
            raise ValueError("Each fixture needs a transcript and recorded result")
        seconds = row["result"].get("latency_sec")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Every fixture needs a finite nonnegative recorded latency")
    return result


def case_rows(result):
    return [{"id": row["id"], "group": row.get("group", "unspecified"),
             "raw_transcript": row["transcript"], "recovery_output": row["result"].get("text", ""),
             "expected_text": row.get("expected_text"), "status": row["result"]["status"],
             "expected_status": row.get("expected_status"), "reason": row["result"].get("reason"),
             "target_match": row.get("text_matches_expected"),
             "status_match": row.get("status_matches_expected"),
             "seconds": row["result"]["latency_sec"]} for row in result["cases"]]


def group_summary(result):
    rows = case_rows(result)
    groups = list(dict.fromkeys(row["group"] for row in rows))
    return [{"group": group, "cases": len(selected := [row for row in rows if row["group"] == group]),
             "selected_outputs": sum(row["status"] in {"accepted", "unchanged"} for row in selected),
             "target_matches": sum(row["target_match"] is True for row in selected),
             "target_mismatches": sum(row["target_match"] is False for row in selected),
             "not_target_scored": sum(row["target_match"] is None for row in selected),
             "expected_status_matches": sum(row["status_match"] is True for row in selected)}
            for group in groups]


def failure_rows(result):
    return [row for row in case_rows(result)
            if row["target_match"] is False or row["status_match"] is False]


def plot_results(result):
    import matplotlib.pyplot as plt
    rows = case_rows(result)
    groups = list(dict.fromkeys(row["group"] for row in rows))
    statuses = list(dict.fromkeys(row["status"] for row in rows))
    figure, (counts, latency) = plt.subplots(1, 2, figsize=(13, max(4.8, .33 * len(rows))))
    left = [0] * len(groups)
    for status in statuses:
        values = [sum(row["status"] == status and row["group"] == group for row in rows) for group in groups]
        counts.barh([GROUPS.get(group, group) for group in groups], values, left=left,
                    label=status, color=COLORS.get(status, "#94a3b8"))
        left = [before + value for before, value in zip(left, values)]
    counts.set_title("Recorded status counts by fixture group")
    counts.set_xlabel("Text fixtures (counts, not accuracy)")
    counts.legend(fontsize=8, loc="best")
    latency.barh([row["id"] for row in rows], [row["seconds"] for row in rows],
                 color=[COLORS.get(row["status"], "#94a3b8") for row in rows])
    latency.set_title("Recorded recovery latency per text fixture")
    latency.set_xlabel("Seconds; excludes ASR and TTS")
    latency.invert_yaxis()
    for axis in (counts, latency):
        axis.set_facecolor("#faf8ff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.grid(axis="x", alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle("Local Qwen text recovery diagnostics — not an audio benchmark", color="#4c1d95")
    figure.tight_layout()
    return figure


def escaped(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return html.escape(str(value))


def table_html(rows, columns):
    headings = "".join("<th>" + escaped(label) + "</th>" for _, label in columns)
    body = "".join("<tr>" + "".join("<td>" + escaped(row.get(key)) + "</td>" for key, _ in columns)
                   + "</tr>" for row in rows)
    return '<div class="scroll"><table><thead><tr>' + headings + "</tr></thead><tbody>" + body + "</tbody></table></div>"


def write_html(result, source, output):
    import matplotlib.pyplot as plt
    figure = plot_results(result)
    image_path = output / "recovery-diagnostics.png"
    figure.savefig(image_path, dpi=160, facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Transcript recovery diagnostics</title><style>'
            'body{margin:0;background:#faf8ff;color:#30233f;font:15px/1.6 system-ui,sans-serif}'
            'main{max-width:1400px;margin:auto;padding:28px}h1,h2{color:#5b21b6}'
            'section,details{background:white;border:1px solid #e9ddfb;border-radius:12px;padding:18px;margin:18px 0}'
            '.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}'
            'th,td{padding:10px;text-align:left;border-bottom:1px solid #eee;vertical-align:top;min-width:85px}'
            'th{background:#f3edff}td{max-width:320px;overflow-wrap:anywhere}'
            'img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}'
            'summary{cursor:pointer;color:#6d28d9}code{overflow-wrap:anywhere}.notice{border-left:4px solid #7c3aed}'
            '</style><main><h1>Local Qwen text recovery</h1>'
            '<section class="notice"><strong>Text fixtures only — not an audio or WER benchmark.</strong><ul>'
            + "".join("<li>" + escaped(item) + "</li>" for item in LIMITS) + "</ul></section>")
    page += '<section><h2>Recorded experiment</h2><p>' + escaped(result.get("scope")) + "</p><p>Model: <code>" + escaped(result.get("model_id")) + "</code> · Version: <code>" + escaped(result.get("version")) + "</code> · Recorded at: " + escaped(result.get("created_at_utc")) + "</p></section>"
    page += "<section><h2>Groups reported separately</h2>" + table_html(group_summary(result), [
        ("group", "Fixture group"), ("cases", "Cases"), ("selected_outputs", "Selected outputs"),
        ("target_matches", "Target matches"), ("target_mismatches", "Target mismatches"),
        ("not_target_scored", "Not target scored"), ("expected_status_matches", "Expected status matches")])
    page += "<p>Target matches count returned-text agreement among these exposed cases, including unchanged raw fallbacks. They do not imply an accepted recovery or correct status. A missing target score means the fixture supplies no target sentence.</p></section>"
    columns = [("id", "Case"), ("group", "Group"), ("raw_transcript", "Raw ASR-like text"),
               ("recovery_output", "Effective text (raw fallback included)"), ("expected_text", "Fixture target"),
               ("status", "Status"), ("expected_status", "Expected status"), ("reason", "Reason"),
               ("target_match", "Target match"), ("status_match", "Status match"), ("seconds", "Seconds")]
    page += '<section><h2>All recorded cases</h2><p>Every case is shown, including failures. An ambiguous or rejected output must not replace the raw transcript.</p>' + table_html(case_rows(result), columns) + "</section>"
    failures = failure_rows(result)
    page += '<section><h2>Target or status mismatches</h2>' + (table_html(failures, columns) if failures else "<p>No recorded fixture mismatch. This does not establish audio recognition quality.</p>") + "</section>"
    page += '<section><h2>Status and timing</h2><img alt="Recorded text fixture status counts and recovery latency" src="data:image/png;base64,' + encoded + '"></section>'
    for case in result["cases"]:
        record = {"history": case.get("history", []), "result": case["result"],
                  "raw_model_output": case.get("raw_model_output"), "generation": case.get("generation")}
        page += "<details><summary>Raw model output: " + escaped(case["id"]) + "</summary><pre>" + escaped(json.dumps(record, indent=2, ensure_ascii=False)) + "</pre></details>"
    provenance = {"results_path": str(source.resolve()), "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  **{key: value for key, value in result.items() if key not in {"cases", "summary"}}}
    page += "<details><summary>Source and provenance</summary><pre>" + escaped(json.dumps(provenance, indent=2, ensure_ascii=False)) + "</pre></details></main></html>\n"
    (output / "report.html").write_text(page, encoding="utf-8")


def notebook_path_source(path):
    try:
        return "ROOT / " + repr(path.resolve().relative_to(ROOT).as_posix())
    except ValueError:
        return "Path(" + repr(str(path.resolve())) + ")"


def make_notebook(source):
    import nbformat
    code = nbformat.v4.new_code_cell
    intro = "# Local transcript recovery: saved text diagnostics\n\n" + "\n\n".join(LIMITS)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(intro),
        code("from pathlib import Path\nimport hashlib, json, sys\nimport pandas as pd\n"
             "import matplotlib.pyplot as plt\nfrom IPython.display import display\n"
             "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/report_transcript_recovery.py').is_file())\n"
             "sys.path.insert(0, str(ROOT))\n"
             "from scripts.report_transcript_recovery import load_results, case_rows, group_summary, failure_rows, plot_results\n"
             f"SOURCE = {notebook_path_source(source)}\n"
             "result = load_results(SOURCE)\nsource_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()\n"
             "print('Recorded TEXT fixtures:', len(result['cases']))\n"
             "print('Model:', result.get('model_id'), '| Version:', result.get('version'))\n"
             "print('Source SHA256:', source_sha256)\n"
             "print('No audio, model inference, API calls, or training run in this notebook.')"),
        code("print('Exposed fixture groups — counts only, not population accuracy')\n"
             "display(pd.DataFrame(group_summary(result)))\n"
             "print('Prompt examples and known user examples are not independent test cases.')"),
        code("rows = pd.DataFrame(case_rows(result))\n"
             "with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):\n"
             "    display(rows)\n"
             "print('Raw text is preserved. An inferred recovery is not a verified spoken-word reference.')"),
        code("figure = plot_results(result)\ndisplay(figure)\nplt.close(figure)\n"
             "print('These latencies exclude ASR, conversation, pronunciation assessment, and TTS.')"),
        code("failures = failure_rows(result)\nprint('Fixture target or status mismatches:', len(failures))\n"
             "display(pd.DataFrame(failures))\n"
             "for case in result['cases']:\n"
             "    print(json.dumps({'id': case['id'], 'group': case.get('group'), 'history': case.get('history', []), "
             "'result': case['result'], 'raw_model_output': case.get('raw_model_output'), 'generation': case.get('generation')}, "
             "ensure_ascii=False, indent=2))\n"
             "assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha256\n"
             "print('Results file remained unchanged. No accuracy or WER claim is made.')"),
    ])
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved evaluate_transcript_recovery.py results.json")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/transcript_recovery_validation.ipynb")
    parser.add_argument("--no-execute", action="store_true", help="Write notebook source without executing its offline analysis cells")
    args = parser.parse_args(argv)
    source = args.input.resolve()
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = load_results(source)
    output = source.parent
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    import nbformat
    notebook = make_notebook(source)
    if not args.no_execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=120, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, args.notebook)
    write_html(result, source, output)
    if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Recorded input results changed while the report was being generated")
    print(json.dumps({"notebook": str(args.notebook.resolve()), "report": str(output / "report.html"),
                      "executed": not args.no_execute, "code_cells": 5, "text_fixtures": len(result["cases"]),
                      "fixture_mismatches": len(failure_rows(result)), "results_sha256": original_hash,
                      "model_calls": 0, "api_calls": 0, "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
