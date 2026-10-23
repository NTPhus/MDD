"""Render an executed notebook and HTML from an already recorded ai33 probe.

This script only reads summary.json and writes local report files. It does not
import the ai33 client, read .env, submit tasks, load models or start training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/generated/ai33-opening-latency-v1"
DEFAULT_SUMMARY = DEFAULT_OUTPUT / "summary.json"
NOTEBOOK = ROOT / "notebooks/ai33_opening_latency.ipynb"
PURPLE = "#7c3aed"
PALE = "#c4b5fd"
INK = "#30233f"
VOICE_NAMES = {"elevenlabs_EXAVITQu4vr4xnSDxMaL": "Sarah — Mature, Reassuring, Confident"}
LIMITS = [
    "One authored opening (n=1), followed by one exact cached repeat (n=1). This is not an estimate of typical or worst-case provider latency.",
    "Timings cover the backend provider client: task submission, task-status polling, audio download and local cache lookup. Browser HTTP, ngrok, decoding, playback start and word highlighting are not measured.",
    "The wait stage includes provider processing, status HTTP requests and the configured polling sleeps. It is not a direct measurement of pure synthesis time.",
    "Only one new ai33 task was submitted. The cached repeat reused the same audio bytes and made no provider status or submission requests, and no audio download.",
    "No learner recording was sent. The measured text is an authored topic opening; no LLM call or training was part of this probe.",
    "Adding a UI pause changes presentation timing, not the measured provider generation time. Report generation itself makes no API calls.",
]


def read_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    runs = record.get("runs") or []
    if record.get("status") != "passed" or len(runs) != 2:
        raise ValueError("A passed two-request opening probe is required")
    if (record.get("learner_audio_sent") is not False or record.get("llm_calls") != 0
            or record.get("training_started") is not False):
        raise ValueError("The input must describe authored-text TTS without learner audio, LLM or training")
    if record.get("max_new_tasks") != 1 or not 0 <= record.get("new_task_posts_total", -1) <= 1:
        raise ValueError("The recorded probe must respect the one-new-task budget")
    if not record.get("text") or not record.get("voice_id"):
        raise ValueError("The recorded text and public voice ID are required")
    for run in runs:
        for value in [run.get("elapsed_sec"), *[run.get("timing", {}).get(key) for key in
                     ("total_sec", "cache_lookup_sec", "submit_sec", "wait_sec", "download_sec")]]:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("All recorded stage times must be finite and nonnegative")
    initial, repeat = runs
    if not initial.get("audio_sha256") or initial["audio_sha256"] != repeat.get("audio_sha256"):
        raise ValueError("The initial and repeated request must return identical audio bytes")
    if (repeat.get("new_tasks_submitted") != 0 or repeat.get("timing", {}).get("cache_hit") is not True
            or repeat.get("timing", {}).get("request_count") != 0
            or repeat.get("timing", {}).get("download_attempts") != 0 or repeat.get("requests") != []):
        raise ValueError("The repeated request must be a verified local cache hit")
    return record


def stage_rows(record: dict) -> list[dict]:
    labels = {"cache_lookup_sec": "Local cache lookup", "submit_sec": "Submit ai33 task",
              "wait_sec": "Wait / status polling", "download_sec": "Download audio"}
    timing = record["runs"][0]["timing"]
    rows = [{"stage": label, "seconds": timing[key]} for key, label in labels.items()]
    rows.append({"stage": "Other measured bookkeeping", "seconds": timing["total_sec"] - sum(row["seconds"] for row in rows)})
    rows.append({"stage": "Total backend client", "seconds": timing["total_sec"]})
    return rows


def request_rows(record: dict) -> list[dict]:
    return [{"request": run["label"], "elapsed seconds": run["elapsed_sec"],
             "cache hit": run["timing"]["cache_hit"], "new tasks": run["new_tasks_submitted"],
             "provider requests": run["timing"]["request_count"],
             "audio downloads": run["timing"]["download_attempts"],
             "audio duration seconds": run.get("audio_duration_sec")}
            for run in record["runs"]]


def metadata(record: dict) -> dict:
    return {"measurement started (UTC)": record.get("started_at_utc"), "topic": record.get("topic_id"),
            "voice": VOICE_NAMES.get(record["voice_id"], "Recorded voice"),
            "public voice ID": record["voice_id"], "speed": record.get("speed"),
            "authored opening": record["text"], "word count (whitespace separated)": len(record["text"].split()),
            "audio duration seconds": record["runs"][0].get("audio_duration_sec"),
            "provider polling interval seconds": record.get("provider_poll_interval_sec"),
            "new tasks submitted": record["new_task_posts_total"]}


def plot_stages(record: dict):
    import matplotlib.pyplot as plt
    rows = stage_rows(record)
    # Keep small stages visible in the table, rather than making millisecond
    # cache lookup look as large as multi-second provider waiting on the chart.
    visible = [row for row in rows[:-1] if abs(row["seconds"]) >= .005]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.55, 1]})
    bars = axes[0].barh([row["stage"] for row in visible], [row["seconds"] for row in visible], color=PURPLE)
    axes[0].bar_label(bars, labels=[f'{row["seconds"]:.3f} s' for row in visible], padding=4, color=INK)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(row["seconds"] for row in visible) * 1.23)
    axes[0].set(title="Initial call: recorded stages (n=1)", xlabel="Seconds")
    elapsed = [run["elapsed_sec"] for run in record["runs"]]
    bars = axes[1].bar(["Initial call", "Cached repeat"], elapsed, color=[PURPLE, PALE], width=.58)
    axes[1].bar_label(bars, labels=[f"{value:.4f} s" for value in elapsed], padding=4, color=INK)
    axes[1].set_ylim(0, max(elapsed) * 1.25)
    axes[1].set(title="Same authored opening", ylabel="Backend elapsed seconds")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.15)
    figure.suptitle("ai33 / Sarah: one submission, then a local cache hit", color=INK)
    figure.tight_layout()
    return figure


def dataframe_html(rows: list[dict]) -> str:
    import pandas as pd
    return pd.DataFrame(rows).to_html(index=False, float_format=lambda number: f"{number:.4f}", escape=True)


def write_html(record: dict, summary: Path, output: Path) -> None:
    initial, repeat = record["runs"]
    meta = metadata(record)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ai33 opening voice latency</title>'
    page += '<style>body{font:16px/1.55 system-ui,sans-serif;color:#30233f;background:white;max-width:1050px;margin:35px auto;padding:0 20px}h1,h2{color:#7c3aed}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e9d5ff;padding:9px}th{background:#faf5ff}img{max-width:100%;height:auto}.scroll{overflow:auto}.notice{padding:15px 18px;background:#faf5ff;border-left:4px solid #7c3aed;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}li{margin:9px 0}</style>'
    page += '<h1>Opening voice: ai33 / Sarah</h1><p>Recorded provider-client measurements. This report does not call any API.</p>'
    page += f'<p class="notice">Initial request: <strong>{initial["elapsed_sec"]:.4f} seconds</strong>. Exact cached repeat: <strong>{repeat["elapsed_sec"]:.4f} seconds</strong>, with {repeat["timing"]["request_count"]} provider requests. One authored opening, one new task.</p>'
    page += '<h2>Voice and input</h2><pre>' + html.escape(json.dumps(meta, ensure_ascii=False, indent=2)) + '</pre>'
    page += '<h2>Initial-call stages</h2><p>The wait includes status requests and polling sleeps. Small bookkeeping stages remain in the table.</p><div class="scroll">' + dataframe_html(stage_rows(record)) + '</div>'
    page += '<img src="stages.png" alt="Recorded initial-call stage times and exact cached repeat">'
    page += '<h2>Initial call and cached repeat</h2><div class="scroll">' + dataframe_html(request_rows(record)) + '</div>'
    page += '<h2>Limits</h2><ul>' + ''.join('<li>' + html.escape(item) + '</li>' for item in LIMITS) + '</ul>'
    provenance = {"summary": str(summary.resolve()), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                  "report_generated_utc": datetime.now(timezone.utc).isoformat(), "audio_sha256": initial["audio_sha256"]}
    page += '<details><summary>Recorded source provenance</summary><pre>' + html.escape(json.dumps(provenance, indent=2)) + '</pre></details></html>'
    (output / "report.html").write_text(page + "\n", encoding="utf-8")


def make_notebook(summary: Path, output: Path):
    import nbformat
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    def project_path(path: Path) -> str:
        try:
            return "ROOT / " + repr(path.resolve().relative_to(ROOT).as_posix())
        except ValueError:
            return "Path(" + repr(str(path.resolve())) + ")"
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        md("# ai33 opening voice: measured latency\n\nRead saved results, inspect stages and draw charts. "
           "Running this notebook does **not** call ai33, read API keys, start a model or train anything."),
        code("from pathlib import Path\nimport sys, json, hashlib\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
             "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/report_ai33_latency.py').is_file())\n"
             "sys.path.insert(0, str(ROOT))\n"
             "from scripts.report_ai33_latency import read_record, metadata, stage_rows, request_rows, plot_stages, LIMITS\n"
             f"SUMMARY = {project_path(summary)}\nOUT = {project_path(output)}\n"
             "OUT.mkdir(parents=True, exist_ok=True)\nrecord = read_record(SUMMARY)\n"
             "print('Loaded recorded results:', SUMMARY)"),
        md("## Voice and authored opening\n\nSarah's public voice ID, speed, input length and audio duration come from the saved probe. "
           "No learner recording was sent."),
        code("display(pd.DataFrame(metadata(record).items(), columns=['field', 'recorded value']))"),
        md("## Initial call: stage table\n\nSubmission, provider waiting/status polling and audio download are separate timers. "
           "Waiting includes the configured polling interval; it is not pure synthesis time."),
        code("stages = pd.DataFrame(stage_rows(record))\ndisplay(stages.round(4))\n"
             "print('Provider polling interval:', record['provider_poll_interval_sec'], 'seconds')\n"
             "print('Status polls:', record['runs'][0]['timing']['poll_count'])"),
        md("## Purple / white latency chart\n\nEach bar is a recorded observation, not an average across many calls. "
           "The exact cached repeat is too small to appear at the initial-call scale; its label and table retain the measured time."),
        code("figure = plot_stages(record)\nfigure.savefig(OUT / 'stages.png', dpi=160, facecolor='white')\nplt.show()"),
        md("## Exact cached repeat\n\nThe same text, voice and speed reused the same audio bytes. "
           "This measures the backend's local cache path, not browser playback or a second voice-generation request."),
        code("display(pd.DataFrame(request_rows(record)).round(4))\n"
             "print('Same audio SHA256:', record['runs'][0]['audio_sha256'] == record['runs'][1]['audio_sha256'])\n"
             "print('Total new tasks:', record['new_task_posts_total'])"),
        md("## Interpretation limits\n\n" + "\n".join("- " + item for item in LIMITS)),
        code("print(json.dumps({'source': str(SUMMARY), 'source_sha256': hashlib.sha256(SUMMARY.read_bytes()).hexdigest(), "
             "'measurement_started_utc': record['started_at_utc'], 'scope': record['scope']}, indent=2))\n"
             "print('This notebook submitted no tasks and made no provider requests.')"),
        md("## Optional: a separately authorized new measurement\n\nThe following command is documentation only; "
           "it is deliberately not an executable notebook cell. It uses the configured API and can incur a charge. "
           "The benchmark enforces **at most one new ai33 task**, followed by an exact cached repeat. "
           "An existing receipt/cache may mean no new task is needed. Use a new output folder to keep the current evidence.\n\n"
           "```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\benchmark_ai33_opening.py --run --topic auto "
           "--output data/generated/ai33-opening-latency-next\n```\n\n"
           "To regenerate this report without calling any API:\n\n"
           "```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\report_ai33_latency.py\n```"),
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    import nbformat
    from nbclient import NotebookClient
    record = read_record(args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    notebook = make_notebook(args.summary, args.output)
    nbformat.write(notebook, NOTEBOOK)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, NOTEBOOK)
    write_html(record, args.summary, args.output)
    errors = sum(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
    print(json.dumps({"status": "passed", "notebook": str(NOTEBOOK), "report": str(args.output / "report.html"),
                      "executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                      "notebook_errors": errors, "report_api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
