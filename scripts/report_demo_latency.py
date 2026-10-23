"""Create a recorded-results-only latency comparison and executed notebook.

No model, web server, training process, audio decoder or API is invoked. The
input files must have been produced separately by benchmark_demo_latency.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CPU = ROOT / "data/generated/demo-latency-cpu/summary.json"
DEFAULT_GPU = ROOT / "data/generated/demo-latency-gpu/summary.json"
DEFAULT_OUTPUT = ROOT / "data/generated/demo-latency-comparison"
PURPLE = "#7c3aed"
PALE = "#c4b5fd"
INK = "#30233f"
LIMITATIONS = [
    "Small diagnostic only: one repeated CPU request and two repeated GPU requests on one 2.97-second synthetic English recording; not a speech-quality benchmark.",
    "The primary comparison excludes each session's first measured request. The CPU worker was already running; its first measured request is not a cold-start measurement. The GPU worker was restarted, and its first measured request took about 25 seconds to publish the reply.",
    "Repeated audio reuses dictionary, pronunciation and model caches. New words, new audio or a restarted process may take longer. Zero recorded stage time can reflect a warm cache or timer precision, not absence of work.",
    "Conversation sampling is enabled. CPU and GPU generated different replies and different completion-token counts; these results do not isolate hardware speed or establish a guaranteed speedup.",
    "Qwen changed from CPU FP32/eager in the existing worker to GPU BF16/SDPA after restart. ASR and phoneme-B remain on CPU with two threads. This compares the full runtime configuration, not GPU alone.",
    "No voice API was called. ai33/Sarah synthesis, audio download/playback, microphone capture, browser rendering and ngrok network delays are excluded.",
    "Reply-ready is observed after a completed reply becomes available; it is not time to first generated token. HTTP polling at 0.15 seconds limits wall-clock precision.",
    "Stage measurements are instrumented only in the GPU session. CPU pure-generation time cannot be recovered from reply_latency_sec, which includes audio analysis and normalization.",
    "Source hashes in benchmark summaries describe files on disk. The CPU measurement used an older worker already loaded before those files changed; disk hashes do not identify that worker's loaded code.",
]


def read_record(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "passed" or len(value.get("runs", [])) < 2:
        raise ValueError(f"A passed benchmark with at least two requests is required: {path}")
    if (value.get("voice_api_calls") != 0 or value.get("audio_sent_externally") is not False
            or value.get("training_started") is not False):
        raise ValueError(f"This report requires local inference without voice calls or training: {path}")
    for run in value["runs"]:
        numbers = [run["full_elapsed_sec"], run["reply_ready"]["elapsed_sec"]]
        if run.get("status") != "ready" or any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError("All measured requests must complete with finite nonnegative times")
    return value


def measured_rows(cpu, gpu):
    rows = []
    for session, record in (("CPU FP32/eager", cpu), ("GPU BF16/SDPA", gpu)):
        for position, run in enumerate(record["runs"]):
            meta = run.get("metadata") or {}
            rows.append({"session": session, "request": position + 1,
                         "group": "first measured request" if position == 0 else "repeat request",
                         "reply_ready_sec": run["reply_ready"]["elapsed_sec"],
                         "full_elapsed_sec": run["full_elapsed_sec"],
                         "normalization_sec": (run.get("normalization") or {}).get("latency_sec"),
                         "rewrite_sec": meta.get("rewrite_latency_sec"),
                         "completion_tokens": (meta.get("usage") or {}).get("completion_tokens"),
                         "reply_generation_attempts": meta.get("reply_generation_attempts"),
                         "transcript": run.get("transcript"), "reply": run.get("reply")})
    return rows


def repeat_statistics(rows):
    result = {}
    for session in ("CPU FP32/eager", "GPU BF16/SDPA"):
        selected = [row for row in rows if row["session"] == session and row["group"] == "repeat request"]
        result[session] = {"n": len(selected)}
        for metric in ("reply_ready_sec", "full_elapsed_sec"):
            values = [row[metric] for row in selected]
            result[session][metric] = {"mean": mean(values), "min": min(values), "max": max(values)}
    improvement = {}
    for metric in ("reply_ready_sec", "full_elapsed_sec"):
        before = result["CPU FP32/eager"][metric]["mean"]
        after = result["GPU BF16/SDPA"][metric]["mean"]
        improvement[metric] = {"observed_ratio": before / after,
                               "reduction_percent": 100 * (before - after) / before}
    return {"sessions": result, "observed_change": improvement,
            "selection": "Exclude the first measured request from each session; arithmetic mean of repeats only."}


def gpu_stage_rows(gpu):
    stages = []
    labels = {"audio_decode_sec": "Decode audio", "asr_sec": "ASR on CPU",
              "reference_setup_sec": "Reference setup", "phoneme_sec": "Phoneme-B on CPU",
              "assessment_sec": "Pronunciation assessment", "reference_resolution_sec": "Reference resolution",
              "phonetic_candidates_sec": "Sound-neighbour lookup"}
    for run in gpu["runs"][1:]:
        timing = (run.get("speech_timings") or {}).get("timings")
        if not isinstance(timing, dict):
            raise ValueError("GPU repeats need speech.timings instrumentation")
        measured = {labels[key]: float(timing[key]) for key in labels if key in timing}
        meta = run.get("metadata") or {}
        measured["Transcript normalization"] = float(run["normalization"]["latency_sec"])
        measured["Qwen reply generation"] = float(meta["generation_latency_sec"])
        measured["Grammar + natural wording"] = float(meta["rewrite_latency_sec"])
        # Residual is not another model stage: include IPC/polling and any
        # uninstrumented work. Preserve its sign rather than hiding overlaps.
        measured["Other / observation residual"] = float(run["full_elapsed_sec"]) - sum(measured.values())
        for label, seconds in measured.items():
            stages.append({"request": run["index"], "stage": label, "seconds": seconds})
    return stages


def build_comparison(cpu_path=DEFAULT_CPU, gpu_path=DEFAULT_GPU):
    cpu, gpu = read_record(cpu_path), read_record(gpu_path)
    digest = cpu.get("audio", {}).get("submitted_wav_sha256")
    if not digest or digest != gpu.get("audio", {}).get("submitted_wav_sha256"):
        raise ValueError("CPU and GPU must use the same submitted audio bytes")
    for key in ("asr_model_id", "grammar_model_id", "reply_model_id"):
        if cpu.get("config", {}).get(key) != gpu.get("config", {}).get(key):
            raise ValueError("The model selection differs between sessions: " + key)
    rows = measured_rows(cpu, gpu)
    provenance = {}
    for name, path, record in (("cpu", cpu_path, cpu), ("gpu", gpu_path, gpu)):
        provenance[name] = {"summary_path": str(Path(path).resolve()),
                            "summary_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            "started_at_utc": record.get("started_at_utc"), "config": record.get("config"),
                            "poll_interval_sec": record.get("poll_interval_sec"),
                            "runtime_source_changed": record.get("runtime_source_changed"),
                            "source_files_sha256": record.get("source_sha256_before"),
                            "reported_qwen_runtime": [run.get("metadata", {}).get("qwen_runtime") for run in record["runs"]]}
    return {"status": "recorded_results", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Small local latency diagnostic, not quality validation or a controlled hardware-only benchmark.",
            "audio": cpu["audio"], "history": "empty for every request", "runs": rows,
            "repeats_only": repeat_statistics(rows),
            "first_measured_requests": [row for row in rows if row["request"] == 1],
            "gpu_repeat_stage_measurements": gpu_stage_rows(gpu),
            "limitations": LIMITATIONS, "provenance": provenance,
            "voice_api_calls": 0, "audio_sent_externally": False, "training_started": False}


def plot_speed(comparison):
    import matplotlib.pyplot as plt
    statistics = comparison["repeats_only"]["sessions"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ("reply_ready_sec", "full_elapsed_sec"),
                              ("Completed reply available", "Reply + optional feedback finished")):
        values = [statistics[name][key]["mean"] for name in statistics]
        labels = [f"CPU\nrepeats n={statistics['CPU FP32/eager']['n']}",
                  f"GPU\nrepeats n={statistics['GPU BF16/SDPA']['n']}"]
        bars = ax.bar(labels, values, color=[PALE, PURPLE], width=.58)
        ax.bar_label(bars, labels=[f"{v:.2f} s" for v in values], padding=4, color=INK)
        ax.set_ylim(0, max(values) * 1.23)
        ax.set(title=title, ylabel="Observed wall seconds (lower is faster)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
    figure.suptitle("Same audio; repeated requests only", color=INK)
    figure.tight_layout()
    return figure


def plot_stages(comparison):
    import matplotlib.pyplot as plt
    import pandas as pd
    measurements = pd.DataFrame(comparison["gpu_repeat_stage_measurements"])
    stages = measurements.groupby("stage", sort=False)["seconds"].mean()
    # Keep zero values in the table, but omit them from the plot to preserve
    # readability. They can be cache hits or timing precision.
    visible = stages[stages.abs() >= .0005]
    figure, ax = plt.subplots(figsize=(9, max(3.5, len(visible) * .44)))
    bars = ax.barh(visible.index, visible.values, color=PURPLE)
    ax.bar_label(bars, labels=[f"{v:.3f} s" for v in visible.values], padding=4)
    ax.invert_yaxis()
    ax.set_xlim(min(0, visible.min() * 1.2), max(visible.max() * 1.25, .1))
    ax.set(xlabel="Mean recorded seconds per repeated GPU request (n=2)",
           title="GPU session: stage diagnostics")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def dataframe_html(records, columns=None):
    import pandas as pd
    frame = pd.DataFrame(records)
    return frame[columns].to_html(index=False, float_format=lambda value: f"{value:.3f}", escape=True) if columns else frame.to_html(index=False, float_format=lambda value: f"{value:.3f}", escape=True)


def write_html(comparison, output):
    stats = comparison["repeats_only"]
    rows = [{"session": name, "repeats": row["n"], "reply-ready mean (s)": row["reply_ready_sec"]["mean"],
             "full mean (s)": row["full_elapsed_sec"]["mean"]} for name, row in stats["sessions"].items()]
    reply_ratio = stats["observed_change"]["reply_ready_sec"]["observed_ratio"]
    full_ratio = stats["observed_change"]["full_elapsed_sec"]["observed_ratio"]
    cpu, gpu = stats["sessions"]["CPU FP32/eager"], stats["sessions"]["GPU BF16/SDPA"]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local demo latency comparison</title>'
    page += '<style>body{font:16px/1.55 system-ui,sans-serif;color:#30233f;background:white;max-width:1080px;margin:35px auto;padding:0 20px}h1,h2{color:#7c3aed}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e9d5ff;padding:9px}th{background:#faf5ff}img{display:block;max-width:100%;height:auto;margin:24px auto}.scroll{overflow:auto}.notice{padding:15px 18px;background:#faf5ff;border-left:4px solid #7c3aed;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}li{margin:9px 0}</style>'
    page += '<h1>Local speech demo: CPU / GPU latency</h1><p>Recorded results only. Same 2.97-second synthetic recording, empty history, local audio processing and no voice synthesis.</p>'
    page += (f'<p class="notice">Repeated-request reply-ready mean: <strong>{cpu["reply_ready_sec"]["mean"]:.2f} → '
             f'{gpu["reply_ready_sec"]["mean"]:.2f} seconds</strong> ({reply_ratio:.2f}× observed ratio). Full processing: '
             f'<strong>{cpu["full_elapsed_sec"]["mean"]:.2f} → {gpu["full_elapsed_sec"]["mean"]:.2f} seconds</strong> '
             f'({full_ratio:.2f}×). CPU n={cpu["n"]}; GPU n={gpu["n"]}. This small diagnostic does not establish a '
             'guaranteed speedup or improved answer quality.</p>')
    page += '<h2>Repeated requests only</h2><div class="scroll">' + dataframe_html(rows) + '</div><img src="before-after.png" alt="Repeated-request latency before and after the runtime change">'
    page += '<h2>First measured requests, separately</h2><p>CPU: existing worker; do not interpret as cold start. GPU: first request after worker restart. Loading and caches make the first GPU response much slower than subsequent repeats.</p><div class="scroll">'
    page += dataframe_html(comparison["first_measured_requests"], ["session", "reply_ready_sec", "full_elapsed_sec", "completion_tokens"]) + '</div>'
    page += '<h2>GPU repeated-request stage diagnostics</h2><p>ASR and phoneme-B still run on CPU with two threads. Normalization, conversation and the grammar adapter run through Qwen on GPU BF16/SDPA. Other/residual includes uninstrumented work, IPC, rounding and HTTP polling; it is not another model.</p><img src="gpu-stage-breakdown.png" alt="Mean recorded stage times for repeated GPU requests">'
    page += '<h2>All measured requests</h2><div class="scroll">' + dataframe_html(comparison["runs"], ["session", "request", "group", "reply_ready_sec", "full_elapsed_sec", "normalization_sec", "rewrite_sec", "completion_tokens"]) + '</div>'
    page += '<h2>Interpretation limits</h2><ul>' + ''.join('<li>' + html.escape(item) + '</li>' for item in comparison["limitations"]) + '</ul>'
    page += '<details><summary>Recorded replies and provenance</summary><pre>' + html.escape(json.dumps({"runs": comparison["runs"], "audio": comparison["audio"], "provenance": comparison["provenance"]}, indent=2, ensure_ascii=False)) + '</pre></details></html>'
    (output / "report.html").write_text(page + "\n", encoding="utf-8")


def make_notebook(cpu_path, gpu_path, output):
    import nbformat
    notebook = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    def project_path(path):
        try:
            return "ROOT / " + repr(str(path.resolve().relative_to(ROOT)))
        except ValueError:
            return "Path(" + repr(str(path.resolve())) + ")"
    notebook.cells = [
        md("# Local demo latency: recorded CPU / GPU comparison\n\n"
           "This notebook only reads saved JSON and draws charts. It never starts a model, server, training job or API call. "
           "The main comparison uses **repeated requests only**: CPU n=1 and GPU n=2. "
           "The first CPU measurement used an existing worker; the first GPU measurement followed a restart."),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
             "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/report_demo_latency.py').is_file())\n"
             "sys.path.insert(0, str(ROOT))\n"
             "from scripts.report_demo_latency import build_comparison, plot_speed, plot_stages\n"
             f"CPU = {project_path(cpu_path)}\nGPU = {project_path(gpu_path)}\nOUT = {project_path(output)}\n"
             "OUT.mkdir(parents=True, exist_ok=True)\ncomparison = build_comparison(CPU, GPU)\n"
             "print('Recorded audio:', comparison['audio'])\n"
             "print('No TTS, no external audio, no training. Same submitted WAV SHA256 verified.')"),
        md("## Per-request measurements\n\nSame audio and empty history, sampled conversation. Completion-token lengths differ; "
           "do not interpret these as a controlled hardware-only comparison. Reply-ready measures a completed reply, not first token."),
        code("requests = pd.DataFrame(comparison['runs'])\n"
             "display(requests[['session','request','group','reply_ready_sec','full_elapsed_sec',"
             "'normalization_sec','rewrite_sec','completion_tokens']].round(3))\n"
             "repeat_rows = []\nfor name, stats in comparison['repeats_only']['sessions'].items():\n"
             "    repeat_rows.append({'session': name, 'n': stats['n'], 'reply-ready mean (s)': stats['reply_ready_sec']['mean'], "
             "'full mean (s)': stats['full_elapsed_sec']['mean']})\n"
             "display(pd.DataFrame(repeat_rows).round(3))\n"
             "print(json.dumps(comparison['repeats_only']['observed_change'], indent=2))"),
        md("## Repeats-only speed chart\n\nArithmetic means. One repeated CPU measurement does not establish variability or a confidence interval."),
        code("figure = plot_speed(comparison)\nfigure.savefig(OUT / 'before-after.png', dpi=160, facecolor='white')\nplt.show()"),
        md("## First measured requests: loading and cache effects\n\n**CPU first measured is not a cold start.** "
           "The restarted GPU worker took about 25 seconds to publish its first reply; this cost is excluded from the repeats-only means."),
        code("display(pd.DataFrame(comparison['first_measured_requests'])[["
             "'session','reply_ready_sec','full_elapsed_sec','normalization_sec','completion_tokens']].round(3))\n"
             "print('First requests are reported separately, never pooled into repeated-request means.')"),
        md("## GPU repeated-request stage breakdown\n\nQwen uses GPU BF16/SDPA; ASR and phoneme-B remain on CPU. "
           "Stage timers are unavailable in the older CPU worker. Zero values can reflect cache hits or timer precision. "
           "The residual includes IPC, polling and uninstrumented work; measurements are not an exact trace."),
        code("stage_rows = pd.DataFrame(comparison['gpu_repeat_stage_measurements'])\n"
             "display(stage_rows.pivot(index='stage', columns='request', values='seconds').round(3))\n"
             "display(stage_rows.groupby('stage', sort=False).seconds.mean().rename('mean seconds').to_frame().round(3))\n"
             "figure = plot_stages(comparison)\nfigure.savefig(OUT / 'gpu-stage-breakdown.png', dpi=160, facecolor='white')\nplt.show()"),
        md("## Limits and provenance\n\n" + "\n".join("- " + item for item in LIMITATIONS)),
        code("print(json.dumps({'audio': comparison['audio'], 'provenance': comparison['provenance']}, indent=2))\n"
             "print('No voice-generation latency or speech-quality result is inferred from these timings.')"),
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--gpu", type=Path, default=DEFAULT_GPU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    import nbformat
    from nbclient import NotebookClient
    comparison = build_comparison(args.cpu, args.gpu)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    notebook = make_notebook(args.cpu, args.gpu, args.output)
    notebook_path = ROOT / "notebooks/mdd_latency_comparison.ipynb"
    nbformat.write(notebook, notebook_path)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, notebook_path)
    write_html(comparison, args.output)
    print(json.dumps({"status": "passed", "notebook": str(notebook_path), "report": str(args.output / "report.html"),
                      "executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                      "repeat_comparison": comparison["repeats_only"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
