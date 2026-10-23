"""Render the recorded Vietnamese-accented English pipeline evaluation.

This report reads local frozen results only: it never calls a model, network API,
training process or voice synthesis service. Audio stays in local artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/generated/vi-voice-pipeline-v1/current"
DEFAULT_MANIFEST = ROOT / "data/generated/vi-voice-pipeline-v1/manifest.json"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tokens(text):
    """Same punctuation/case-insensitive tokenization as the evaluation runner."""
    return re.findall(r"[a-z]+(?:'[a-z]+)*|\d+(?:[.:]\d+)*", text.lower().replace("’", "'"))


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def attach_text_reviews(results, results_path):
    """Verify a separate review sidecar and attach only to a presentation copy."""
    reviewed = copy.deepcopy(results)
    sidecar = Path(results_path).with_name("assistant_text_review.json")
    if not sidecar.is_file():
        return reviewed
    review = load_json(sidecar)
    if review.get("results_sha256") != digest(results_path):
        raise ValueError("Assistant review is not bound to these exact frozen results")
    entries = review.get("reviews", [])
    by_id = {entry["id"]: entry for entry in entries}
    if len(entries) != len(by_id) or set(by_id) != {row["id"] for row in results["records"]}:
        raise ValueError("Assistant reviews must cover every recorded case exactly once")
    for row in reviewed["records"]:
        entry = by_id[row["id"]]
        for field, hash_field in [("reference_text", "reference_sha256"),
                                  ("transcript", "transcript_sha256"), ("reply", "reply_sha256")]:
            if hashlib.sha256(row[field].encode("utf-8")).hexdigest() != entry.get(hash_field):
                raise ValueError("Assistant review text SHA mismatch: " + row["id"])
        row["assistant_text_review"] = entry
    reviewed["assistant_review_metadata"] = {key: value for key, value in review.items() if key != "reviews"}
    return reviewed


def checked_audio(path):
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Audio must remain inside the workspace")
    return resolved


def audit_sources(manifest, results, manifest_path=DEFAULT_MANIFEST):
    import pandas as pd
    manifest_cases = {case["id"]: case for case in manifest["cases"]}
    records = results.get("records", [])
    ids = [row["id"] for row in records]
    if len(set(ids)) != len(ids) or set(ids) != set(manifest_cases):
        raise ValueError("Report must preserve every fixed case exactly once")
    if digest(manifest_path) != results["identity"]["manifest_sha256"]:
        raise ValueError("Source manifest differs from the frozen inference manifest")
    rows = []
    for row in records:
        case = manifest_cases[row["id"]]
        for field in ("audio", "audio_sha256", "corpus", "speaker_id", "reference_text"):
            if row[field] != case[field]:
                raise ValueError("Recorded source differs from frozen case: " + case["id"])
        matches = digest(checked_audio(case["audio"])) == case["audio_sha256"]
        if not matches or row["audio_sha256"] != case["audio_sha256"]:
            raise ValueError("Audio bytes differ from the frozen inference input: " + case["id"])
        rows.append({"id": case["id"], "corpus": case["corpus"], "speaker": case["speaker_id"],
                     "seconds": case["duration_sec"], "reference_words": len(tokens(case["reference_text"])),
                     "audio_sha_verified": matches, "status": row["status"],
                     "source": case["source_url"], "license": (case.get("license") or {}).get("id")})
    return pd.DataFrame(rows)


def metric_table(records):
    """Micro-average edit counts; failed cases are counted but never scored as silence."""
    import pandas as pd
    rows = []
    groups = ["All clips", *sorted({row["corpus"] for row in records})]
    stages = [("transcript", "Raw ASR"), ("normalized", "Normalized")]
    if any("interpretation" in row for row in records):
        stages.append(("conversation_text", "Conversation interpretation"))
    for corpus in groups:
        selected = [r for r in records if corpus == "All clips" or r["corpus"] == corpus]
        ready = [r for r in selected if r["status"] == "ready"]
        reference_words = sum(len(tokens(r["reference_text"])) for r in ready)
        reference_chars = sum(len(" ".join(tokens(r["reference_text"]))) for r in ready)
        for field, label in stages:
            word_errors = sum(edit_distance(tokens(r["reference_text"]), tokens(r.get(field, ""))) for r in ready)
            char_errors = sum(edit_distance(" ".join(tokens(r["reference_text"])),
                                           " ".join(tokens(r.get(field, "")))) for r in ready)
            exact = sum(tokens(r["reference_text"]) == tokens(r.get(field, "")) for r in ready)
            rows.append({"corpus": corpus, "text_stage": label, "selected_clips": len(selected),
                         "ready_clips": len(ready), "failed_clips": len(selected) - len(ready),
                         "reference_words": reference_words, "word_errors": word_errors,
                         "WER (%)": 100 * word_errors / reference_words if reference_words else float("nan"),
                         "reference_chars": reference_chars, "char_errors": char_errors,
                         "CER (%)": 100 * char_errors / reference_chars if reference_chars else float("nan"),
                         "exact_prompt_matches": exact})
    return pd.DataFrame(rows)


def latency_table(records):
    import pandas as pd
    rows = []
    for row in records:
        latency = row.get("latencies") or {}
        interpretation = row.get("interpretation") or {}
        rows.append({"id": row["id"], "corpus": row["corpus"], "status": row["status"],
                     "Reply ready (s)": (latency.get("reply_ready") or {}).get("elapsed_sec"),
                     "All feedback ready (s)": latency.get("full_elapsed_sec"),
                     "Submission wait (s)": latency.get("submission_wait_sec"),
                     "Normalization (s)": (row.get("normalization") or {}).get("latency_sec"),
                     "Interpretation (s)": interpretation.get("latency_sec"),
                     "Recovery provider": interpretation.get("provider", "not_recorded"),
                     "Recovery model": interpretation.get("model_id", "not_recorded"),
                     "Recovery device": interpretation.get("device", "not_recorded")})
    return pd.DataFrame(rows)


def per_case_table(records):
    import pandas as pd
    rows = []
    for row in records:
        reference = tokens(row["reference_text"])
        interpretation = row.get("interpretation") or {}
        conversation = row.get("conversation_text") or row.get("normalized") or row.get("transcript", "")
        rows.append({"id": row["id"], "status": row["status"], "reference": row["reference_text"],
                     "raw_asr": row.get("transcript", ""), "normalized": row.get("normalized", ""),
                     "conversation_text": conversation,
                     "interpretation_status": interpretation.get("status", "not_recorded"),
                     "recovery_provider": interpretation.get("provider", "not_recorded"),
                     "recovery_model_id": interpretation.get("model_id", "not_recorded"),
                     "recovery_revision": interpretation.get("revision", "not_recorded"),
                     "recovery_latency_sec": interpretation.get("latency_sec"),
                     "interpretation_reason": interpretation.get("reason"),
                     "interpretation_requires_confirmation": interpretation.get("requires_confirmation"),
                     "conversation_source": row.get("conversation_source"),
                     "raw_word_errors": edit_distance(reference, tokens(row.get("transcript", ""))) if row["status"] == "ready" else None,
                     "conversation_word_errors": edit_distance(reference, tokens(conversation)) if row["status"] == "ready" else None,
                     "reference_words": len(reference), "reply": row.get("reply", ""),
                     "error": row.get("error")})
    return pd.DataFrame(rows)


def recovery_table(records):
    """Count every selected case, retaining missing/failed recovery states."""
    import pandas as pd
    rows = [{"request_status": row["status"],
             "provider": (row.get("interpretation") or {}).get("provider", "not_recorded"),
             "model_id": (row.get("interpretation") or {}).get("model_id", "not_recorded"),
             "interpretation_status": (row.get("interpretation") or {}).get("status", "not_recorded"),
             "reason": (row.get("interpretation") or {}).get("reason") or "none",
             "requires_confirmation": bool((row.get("interpretation") or {}).get("requires_confirmation"))}
            for row in records]
    return pd.DataFrame(rows).groupby(list(rows[0]), dropna=False).size().rename("clips").reset_index()


def baseline_comparison(results, baseline):
    """Pair the same audio bytes; never imply a retrained ASR or semantic score."""
    import pandas as pd
    current = {row["id"]: row for row in results["records"]}
    previous = {row["id"]: row for row in baseline["records"]}
    if (len(current) != len(results["records"]) or len(previous) != len(baseline["records"])
            or set(current) != set(previous)):
        raise ValueError("Baseline must contain the same unique case IDs")
    if results["identity"]["manifest_sha256"] != baseline["identity"]["manifest_sha256"]:
        raise ValueError("Baseline must use the same frozen source manifest")
    rows = []
    for key, row in current.items():
        old = previous[key]
        if any(row[field] != old[field] for field in ("audio", "audio_sha256", "reference_text")):
            raise ValueError("Baseline audio/reference mismatch: " + key)
        reference = tokens(row["reference_text"])
        both_ready = row["status"] == old["status"] == "ready"
        current_text = row.get("conversation_text") or row.get("normalized") or row.get("transcript", "")
        previous_text = old.get("conversation_text") or old.get("normalized") or old.get("transcript", "")
        rows.append({"id": key, "baseline_status": old["status"], "current_status": row["status"],
                     "paired_ready": both_ready, "baseline_raw_asr": old.get("transcript", ""),
                     "current_raw_asr": row.get("transcript", ""),
                     "raw_asr_tokens_equal": tokens(row.get("transcript", "")) == tokens(old.get("transcript", "")),
                     "baseline_conversation_text": previous_text, "current_conversation_text": current_text,
                     "baseline_recovery_provider": (old.get("interpretation") or {}).get("provider", "not_recorded"),
                     "current_recovery_provider": (row.get("interpretation") or {}).get("provider", "not_recorded"),
                     "baseline_recovery_model": (old.get("interpretation") or {}).get("model_id", "not_recorded"),
                     "current_recovery_model": (row.get("interpretation") or {}).get("model_id", "not_recorded"),
                     "baseline_recovery_sec": (old.get("interpretation") or {}).get("latency_sec"),
                     "current_recovery_sec": (row.get("interpretation") or {}).get("latency_sec"),
                     "baseline_word_errors": edit_distance(reference, tokens(previous_text)) if both_ready else None,
                     "current_word_errors": edit_distance(reference, tokens(current_text)) if both_ready else None,
                     "reference_words": len(reference), "baseline_reply": old.get("reply", ""),
                     "current_reply": row.get("reply", "")})
    return pd.DataFrame(rows)


def plot_results(metrics, latencies, output):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#382c4d", "text.color": "#382c4d",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    colors = ["#c4b5fd", "#a78bfa", "#6d28d9"]
    stages = [label for label in ["Raw ASR", "Normalized", "Conversation interpretation"]
              if label in set(metrics["text_stage"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["WER (%)", "CER (%)"]):
        wide = metrics.pivot(index="corpus", columns="text_stage", values=metric)
        wide = wide.reindex(["All clips", *[name for name in wide.index if name != "All clips"]])
        wide[stages].plot.bar(ax=ax, color=colors[:len(stages)], rot=0)
        ax.set(ylabel=metric, xlabel="", title=f"{metric}: differences from reading prompt")
        ax.tick_params(axis="x", labelsize=8)
        ax.get_legend().remove()
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(stages), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "transcription-errors.png", dpi=160)
    plt.show()
    fig, ax = plt.subplots(figsize=(11, 4))
    positions = np.arange(len(latencies))
    for offset, column, color in [(-.18, "Reply ready (s)", colors[0]), (.18, "All feedback ready (s)", colors[1])]:
        values = latencies[column].astype(float)
        ax.bar(positions + offset, values, width=.36, label=column, color=color)
    ax.set_xticks(positions, latencies["id"], rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="Observed elapsed seconds", title="Each local request, including HTTP polling; no ai33")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "latencies.png", dpi=160)
    plt.show()


def escape(value):
    return html.escape(str(value))


def json_block(value):
    return "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def case_html(row, case, output, *, include_audio=True):
    pieces = [f'<article id="{escape(row["id"])}"><h3>{escape(row["id"])}</h3>',
              f'<p class="meta">{escape(row["corpus"])} · {escape(row["speaker_id"])} · '
              f'{case["duration_sec"]:.2f} s · status: {escape(row["status"])}</p>']
    if include_audio:
        path = checked_audio(row["audio"])
        relative = quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")
        media = mimetypes.guess_type(path.name)[0] or "audio/wav"
        pieces.append(f'<audio controls preload="none"><source src="{relative}" type="{escape(media)}"></audio>')
    fields = [("Published reading prompt", row["reference_text"]), ("Raw ASR", row.get("transcript", "")),
              ("Normalized text", row.get("normalized", "")),
              ("Text received by conversation model", row.get("conversation_text", "")),
              ("Bot reply", row.get("reply", ""))]
    for label, value in fields:
        pieces.append(f'<p><b>{label}</b><br>{escape(value) if value else "<em>Unavailable</em>"}</p>')
    interpretation = row.get("interpretation")
    if interpretation is not None:
        pieces.append('<p><b>Interpretation status</b>: '
                      + escape(interpretation.get("status", "not_recorded"))
                      + ' · reason: ' + escape(interpretation.get("reason") or "none")
                      + ' · confirmation required: '
                      + escape(bool(interpretation.get("requires_confirmation"))) + '</p>')
        pieces.append('<p class="meta"><b>Recovery model</b>: '
                      + escape(interpretation.get("provider", "not_recorded")) + ' · '
                      + escape(interpretation.get("model_id", "not_recorded")) + ' · '
                      + escape(interpretation.get("device", "not_recorded")) + ' · '
                      + escape(interpretation.get("latency_sec", "not_recorded")) + ' s</p>')
        pieces.append('<details><summary>Recorded recovery decision and effective text</summary>'
                      + json_block(interpretation) + '</details>')
    rewrite = row.get("rewrite") or {}
    pieces.append('<details><summary>Separate grammar/natural wording feedback</summary>' + json_block(rewrite) + '</details>')
    review = row.get("assistant_text_review")
    if review is not None:
        pieces.append('<div class="review"><b>Assistant text review — subjective; no listening or expert validation</b>'
                      + json_block(review) + '</div>')
    pieces.append('<p class="meta">Expert audio/semantic review: '
                  + escape((row.get("human_review") or {}).get("status", "pending")) + '</p>')
    if row.get("error"):
        pieces.append('<div class="error"><b>Recorded failure</b>' + json_block(row["error"]) + '</div>')
    provenance = {key: row.get(key) for key in ["models", "qwen_runtime", "reply_diagnostics", "assessment", "transcript_clarity", "latencies"]}
    pieces.append('<details><summary>Per-case model, timing and diagnostic details</summary>' + json_block(provenance) + '</details>')
    pieces.append(f'<p class="meta"><a href="{escape(case["source_url"])}">Original source</a> · '
                  + escape((case.get("license") or {}).get("id", "See source terms")) + '</p></article>')
    return "".join(pieces)


INTRO = (
    "This is a fixed exploratory test of Vietnamese speakers reading English through the current local demo. "
    "Raw ASR, recorded normalization (when executed), conversational interpretation, grammar suggestions and bot replies are shown separately. "
    "A model interpretation is inferred text, not a verified transcript or a pronunciation reference. "
    "The published reading prompt is an intended reference, not an expert transcription of every realized word. "
    "WER/CER count differences from that prompt; they do not prove pronunciation or conversational understanding. "
    "The four Speech Accent Archive clips share one paragraph. The six L2-ARCTIC clips share one held-out speaker. "
    "Ten clips therefore are not ten independent speakers or a broad conversational benchmark."
)


def build_html(manifest, results, sources, metrics, latencies, output, baseline=None):
    ready = sum(r["status"] == "ready" for r in results["records"])
    cases = {case["id"]: case for case in manifest["cases"]}
    pieces = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
              '<meta name="viewport" content="width=device-width,initial-scale=1">',
              '<title>Vietnamese English: current pipeline test</title>',
              '<style>body{margin:0;background:#faf8ff;color:#352b43;font:16px/1.55 system-ui,sans-serif}'
              'main{max-width:1160px;margin:30px auto;padding:0 22px}h1,h2{color:#6d28d9}'
              'h1{font-size:30px}h2{margin-top:32px}h3{margin-top:0}article,.panel{background:white;'
              'border:1px solid #e7ddfa;border-radius:14px;padding:20px;margin:18px 0}'
              'table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;'
              'padding:9px;border-bottom:1px solid #e8e0f3;vertical-align:top}th{background:#f4eeff}'
              '.scroll{overflow-x:auto}img{max-width:100%;height:auto}audio{width:min(100%,650px)}'
              'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f4fc;padding:12px;border-radius:8px;font-size:12px}'
              '.meta{font-size:13px;color:#746584}.review{border-left:3px solid #8b5cf6;padding:10px;margin:15px 0}'
              '.error{color:#9b2525}summary{cursor:pointer;padding:8px 0}a{color:#6d28d9}b{font-weight:650}'
              '</style></head><body><main>',
              '<h1>Vietnamese-accented English: current pipeline</h1>',
              '<p>' + escape(INTRO) + '</p>',
              f'<p><b>{ready}/{len(results["records"])} requests completed</b> · '
              f'{sources.speaker.nunique()} speakers · {int(sources.reference_words.sum())} intended words · '
              f'{sources.seconds.sum():.1f} seconds of audio. Run status: {escape(results["status"])}.</p>',
              '<div class="panel"><b>Inference protocol</b><p>' + escape(manifest.get("protocol", "")) + '</p>'
              '<p>Each input has empty conversation history. References were not passed to ASR, recovery or Qwen. '
              'No training, ai33 synthesis or external audio submission occurs in this evaluation.</p></div>',
              '<h2>Sources and fixed selection</h2><p>' + escape(manifest.get("selection", "")) + '</p>',
              '<div class="scroll">' + sources.round(3).to_html(index=False, escape=True) + '</div>',
              '<details><summary>Downloads excluded before inference</summary>'
              + json_block(manifest.get("excluded_downloads", [])) + '</details>',
              '<h2>Transcription diagnostics</h2><p>Micro WER/CER sum edit counts and reference lengths over ready requests. '
              'Failed requests remain visible and are excluded from denominators. Case and punctuation are ignored. '
              'Raw, normalized and conversation interpretation are pipeline stages, not a before/after training comparison. '
              'A lower prompt edit count alone does not establish that an inferred interpretation matches the recording.</p>',
              '<div class="scroll">' + metrics.round(3).to_html(index=False, escape=True) + '</div>',
              '<img src="transcription-errors.png" alt="Prompt edit counts by text stage and corpus">',
              '<h2>Recovery decisions, including failures</h2><p>Accepted means a guarded model proposal was used; '
              'it does not mean a listener confirmed its meaning. Unchanged is also not proof that ASR was correct.</p>',
              '<div class="scroll">' + recovery_table(results["records"]).to_html(index=False, escape=True) + '</div>',
              '<h2>Observed latency</h2><p>Reply ready and all feedback ready are measured from local request submission. '
              'They include polling and any request queue wait. They do not include ai33 voice generation. '
              'The first request is not assumed cold; longer clips naturally require more acoustic processing.</p>',
              '<div class="scroll">' + latencies.round(3).to_html(index=False, escape=True) + '</div>',
              '<img src="latencies.png" alt="Per-case reply and full feedback observed latency">',
              '<h2>Listen and inspect every case</h2><p>Audio players read original full local clips. '
              'Assistant text reviews, when present, compare text only; they are subjective and do not replace '
              'a listener checking the recording. Expert listening and semantic review remain pending.</p>']
    if baseline is not None:
        paired = baseline_comparison(results, baseline)
        pieces.extend(['<h2>Previous run versus current recovery pipeline</h2>',
            '<p>The same fixed audio IDs, audio SHA256 and reading prompts are verified before pairing. '
            'The runs differ in code/configuration; this is not an isolated training or causal comparison. '
            'Actual old and new ASR text is shown, including differences, and failed pairs remain visible.</p>',
            '<div class="scroll">' + paired.to_html(index=False, escape=True) + '</div>',
            '<details><summary>Baseline and current model/configuration provenance</summary>'
            + json_block({"baseline": baseline.get("identity"), "current": results.get("identity")}) + '</details>'])
    if (any("interpretation" in row for row in results["records"])
            and not any(row.get("normalization") for row in results["records"])):
        pieces.append('<div class="panel"><b>Normalization compatibility column</b><p>'
                      'This recovery run did not execute the earlier strict normalization operation. '
                      'The Normalized column is retained for compatibility and equals the raw ASR fallback. '
                      'It is not a second LLM pass. Compare Raw ASR with Conversation interpretation '
                      'to inspect the operation actually recorded.</p></div>')
    for row in results["records"]:
        pieces.append(case_html(row, cases[row["id"]], output))
    provenance = {key: results.get(key) for key in ["schema", "status", "started_at_utc", "finished_at_utc",
                  "identity", "identity_sha256", "voice_api_calls", "audio_sent_externally", "training_started",
                  "references_submitted_to_models", "runtime_source_changed", "manifest_changed", "limitations",
                  "assistant_review_metadata"]}
    pieces.append('<h2>Frozen run provenance</h2>' + json_block(provenance) + '</main></body></html>')
    (output / "report.html").write_text("".join(pieces), encoding="utf-8")


def build_notebook(manifest_path, results_path, output, notebook_path, *, execute=True, baseline_path=None):
    import nbformat
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    manifest_relative = manifest_path.relative_to(ROOT).as_posix()
    results_relative = results_path.relative_to(ROOT).as_posix()
    output_relative = output.relative_to(ROOT).as_posix()
    notebook = nbformat.v4.new_notebook(cells=[
        markdown('# Vietnamese-accented English: current pipeline test\n\n' + INTRO + '\n\n'
                 'These cells read recorded results and local audio only. They do not call models, APIs or training.'),
        code("from pathlib import Path\nimport json, sys\nimport pandas as pd\nfrom IPython.display import Audio, HTML, Markdown, display\n"
             "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'scripts/report_vi_voice_pipeline.py').is_file())\n"
             "sys.path.insert(0, str(ROOT))\n"
             "from scripts.report_vi_voice_pipeline import (attach_text_reviews, audit_sources, metric_table, latency_table, per_case_table, "
             "plot_results, case_html, checked_audio, recovery_table, baseline_comparison)\n"
             f"MANIFEST = ROOT/{manifest_relative!r}\nRESULTS = ROOT/{results_relative!r}\nOUT = ROOT/{output_relative!r}\n"
             "manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))\n"
             "results = attach_text_reviews(json.loads(RESULTS.read_text(encoding='utf-8')), RESULTS)\n"
             "print('Run:', results['status'], '|', results.get('started_at_utc'))\n"
             "sources = audit_sources(manifest, results, MANIFEST)\ndisplay(sources)"),
        markdown('## Source and protocol audit\n\nSelection is frozen before inference. Original full clips are used; '
                 'downloads longer than the demo 30-second input limit were excluded rather than cropped. '
                 'The same paragraph is repeated across Speech Accent Archive speakers. TLV is held out from '
                 'ASR-v1 training but is part of an existing test corpus, not a newly independent dataset.'),
        code("print(manifest['selection'])\nprint(manifest['protocol'])\n"
             "display(sources.groupby('corpus').agg(clips=('id','size'), speakers=('speaker','nunique'), "
             "audio_seconds=('seconds','sum'), intended_words=('reference_words','sum')).round(3))\n"
             "display(pd.DataFrame(manifest.get('excluded_downloads', [])))\n"
             "display({key: results.get(key) for key in ['voice_api_calls','audio_sent_externally','training_started',"
             "'references_submitted_to_models','runtime_source_changed','manifest_changed']})\n"
             "display(results['identity']['config'])"),
        markdown('## WER, CER and exact prompt matches\n\nMicro averages sum edits and reference lengths over ready '
                 'requests. Failures remain in the report and are excluded from scoring denominators. '
                 'Case and punctuation are ignored. Text stages are not a before/after training comparison. '
                 'Recovered text is inferred; a lower prompt edit count does not verify its spoken meaning. '
                 'When normalization metadata is empty, the Normalized compatibility column contains raw ASR '
                 'fallback; it does not represent another executed normalization/LLM operation. '
                 'Grammar suggestions and bot replies are not treated as ASR transcripts.'),
        code("metrics = metric_table(results['records'])\ndisplay(metrics.round(3))\n"
             "metrics.to_csv(OUT/'metrics.csv', index=False)\n"
             "cases = per_case_table(results['records'])\ndisplay(cases)\n"
             "cases.to_csv(OUT/'per-case.csv', index=False)\n"
             "recoveries = recovery_table(results['records'])\ndisplay(recoveries)\n"
             "recoveries.to_csv(OUT/'recovery-status.csv', index=False)"),
        markdown('## Response time and visualizations\n\nObserved times include local HTTP polling and queue wait. '
                 'No ai33 call is included. The first request is not assumed to be cold.'),
        code("latencies = latency_table(results['records'])\ndisplay(latencies.round(3))\n"
             "latencies.to_csv(OUT/'latencies.csv', index=False)\nplot_results(metrics, latencies, OUT)"),
        markdown('## Original audio, transcripts and actual bot replies\n\nEach player embeds the complete original '
                 'recording in this notebook. Published reference, raw ASR, normalized input, model recovery, grammar feedback and '
                 'actual bot reply remain separate. Assistant text reviews, when present, are subjective text-only '
                 'observations. They are not expert listening, semantic accuracy scores or pronunciation labels. '
                 'Expert audio/semantic review remains pending.'),
        code("by_id = {case['id']: case for case in manifest['cases']}\n"
             "for row in results['records']:\n"
             "    display(Markdown('### ' + row['id']))\n"
             "    display(Audio(filename=str(checked_audio(row['audio'])), embed=True))\n"
             "    display(HTML(case_html(row, by_id[row['id']], OUT, include_audio=False)))"),
        markdown('## Runtime provenance and limitations\n\nThe conversation model receives recognized text or a guarded '
                 'model interpretation, without listening to the audio itself. ASR and interpretation errors can '
                 'therefore change the subject of a fluent reply. No automatic '
                 'semantic accuracy score is derived from WER. These read-speech clips with empty history do not '
                 'establish natural multi-turn conversation quality.'),
        code("display(results['identity'])\n"
             "display(results.get('assistant_review_metadata', {}))\n"
             "display(pd.DataFrame([{'id': r['id'], 'models': r.get('models'), 'qwen_runtime': r.get('qwen_runtime'), "
             "'reply_diagnostics': r.get('reply_diagnostics'), 'error': r.get('error')} for r in results['records']]))\n"
             "for limitation in results.get('limitations', []):\n    print('-', limitation)"),
    ])
    if baseline_path is not None:
        baseline_relative = baseline_path.relative_to(ROOT).as_posix()
        notebook.cells.extend([
            markdown('## Same-audio previous-run comparison\n\nBoth runs must contain the same unique IDs, '
                     'audio bytes and intended prompts. This compares recorded pipeline versions, not a new ASR '
                     'training checkpoint or a controlled causal experiment. Failed pairs remain visible. '
                     'Prompt edit counts do not establish semantic correctness.'),
            code(f"BASELINE = ROOT/{baseline_relative!r}\n"
                 "baseline = json.loads(BASELINE.read_text(encoding='utf-8'))\n"
                 "paired = baseline_comparison(results, baseline)\ndisplay(paired)\n"
                 "paired.to_csv(OUT/'baseline-comparison.csv', index=False)\n"
                 "display({'baseline_config': baseline['identity']['config'], 'current_config': results['identity']['config']})\n"
                 "eligible = paired[paired['paired_ready']]\n"
                 "print('Paired ready:', len(eligible), '/', len(paired))\n"
                 "print('Raw ASR token sequences unchanged:', int(paired['raw_asr_tokens_equal'].sum()), '/', len(paired))\n"
                 "denominator = int(eligible['reference_words'].sum())\n"
                 "if denominator:\n"
                 "    for field in ['baseline_word_errors', 'current_word_errors']:\n"
                 "        print(field, ':', int(eligible[field].sum()), '/', denominator, '=', round(100*eligible[field].sum()/denominator, 3), '%')")])
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    nbformat.write(notebook, notebook_path)
    if execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(notebook, notebook_path)
    return notebook


def workspace_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise argparse.ArgumentTypeError("Report paths must stay within the workspace")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=workspace_path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=workspace_path, default=DEFAULT_OUT / "results.json")
    parser.add_argument("--output", type=workspace_path, default=DEFAULT_OUT)
    parser.add_argument("--notebook", type=workspace_path, default=ROOT / "notebooks/vi_voice_pipeline_validation.ipynb")
    parser.add_argument("--baseline", type=workspace_path, help="Optional frozen previous run with exactly the same audio cases")
    parser.add_argument("--no-execute", action="store_true", help="Write notebook cells without starting a local kernel")
    args = parser.parse_args(argv)
    manifest = load_json(args.manifest)
    results = attach_text_reviews(load_json(args.results), args.results)
    if results.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("Wait until evaluation completes before rendering final artifacts")
    baseline = load_json(args.baseline) if args.baseline else None
    if baseline is not None:
        if baseline.get("status") not in {"completed", "completed_with_errors"}:
            raise ValueError("Baseline must be a completed recorded run")
        audit_sources(manifest, baseline, args.manifest)
        paired = baseline_comparison(results, baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    sources = audit_sources(manifest, results, args.manifest)
    metrics = metric_table(results["records"])
    latencies = latency_table(results["records"])
    metrics.to_csv(args.output / "metrics.csv", index=False)
    latencies.to_csv(args.output / "latencies.csv", index=False)
    recovery_table(results["records"]).to_csv(args.output / "recovery-status.csv", index=False)
    if baseline is not None:
        paired.to_csv(args.output / "baseline-comparison.csv", index=False)
    per_case_table(results["records"]).to_csv(args.output / "per-case.csv", index=False)
    notebook = build_notebook(args.manifest, args.results, args.output, args.notebook,
                              execute=not args.no_execute, baseline_path=args.baseline)
    if args.no_execute:
        import matplotlib
        matplotlib.use("Agg")
        plot_results(metrics, latencies, args.output)
    build_html(manifest, results, sources, metrics, latencies, args.output, baseline=baseline)
    print(json.dumps({"notebook": str(args.notebook), "executed": not args.no_execute,
                      "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                      "report": str(args.output / "report.html"), "metrics": str(args.output / "metrics.csv"),
                      "run_status": results["status"], "cases": len(results["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
