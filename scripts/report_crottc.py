"""Read-only report for the frozen CROTTC/B pilot; no model or API calls."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_crottc import MODELS, canonical_hash, digest, summary

LABELS = {"cascade-B": "Local wav2vec2 B", "crottc-if": "Official CROTTC-IF"}


def live_examples(value):
    """Extract only transcript/timing fields from an optional saved live check."""
    output, seen = [], set()
    def walk(item):
        if isinstance(item, dict):
            if isinstance(item.get("fluency"), dict) and isinstance(item.get("transcript"), str):
                row = {"original_whisper_transcript": item.get("asr_raw_transcript", item["transcript"]),
                       "selected_transcript": item["transcript"], "asr_model": item.get("asr_model"),
                       "fluency": {key: item["fluency"].get(key) for key in ("status", "source", "method", "metrics", "measured_transcript")}}
                key = canonical_hash(row)
                if key not in seen:
                    seen.add(key); output.append(row)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return output


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if results.get("status") not in {"completed", "partial_errors"}:
        raise ValueError("Wait for the recorded inference run to finish")
    if results["manifest_sha256"] != canonical_hash(manifest):
        raise ValueError("Pilot manifest identity differs")
    if results["summary"] != summary(manifest, results["predictions"]):
        raise ValueError("Stored summary differs from recorded per-case counts")
    live_path = ROOT / "data/generated/crottc-live-20261023/verification.json"
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.is_file() else None
    notes_path = directory / "execution_notes.json"
    return {"directory": directory, "manifest": manifest, "results": results,
            "results_sha256": digest(directory / "results.json"),
            "live_examples": live_examples(live) if live else [],
            "live_verification_sha256": digest(live_path) if live else None,
            "execution_notes": json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.is_file() else None}


def research_rows(bundle):
    rows = []
    for model, values in bundle["results"]["summary"]["models"].items():
        for inventory in ("raw", "shared_inventory"):
            score = values[inventory]
            rows.append({"model": model, "inventory": inventory, **score["counts"], **score["metrics"]})
    return rows


def ui_rows(bundle):
    return [{"model": model, **values["ui_review"], "inference_sec": values["inference_sec"]}
            for model, values in bundle["results"]["summary"]["models"].items()]


def findings(bundle):
    values = bundle["results"]["summary"]["models"]
    percent = lambda value: f"{value:.2%}" if value is not None else "undefined"
    output = []
    for model in MODELS:
        research, review = values[model]["raw"], values[model]["ui_review"]
        output.append(f"{LABELS[model]}: raw phone-detection F1 {percent(research['metrics']['f1'])}, "
                      f"perceived-phone error rate {percent(research['metrics']['per'])}; "
                      f"{research['counts']['PER_I']} inserted phones against {research['counts']['PER_N']} expert-perceived phones. "
                      f"Optional review hints: precision {percent(review['hint_precision'])}, recall {percent(review['hint_recall'])}, "
                      f"false-positive rate {percent(review['hint_false_positive_rate'])}; "
                      f"{review['calibrated_decided_words']} calibrated decisions.")
    output.append("Review-list F1 alone can hide a tradeoff between finding more errors and flagging more correctly spoken words. The false-positive count and unknown pretraining overlap must be considered before choosing a demo default.")
    return output


def case_rows(bundle):
    rows = []
    for case in bundle["manifest"]["cases"]:
        expert = case.get("annotation")
        for model in MODELS:
            result = bundle["results"]["predictions"].get(model, {}).get(case["id"], {})
            fields = result.get("ui_assessment", {})
            hints = fields.get("sound_candidates", {})
            rows.append({"id": case["id"], "speaker": case["speaker_id"], "group": case["group"],
                         "stratum": case.get("stratum"), "reference_text": case["reference_text"], "model": model,
                         "status": result.get("status", "missing"), "error": result.get("error"),
                         "canonical": " ".join(expert["canonical_phones"]) if expert else None,
                         "expert_perceived": " ".join(expert["perceived_phones"]) if expert else None,
                         "predicted": " ".join(result.get("phonemes", [])),
                         "decoded_labels": " ".join(result.get("decoded_labels") or []),
                         "raw_f1": result.get("research_raw", {}).get("metrics", {}).get("f1"),
                         "raw_per": result.get("research_raw", {}).get("metrics", {}).get("per"),
                         "shared_f1": result.get("research_shared", {}).get("metrics", {}).get("f1"),
                         "shared_per": result.get("research_shared", {}).get("metrics", {}).get("per"),
                         "inference_sec": result.get("inference_sec"),
                         "review_status": hints.get("status"),
                         "review_reason": hints.get("reason"),
                         "review_words": ", ".join(word["text"] for word in hints.get("words", [])),
                         "calibrated": fields.get("assessment", {}).get("calibration", {}).get("applied"),
                         "audio_sha256": case["audio_sha256"]})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart(bundle, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x, width = np.arange(2), .34
    for axis, key, title in ((axes[0], "f1", "Phone error detection F1 · higher is better"),
                             (axes[1], "per", "Perceived phone error rate · lower is better")):
        for index, model in enumerate(MODELS):
            values = [bundle["results"]["summary"]["models"][model][inventory]["metrics"][key]
                      for inventory in ("raw", "shared_inventory")]
            values = [value * 100 if value is not None else float("nan") for value in values]
            bars = axis.bar(x + (index - .5) * width, values, width, color=("#9c92b7", "#6533a5")[index], label=LABELS[model])
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
        axis.set(xticks=x, xticklabels=["Original inventory", "Shared 39-phone inventory"], title=title, ylabel="Percent")
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False)
    figure.suptitle("24-case stratified exploratory pilot · project protocol, not a paper score", fontsize=11)
    figure.tight_layout()
    figure.savefig(directory / "research_metrics.png", dpi=170)
    plt.close(figure)


def render(bundle, directory):
    import pandas as pd
    esc = html.escape
    research, reviews, cases = research_rows(bundle), ui_rows(bundle), case_rows(bundle)
    write_csv(directory / "research_metrics.csv", research)
    write_csv(directory / "ui_review_metrics.csv", reviews)
    write_csv(directory / "per_case.csv", cases)
    chart(bundle, directory)
    panels = []
    for case in bundle["manifest"]["cases"]:
        rows = [row for row in cases if row["id"] == case["id"]]
        body = [f'<p><b>Intended reference:</b> {esc(case["reference_text"])}</p>']
        if case.get("annotation"):
            body.append(f'<p><b>Expert perceived phones:</b> {esc(" ".join(case["annotation"]["perceived_phones"]))}</p>')
        else:
            body.append('<p>No expert phone/error labels: outputs are development examples, with no accuracy score.</p>')
        for row in rows:
            body.append(f'<article><b>{esc(LABELS[row["model"]])}</b><p>{esc(row["predicted"] or row["error"] or "No phones")}</p>'
                        f'<p>Review words: {esc(row["review_words"] or "None suggested / unavailable")} · status: {esc(str(row["review_status"]))}</p>'
                        f'<small>Inference seconds: {row["inference_sec"]}; calibrated decisions enabled: {row["calibrated"]}</small></article>')
        panels.append(f'<details><summary>{esc(case["id"])} · {esc(case["group"])}</summary>{"".join(body)}</details>')
    metric_df = pd.DataFrame(research)[["model", "inventory", "UTTERANCES", "precision", "recall", "f1", "per", "word_precision", "word_recall", "word_f1", "frr", "far", "DIAG_EXCLUDED"]]
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CROTTC-IF versus B pilot</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;background:#f8f6fc;color:#30273d}}h1,h2{{color:#6533a5}}section,details{{background:white;border:1px solid #e3deed;border-radius:10px;margin:16px 0;padding:18px}}summary{{cursor:pointer;font-weight:650}}article{{background:#f5f2fa;border-radius:8px;padding:12px;margin:12px 0}}article p{{overflow-wrap:anywhere}}table{{border-collapse:collapse;font-size:12px}}td,th{{border-bottom:1px solid #e2ddeb;padding:8px;text-align:left}}.table{{overflow:auto}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>CROTTC-IF versus local wav2vec2 B</h1><p>{bundle["results"]["status"]} · 24 expert-labelled test clips, six speakers, and three user development recordings. Full audio only; no ASR, Qwen, TTS, training, reference injection or threshold fitting.</p>
<section><h2>What was recorded</h2>{''.join('<p>' + esc(statement) + '</p>' for statement in findings(bundle))}</section>
<section><h2>Research phoneme metrics</h2><p>{bundle["results"]["summary"]["paired_completed_expert_cases"]}/24 paired expert cases. Raw inventory and shared-inventory scores are both shown; neither is a reproduction of the paper's published F1. Explicit expert error events remain errors even when normalization collapses the phone labels.</p><div class="table">{metric_df.to_html(index=False, border=0)}</div><img src="research_metrics.png" alt="Phone error detection F1 and PER, original and shared inventory"></section>
<section><h2>Actual optional review-list behaviour</h2><p>Both models use their actual demo assessment helper and the same CMU variant dictionary. Hint precision/recall compare tentative review words against manual word_index error labels. An absent hint is not an expert acceptance, and unavailable assessments are reported separately. These counts are distinct from research phone alignment, which attributes insertion gaps using canonical positions.</p><div class="table">{pd.DataFrame(reviews).to_html(index=False, border=0)}</div><p>Calibration is not fitted here. Review hints can exist while every calibrated decision abstains. No pronunciation pass/fail grade is inferred from these hints.</p></section>
<section><h2>Separate live Whisper / fluency verification</h2><p>This panel reads a separately saved live-system check; it is not part of the audio-only phoneme pilot.</p>{'<pre>' + esc(json.dumps(bundle["live_examples"], ensure_ascii=False, indent=2)) + '</pre>' if bundle["live_examples"] else '<p>Pending saved live verification. No live inference is triggered by this report.</p>'}</section>
<section><h2>Selection and limits</h2><p>Two cases from each speaker's lower manual-error-density half and two from its upper half are selected deterministically by hashed ID. This label-stratified 24-case pilot is exploratory and does not estimate full-corpus prevalence or overall accuracy. No thresholds or model choices are tuned on these cases.</p><p>The released CROTTC model was trained for L2-ARCTIC; overlap with our local held-out split is unknown. Scores therefore do not establish independent speaker generalization or fair pretraining-data control. B uses greedy CTC; CROTTC uses its official AR/CTC beam decoder. CROTTC has no phone times/posterior matrix; the UI uses sequence review, while B uses posterior-based assessment.</p><p>Shared normalization is applied equally to canonical/perceived/predicted sequences and explicit event labels: remove stress/distortion suffixes, AX→AH, unknown→[UNK]. Distortion distinctions are lost in this secondary view; primary raw results remain available. Unknown labels remain in PER/detection and are excluded only from diagnosis by the existing metric protocol.</p><p>User recordings have intended text only and receive no expert accuracy metrics. See every output below, including failures.</p><p><a href="research_metrics.csv">Research CSV</a> · <a href="ui_review_metrics.csv">Review-list CSV</a> · <a href="per_case.csv">All cases CSV</a></p></section>
<section><h2>Recorded identities</h2><p>Manifest: {bundle["results"]["manifest_sha256"]}<br>Results file SHA256: {bundle["results_sha256"]}</p><details><summary>Runtime and frozen model/source protocol</summary><pre>{esc(json.dumps({"runtime": bundle["results"]["runtime"], "models": bundle["manifest"]["models"], "selection": bundle["manifest"]["selection"], "protocol": bundle["manifest"]["protocol"], "execution_notes": bundle["execution_notes"]}, ensure_ascii=False, indent=2))}</pre></details></section><h2>Every recording</h2>{"".join(panels)}</html>'''
    (directory / "report.html").write_text(body, encoding="utf-8")


def notebook(bundle, destination):
    import nbformat as nb
    relative = bundle["directory"].relative_to(ROOT).as_posix()
    cells = [nb.v4.new_markdown_cell("# CROTTC-IF versus wav2vec2 B: frozen exploratory pilot\n\nAll cells read existing outputs; no model inference, APIs, training or threshold tuning. The user recordings have no expert phone labels.")]
    codes = [f'''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown, Image, Audio
ROOT = Path.cwd().resolve()
while not (ROOT / "scripts/report_crottc.py").is_file():
    if ROOT.parent == ROOT: raise RuntimeError("Open from the MDD workspace")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.report_crottc import load_bundle, research_rows, ui_rows, case_rows
OUT = ROOT / {relative!r}
bundle = load_bundle(OUT)
manifest, results = bundle["manifest"], bundle["results"]
display({{"status": results["status"], "manifest_sha256": results["manifest_sha256"], "expert_paired_cases": results["summary"]["paired_completed_expert_cases"]}})''',
'''display(Markdown("## Selection: fixed before inference"))
display(manifest["selection"])
display(pd.DataFrame([{key: case.get(key) for key in ("id", "speaker_id", "group", "stratum", "manual_error_density", "duration_sec")} for case in manifest["cases"]]))''',
'''display(Markdown("## Raw and shared-inventory research scores"))
with pd.option_context("display.max_columns", None): display(pd.DataFrame(research_rows(bundle)))
display(Image(filename=str(OUT / "research_metrics.png")))''',
'''display(Markdown("## Actual optional UI review hints, separately from calibrated decisions"))
with pd.option_context("display.max_columns", None): display(pd.DataFrame(ui_rows(bundle)))
display(Markdown("An absent review hint is not expert acceptance. Unavailable assessments and calibrated abstentions remain explicit. These are exploratory word-index review-list counts, not the research phone F1."))''',
'''per_case = pd.DataFrame(case_rows(bundle))
display(Markdown("## All 54 model/case outputs"))
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None): display(per_case)''',
'''display(Markdown("## Listen to three deterministic examples"))
examples = [manifest["cases"][0], manifest["cases"][4], next(case for case in manifest["cases"] if case["id"] == "user-road")]
for case in examples:
    display(Markdown(f"### {case['id']}\\n**Intended text:** {case['reference_text']}"))
    display(Audio(filename=str(ROOT / case["audio"]), embed=True))
    display(per_case.loc[per_case.id == case["id"], ["model", "expert_perceived", "predicted", "review_words", "raw_f1", "shared_f1", "inference_sec"]])''',
'''display(Markdown("## User recordings: development outputs, no expert accuracy claims"))
with pd.option_context("display.max_colwidth", None, "display.max_columns", None): display(per_case.loc[per_case.group == "user_development"])
display(Markdown("Intended text does not identify which sounds were actually mispronounced. Human phone/error labels would be needed to score these recordings."))''',
'''display(Markdown("## Provenance and limits"))
display(results["runtime"])
display(bundle["execution_notes"])
display(manifest["models"])
display(manifest["protocol"])
display(Markdown("This project protocol is not the paper's scorer. The 24 cases are a label-stratified pilot across six previously used test speakers. Released CROTTC pretraining overlap with this local split is unknown. No thresholds or labels were sent into decoding; gold labels are used only for frozen sampling and post-inference scoring. Calibration is not fitted on this pilot."))''',
'''display(Markdown("## Separate saved live-system verification: Whisper and descriptive fluency"))
if bundle["live_examples"]:
    display(bundle["live_examples"])
    display({"verification_sha256": bundle["live_verification_sha256"]})
else:
    display(Markdown("Pending saved live verification. Running this cell does not call models, APIs, microphones or TTS."))''']
    cells.extend(nb.v4.new_code_cell(code) for code in codes)
    document = nb.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nb.write(document, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/crottc_pronunciation_pilot.ipynb")
    args = parser.parse_args()
    directory, destination = args.input.resolve(), args.notebook.resolve()
    if not directory.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
        parser.error("Keep artifacts in the workspace")
    bundle = load_bundle(directory)
    render(bundle, directory)
    notebook(bundle, destination)
    print(json.dumps({"report": str(directory / "report.html"), "notebook": str(destination), "inference": False}))


if __name__ == "__main__":
    main()
