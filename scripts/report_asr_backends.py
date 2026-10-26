"""Render frozen offline ASR comparison results; no inference or network access."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_asr_backends import MODEL_IDS, aggregate, canonical_hash, digest

NAMES = {MODEL_IDS[0]: "wav2vec2 base", MODEL_IDS[1]: "wav2vec2 fine-tuned", MODEL_IDS[2]: "Whisper small.en"}
GROUPS = {"tlv_previously_observed_heldout": "TLV · 111 previously observed held-out clips",
          "saa_external_same_prompt": "SAA · 4 external speakers, same paragraph",
          "user_development": "User · 3 development recordings"}
COLORS = ("#a6a9b8", "#9473d2", "#562aaa")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest, results = read_json(directory / "manifest.json"), read_json(directory / "results.json")
    if results.get("status") != "completed":
        raise ValueError("Render the final report only after all comparison cases are completed")
    if canonical_hash(manifest) != results["manifest_sha256"]:
        raise ValueError("Inference manifest hash does not match the report inputs")
    ids = {case["id"] for case in manifest["cases"]}
    if len(ids) != len(manifest["cases"]):
        raise ValueError("Duplicate manifest case IDs")
    for model in MODEL_IDS:
        predictions = results["predictions"].get(model, {})
        if set(predictions) != ids:
            raise ValueError("Results must cover every fixed case exactly once: " + model)
        for case in manifest["cases"]:
            row = predictions[case["id"]]
            if row["status"] != "completed" or row["audio_sha256"] != case["audio_sha256"]:
                raise ValueError("Prediction audio provenance differs: " + case["id"])
    actual_groups = aggregate(manifest["cases"], results["predictions"])
    if actual_groups != results["groups"]:
        raise ValueError("Stored group metrics differ from the per-case counts")
    training_path = directory / "wav2vec2_training_provenance.json"
    training = read_json(training_path) if training_path.is_file() else None
    identities = {name: digest(directory / name) for name in ("manifest.json", "results.json")}
    if training_path.is_file():
        identities[training_path.name] = digest(training_path)
    return {"directory": directory, "manifest": manifest, "results": results, "training": training,
            "input_sha256": identities}


def metric_rows(bundle):
    rows = []
    for group, result in bundle["results"]["groups"].items():
        for model, row in result["models"].items():
            rows.append({"group": group, "model": model, "cases": row["cases"],
                         "strict_word_errors": row["strict"]["word_errors"],
                         "strict_reference_words": row["strict"]["reference_words"],
                         "strict_wer": row["strict"]["wer"],
                         "number_normalized_word_errors": row["number_normalized"]["word_errors"],
                         "number_normalized_reference_words": row["number_normalized"]["reference_words"],
                         "number_normalized_wer": row["number_normalized"]["wer"],
                         "substitutions": row["strict"]["substitutions"],
                         "deletions": row["strict"]["deletions"], "insertions": row["strict"]["insertions"],
                         "exact_matches": row["exact_matches"], "inference_seconds": row["inference_seconds"],
                         "mean_inference_seconds": row["inference_seconds"] / row["cases"],
                         "audio_seconds": row["audio_seconds"], "real_time_factor": row["real_time_factor"]})
    return rows


def case_rows(bundle):
    rows = []
    for case in bundle["manifest"]["cases"]:
        for model in MODEL_IDS:
            prediction = bundle["results"]["predictions"][model][case["id"]]
            rows.append({"id": case["id"], "group": case["group"], "speaker_id": case["speaker_id"],
                         "duration_sec": case["duration_sec"], "reference": case["reference_text"],
                         "reference_kind": case["reference_kind"], "model": model,
                         "raw_transcript": prediction["text"], "production_text": prediction["production_text"],
                         **prediction["strict"], "number_normalized_wer": prediction["number_normalized"]["wer"],
                         "inference_sec": prediction["inference_sec"], "audio_sha256": prediction["audio_sha256"]})
    return rows


def conclusions(bundle):
    statements = []
    for group, result in bundle["results"]["groups"].items():
        models = result["models"]
        base, trained, whisper = (models[model]["strict"]["wer"] for model in MODEL_IDS)
        change = (base - trained) * 100
        direction = "reduced" if change >= 0 else "increased"
        relative = (base - trained) / base * 100 if base else 0
        statements.append(f"{GROUPS.get(group, group)}: fine-tuning {direction} WER by {abs(change):.2f} percentage points "
                          f"versus its own wav2vec2 base ({base:.2%} → {trained:.2%}; signed relative error reduction {relative:.1f}%). "
                          f"Whisper small.en WER is {whisper:.2%} in this separate pretrained-model comparison.")
    statements.append("Fine-tuning benefit is measured against the same base model. A stronger separate model can still outperform a fine-tuned smaller-data system; this benchmark does not identify which architecture or training-data difference caused the gap.")
    return statements


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def charts(bundle, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    groups = list(bundle["results"]["groups"])
    labels = ["TLV held-out\n(previously observed)" if group.startswith("tlv") else "SAA external\n(same paragraph)" if group.startswith("saa") else "User\n(development)" for group in groups]
    x, width = np.arange(len(groups)), .24
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, metric, title in zip(axes, ("strict", "number_normalized"), ("Strict word error rate", "Secondary: number-normalized WER")):
        for index, model in enumerate(MODEL_IDS):
            values = [bundle["results"]["groups"][group]["models"][model][metric]["wer"] * 100 for group in groups]
            bars = axis.bar(x + (index - 1) * width, values, width=width, color=COLORS[index], label=NAMES[model])
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
        axis.set(xticks=x, xticklabels=labels, title=title, ylabel="WER (%) · lower is better")
        axis.set_ylim(0, max(axis.get_ylim()[1], 105))
        axis.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "wer_by_group.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, key, title in ((axes[0], "mean", "Mean ASR inference seconds per clip"),
                             (axes[1], "rtf", "ASR real-time factor · lower is faster")):
        for index, model in enumerate(MODEL_IDS):
            rows = [bundle["results"]["groups"][group]["models"][model] for group in groups]
            values = [row["inference_seconds"] / row["cases"] if key == "mean" else row["real_time_factor"] for row in rows]
            bars = axis.bar(x + (index - 1) * width, values, width=width, color=COLORS[index], label=NAMES[model])
            axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
        axis.set(xticks=x, xticklabels=labels, title=title)
        axis.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("CPU 2 threads; inference excludes model load and audio decode; Whisper includes native word timing", fontsize=10)
    figure.tight_layout()
    figure.savefig(directory / "asr_latency.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.4))
    comparison = MODEL_IDS[0] + " -> " + MODEL_IDS[1]
    for index, (key, color, label) in enumerate((("improved_cases", "#549c7a", "Improved"),
                                                ("same_cases", "#adb0bf", "Same"), ("worse_cases", "#d68c64", "Worse"))):
        values = [bundle["results"]["groups"][group]["paired_comparisons"][comparison][key] for group in groups]
        bars = axis.bar(x + (index - 1) * width, values, width=width, color=color, label=label)
        axis.bar_label(bars, padding=3)
    axis.set(xticks=x, xticklabels=labels, title="Fine-tuned wav2vec2 versus its own base: paired word-error counts", ylabel="Clips")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(directory / "fine_tuning_paired_cases.png", dpi=170)
    plt.close(figure)


def render_html(bundle, directory):
    import pandas as pd
    esc = html.escape
    metrics = pd.DataFrame(metric_rows(bundle))
    display_metrics = metrics[["group", "model", "cases", "strict_word_errors", "strict_reference_words", "strict_wer", "number_normalized_wer", "exact_matches", "mean_inference_seconds", "real_time_factor"]].copy()
    for column in ("strict_wer", "number_normalized_wer"):
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.2%}")
    for column in ("mean_inference_seconds", "real_time_factor"):
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.3f}")
    sections = []
    for case in bundle["manifest"]["cases"]:
        models = []
        for model in MODEL_IDS:
            row = bundle["results"]["predictions"][model][case["id"]]
            production = ("<p><b>Production formatting:</b> " + esc(row["production_text"]) + "</p>"
                          if row["production_text"] != row["text"] else "")
            models.append(f'<article><b>{esc(NAMES[model])}</b><p>{esc(row["text"])}</p>{production}'
                          f'<small>Strict WER {row["strict"]["wer"]:.2%} · S/D/I {row["strict"]["substitutions"]}/{row["strict"]["deletions"]}/{row["strict"]["insertions"]} '
                          f'· Number-normalized WER {row["number_normalized"]["wer"]:.2%} · {row["inference_sec"]:.3f} s</small></article>')
        sections.append(f'<details><summary>{esc(case["id"])} · {esc(case["group"])} · {case["duration_sec"]:.2f}s</summary>'
                        f'<p><b>Reference:</b> {esc(case["reference_text"])}</p><div class="models">{"".join(models)}</div></details>')
    training = bundle["training"]
    training_html = ""
    if training:
        config, dataset = training.get("config", {}), training.get("dataset", {})
        training_html = (f'<p>Existing wav2vec2 fine-tuning: {dataset.get("parts", {}).get("train", {}).get("count", "?")} training clips, '
                         f'{dataset.get("parts", {}).get("train", {}).get("hours", 0):.3f} hours, '
                         f'{config.get("epochs", "?")} epochs, train speakers {esc(str(config.get("train_speakers", [])))}; '
                         f'CNN frozen: {config.get("freeze_cnn")}; {training.get("trainable_parameters", "?")} trainable parameters. '
                         'No new training was run for this comparison.</p>'
                         '<p>Historical training logs evaluated on GPU with BF16 autocast; this comparison evaluates both CTC models on CPU FP32 with the same current strict tokenization. Small differences from historical WER values can follow numerical precision and tokenization. The original training logs are preserved; use the new paired numbers consistently.</p>')
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whisper small.en vs wav2vec2: frozen offline comparison</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#272238;background:#f8f6fc;max-width:1240px;margin:32px auto;padding:0 24px}}h1,h2{{color:#562aaa}}.card,details{{background:white;border:1px solid #e4deee;border-radius:12px;padding:18px;margin:16px 0}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #e8e3f0;text-align:left}}img{{max-width:100%;height:auto}}.models{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}article{{background:#f8f6fc;padding:12px;border-radius:8px}}small{{color:#655b73}}summary{{cursor:pointer;font-weight:600}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{.models{{grid-template-columns:1fr}}}}
</style><h1>Raw ASR: Whisper small.en vs wav2vec2</h1>
<p>{len(bundle["manifest"]["cases"])} full recordings · CPU 2 threads · no BERT/Qwen · no target text, history or hotwords supplied to ASR.</p>
<div class="card"><h2>What the recorded results show</h2>{''.join('<p>'+esc(item)+'</p>' for item in conclusions(bundle))}{training_html}</div>
<div class="card"><h2>Group metrics</h2><p>WER is micro-averaged edit counts, not mean clip WER. Strict WER is primary; number normalization is applied uniformly to reference and hypothesis for every backend.</p><div class="table">{display_metrics.to_html(index=False, border=0, escape=True)}</div>
<p><a href="metrics.csv">Metrics CSV</a> · <a href="per_case.csv">Every transcript and edit count CSV</a></p></div>
<div class="card"><img src="wer_by_group.png" alt="Strict and number-normalized WER by source group"><img src="asr_latency.png" alt="Inference latency and real-time factor"><img src="fine_tuning_paired_cases.png" alt="Paired fine-tuning changes"></div>
<div class="card"><h2>Limits and reproducibility</h2><p>TLV is one speaker and a previously observed speaker-and-prompt-held-out split, not a fresh unseen benchmark. SAA includes four speakers reading the same paragraph. The three user clips are previously inspected development cases. Keep these groups separate.</p>
<p>References are intended corpus/user reading text, not independently annotated realized words. Whisper pretraining overlap with these public corpora is unknown. Different architectures, pretraining data and decoders are compared: wav2vec2 greedy FP32 versus Whisper beam 5 INT8 with word timing. The experiment cannot isolate which difference caused the result.</p>
<p>WER evaluates transcript similarity; it does not validate pronunciation diagnosis, semantic correctness, Qwen replies or end-to-end latency. First-use overhead, model loading and decoding/resampling are recorded separately. Memory readings are sampled RSS and cumulative process peak, not isolated model maxima.</p>
<p>Manifest SHA256: <code>{esc(bundle["results"]["manifest_sha256"])}</code><br>Results file SHA256: <code>{bundle["input_sha256"]["results.json"]}</code></p>
<details><summary>Runtime, model weight hashes and split audit</summary><pre>{esc(json.dumps({"runtime": bundle["results"]["runtime"], "packages": bundle["results"]["package_versions"], "models": bundle["manifest"]["models"], "split_audit": bundle["manifest"]["split_audit"]}, ensure_ascii=False, indent=2))}</pre></details></div>
<h2>Every recording: raw model output</h2>{''.join(sections)}</html>'''
    (directory / "report.html").write_text(text, encoding="utf-8")


def notebook(bundle, destination):
    import nbformat as nbf
    relative = bundle["directory"].relative_to(ROOT).as_posix()
    cells = [nbf.v4.new_markdown_cell("# Whisper small.en vs wav2vec2\n\nFrozen raw-ASR results. Running these cells reads existing files and displays local audio; it does not run models, train, call APIs, or upload audio.")]
    cells.append(nbf.v4.new_code_cell(f'''from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display, Markdown, Image, Audio
ROOT = Path.cwd().resolve()
while not (ROOT / "scripts/compare_asr_backends.py").is_file():
    if ROOT.parent == ROOT: raise RuntimeError("Open this notebook from the MDD workspace")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from scripts.report_asr_backends import load_bundle, metric_rows, case_rows, conclusions
OUT = ROOT / {relative!r}
bundle = load_bundle(OUT)
manifest, results = bundle["manifest"], bundle["results"]
display(Markdown(f"**{{len(manifest['cases'])}} recordings; all three backends completed.** Manifest: `{{results['manifest_sha256']}}`"))'''))
    cells.append(nbf.v4.new_markdown_cell("## Group metrics\n\nMicro WER = summed substitutions, deletions and insertions / summed reference words. Strict WER is primary; the secondary score applies the same English number normalization to all models."))
    cells.append(nbf.v4.new_code_cell('''metrics = pd.DataFrame(metric_rows(bundle))
with pd.option_context("display.max_columns", None, "display.max_rows", None):
    display(metrics)
for statement in conclusions(bundle): display(Markdown(statement))'''))
    cells.append(nbf.v4.new_markdown_cell("## Visual comparison\n\nRaw ASR latency excludes model loading and audio decoding; Whisper includes native word timestamps."))
    cells.append(nbf.v4.new_code_cell('''for filename in ("wer_by_group.png", "asr_latency.png", "fine_tuning_paired_cases.png"):
    display(Image(filename=str(OUT / filename)))'''))
    cells.append(nbf.v4.new_markdown_cell("## Every recording and transcript\n\nWhisper `raw_transcript` is scored. `production_text` additionally spells a small set of unambiguous cardinal numbers for the pronunciation dictionary."))
    cells.append(nbf.v4.new_code_cell('''per_case = pd.DataFrame(case_rows(bundle))
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(per_case[["id", "group", "reference", "model", "raw_transcript", "production_text", "word_errors", "reference_words", "wer", "number_normalized_wer", "inference_sec"]])'''))
    cells.append(nbf.v4.new_markdown_cell("## Listen to three fixed examples\n\nAudio is embedded from the frozen local source files. Examples are selected by ID rather than chosen for favourable scores."))
    cells.append(nbf.v4.new_code_cell('''examples = [manifest["cases"][0]["id"], "user-stella", "user-road"]
for identifier in examples:
    case = next(item for item in manifest["cases"] if item["id"] == identifier)
    display(Markdown(f"### {identifier}\\n**Reference:** {case['reference_text']}"))
    display(Audio(filename=str(ROOT / case["audio"]), embed=True))
    display(per_case.loc[per_case.id == identifier, ["model", "raw_transcript", "wer", "inference_sec"]])'''))
    cells.append(nbf.v4.new_markdown_cell("## Provenance and limits\n\nTLV: one previously observed held-out speaker. SAA: four speakers reading one shared paragraph. User files: development cases. Corpus/user prompts are not verified realized-word annotations. Whisper's pretraining overlap is unknown. WER does not measure pronunciation diagnosis or Qwen dialogue quality."))
    cells.append(nbf.v4.new_code_cell('''display(Markdown("**Runtime and model loading:**"))
display(pd.DataFrame(results["runtime"]).T)
display(Markdown("**Split audit:**"))
display(manifest["split_audit"])
display(Markdown("**Model files and SHA256:**"))
display(manifest["models"])
if bundle["training"]:
    training = bundle["training"]
    display({"training_dataset": training.get("dataset"), "configuration": training.get("config"),
             "base_sha256": training.get("base_sha256"), "trained_sha256": training.get("results", {}).get("trained_sha256")})
    display(Markdown("Historical training-log evaluation used GPU BF16; this paired rerun uses CPU FP32 and current strict tokenization. Small WER differences from old logs are not additional training. Original logs are preserved."))
display(Markdown("Fine-tuning improvement is evaluated against the same wav2vec2 base. Comparing against Whisper tests a separate pretrained model and decoding configuration; it does not show that fine-tuning had no benefit."))'''))
    document = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbf.write(document, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/whisper_vs_wav2vec2.ipynb")
    args = parser.parse_args()
    directory = args.input.resolve()
    destination = args.notebook.resolve()
    if not directory.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
        parser.error("Report artifacts must remain in the workspace")
    bundle = load_bundle(directory)
    save_csv(directory / "metrics.csv", metric_rows(bundle))
    save_csv(directory / "per_case.csv", case_rows(bundle))
    charts(bundle, directory)
    render_html(bundle, directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook(bundle, destination)
    print(json.dumps({"report": str(directory / "report.html"), "notebook": str(destination),
                      "cases": len(bundle["manifest"]["cases"]), "inference": False, "input_sha256": bundle["input_sha256"]}))


if __name__ == "__main__":
    main()
