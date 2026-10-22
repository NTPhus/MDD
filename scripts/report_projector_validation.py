"""Verified report for available checkpoints; full tests and generation pilots stay separate."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projector_data import load, read_json, read_rows, save
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest
from scripts.evaluate_projector_checkpoint import select_balanced
from scripts.published_feedback_metrics import summarize, paired_bootstrap
from scripts.validation_records import read_json_lines, word_counts


def verified_case(folder, expected, manifest):
    meta = read_json(folder / "metadata.json")
    rows = read_json_lines(folder / "predictions.jsonl")
    identity = meta["identity"]
    if (meta["status"] != "completed" or identity["dataset"] != manifest
            or len(rows) != len(expected) or identity["rows"] != len(expected)
            or digest(folder / "predictions.jsonl") != meta["predictions_sha256"]
            or [(r["utterance_id"], r["source_row_sha256"]) for r in rows] !=
               [(r["utterance_id"], r["source_row_sha256"]) for r in expected]):
        raise ValueError("Incomplete, altered or unpaired result: " + str(folder))
    for row, source in zip(rows, expected):
        if "reference_feedback" in row and row["reference_feedback"] != source["feedback"]:
            raise ValueError("Feedback reference changed")
        if "reference_text" in row and row["reference_text"] != source["reference_text"]:
            raise ValueError("Transcript reference changed")
    return meta, rows


def likelihood(rows):
    tokens = sum(r["target_tokens"] for r in rows)
    nll = sum(r["nll_sum"] for r in rows) / tokens
    return {"rows": len(rows), "target_tokens": tokens, "target_token_nll": nll, "perplexity": math.exp(nll)}


def transcription(rows):
    # Rescore immutable raw outputs, including foreign-script hallucinations.
    rows = [{**r, **word_counts(r["reference_text"], r["raw_text"])} for r in rows]
    words = sum(r["reference_words"] for r in rows)
    return {"rows": len(rows), "reference_words": words,
            "wer_percent": 100 * sum(r["word_errors"] for r in rows) / words,
            "exact_sentence_percent": 100 * sum(r["exact_sentence"] for r in rows) / len(rows),
            "eos_percent": 100 * sum(r["finished_with_eos"] for r in rows) / len(rows),
            "mean_generation_sec": sum(r["generation_sec"] for r in rows) / len(rows)}


def paired_nll(left, right, repetitions=2000):
    import numpy as np
    if [(r["utterance_id"], r["target_tokens"]) for r in left] != [(r["utterance_id"], r["target_tokens"]) for r in right]:
        raise ValueError("Likelihood bootstrap is not paired")
    speakers = sorted({r["speaker_id"] for r in left})
    groups = {s: [i for i, r in enumerate(left) if r["speaker_id"] == s] for s in speakers}
    rng = np.random.default_rng(42)
    changes = []
    for _ in range(repetitions):
        indices = [i for s in rng.choice(speakers, len(speakers), replace=True) for i in groups[s]]
        changes.append(sum(right[i]["nll_sum"] - left[i]["nll_sum"] for i in indices) / sum(left[i]["target_tokens"] for i in indices))
    return {"delta_nll_after_minus_before": likelihood(right)["target_token_nll"] - likelihood(left)["target_token_nll"],
            "ci95": np.percentile(changes, [2.5, 97.5]).tolist(), "clusters": len(speakers), "resampling_unit": "speaker"}


def report(settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    config = read_json(rooted(settings["projector_config"]))
    parts, manifest = load(config)
    root = rooted(settings["output_dir"])
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    figures, nll_results, transcript_results, feedback_results, speaker_results, examples = [], [], [], [], [], []
    benchmark_manifest = read_json(root / "benchmark_manifest.json") if (root / "benchmark_manifest.json").exists() else None
    if benchmark_manifest is not None and benchmark_manifest["settings"] != settings:
        raise ValueError("Report settings differ from the frozen benchmark")
    if benchmark_manifest is not None:
        if benchmark_manifest["projector_training"]["identity"]["config"] != config:
            raise ValueError("Projector settings changed after evaluation")
        for name, expected in benchmark_manifest["projector_training"]["best_weights"].items():
            if digest(rooted(config["run_root"]) / "alignment/best" / name) != expected:
                raise ValueError("Evaluated projector weights changed")
        source = ROOT / "scripts/evaluate_projector_checkpoint.py"
        if digest(source) != benchmark_manifest["evaluation_code_sha256"]:
            source = root / "source/evaluate_projector_checkpoint.py"
        if not source.exists() or digest(source) != benchmark_manifest["evaluation_code_sha256"]:
            raise ValueError("The benchmark's actual evaluation source is unavailable or changed")
        if digest(ROOT / "scripts/projector_inference.py") != benchmark_manifest["inference_code_sha256"]:
            raise ValueError("Inference code differs from the evaluated version")
    def checked_case(folder, expected):
        meta, rows = verified_case(folder, expected, manifest)
        if benchmark_manifest is None or any(meta["identity"].get(key) != value for key, value in benchmark_manifest.items()):
            raise ValueError("Case provenance differs from the benchmark manifest")
        return meta, rows
    state = {"status": "partial", "split_counts": manifest["counts"], "test_speakers": dict(Counter(r["speaker_id"] for r in parts["test"])),
             "projector_alignment": read_json(rooted(config["run_root"]) / "alignment/training_provenance.json"),
             "feedback_projector_stage": "not_trained", "test_used_for_checkpoint_selection": False,
             "transcript_metric": "Unicode words; casefold, normalize curly apostrophes, remove punctuation; score immutable raw text; WER can exceed 100%",
             "report_code_sha256": digest(Path(__file__)), "transcript_metric_code_sha256": digest(ROOT / "scripts/validation_records.py"),
             "limitations": ["Projector has transcript alignment only; no P1 feedback-stage weights exist.",
                 "Random baseline uses a reproducible new initialization, not an exact saved step-0 checkpoint.",
                 "Transcript NLL is teacher forced and is not a transcription accuracy score.",
                 "Generation metrics are balanced pilots, not a full 900-row generation benchmark.",
                 "Word-key F1 and lexical ROUGE measure agreement with published feedback, not phone F1 or pedagogical correctness.",
                 "Oracle Qwen uses gold phone labels and is diagnostic only, not a deployable audio pipeline.",
                 "Speaker holdout allows shared read prompts; one training seed; 6 held-out speakers.",
                 "Training feedback is authors' GPT-4o supervision; test feedback is authors' human-checked reference."]}
    def save_figure(name):
        plt.tight_layout()
        for extension in ("png", "svg"):
            plt.savefig(output / (name + "." + extension), dpi=150)
        plt.close()
        figures.append(name + ".png")
    dataset_rows = [{"split": role, "speaker": speaker, "utterances": len([r for r in values if r["speaker_id"] == speaker])}
                    for role, values in parts.items() for speaker in sorted({r["speaker_id"] for r in values})]
    pd.DataFrame(dataset_rows).to_csv(output / "dataset_counts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    pd.DataFrame(dataset_rows).pivot(index="speaker", columns="split", values="utterances").fillna(0).plot.bar(stacked=True, ax=axes[0])
    axes[0].set_title("Frozen speaker-disjoint split")
    history = read_rows(rooted(config["run_root"]) / "alignment/training_history.jsonl")
    history = sorted({r["step"]: r for r in history}.values(), key=lambda r: r["step"])
    axes[1].plot([r["step"] for r in history], [r["loss"] for r in history], alpha=.5, label="Train transcript loss")
    validation = [r for r in history if "validation_loss" in r]
    axes[1].plot([r["step"] for r in validation], [r["validation_loss"] for r in validation], "o-", label="Validation transcript loss")
    axes[1].set(xlabel="Optimizer updates", ylabel="Token cross entropy", title="Actual alignment training")
    axes[1].legend()
    save_figure("data_and_training")
    cascade = read_json(rooted(config["cascade_config"]))
    audio_results = []
    for stage in ("A", "B"):
        folder = rooted(cascade["report_root"]) / "audio" / stage
        if not (folder / "metrics.json").exists():
            continue
        audio = read_json(folder / "metrics.json")
        predictions = read_rows(folder / "predictions.jsonl")
        if (audio["utterance_count"] != 900 or audio["split"] != "cascade_test"
                or audio["limited_technical_subset"] or audio["purpose"] != "scientific_training"
                or digest(folder / "predictions.jsonl") != audio["prediction_sha256"]
                or audio["checkpoint_sha256"] != digest(rooted(cascade["run_root"]) / stage / "best_checkpoint/model.safetensors")
                or {r["utterance_id"] for r in predictions} != {r["utterance_id"] for r in parts["test"]}
                or audio["protocol"]["files"]["protocol"]["sha256"] != manifest["protocol_sha256"]):
            raise ValueError("Invalid full audio test")
        audio_results.append({"case": stage, "rows": 900, **{key + "_percent": value * 100 for key, value in audio["metrics"].items() if value is not None}})
        for speaker, values in audio["by_speaker"].items():
            speaker_results.append({"case": "audio_" + stage, "speaker": speaker, "rows": values["counts"]["UTTERANCES"],
                                    "phone_f1_percent": values["metrics"]["f1"] * 100})
    state["audio_full_test"] = audio_results
    if audio_results:
        pd.DataFrame(audio_results).to_csv(output / "audio_full_test.csv", index=False)
        pd.DataFrame([r for r in speaker_results if r["case"].startswith("audio_")]).pivot(index="speaker", columns="case", values="phone_f1_percent").plot.bar(figsize=(10, 4), rot=0)
        plt.ylabel("Phone error detection F1 (%) — 150 utterances per speaker")
        save_figure("audio_by_speaker")
    nll_rows = {}
    for case in ("transcript_nll_random", "transcript_nll_aligned", "transcript_nll_muted"):
        folder = root / case
        if not (folder / "metadata.json").exists() or read_json(folder / "metadata.json")["status"] != "completed":
            continue
        meta, rows = checked_case(folder, parts["test"])
        if meta["identity"]["purpose"] != "full_held_out_teacher_forced_nll":
            raise ValueError("Full transcript likelihood requires all 900 test rows")
        nll_rows[case] = rows
        nll_results.append({"case": case, **likelihood(rows)})
    state["transcript_full_nll"] = nll_results
    if len(nll_rows) == 3:
        state["alignment_delta"] = paired_nll(nll_rows["transcript_nll_random"], nll_rows["transcript_nll_aligned"])
        state["audio_ablation_delta"] = paired_nll(nll_rows["transcript_nll_muted"], nll_rows["transcript_nll_aligned"])
    transcript_samples = select_balanced(parts["test"], settings["transcription_per_speaker"], settings["seed"], "transcription")
    feedback_samples = select_balanced(parts["test"], settings["feedback_per_speaker"], settings["seed"], "feedback")
    for case in ("transcript_generation_random", "transcript_generation_aligned", "transcript_generation_muted"):
        folder = root / case
        if not (folder / "metadata.json").exists() or read_json(folder / "metadata.json")["status"] != "completed":
            continue
        _, rows = checked_case(folder, transcript_samples)
        transcript_results.append({"case": case, **transcription(rows)})
        for row in rows:
            examples.append({"case": case, **row, "reported_word_counts": word_counts(row["reference_text"], row["raw_text"])})
        for speaker in state["test_speakers"]:
            speaker_results.append({"case": case, "speaker": speaker, **transcription([r for r in rows if r["speaker_id"] == speaker])})
    state["transcript_generation_pilot"] = transcript_results
    if nll_results or transcript_results:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        if nll_results:
            pd.DataFrame(nll_results).set_index("case")["target_token_nll"].plot.bar(ax=axes[0], rot=15)
        axes[0].set_title("Full 900-row transcript NLL (lower is better)")
        if transcript_results:
            pd.DataFrame(transcript_results).set_index("case")["wer_percent"].plot.bar(ax=axes[1], rot=15)
        axes[1].set_title(f"Transcription pilot WER — {len(transcript_samples)} rows")
        save_figure("projector_transcription")
    feedback_scores, diagnostics = {}, []
    for case in ("feedback_projector_P0", "feedback_cascade_B_Q0", "feedback_cascade_B_Q1", "feedback_oracle_Q1",
                 "feedback_projector_cascade_adapter_transfer"):
        folder = root / case
        if not (folder / "metadata.json").exists() or read_json(folder / "metadata.json")["status"] != "completed":
            continue
        case_meta, rows = checked_case(folder, feedback_samples)
        if "extension_evaluation_code_sha256" in case_meta["identity"]:
            if digest(ROOT / "scripts/evaluate_projector_adapter_transfer.py") != case_meta["identity"]["extension_evaluation_code_sha256"]:
                raise ValueError("Adapter transfer source differs")
        summary, feedback_scores[case] = summarize(rows)
        feedback_results.append({"case": case, **summary})
        diagnostics.append({"case": case, "rows": len(rows),
            "invalid_json_or_schema": sum(not r["valid_json"] for r in rows),
            "generation_without_eos": sum(not r["finished_with_eos"] for r in rows),
            "markdown_fences_in_raw": sum(r["raw_text"].lstrip().startswith("```") for r in rows),
            "valid_but_unlisted_word_keys": sum(r["valid_json"] and bool(set(r["prediction"]) - set(r["word_order"])) for r in rows)})
        examples.extend({"case": case, **row} for row in rows)
        for speaker in state["test_speakers"]:
            speaker_results.append({"case": case, "speaker": speaker, **summarize([r for r in rows if r["speaker_id"] == speaker])[0]})
    state["feedback_generation_pilot"] = feedback_results
    state["feedback_failure_diagnostics"] = diagnostics
    if feedback_results:
        labels = {"feedback_projector_P0": "Projector P0", "feedback_cascade_B_Q0": "Cascade B + base Qwen",
                  "feedback_cascade_B_Q1": "Cascade B + Qwen LoRA", "feedback_oracle_Q1": "Oracle + Qwen LoRA",
                  "feedback_projector_cascade_adapter_transfer": "Projector + cascade LoRA"}
        pd.DataFrame(feedback_results).set_index("case")[["json_valid_percent", "feedback_word_f1_percent", "suggestion_rougeL_on_reference_error_rows"]].rename(
            index=labels, columns={"json_valid_percent": "Valid JSON (%)", "feedback_word_f1_percent": "Word-key F1 (%)",
                                  "suggestion_rougeL_on_reference_error_rows": "Suggestion ROUGE-L"}).plot.bar(figsize=(12, 4.5), rot=10)
        plt.title(f"Feedback pilot — SAME {len(feedback_samples)} held-out utterances; oracle is diagnostic")
        plt.ylabel("Percent / lexical agreement score")
        save_figure("feedback_pilot")
    if all(case in feedback_scores for case in ("feedback_cascade_B_Q0", "feedback_cascade_B_Q1")):
        state["cascade_qwen_pilot_delta"] = paired_bootstrap(feedback_scores["feedback_cascade_B_Q0"], feedback_scores["feedback_cascade_B_Q1"])
    raw_folder = root / "raw_audio_parity"
    if (raw_folder / "metadata.json").exists() and read_json(raw_folder / "metadata.json")["status"] == "completed":
        _, raw = checked_case(raw_folder, select_balanced(parts["test"], 1, settings["seed"], "raw_audio_parity"))
        state["raw_audio_parity"] = {"rows": len(raw), "all_exact_match": all(r["raw_cache_equal"] for r in raw)}
        examples.extend({"case": "raw_audio_parity", **r} for r in raw)
    for name, values in (("transcript_full_nll", nll_results), ("transcript_generation_pilot", transcript_results),
                         ("feedback_generation_pilot", feedback_results), ("feedback_failure_diagnostics", diagnostics), ("per_speaker", speaker_results)):
        if values:
            pd.DataFrame(values).to_csv(output / (name + ".csv"), index=False)
    save(output / "examples.json", examples)
    execution_path = root / "execution.json"
    if execution_path.exists() and read_json(execution_path)["status"] == "completed":
        if not (len(audio_results) == 2 and len(nll_results) == 3 and len(transcript_results) == 3
                and len(feedback_results) >= 4 and state.get("raw_audio_parity", {}).get("all_exact_match")):
            raise ValueError("Execution claims completion but required verified results are missing")
        state.update(status="completed_available_checkpoint_validation", execution=read_json(execution_path))
    transfer_path = root / "transfer_execution.json"
    if transfer_path.exists():
        if "feedback_projector_cascade_adapter_transfer" not in feedback_scores:
            raise ValueError("Transfer completion exists without verified predictions")
        state["projector_cascade_adapter_transfer"] = read_json(transfer_path)
    conclusions = ["Đã hoàn tất alignment, chưa có checkpoint task tuning P1. Không có lượt train mới trong pipeline kiểm thử."]
    if "projector_cascade_adapter_transfer" in state:
        conclusions.append("Có thử ghép Projector đã alignment với LoRA Qwen cascade đã train. Đây là chuyển giao sang đầu vào audio mới, không phải checkpoint P1 được train trên audio + feedback.")
    if len(audio_results) == 2:
        conclusions.append(f"Cùng 900 test: F1 âm vị A {audio_results[0]['f1_percent']:.2f}%, B {audio_results[1]['f1_percent']:.2f}%. "
                           "A/B đều đã train; các điểm này khác phép đo trên tập test cũ.")
    if len(nll_results) == 3:
        conclusions.append(f"NLL transcript: random {nll_results[0]['target_token_nll']:.3f}, aligned {nll_results[1]['target_token_nll']:.3f}, "
                           f"muted {nll_results[2]['target_token_nll']:.3f}. Alignment đã học dùng âm thanh, nhưng NLL không đo trực tiếp khả năng bắt lỗi.")
    aligned = next((r for r in transcript_results if r['case']=='transcript_generation_aligned'), None)
    if aligned:
        conclusions.append(f"Sinh transcript trên {aligned['rows']} mẫu: WER {aligned['wer_percent']:.2f}%, câu khớp hoàn toàn {aligned['exact_sentence_percent']:.2f}%. "
                           "Nếu còn cao, chưa thể coi acoustic alignment là ổn định dù loss giảm.")
    if feedback_results:
        conclusions.append(f"Phản hồi được thử trên cùng {len(feedback_samples)} mẫu cân bằng speaker. JSON đúng cấu trúc chưa bảo đảm chẩn đoán đúng âm hoặc lời khuyên đúng; xem raw output và nhãn cạnh nhau.")
        if not any(not r['feedback'] for r in feedback_samples):
            state['limitations'].append("All feedback pilot samples have reference errors; this pilot does not measure false alarms on fully correct speech.")
        conclusions.append("So sánh Qwen gốc/LoRA gồm cả khả năng tuân thủ schema: output sai schema bị chấm 0. Mức tăng feedback F1 không có nghĩa wav2vec2 nhận âm chính xác hơn.")
    state["conclusions_vi"] = conclusions
    state["next_manual_step"] = "Chưa chứng minh nhánh Projector sửa phát âm tốt. Người dùng có thể tự chạy stage feedback, sau đó benchmark P0/P1 đủ 900 mẫu; không chọn checkpoint hoặc tinh chỉnh bằng test này."
    save(output / "summary.json", state)
    tables = []
    for title, values in (("Audio: đủ 900 mẫu / mỗi model", audio_results), ("Projector: NLL transcript trên đủ 900 mẫu", nll_results),
                          (f"Sinh transcript: pilot {len(transcript_samples)} mẫu", transcript_results),
                          (f"Phản hồi lỗi: pilot {len(feedback_samples)} mẫu", feedback_results)):
        tables.append("<h2>" + html.escape(title) + "</h2>" + (pd.DataFrame(values).round(4).to_html(index=False) if values else "<p>Chưa hoàn tất.</p>"))
    if diagnostics:
        tables.append("<h2>Phân loại lỗi output</h2>" + pd.DataFrame(diagnostics).to_html(index=False))
    images = "".join('<img alt="Evaluation figure" src="data:image/png;base64,' + base64.b64encode((output / name).read_bytes()).decode() + '">' for name in figures)
    note = "Projector hiện chỉ học căn chỉnh audio → transcript. Chưa train bước phản hồi phát âm (P1). Điểm pilot không đại diện cho toàn bộ 900 test."
    (output / "report.html").write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Projector: kiểm thử checkpoint hiện có</title><style>body{font:16px system-ui;margin:2rem;color:#182333}h2{margin-top:2rem}'
        'table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:8px;border:1px solid #ccd5df}img{max-width:100%}'
        '.note{background:#fff2cd;padding:1rem;border-left:4px solid #ac7510}pre{white-space:pre-wrap}</style>'
        '<h1>wav2vec2 → Projector → Qwen3-0.6B</h1><p class="note">' + note + '</p><p>Chỉ suy luận và đánh giá; không khởi chạy train, không gọi API ngoài.</p>'
        + '<ul>' + ''.join('<li>' + html.escape(s) + '</li>' for s in conclusions) + '</ul>'
        + ''.join(tables) + images + '<h2>Phạm vi và giới hạn</h2><ul>' + ''.join('<li>' + html.escape(s) + '</li>' for s in state["limitations"])
        + '</ul><p>Raw outputs: ../&lt;case&gt;/predictions.jsonl. CSV và examples.json nằm cạnh report.</p></html>', encoding="utf-8")
    print(json.dumps({"status": state["status"], "report": str(output / "report.html"), "completed_feedback_cases": len(feedback_results)}, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/projector_validation.json")
    args = parser.parse_args()
    report(read_json(args.config))
