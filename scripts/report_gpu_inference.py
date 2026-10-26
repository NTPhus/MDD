"""Render completed CPU/CUDA measurements; never load models or call an API."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"whisper": "Whisper small.en", "phoneme-B": "wav2vec2 B"}


def load_results(directory):
    directory = Path(directory)
    paths = {"models": directory / "models/results.json", "live": directory / "live/verification.json"}
    values = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    models, live = values["models"], values["live"]
    if models.get("status") != "completed" or live.get("status") != "completed":
        raise ValueError("Both saved measurements must be completed")
    ids = {row["id"] for row in models["cases"]}
    if len(ids) != len(models["cases"]):
        raise ValueError("Duplicate audio case IDs")
    for name, rows in models["predictions"].items():
        if len(rows) != len(ids) or {row["id"] for row in rows} != ids:
            raise ValueError("Incomplete paired recordings: " + name)
        if abs(sum(row["elapsed_sec"] for row in rows) - models["summaries"][name]["total_sec"]) > .001:
            raise ValueError("Summary timing differs from per-case measurements")
    if models.get("training_started") is not False or live.get("training_started") is not False:
        raise ValueError("This report is for inference only")
    if models.get("remote_api_calls") != 0 or live.get("external_api_calls") != 0:
        raise ValueError("This local-only report cannot include external API calls")
    return {**values, "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                for key, path in paths.items()}}


def latency_chart(bundle, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = bundle["models"]["summaries"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, (key, label) in zip(axes, MODEL_NAMES.items()):
        values = [results[key + "-" + mode]["total_sec"] for mode in ("cpu", "cuda")]
        bars = axis.bar(["CPU", "CUDA"], values, color=["#b6a9ca", "#7546c8"], width=.55)
        axis.bar_label(bars, labels=[f"{value:.3f} s" for value in values], padding=4)
        axis.set_title(label)
        axis.set_ylabel("Total inference time · 6 clips (seconds)")
        axis.set_ylim(0, max(values) * 1.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Warm, paired development recordings · CPU then CUDA")
    fig.savefig(target, dpi=160)
    plt.close(fig)


def render(bundle, directory):
    directory = Path(directory)
    chart = directory / "cpu_cuda_latency.png"
    latency_chart(bundle, chart)
    models, live = bundle["models"], bundle["live"]
    summaries = models["summaries"]
    summary_rows = "".join(
        f"<tr><td>{label}</td><td>{summaries[key+'-cpu']['total_sec']:.4f}</td>"
        f"<td>{summaries[key+'-cuda']['total_sec']:.4f}</td>"
        f"<td>{summaries[key+'-speedup']:.3f}×</td></tr>" for key, label in MODEL_NAMES.items())
    jobs = "".join(
        f"<tr><td>{html.escape(row['case'])}</td><td>{html.escape(row['pronunciation_model'])}</td>"
        f"<td>{row['reply_sec']:.3f}</td><td>{row['voice_ready_sec']:.3f}</td>"
        f"<td>{row['full_elapsed_sec']:.3f}</td></tr>" for row in live["jobs"])
    cpu = {row["id"]: row for row in models["predictions"]["whisper-cpu"]}
    cuda = {row["id"]: row for row in models["predictions"]["whisper-cuda"]}
    changes = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(cpu[key]['text'])}</td>"
        f"<td>{html.escape(cuda[key]['text'])}</td></tr>" for key in models["text_changes_cpu_cuda"])
    crottc = [row for row in live["jobs"] if row["pronunciation_model"].startswith("crottc")]
    crottc_note = ""
    if crottc:
        runtime = crottc[-1]["pronunciation_runtime"]
        crottc_note = (f"CROTTC-IF dùng CPU: VRAM vật lý còn {runtime['cuda_physical_free_mib']:.0f} MiB, "
                       f"thấp hơn ngưỡng {runtime['min_free_cuda_mib']:.0f} MiB. "
                       "Đây là chuyển thiết bị có ghi nhận lý do, không phải chạy CUDA thành công.")
    body = f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>GPU inference · Kiểm chứng triển khai</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#292138;background:#fff}}
h1,h2{{color:#6435ad}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;text-align:left;border:1px solid #e6def0;vertical-align:top}}
th{{background:#f4effa}}img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}small{{color:#665d73}}
</style><h1>Đưa inference lên GPU</h1>
<p>Sáu bản thu phát triển được đo khi model đã nạp, cùng audio, CPU chạy trước CUDA. "Tăng tốc" dưới đây chỉ áp dụng từng model; không phải tốc độ toàn bộ hội thoại.</p>
<table><tr><th>Model</th><th>CPU · tổng giây</th><th>CUDA · tổng giây</th><th>Tỷ lệ CPU/CUDA</th></tr>{summary_rows}</table>
<img src="{chart.name}" alt="Thời gian CPU và CUDA trên cùng sáu bản thu">
<p>Whisper: CPU INT8 → CUDA INT8_FLOAT16; wav2vec2 B: CPU FP32 → CUDA FP16. Có {len(models['text_changes_cpu_cuda'])}/6 transcript khác nhau; chuỗi âm vị B thay đổi {len(models['phone_changes_cpu_cuda'])}/6. Khác precision có thể đổi kết quả; thử nghiệm này không chứng minh độ chính xác được cải thiện.</p>
<h2>Kiểm chứng demo thật</h2><p>Whisper chạy CUDA, B chạy CUDA FP16, Qwen chạy CUDA BF16 và Kokoro chạy CUDA. Thời gian tính từ lúc gửi bản thu. Full gồm cả các phần xử lý trong collapse và tạo giọng cho cả câu.</p>
<table><tr><th>Audio</th><th>Model phát âm</th><th>Có câu trả lời · giây</th><th>Voice sẵn · giây</th><th>Hoàn tất · giây</th></tr>{jobs}</table>
<p>{html.escape(crottc_note)} Sau lượt kiểm tra, model B mặc định đã được khôi phục: <strong>{live['default_B_restored']}</strong>.</p>
<p>VRAM cao nhất quan sát được: <strong>{live['observed_peak_gpu_used_mib']:.0f} MiB</strong> trong demo; <strong>{models['observed_peak_gpu_used_mib']:.0f} MiB</strong> trong benchmark model. Đây là mẫu lấy định kỳ, có thể bỏ sót đỉnh rất ngắn.</p>
<h2>Các transcript thay đổi</h2><table><tr><th>Audio</th><th>CPU</th><th>CUDA</th></tr>{changes}</table>
<h2>Giới hạn và nguồn số liệu</h2><p>Chỉ sáu bản thu phát triển; không phải tập test độc lập. Live chỉ có một lượt mỗi case, có khác biệt về tải máy, model đã nạp và cache. Không có đối chứng end-to-end CPU tương ứng; không suy ra tỷ lệ tăng tốc toàn luồng. Câu trả lời tự nhiên và lỗi ASR vẫn có thể sai dù chạy nhanh hơn.</p>
<p>Không chạy training và không gọi API trả phí. Báo cáo chỉ đọc hai JSON đã hoàn thành.</p>
<p><a href="models/results.json">Model results</a> · <a href="live/verification.json">Live verification</a></p>
<small>SHA256 models: <code>{bundle['sha256']['models']}</code><br>SHA256 live: <code>{bundle['sha256']['live']}</code></small></html>"""
    (directory / "report.html").write_text(body, encoding="utf-8")
    return chart


def make_notebook(target):
    import nbformat as nbf
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("# GPU inference · CPU/CUDA\n\nChỉ đọc kết quả đã lưu và hiển thị biểu đồ. Không nạp model, không train, không gọi API. Sáu audio phát triển; tỷ lệ tăng tốc từng model không phải tốc độ toàn pipeline."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, sys\nfrom IPython.display import display, Markdown, Image, HTML\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom scripts.report_gpu_inference import load_results\nOUT = ROOT / 'data/generated/gpu-inference-v1'\nbundle = load_results(OUT)\ndisplay(Markdown('**6 audio · CPU và CUDA đã hoàn thành.** Không có training hoặc API trả phí.'))"),
        nbf.v4.new_code_cell("summary = bundle['models']['summaries']\nfor key in ('whisper', 'phoneme-B'):\n    print(key, 'CPU:', summary[key+'-cpu']['total_sec'], 's; CUDA:', summary[key+'-cuda']['total_sec'], 's; speedup:', summary[key+'-speedup'], 'x')\ndisplay(Image(filename=str(OUT / 'cpu_cuda_latency.png')))"),
        nbf.v4.new_code_cell("cpu = {r['id']: r for r in bundle['models']['predictions']['whisper-cpu']}\ngpu = {r['id']: r for r in bundle['models']['predictions']['whisper-cuda']}\nfor key in bundle['models']['text_changes_cpu_cuda']:\n    print(key, '\\nCPU: ', cpu[key]['text'], '\\nCUDA:', gpu[key]['text'], '\\n')\nprint('B phone-sequence changes:', bundle['models']['phone_changes_cpu_cuda'])"),
        nbf.v4.new_code_cell("live = bundle['live']\nfor job in live['jobs']:\n    print(job['case'], job['pronunciation_model'], 'reply / voice / full (s):', job['reply_sec'], job['voice_ready_sec'], job['full_elapsed_sec'])\n    print('ASR:', job['asr_runtime']['device'], job['asr_runtime']['compute_type'])\n    print('Pronunciation:', job['pronunciation_runtime'])\nprint('Observed live peak VRAM:', live['observed_peak_gpu_used_mib'], 'MiB')\nprint('B default restored:', live['default_B_restored'])"),
        nbf.v4.new_code_cell("display(HTML((OUT / 'report.html').read_text(encoding='utf-8')))\nprint('Input SHA256:', bundle['sha256'])")]
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}}
    nbf.write(notebook, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/gpu-inference-v1")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/gpu_inference_comparison.ipynb")
    args = parser.parse_args()
    bundle = load_results(args.output)
    render(bundle, args.output)
    make_notebook(args.notebook)
    print(args.output / "report.html")
    print(args.notebook)


if __name__ == "__main__":
    main()
