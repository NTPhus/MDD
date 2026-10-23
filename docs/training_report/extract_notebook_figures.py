"""Extract saved notebook figures and readable evidence without executing cells.

Uses only the Python standard library. The original notebook bytes are never
modified. All image payloads are decoded and retained in their original format.
"""
from __future__ import annotations

import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOKS = PROJECT / "notebooks"
REPORT = Path(__file__).resolve().parent
FIGURES = REPORT / "figures"
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/svg+xml": ".svg"}
TOPICS = {
    "feedback_review_and_training": ("feedback_training", "Huấn luyện từ phản hồi"),
    "l2arctic_plus_dataset_audit": ("dataset_audit", "Kiểm tra dữ liệu L2-ARCTIC+"),
    "mdd_asr_vi_finetuning": ("wav2vec2_asr_training", "Fine-tuning ASR tiếng Việt"),
    "mdd_cascade_training_comparison": ("cascade_training_comparison", "So sánh huấn luyện cascade MDD"),
    "mdd_chat_error_checks": ("system_error_validation", "Kiểm tra lỗi trong phản hồi MDD"),
    "mdd_external_validation": ("external_validation", "Đánh giá trên dữ liệu ngoài"),
    "mdd_local_demo_comparison": ("local_model_comparison", "So sánh mô hình và demo cục bộ"),
    "mdd_projector_training_comparison": ("projector_training", "So sánh huấn luyện projector"),
    "mdd_projector_validation": ("projector_validation", "Đánh giá projector"),
    "mdd_pronunciation_calibration": ("pronunciation_calibration", "Hiệu chỉnh đánh giá phát âm"),
    "phonetic_candidates_validation": ("phonetic_candidates", "Đánh giá ứng viên âm vị"),
    "qwen_grammar_training": ("qwen_grammar_training", "Fine-tuning Qwen cho ngữ pháp"),
    "train_mdd_colab": ("colab_training_recipe", "Quy trình huấn luyện trên Colab"),
    "transcript_normalization_validation": ("transcript_normalization", "Chuẩn hóa bản chép lời"),
}


def as_text(value):
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return str(value) if value is not None else ""


def dimensions(data, mime):
    if mime == "image/png" and data[:8] == b"\x89PNG\r\n\x1a\n":
        return dict(zip(("width", "height"), struct.unpack(">II", data[16:24])))
    if mime == "image/jpeg":
        i = 2
        while i + 9 < len(data):
            if data[i] != 255:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            size = int.from_bytes(data[i:i + 2], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return {"width": int.from_bytes(data[i + 5:i + 7], "big"), "height": int.from_bytes(data[i + 3:i + 5], "big")}
            if size < 2:
                break
            i += size
    if mime == "image/svg+xml":
        try:
            root = ET.fromstring(data)
            return {key: root.attrib[key] for key in ("width", "height", "viewBox") if key in root.attrib}
        except ET.ParseError:
            return {}
    return {}


def plot_titles(source):
    titles = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            values = []
            if name in {"title", "set_title", "suptitle"} and node.args:
                values.append(node.args[0])
            values.extend(keyword.value for keyword in node.keywords if keyword.arg == "title")
            for value in values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    titles.append(value.value)
                else:
                    titles.append(ast.unparse(value))
    except (SyntaxError, ValueError):
        pass
    return list(dict.fromkeys(titles))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = []
    notebooks = []
    warnings = []
    digest_first = {}
    for notebook_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        topic, topic_label = TOPICS.get(notebook_path.stem, (notebook_path.stem, notebook_path.stem))
        relative_notebook = notebook_path.relative_to(PROJECT).as_posix()
        summary = {
            "notebook": relative_notebook,
            "notebook_absolute_path": str(notebook_path),
            "topic": topic,
            "topic_label_vi": topic_label,
            "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
            "nbformat": notebook.get("nbformat"),
            "kernel": notebook.get("metadata", {}).get("kernelspec", {}),
            "cells_count": len(notebook.get("cells", [])),
            "headings": [],
            "markdown_cells": [],
            "code_cells": [],
            "error_outputs": [],
            "output_mime_counts": Counter(),
            "execution_counts": [],
            "outputs_count": 0,
            "figures_count": 0,
            "linked_images": [],
        }
        nearest_markdown = ""
        nearest_heading = ""
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            source = as_text(cell.get("source", ""))
            cell_number = cell_index + 1
            titles = plot_titles(source)
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                nearest_markdown = source
                headings = re.findall(r"(?m)^\s{0,3}(#{1,6})\s+(.+)$", source)
                if headings:
                    nearest_heading = headings[-1][1].strip()
                for level, title in headings:
                    summary["headings"].append({"cell_index_0": cell_index, "cell_number_1": cell_number, "level": len(level), "title": title.strip()})
                summary["markdown_cells"].append({"cell_index_0": cell_index, "cell_number_1": cell_number, "source": source})
                for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source):
                    summary["linked_images"].append({"cell_index_0": cell_index, "target": target, "is_embedded_attachment": target.startswith("attachment:")})
            outputs = cell.get("outputs", [])
            if cell_type == "code":
                summary["execution_counts"].append({"cell_index_0": cell_index, "cell_number_1": cell_number, "execution_count": cell.get("execution_count")})
                readable_outputs = []
                for output_index, output in enumerate(outputs):
                    summary["outputs_count"] += 1
                    item = {"output_index_0": output_index, "output_type": output.get("output_type"), "execution_count": output.get("execution_count")}
                    if "text" in output:
                        item["text"] = as_text(output["text"])
                    if "name" in output:
                        item["stream_name"] = output["name"]
                    item["text_mime_data"] = {mime: as_text(value) for mime, value in output.get("data", {}).items() if mime.startswith("text/")}
                    for mime in output.get("data", {}):
                        summary["output_mime_counts"][mime] += 1
                    if output.get("output_type") == "error":
                        item.update({key: output.get(key) for key in ("ename", "evalue", "traceback")})
                        summary["error_outputs"].append({"cell_index_0": cell_index, "cell_number_1": cell_number, **item})
                    readable_outputs.append(item)
                summary["code_cells"].append({"cell_index_0": cell_index, "cell_number_1": cell_number, "execution_count": cell.get("execution_count"), "nearest_heading": nearest_heading, "source": source, "outputs": readable_outputs})

            def save_image(mime, value, output_index, origin, suffix=""):
                nonlocal manifest
                raw = as_text(value)
                try:
                    payload = raw.encode("utf-8") if mime == "image/svg+xml" else base64.b64decode(raw)
                except (ValueError, TypeError) as exc:
                    warnings.append({"notebook": relative_notebook, "cell_index_0": cell_index, "mime": mime, "error": str(exc)})
                    return
                file_name = f"{notebook_path.stem}__cell{cell_number:03d}__{origin}{output_index + 1:02d}{suffix}{IMAGE_TYPES[mime]}"
                destination = FIGURES / file_name
                destination.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                duplicate = digest_first.get(digest)
                digest_first.setdefault(digest, file_name)
                caption = f"{topic_label}: " + ("; ".join(titles) if titles else nearest_heading or f"hình tại ô {cell_number}")
                entry = {
                    "figure_id": len(manifest) + 1,
                    "notebook": relative_notebook,
                    "notebook_absolute_path": str(notebook_path),
                    "cell_index_0": cell_index,
                    "cell_number_1": cell_number,
                    "execution_count": cell.get("execution_count"),
                    "output_index_0": output_index if origin == "output" else None,
                    "origin": origin,
                    "mime": mime,
                    "file": destination.relative_to(REPORT).as_posix(),
                    "absolute_path": str(destination),
                    "byte_count": len(payload),
                    "dimensions": dimensions(payload, mime),
                    "sha256": digest,
                    "duplicate_of_file": f"figures/{duplicate}" if duplicate else None,
                    "topic": topic,
                    "topic_label_vi": topic_label,
                    "caption_suggestion_vi": caption,
                    "plot_titles": titles,
                    "nearest_heading": nearest_heading,
                    "nearest_markdown": nearest_markdown,
                    "cell_source": source,
                }
                manifest.append(entry)
                summary["figures_count"] += 1

            for output_index, output in enumerate(outputs):
                for mime, value in output.get("data", {}).items():
                    if mime in IMAGE_TYPES:
                        save_image(mime, value, output_index, "output")
            for attachment_index, (attachment_name, attachment) in enumerate(cell.get("attachments", {}).items()):
                for mime, value in attachment.items():
                    if mime in IMAGE_TYPES:
                        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", attachment_name)
                        save_image(mime, value, attachment_index, "attachment", "__" + safe_name)
        summary["output_mime_counts"] = dict(summary["output_mime_counts"])
        summary["executed_code_cells_count"] = sum(item["execution_count"] is not None for item in summary["execution_counts"])
        summary["unexecuted_code_cells_count"] = len(summary["execution_counts"]) - summary["executed_code_cells_count"]
        notebooks.append(summary)

    duplicates = defaultdict(list)
    for item in manifest:
        duplicates[item["sha256"]].append(item["file"])
    generated = datetime.now(timezone.utc).isoformat()
    manifest_result = {
        "generated_at_utc": generated,
        "method": "Read existing notebook JSON only; no cells executed. Image bytes decoded without resizing or editing.",
        "notebooks_directory": str(NOTEBOOKS),
        "figures_count": len(manifest),
        "unique_figures_count": len(duplicates),
        "mime_counts": dict(Counter(item["mime"] for item in manifest)),
        "duplicate_groups": [files for files in duplicates.values() if len(files) > 1],
        "warnings": warnings,
        "figures": manifest,
    }
    inventory_result = {
        "generated_at_utc": generated,
        "notebooks_count": len(notebooks),
        "outputs_count": sum(n["outputs_count"] for n in notebooks),
        "saved_figures_count": len(manifest),
        "error_outputs_count": sum(len(n["error_outputs"]) for n in notebooks),
        "warnings": warnings,
        "notebooks": notebooks,
    }
    for name, value in (("figures_manifest.json", manifest_result), ("notebook_inventory.json", inventory_result)):
        (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"notebooks": len(notebooks), "outputs": inventory_result["outputs_count"], "figures": len(manifest), "unique_figures": len(duplicates), "errors": inventory_result["error_outputs_count"], "warnings": warnings, "figure_counts": {n["notebook"]: n["figures_count"] for n in notebooks}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
