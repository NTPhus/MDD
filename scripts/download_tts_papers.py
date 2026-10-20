"""Download public TTS papers, verify PDF titles and pages, and record SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
    ("tts1", "2018_shen_tacotron2.pdf", "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions",
     "ICASSP 2018; arXiv preprint 2017", "https://arxiv.org/pdf/1712.05884", ["naturalttssynthesis", "melspectrogram"]),
    ("tts2", "2021_ren_fastspeech2.pdf", "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech",
     "ICLR 2021; arXiv preprint 2020", "https://arxiv.org/pdf/2006.04558", ["fastspeech2", "highquality"]),
    ("tts3", "2020_kong_hifi_gan.pdf", "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis",
     "NeurIPS 2020", "https://papers.neurips.cc/paper/2020/file/c5d736809766d46260d816d8dbc9eb44-Paper.pdf", ["hifigan", "generativeadversarialnetworks"]),
    ("tts4", "2021_kim_vits.pdf", "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech",
     "ICML 2021", "https://proceedings.mlr.press/v139/kim21f/kim21f.pdf", ["conditionalvariationalautoencoder", "adversariallearning"]),
    ("tts5", "2023_wang_vall_e.pdf", "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
     "arXiv preprint 2023", "https://arxiv.org/pdf/2301.02111", ["neuralcodeclanguagemodels", "synthesizers"]),
    ("tts6", "2022_ao_speecht5.pdf", "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing",
     "ACL 2022", "https://aclanthology.org/2022.acl-long.393.pdf", ["speecht5", "spokenlanguageprocessing"]),
    ("tts7", "2025_chen_f5_tts.pdf", "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching",
     "ACL 2025; arXiv preprint 2024", "https://aclanthology.org/2025.acl-long.313.pdf", ["f5tts", "flowmatching"]),
]


def inspect_pdf(path, title_fragments):
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise ValueError(f"Not a PDF: {path.name}")
    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    page_count = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if page_count is None:
        raise ValueError(f"No page count: {path.name}")
    first = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(path), "-"],
                           check=True, capture_output=True, text=True).stdout
    normalized = re.sub(r"[^a-z0-9]", "", first.lower())
    if not all(fragment in normalized for fragment in title_fragments):
        raise ValueError(f"First-page title mismatch: {path.name}")
    return {"status": "verified", "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "page_count": int(page_count[1]), "verified_title": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "papers/tts")
    parser.add_argument("--verify-only", action="store_true", help="Check local files against the saved manifest without network access")
    args = parser.parse_args()
    manifest_path = args.destination / "manifest.json"
    previous = {}
    if args.verify_only:
        previous = {item["filename"]: item for item in json.loads(manifest_path.read_text())}
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for reference, filename, title, publication, url, fragments in PAPERS:
        path = args.destination / filename
        resolved_url = previous.get(filename, {}).get("resolved_url", url)
        if not path.exists() and not args.verify_only:
            request = Request(url, headers={"User-Agent": "MDD-TTS-Paper-Reader/1.0"})
            with urlopen(request, timeout=60) as response:
                content = response.read(40 * 1024 * 1024 + 1)
                resolved_url = response.geturl()
            if len(content) > 40 * 1024 * 1024 or not content.startswith(b"%PDF-"):
                raise ValueError(f"Unexpected download for {filename}")
            path.write_bytes(content)
        metadata = inspect_pdf(path, fragments)
        if args.verify_only:
            for field in ("sha256", "size_bytes", "page_count"):
                if metadata[field] != previous[filename][field]:
                    raise ValueError(f"Manifest mismatch: {filename} {field}")
        entries.append({"reference": reference, "filename": filename, "title": title,
                        "publication": publication, "source_url": url, "resolved_url": resolved_url, **metadata})
        print(f"Verified {filename}: {metadata['page_count']} pages, {metadata['size_bytes']} bytes", flush=True)
    if not args.verify_only:
        manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(entries)} TTS papers.")


if __name__ == "__main__":
    main()
