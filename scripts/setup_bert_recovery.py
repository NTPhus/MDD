"""Download and verify the pinned official BERT masked-language model.

This command does not train a model, read API keys, or install packages.  The
demo uses these files locally on the CPU; --smoke is an optional real forward
pass and needs additional RAM while the demo is running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/pretrained/bert-base-uncased"
MODEL_ID = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
# Git blob identifiers and the LFS SHA256 were read from the official repository
# metadata for this exact revision.  Git blob SHA1 includes its standard header.
FILES = {
    "config.json": {"bytes": 570, "git_blob": "45a2321a7ecfdaaf60a6c1fd7f5463994cc8907d"},
    "tokenizer.json": {"bytes": 466062, "git_blob": "949a6f013d67eb8a5b4b5b46026217b888021b88"},
    "tokenizer_config.json": {"bytes": 48, "git_blob": "e5c73d8a50df1f56fb5b0b8002d7cf4010afdccb"},
    "vocab.txt": {"bytes": 231508, "git_blob": "fb140275c155a9c7c5a3b3e0e77a9e839594a938"},
    "LICENSE": {"bytes": 11356, "git_blob": "f49a4e16e68b128803cc2dcea614603632b04eac"},
    "README.md": {"bytes": 10517, "git_blob": "40a2aaca31dd005eb5f6ffad07b5ffed0a31d1f6"},
    "model.safetensors": {
        "bytes": 440449768,
        "sha256": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
    },
}


def file_record(path: Path, expected: dict) -> dict:
    """Check against repository pins, rather than trusting a local manifest."""
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing or wrong-sized BERT file: {path.name}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {expected['bytes']}\0".encode("ascii"))
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            sha256.update(chunk)
            git_blob.update(chunk)
    actual_sha = sha256.hexdigest()
    if expected.get("sha256") and actual_sha != expected["sha256"]:
        raise ValueError(f"Official BERT SHA256 mismatch: {path.name}")
    if expected.get("git_blob") and git_blob.hexdigest() != expected["git_blob"]:
        raise ValueError(f"Official BERT Git blob mismatch: {path.name}")
    return {
        "sha256": actual_sha,
        "bytes": expected["bytes"],
        "url": BASE_URL + path.name,
        **({"git_blob": expected["git_blob"]} if expected.get("git_blob") else {}),
    }


def _save(record: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / "setup_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def download() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in FILES.items():
        path = MODEL_DIR / name
        try:
            files[name] = file_record(path, expected)
        except ValueError:
            temporary = path.with_name(path.name + ".part")
            print(f"Downloading {name} ({expected['bytes']:,} bytes)", flush=True)
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "MDD-BERT-setup/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            files[name] = file_record(temporary, expected)
            # The temporary filename must not leak into the recorded source URL.
            files[name]["url"] = BASE_URL + name
            temporary.replace(path)
        print(f"Verified {name}", flush=True)
    record = {
        "status": "downloaded",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "task": "masked-language-modeling",
        "device": "cpu",
        "dtype": "float32",
        "license": "Apache-2.0",
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "checksum_source": f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    _save(record)
    return record


def verify() -> dict:
    manifest_path = MODEL_DIR / "setup_manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("model_id") != MODEL_ID or record.get("revision") != REVISION:
        raise ValueError("BERT manifest does not match the pinned official revision")
    if record.get("status") not in {"downloaded", "ready"}:
        raise ValueError("BERT files are not marked downloaded")
    for name, expected in FILES.items():
        actual = file_record(MODEL_DIR / name, expected)
        recorded = record.get("files", {}).get(name, {})
        if any(recorded.get(key) != actual[key] for key in ("sha256", "bytes")):
            raise ValueError(f"BERT manifest checksum mismatch: {name}")
    print(f"Verified official {MODEL_ID} @ {REVISION}", flush=True)
    return record


def smoke() -> dict:
    """One offline CPU forward pass; this is readiness, not an ASR benchmark."""
    record = verify()
    import importlib.metadata
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32,
    ).to("cpu").eval()
    model.requires_grad_(False)
    loaded = time.perf_counter() - started
    encoded = tokenizer("The capital of France is [MASK].", return_tensors="pt")
    position = int((encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0])
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits[0, position]
    if not torch.isfinite(logits).all():
        raise ValueError("BERT smoke forward pass produced non-finite logits")
    ids = torch.topk(logits, 5).indices.tolist()
    record.update(
        status="ready",
        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
        smoke={
            "device": "cpu", "parameter_count": sum(p.numel() for p in model.parameters()),
            "load_sec": round(loaded, 4),
            "forward_sec": round(time.perf_counter() - started, 4),
            "masked_predictions": tokenizer.convert_ids_to_tokens(ids),
            "benchmark": False,
        },
        smoke_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _save(record)
    print(json.dumps({"status": record["status"], "smoke": record["smoke"]}, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download pinned public files; reuse verified existing files")
    parser.add_argument("--verify", action="store_true", help="Verify all file checksums offline")
    parser.add_argument("--smoke", action="store_true", help="Run one offline CPU masked-LM forward pass")
    args = parser.parse_args()
    if not any((args.download, args.verify, args.smoke)):
        parser.error("Select --download, --verify and/or --smoke")
    if args.download:
        download()
    if args.verify:
        verify()
    if args.smoke:
        smoke()


if __name__ == "__main__":
    main()
