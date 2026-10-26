"""Download, verify and smoke-test the pinned official CROTTC inference bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crottc_pronunciation import (
    CROTTCPronunciation, FILE_PINS, MODEL_ID, RELATIVE, REVISION, file_record, verify, runtime_versions,
)


def save(record, root=ROOT):
    folder = Path(root).resolve() / RELATIVE
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "setup_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def download(root=ROOT):
    folder = Path(root).resolve() / RELATIVE
    bundle = folder / "bundle"
    files = {}
    for name, pin in FILE_PINS.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            files[name] = file_record(path, pin)
            continue
        except ValueError:
            pass
        print(f"Downloading {name} ({pin[0]:,} bytes)", flush=True)
        temporary = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "MDD-CROTTC-setup/1.0"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        files[name] = file_record(temporary, pin)
        temporary.replace(path)
    old = {}
    try:
        old = verify(root, require_ready=False)
    except (OSError, ValueError):
        pass
    record = {"status": old.get("status", "downloaded"), "model_id": MODEL_ID, "revision": REVISION,
              "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}", "files": files,
              "device": "cpu", "compute_type": "float32", "training": False,
              "runtime_packages": runtime_versions(root),
              "source_review": "Pinned source inspected; official AR beam decoder retained; weights_only enforced",
              "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    if old.get("smoke"):
        record["smoke"] = old["smoke"]
    save(record, root)
    return record


def install(root=ROOT):
    folder = Path(root).resolve() / RELATIVE
    destination = folder / "python"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "--no-deps", "--target", str(destination),
                    "-r", str(Path(root) / "requirements/crottc.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "--no-deps", "--target", str(destination),
                    "torchaudio==2.7.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)


def smoke(audio=None, root=ROOT):
    record = verify(root, require_ready=False)
    engine = CROTTCPronunciation(root, require_ready=False)
    try:
        engine.load()
        smoke_result = {"load_only": True, "runtime": engine.runtime, "training": False}
        if audio:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
            from math import gcd
            samples, rate = sf.read(audio, dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if rate != 16000:
                divisor = gcd(rate, 16000)
                samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
            result = engine.infer(samples)
            (Path(root) / RELATIVE / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            smoke_result.update(load_only=False, inference_sec=result["inference_sec"],
                                decoded_phone_count=len(result["phonemes"]),
                                unresolved_labels=result["unresolved_labels"],
                                reference_supplied_to_model=False)
        record.update(status="ready", smoke=smoke_result,
                      smoke_at_utc=datetime.now(timezone.utc).isoformat())
        save(record, root)
        return record
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audio", type=Path, help="Optional local WAV/MP3 for one real audio-only inference")
    args = parser.parse_args()
    if args.audio and not args.smoke:
        parser.error("--audio requires --smoke")
    if not any((args.download, args.install, args.verify, args.smoke)):
        parser.error("Select --download, --install, --verify and/or --smoke")
    if args.install:
        install()
    if args.download:
        download()
    if args.verify:
        verify(ROOT, require_ready=False)
    record = smoke(args.audio) if args.smoke else None
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
                      "status": record["status"] if record else "verified", "training": False,
                      "smoke": record.get("smoke") if record else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
