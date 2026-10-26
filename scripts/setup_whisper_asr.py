"""Prepare the pinned local Whisper runtime without changing the demo's .venv.

Reuse this machine's verified experiment with --from-experiment, or install the
three optional packages with --install and fetch pinned weights with --download.
No flag runs inference or training, and inference itself is always offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.whisper_asr import (
    DEPS_RELATIVE, FILES, MODEL_ID, MODEL_RELATIVE, PACKAGES, REVISION,
    file_record, package_versions, verify,
)

EXPERIMENT = ROOT / "data/generated/quick-asr-20261023"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"


def _save(root, source):
    model_dir = root / MODEL_RELATIVE
    files = {name: file_record(model_dir / name, expected) for name, expected in FILES.items()}
    packages = package_versions(root / DEPS_RELATIVE)
    record = {"status": "ready", "model_id": MODEL_ID, "revision": REVISION,
              "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
              "prepared_from": source, "device": "cpu", "compute_type": "int8",
              "packages": packages, "files": files,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "inference_executed": False, "training": False}
    target = model_dir / "setup_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return record


def from_experiment(root=ROOT, experiment=EXPERIMENT):
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    # Validate everything before writing either destination directory.
    for name, expected in FILES.items():
        file_record(experiment / "models/small.en" / name, expected)
    package_versions(experiment / "deps")
    directories = ["av", "av.libs", "ctranslate2", "faster_whisper",
                   *[f"{name.replace('-', '_')}-{version}.dist-info" for name, version in PACKAGES.items()]]
    for name in directories:
        if not (experiment / "deps" / name).is_dir():
            raise ValueError(f"Missing experimental runtime directory: {name}")
    target = root / MODEL_RELATIVE
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        try:
            file_record(target / name, expected)
        except ValueError:
            temporary = target / (name + ".part")
            shutil.copy2(experiment / "models/small.en" / name, temporary)
            file_record(temporary, expected)
            temporary.replace(target / name)
    destination = root / DEPS_RELATIVE
    destination.mkdir(parents=True, exist_ok=True)
    for name in directories:
        shutil.copytree(experiment / "deps" / name, destination / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return _save(root, "verified_local_experiment")


def install(root=ROOT):
    destination = Path(root).resolve() / DEPS_RELATIVE
    destination.mkdir(parents=True, exist_ok=True)
    # Dependencies shared with transformers already live in the existing venv;
    # --no-deps prevents this optional installation from shadowing those packages.
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "--no-deps", "--target", str(destination),
                    *[f"{name}=={version}" for name, version in PACKAGES.items()]], check=True)
    package_versions(destination)


def download(root=ROOT):
    target = Path(root).resolve() / MODEL_RELATIVE
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = target / name
        try:
            file_record(path, expected)
        except ValueError:
            temporary = path.with_name(name + ".part")
            print(f"Downloading {name} ({expected['bytes']:,} bytes)", flush=True)
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "MDD-Whisper-setup/1.0"})
            with urllib.request.urlopen(request, timeout=90) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            file_record(temporary, expected)
            temporary.replace(path)
    return _save(Path(root).resolve(), "pinned_official_repository")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-experiment", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("Select --from-experiment, --install, --download and/or --verify")
    if args.from_experiment:
        from_experiment()
    if args.install:
        install()
    if args.download:
        download()
    if args.verify:
        verify(ROOT)
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
                      "packages": PACKAGES, "training": False, "inference_executed": False}))


if __name__ == "__main__":
    main()
