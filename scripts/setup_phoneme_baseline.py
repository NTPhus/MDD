"""Download a pinned public TIMIT phoneme baseline, without loading or training it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
REVISION = "efb7ae9b88f13db0d42eac8cedbba19739e2a278"
RELATIVE = Path("models/pretrained/wav2vec2-xls-r-300m-timit-phoneme")
PINS = {
    "model.safetensors": (1261987832, "0e3bde20ed7b252a48e69eddf7bd6d7eadc7dbacb06924148514e500ebad4b35"),
    "README.md": (3444, "157102d3d9db69936311fdfe8812cfea651c652fd9f8ed1c9af8c3a27691a577"),
    "config.json": (2068, "17158b6f99c0bef454c7313b021240c7133a4039f229c28d4495ecd207f9f007"),
    "preprocessor_config.json": (214, "60ca5a31e13f69ee2fbf147504c8676db5f6398fd7a6b12294341dff838edfcf"),
    "special_tokens_map.json": (309, "fe9499f3f127cf340419ded12d4175b3a28a58c99c949bf3f8343a6f6b9fa872"),
    "tokenizer_config.json": (287, "3dd1990023adb889ac3029349f80262802c923fe06beb1807118cb1d5e5f8b02"),
    "vocab.json": (486, "12b7de6be6a1bd3132ffdef8d8fe191c8e98f2d70639666e35d9cf6c9d2af77a"),
    "added_tokens.json": (23, "ec58d12af09ef36b4f44ba496b5f3b6cb650b70994ebff692a160f77d2d28c51"),
}


def file_record(path, expected):
    path = Path(path)
    size, checksum = expected
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("Missing or wrong-sized baseline file: " + path.name)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    if sha.hexdigest() != checksum:
        raise ValueError("Baseline checksum mismatch: " + path.name)
    return {"bytes": size, "sha256": checksum}


def verify(root=ROOT):
    folder = Path(root) / RELATIVE
    return {name: file_record(folder / name, pin) for name, pin in PINS.items()}


def download(root=ROOT):
    folder = Path(root) / RELATIVE
    folder.mkdir(parents=True, exist_ok=True)
    for name, pin in PINS.items():
        destination = folder / name
        try:
            file_record(destination, pin)
            continue
        except ValueError:
            pass
        print(json.dumps({"downloading": name, "bytes": pin[0]}), flush=True)
        temporary = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
            headers={"User-Agent": "MDD-phoneme-comparison/1.0"})
        with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        file_record(temporary, pin)
        temporary.replace(destination)
    record = {"model_id": MODEL_ID, "revision": REVISION, "files": verify(root),
              "license": "apache-2.0", "status": "downloaded_verified", "training": False,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
              "scope": "General phoneme recognizer trained on TIMIT; not a calibrated MDD error classifier."}
    (folder / "setup_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    record = download() if args.download else {"model_id": MODEL_ID, "revision": REVISION, "files": verify()}
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
