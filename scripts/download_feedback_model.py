"""Download Qwen3-0.6B at a pinned revision and verify Hub checksums; no API key."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import urlopen

if __package__:
    from .download_models import download_file, write_json
else:
    from download_models import download_file, write_json

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3-0.6B"
FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "model.safetensors", "LICENSE", "README.md")


def verify(directory):
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["repo_id"] != REPO or (directory / "download_revision.txt").read_text().strip() != manifest["revision"]:
        raise ValueError("Feedback model identity/revision mismatch")
    for record in manifest["files"]:
        path = directory / record["filename"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("Feedback model checksum mismatch: " + path.name)
    print(f"Verified {REPO} at {manifest['revision']}", flush=True)
    return manifest


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "download_revision.txt"
    revision = lock.read_text().strip() if lock.is_file() else "main"
    with urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{quote(revision, safe='')}?blobs=true", timeout=30) as response:
        info = json.load(response)
    revision = info["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Hub did not return a pinned revision")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if name not in siblings:
            raise ValueError("Missing required Qwen artifact: " + name)
    lock.write_text(revision + "\n", encoding="utf-8")
    records = []
    for name in FILES:
        if name in siblings:
            print("Downloading/verifying " + name, flush=True)
            url = f"https://huggingface.co/{REPO}/resolve/{revision}/{quote(name)}?download=true"
            records.append(download_file(url, directory / name, siblings[name]))
    write_json(directory / "download_manifest.json", {
        "repo_id": REPO, "revision": revision, "files": records,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Vietnamese coaching from English MDD assessments; not an acoustic detector or TTS",
    })
    return verify(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/pretrained/Qwen3-0.6B")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        (verify if args.verify_only else download)(args.destination.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
