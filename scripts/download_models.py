"""Download pinned public wav2vec2 checkpoints and verify every file; stdlib only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    "wav2vec2-base": ("facebook/wav2vec2-base", "Backbone for MDD fine-tuning; no trained phoneme head/tokenizer"),
    "wav2vec2-base-960h": ("facebook/wav2vec2-base-960h", "English character CTC ASR baseline for demo transcripts"),
}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hashes(path):
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest()


def verify_file(path, expected):
    expected_size = expected.get("size") or expected.get("lfs", {}).get("size")
    if expected_size is None or path.stat().st_size != expected_size:
        raise ValueError(f"File size mismatch: {path.name}")
    sha256, git_sha1 = file_hashes(path)
    remote_sha256 = expected.get("lfs", {}).get("sha256")
    if remote_sha256:
        if sha256 != remote_sha256:
            raise ValueError(f"Hub SHA-256 mismatch: {path.name}")
    elif expected.get("blobId") != git_sha1:
        raise ValueError(f"Hub Git blob SHA-1 mismatch: {path.name}")
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256,
            "hub_sha256": remote_sha256, "hub_blob_id": expected.get("blobId")}


def download_file(url, path, expected):
    if path.is_file():
        try:
            return verify_file(path, expected)
        except ValueError as error:
            # A Windows checkout can change tracked JSON/Markdown to CRLF. Keep
            # the existing file until a byte-exact replacement passes Hub checks.
            print(f"  Replacing invalid {path.name}: {error}", flush=True)
    partial = path.with_name(path.name + ".part")
    expected_size = expected.get("size") or expected.get("lfs", {}).get("size")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and offset == expected_size:
        try:
            result = verify_file(partial, expected)
        except ValueError:
            offset = 0
        else:
            partial.replace(path)
            result["filename"] = path.name
            return result
    elif expected_size is not None and offset > expected_size:
        offset = 0
    # A partial transfer may already contain corruption. Resume first; if the
    # completed candidate fails verification, retry once from byte zero.
    while True:
        headers = {"User-Agent": "MDD-Checkpoint-Downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError(f"Invalid download resume offset: {path.name}")
            transferred = offset if append else 0
            last_report = transferred
            with partial.open("ab" if append else "wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    transferred += len(block)
                    if transferred - last_report >= 64 * 1024 * 1024:
                        print(f"  {path.name}: {transferred / 1024**2:.0f} MiB", flush=True)
                        last_report = transferred
        try:
            result = verify_file(partial, expected)
        except ValueError:
            if not append:
                raise
            print(f"  Restarting invalid resumed download: {path.name}", flush=True)
            offset = 0
            continue
        break
    partial.replace(path)
    result["filename"] = path.name
    return result


def inspect_configuration(directory, name):
    config = json.loads((directory / "config.json").read_text())
    feature = json.loads((directory / "preprocessor_config.json").read_text())
    if (config.get("model_type") != "wav2vec2" or config.get("hidden_size") != 768
            or config.get("num_hidden_layers") != 12 or feature.get("sampling_rate") != 16000):
        raise ValueError(f"Unexpected BASE architecture or sampling rate: {name}")
    result = {"model_type": config["model_type"], "architectures": config.get("architectures", []),
              "hidden_size": 768, "num_hidden_layers": 12, "sampling_rate": 16000}
    if name == "wav2vec2-base-960h":
        vocab = json.loads((directory / "vocab.json").read_text())
        if len(vocab) != config.get("vocab_size"):
            raise ValueError("ASR vocabulary does not match model output size")
        result["asr_vocabulary_size"] = len(vocab)
    return result


def download_model(name, destination):
    repo_id, role = REPOSITORIES[name]
    directory = destination / name
    directory.mkdir(parents=True, exist_ok=True)
    revision_path = directory / "download_revision.txt"
    revision = revision_path.read_text().strip() if revision_path.exists() else "main"
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{quote(revision, safe='')}?blobs=true"
    with urlopen(Request(url, headers={"User-Agent": "MDD-Checkpoint-Downloader/1.0"}), timeout=30) as response:
        info = json.load(response)
    revision = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a pinned Git commit SHA")
    revision_path.write_text(revision + "\n", encoding="utf-8")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    weight = "model.safetensors" if "model.safetensors" in siblings else "pytorch_model.bin"
    if weight not in siblings:
        raise ValueError(f"No supported unsharded weights in {repo_id}")
    selected = sorted(filename for filename in siblings if "/" not in filename and (
        filename == weight or filename.endswith((".json", ".md")) or filename.startswith("LICENSE")
    ))
    for required in ("config.json", "preprocessor_config.json"):
        if required not in selected:
            raise ValueError(f"Missing required model file: {required}")
    if name.endswith("-960h") and "vocab.json" not in selected:
        raise ValueError("Missing ASR vocabulary")
    print(f"Downloading {repo_id} at {revision}; weights={weight}", flush=True)
    records = []
    for filename in selected:
        source = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(filename)}?download=true"
        metadata = download_file(source, directory / filename, siblings[filename])
        records.append({**metadata, "source_url": source})
        print(f"  Verified {filename}: {metadata['size_bytes']} bytes", flush=True)
    manifest = {
        "repo_id": repo_id, "revision": revision, "role": role,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights_file": weight, "files": records,
        "configuration": inspect_configuration(directory, name),
        "verification": {"hub_checksums": True, "local_configuration": True,
                         "inference_load_tested": False},
    }
    write_json(directory / "download_manifest.json", manifest)
    return manifest


def verify_model(name, destination):
    directory = destination / name
    manifest = json.loads((directory / "download_manifest.json").read_text())
    if manifest["repo_id"] != REPOSITORIES[name][0]:
        raise ValueError("Model repository identity mismatch")
    if (directory / "download_revision.txt").read_text().strip() != manifest["revision"]:
        raise ValueError("Revision lock mismatch")
    for item in manifest["files"]:
        path = directory / item["filename"]
        if path.stat().st_size != item["size_bytes"] or file_hashes(path)[0] != item["sha256"]:
            raise ValueError(f"Local model checksum mismatch: {path}")
    inspect_configuration(directory, name)
    print(f"Verified offline: {name}, revision={manifest['revision']}", flush=True)
    return manifest


def update_inventory(destination):
    models = []
    for name in REPOSITORIES:
        path = destination / name / "download_manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            models.append({"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                           "directory": name, "role": manifest["role"],
                           "size_bytes": sum(item["size_bytes"] for item in manifest["files"])})
    write_json(destination / "manifest.json", {"default_models": ["wav2vec2-base"], "models": models})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/pretrained")
    parser.add_argument("--models", nargs="+", choices=list(REPOSITORIES), default=["wav2vec2-base"],
                        help="Checkpoints to download or verify; default: wav2vec2-base only")
    parser.add_argument("--verify-only", action="store_true", help="Verify local checksums without network access")
    args = parser.parse_args()
    manifests = []
    for name in dict.fromkeys(args.models):
        manifest = (verify_model if args.verify_only else download_model)(name, args.destination)
        manifests.append({"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                          "directory": name, "role": manifest["role"],
                          "size_bytes": sum(item["size_bytes"] for item in manifest["files"])})
    if not args.verify_only:
        update_inventory(args.destination)
    print(f"Complete: {len(manifests)} checkpoints; {sum(item['size_bytes'] for item in manifests) / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
