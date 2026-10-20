"""Package only the frozen annotated data, BASE checkpoint and training code for Colab."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/generated/colab/mdd-colab.zip"
CODE_FILES = (
    "README.md", "docs.md", "TRAINING.md",
    "scripts/download_models.py", "scripts/prepare_l2arctic.py",
    "scripts/train_mdd.py", "scripts/evaluate_mdd.py",
    "scripts/compare_mdd_runs.py",
    "scripts/check_training_environment.py", "scripts/check_mdd_forward.py",
    "scripts/export_colab_bundle.py",
    "requirements/training.txt", "requirements/colab.txt",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(root):
    """Use an allowlist; no directory-wide archive of user configuration/secrets."""
    selected = {root / name for name in CODE_FILES}
    for pattern in ("src/mdd/*.py", "configs/*.json", "tests/test_metrics.py",
                    "tests/test_prepare_l2arctic.py", "tests/test_training_runtime.py",
                    "tests/test_compare_mdd_runs.py",
                    "data/protocol/l2arctic-v1/*", "data/processed/l2arctic-v1/*.json",
                    "data/processed/l2arctic-v1/*.jsonl", "data/processed/l2arctic-v1/SHA256SUMS",
                    "notebooks/*.ipynb", "data/generated/setup/scorer_calibration.json",
                    "data/generated/setup/ctc_feasibility.json"):
        selected.update(path for path in root.glob(pattern) if path.is_file())
    manifest_path = root / "data/protocol/l2arctic-v1/manifest.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for kind in ("audio", "annotation", "transcript"):
            path = root / record[kind]["path"]
            if sha256(path) != record[kind]["sha256"]:
                raise ValueError(f"Frozen input checksum mismatch: {record['utterance_id']} {kind}")
            selected.add(path)
    for filename in ("README.md", "LICENSE", "LICENSE.txt", "PROMPTS"):
        path = root / "l2arctic_release_v5.0" / filename
        if path.is_file():
            selected.add(path)
    base = root / "models/pretrained/wav2vec2-base"
    model_manifest = json.loads((base / "download_manifest.json").read_text())
    selected.update({base / "download_manifest.json", base / "download_revision.txt"})
    for item in model_manifest["files"]:
        path = base / item["filename"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Pretrained checkpoint checksum mismatch: {path.name}")
        selected.add(path)
    for path in selected:
        relative = path.resolve().relative_to(root.resolve())
        if not path.is_file() or any(part.startswith(".env") or part == ".venv" for part in relative.parts):
            raise ValueError(f"Invalid bundle file: {relative}")
    return sorted(selected)


def create_bundle(root, output):
    if output.exists():
        raise FileExistsError("Bundle already exists; use a new --output path to preserve provenance.")
    paths = selected_files(root)
    required = [root / "data/processed/l2arctic-v1" / name
                for name in ("annotations.jsonl", "vocab.json", "normalization.json", "audit.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Run prepare_l2arctic.py before exporting the Colab bundle.")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    partial = output.with_name(output.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(root).as_posix()
            archive.write(path, "MDD/" + relative)
            records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
            if index % 1000 == 0:
                print(f"Packed {index}/{len(paths)} files", flush=True)
        manifest = {
            "schema_version": "1.0", "purpose": "colab_training_setup",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": "l2arctic-v1", "default_model": "facebook/wav2vec2-base",
            "file_count": len(records), "files": records,
        }
        archive.writestr("MDD/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    partial.replace(output)
    digest = sha256(output)
    output.with_name(output.name + ".sha256").write_text(digest + "  " + output.name + "\n")
    print(f"Bundle: {output}")
    print(f"Files: {len(records)}; compressed size: {output.stat().st_size / 1024**3:.2f} GiB")
    print(f"SHA-256: {digest}")


def verify_extracted(root):
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for item in manifest["files"]:
        validate_relative_path(item["path"])
        path = root / item["path"]
        path.resolve().relative_to(root.resolve())
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Bundle file checksum mismatch: {item['path']}")
    print(f"Verified extracted bundle: {len(manifest['files'])} files; protocol={manifest['protocol_id']}")


def validate_relative_path(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or not path.parts or path.as_posix() != name):
        raise ValueError(f"Unsafe bundle path: {name}")


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names")
        for name in names:
            validate_relative_path(name)
            if PurePosixPath(name).parts[0] != "MDD":
                raise ValueError(f"Archive member outside MDD: {name}")
        manifest = json.loads(archive.read("MDD/bundle_manifest.json"))
        for item in manifest["files"]:
            validate_relative_path(item["path"])
        expected = {"MDD/" + item["path"] for item in manifest["files"]} | {"MDD/bundle_manifest.json"}
        if len(expected) != len(manifest["files"]) + 1 or set(names) != expected:
            raise ValueError("Unexpected or missing files in Colab archive")
        for item in manifest["files"]:
            name = "MDD/" + item["path"]
            if archive.getinfo(name).file_size != item["size_bytes"]:
                raise ValueError(f"Wrong archived size: {name}")
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Archived checksum mismatch: {name}")
        print(f"Verified archive: {len(manifest['files'])} files; protocol={manifest['protocol_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-extracted", type=Path)
    parser.add_argument("--verify-archive", type=Path)
    args = parser.parse_args()
    if args.verify_archive:
        verify_archive(args.verify_archive.resolve())
    elif args.verify_extracted:
        verify_extracted(args.verify_extracted.resolve())
    else:
        create_bundle(ROOT, args.output.resolve())


if __name__ == "__main__":
    main()
