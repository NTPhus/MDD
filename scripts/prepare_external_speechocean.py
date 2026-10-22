#!/usr/bin/env python3
"""Freeze and extract only SpeechOcean762's official test split from a pinned ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "613968e3b0b789fc33936fb5eba1973176ba7d11"
REPOSITORY = "https://github.com/jimbozhang/speechocean762"
SOURCE_URL = f"https://codeload.github.com/jimbozhang/speechocean762/zip/{COMMIT}"
BASE = ROOT / "data/external/speechocean762"
DATASET = BASE / COMMIT
ARCHIVE = BASE / "downloads" / f"{COMMIT}.zip"
STUDY_CONFIG = ROOT / "configs/external_speechocean762.json"
PAPER = "https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.pdf"
EXPECTED_TEST_COUNT = 2500
EXPECTED_CONTROL_COUNT = 179
METADATA = {
    "README.md": "4f29e91dda24a591bbb7c99b52a9b874779fda6ad4c3346a7430f51e957da3c2",
    "test/wav.scp": "ec911e9319759c73e71ab62dc9cf6a2a00a1d02b7b91a288b39e6d56451e66f3",
    "test/utt2spk": "061291f397f8c281ff1bf6bf388e875ab9c4cc0de66f37d8559e690af08964be",
    "test/spk2age": "bcc719f23f2352b76f3cec80f12a02f97d775ecfa7b77646be136159caf036d3",
    "test/spk2gender": "4c59b3694169b84a2e69b44632c03d2cfb3ef309f29633a40a729b54d5b77485",
    "test/text": "a0cf51d5f21b9e404b405672e85363e341dc2b8d510de41132d326f5f543382c",
    "resource/scores.json": "31c847b8ee29d61d8fe338bcbb1d4c4721425f05481d2144ad57cd19ea1159f8",
    "resource/scores-detail.json": "8a90ff40558fe8032d9bfa919429dcae267c0ec3fbdedf1dca30c0515ea989e1",
}


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value
            or value.startswith("/") or any(piece in {"", ".", ".."} for piece in value.rstrip("/").split("/"))):
        raise ValueError("Archive paths must be plain relative POSIX paths.")
    return PurePosixPath(value.rstrip("/"))


def safe_path(base, relative):
    """Reject redirects through existing symlinks, Windows junctions or reparse points."""
    relative = validate_relative(str(relative))
    base = Path(base)
    if not base.absolute().is_relative_to(ROOT):
        raise ValueError("Output directories must stay within this workspace.")
    target = base.joinpath(*relative.parts)
    for candidate in (base, *base.parents, target, *target.parents):
        if not candidate.absolute().is_relative_to(ROOT):
            continue
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("Symlinks, junctions and reparse points are not accepted in output paths.")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("An output path escaped the dataset directory.")
    return target


def atomic_bytes(base, relative, content, *, immutable=False):
    destination = safe_path(base, relative)
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == content:
            return destination
        if immutable:
            raise ValueError(f"Frozen/existing content differs: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_path(base, relative)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.part")
    safe_path(base, temporary.relative_to(base).as_posix())
    try:
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def download_archive(archive):
    if archive.is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    safe_path(archive.parent, archive.name)
    safe_path(archive.parent, partial.name)
    # Exclusive creation also protects against the separately authorized download.
    with partial.open("xb") as output:
        with urlopen(SOURCE_URL, timeout=90) as response:
            if response.geturl().split("?", 1)[0] != SOURCE_URL:
                raise ValueError("Unexpected redirect for the pinned GitHub archive.")
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise ValueError("The pinned archive exceeds the 1 GiB limit.")
                output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, archive)


def index_archive(package):
    prefix = f"speechocean762-{COMMIT}"
    members, seen = {}, set()
    if len(package.infolist()) > 20000:
        raise ValueError("Unexpectedly many entries in the pinned archive.")
    for member in package.infolist():
        path = validate_relative(member.filename)
        if path.parts[0] != prefix:
            raise ValueError("The archive does not have the pinned repository/commit prefix.")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode) or member.external_attr & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("Archive links and reparse points are not accepted.")
        if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("Archive special files are not accepted.")
        key = path.as_posix().casefold()
        if key in seen:
            raise ValueError("Duplicate or case-colliding archive paths are not accepted.")
        seen.add(key)
        if member.is_dir() or len(path.parts) == 1:
            continue
        relative = PurePosixPath(*path.parts[1:]).as_posix()
        members[relative] = member
    return members


def read_pairs(content, name):
    rows = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in rows:
            raise ValueError(f"Malformed or duplicate entry in {name}.")
        rows[fields[0]] = fields[1]
    return rows


def test_selection(metadata):
    waves = read_pairs(metadata["test/wav.scp"], "wav.scp")
    speakers = read_pairs(metadata["test/utt2spk"], "utt2spk")
    texts = read_pairs(metadata["test/text"], "text")
    ages = read_pairs(metadata["test/spk2age"], "spk2age")
    genders = read_pairs(metadata["test/spk2gender"], "spk2gender")
    if len(waves) != EXPECTED_TEST_COUNT or set(waves) != set(speakers) or set(waves) != set(texts):
        raise ValueError("The official test manifests must agree on all 2500 utterance IDs.")
    for identifier, relative in waves.items():
        path = validate_relative(relative)
        speaker = speakers[identifier]
        expected = f"WAVE/SPEAKER{speaker}/{identifier}.WAV"
        if not re.fullmatch(r"\d{9}", identifier) or not re.fullmatch(r"\d{4}", speaker) or path.as_posix() != expected:
            raise ValueError("A test WAV path does not match its official utterance and speaker IDs.")
        if speaker not in ages or speaker not in genders:
            raise ValueError("A test speaker is missing age or gender metadata.")
    if len(set(waves.values())) != EXPECTED_TEST_COUNT:
        raise ValueError("Multiple test IDs map to the same audio path.")
    return waves, speakers, texts


def tokens(value):
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("Unexpected phoneme sequence in the human annotations.")


def strict_perfect_controls(details, identifiers):
    """All five experts use only identical, undecorated canonical phones."""
    result = []
    for identifier in sorted(identifiers):
        entry = details.get(identifier)
        if not isinstance(entry, dict) or not isinstance(entry.get("words"), list) or not entry["words"]:
            raise ValueError("Missing word-level human annotations for an official test utterance.")
        perfect = True
        for word in entry["words"]:
            canonical = tokens(word.get("ref-phones"))
            raters = word.get("phones")
            if (not canonical or any(not re.fullmatch(r"[A-Z]+[012]?", phone) for phone in canonical)
                    or not isinstance(raters, list) or len(raters) != 5):
                raise ValueError("Every test word must have canonical phones and five expert annotations.")
            if any(tokens(rater) != canonical for rater in raters):
                perfect = False
        if perfect:
            result.append(identifier)
    if len(result) != EXPECTED_CONTROL_COUNT:
        raise ValueError(f"Strict-perfect controls differ from the pre-inference audit: found {len(result)}, expected 179.")
    return result


def prepare(archive, *, download=False):
    archive = Path(archive).absolute()
    safe_path(archive.parent, archive.name)
    if not archive.is_file():
        if not download:
            raise FileNotFoundError("The cached ZIP is not ready. Wait for its .part download, or explicitly use --download.")
        download_archive(archive)
    archive_hash = digest_file(archive)
    previous_path = DATASET / "source-manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    if previous.get("commit", COMMIT) != COMMIT:
        raise ValueError("Existing dataset provenance refers to a different commit.")
    if previous.get("archive", {}).get("sha256", archive_hash) != archive_hash:
        raise ValueError("The cached ZIP changed after its extraction receipt was recorded.")
    if not STUDY_CONFIG.is_file():
        raise FileNotFoundError("The pre-inference study config configs/external_speechocean762.json is required.")
    study_bytes = STUDY_CONFIG.read_bytes()
    study = json.loads(study_bytes)
    if study.get("commit") != COMMIT or not study.get("selection_frozen_before_predictions"):
        raise ValueError("The study config must pin this commit and predeclare the selection.")
    metadata_receipts = {}
    audio_rows = []
    with zipfile.ZipFile(archive) as package:
        members = index_archive(package)
        metadata = {}
        for relative, expected_hash in METADATA.items():
            member = members.get(relative)
            if member is None or member.file_size > 64 * 1024 * 1024:
                raise ValueError("A required metadata file is missing or oversized.")
            content = package.read(member)
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f"Pinned metadata SHA256 mismatch: {relative}")
            metadata[relative] = content
            metadata_receipts[relative] = {"sha256": expected_hash, "bytes": len(content),
                "url": f"https://raw.githubusercontent.com/jimbozhang/speechocean762/{COMMIT}/{relative}"}
        waves, speakers, texts = test_selection(metadata)
        details = json.loads(metadata["resource/scores-detail.json"])
        controls = strict_perfect_controls(details, waves)
        protocol = {"schema_version": 1, "protocol_id": study["protocol_id"], "dataset": "SpeechOcean762",
            "commit": COMMIT, "repository": REPOSITORY, "paper": PAPER,
            "license": "CC BY 4.0 (OpenSLR SLR101)", "split": "official test", "test_count": EXPECTED_TEST_COUNT,
            "official_test_ids": sorted(waves), "strict_perfect_control_ids": controls,
            "strict_perfect_control_count": len(controls), "metadata_sha256": METADATA,
            "archive_sha256": archive_hash, "study_config": "configs/external_speechocean762.json",
            "study_config_sha256": hashlib.sha256(study_bytes).hexdigest(), "study": study,
            "annotation_notation": {"plain": "score 2: correct", "{PHONE}": "score 1: right phone with heavy accent; accepted",
                "(PHONE)": "score 0: incorrect or missed", "[PHONE]": "inserted phone; not a canonical-position token"},
            "gold_phone_error": "At least 3 of 5 experts assign score 0 at a canonical phone position.",
            "gold_word_error": "At least 3 of 5 experts mark any score-0 canonical phone or an inserted phone in that word.",
            "strict_perfect_control": "Every expert provides the exact plain canonical phone sequence for every word; no insertion.",
            "selection_frozen_before_predictions": True, "prediction_based_filtering": False,
            "missing_audio_policy": "Fail visibly; no silent removal of official-test utterances.",
            "no_training": True, "no_llm_or_tts_calls": True}
        # Freeze the complete population and human-only selection BEFORE any prediction.
        atomic_bytes(DATASET, "frozen_protocol.json", json_bytes(protocol), immutable=True)
        for relative, content in metadata.items():
            atomic_bytes(DATASET, relative, content, immutable=True)
        for index, identifier in enumerate(sorted(waves), 1):
            relative = waves[identifier]
            member = members.get(relative)
            if member is None or not 44 <= member.file_size <= 32 * 1024 * 1024:
                raise ValueError("An official test WAV is missing or exceeds the size limit.")
            content = package.read(member)  # ZipFile checks the member's CRC on read.
            if content[:4] != b"RIFF" or content[8:12] != b"WAVE":
                raise ValueError("An official test WAV has an invalid RIFF/WAVE header.")
            atomic_bytes(DATASET, relative, content, immutable=True)
            audio_rows.append({"utterance_id": identifier, "speaker_id": speakers[identifier], "path": relative,
                               "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
            if index % 250 == 0:
                print(f"Verified official-test audio: {index}/{EXPECTED_TEST_COUNT}", flush=True)
    manifest = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in audio_rows)
    atomic_bytes(DATASET, "audio_manifest.jsonl", manifest, immutable=True)
    source = {**previous, "repository": REPOSITORY, "commit": COMMIT, "paper": PAPER,
        "license": "CC BY 4.0", "license_source": "https://openslr.org/101/", "files": metadata_receipts,
        "archive": {"url": SOURCE_URL, "path": archive.relative_to(ROOT).as_posix(),
                    "sha256": archive_hash, "bytes": archive.stat().st_size,
                    "verification": "Pinned GitHub commit over HTTPS; SHA256 receipt, not a publisher signature"},
        "study_config": {"path": "configs/external_speechocean762.json", "sha256": hashlib.sha256(study_bytes).hexdigest()},
        "audio_manifest": {"path": "audio_manifest.jsonl", "sha256": hashlib.sha256(manifest).hexdigest(),
                           "utterances": len(audio_rows), "bytes": sum(row["bytes"] for row in audio_rows)},
        "frozen_protocol": {"path": "frozen_protocol.json", "sha256": hashlib.sha256(json_bytes(protocol)).hexdigest()},
        "extraction": "Only pinned metadata and official-test WAVs; no training WAVs extracted", "status": "complete"}
    atomic_bytes(DATASET, "source-manifest.json", json_bytes(source))
    return {"status": "complete", "dataset_root": DATASET.relative_to(ROOT).as_posix(),
            "official_test_utterances": len(audio_rows), "strict_perfect_controls": len(controls),
            "archive_sha256": archive_hash, "audio_bytes": source["audio_manifest"]["bytes"],
            "frozen_protocol_sha256": source["frozen_protocol"]["sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="Cached pinned codeload ZIP; defaults to the workspace download cache.")
    parser.add_argument("--download", action="store_true", help="Explicitly download a missing ZIP; refuses an existing .part file.")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.archive, download=args.download), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"SpeechOcean762 preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
