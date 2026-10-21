"""Plan or fully extract the local nested L2-ARCTIC ZIP; never delete source archives."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
CORPUS = "l2arctic_release_v5.0"
BLOCK_SIZE = 1024 * 1024
_CHECKED_DIRECTORIES = set()


def safe_name(name, *, directory=False):
    candidate = name[:-1] if directory and name.endswith("/") else name
    path = PurePosixPath(candidate)
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    if (not candidate or path.is_absolute() or path.as_posix() != candidate or ".." in path.parts
            or any(character in candidate for character in '\\:<>"|?*\0')
            or any(ord(character) < 32 for character in candidate)
            or any(part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in reserved
                   for part in path.parts)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def reject_links(path, boundary):
    current = path
    while current != boundary.parent:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"Destination contains a link or junction: {current}")
        if current == boundary:
            return
        current = current.parent
    raise ValueError("Destination is outside the project")


def destination_path(destination, *, root=ROOT):
    root = Path(root).resolve()
    destination = Path(os.path.abspath(destination))
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("Dataset destination must be a directory inside the MDD project")
    reject_links(destination, root)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Dataset destination is not a directory")
    # Each run rechecks its destination. Within a run only this extractor creates directories.
    _CHECKED_DIRECTORIES.clear()
    return destination


def target_path(destination, name):
    relative = safe_name(name)
    target = destination.joinpath(*relative.parts)
    if not target.is_relative_to(destination):
        raise ValueError(f"Archive path escapes the destination: {name}")
    current = target.parent
    while current not in _CHECKED_DIRECTORIES:
        try:
            info = current.lstat()
        except FileNotFoundError:
            pass
        else:
            reject_reparse(current, info)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Output parent is not a directory: {current}")
            _CHECKED_DIRECTORIES.add(current)
        if current == destination:
            break
        current = current.parent
    try:
        reject_reparse(target, target.lstat())
    except FileNotFoundError:
        pass
    return target


def reject_reparse(path, info):
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError(f"Destination contains a link or junction: {path}")


def remember_created_directories(directory, destination):
    while directory.is_relative_to(destination):
        _CHECKED_DIRECTORIES.add(directory)
        if directory == destination:
            break
        directory = directory.parent


def entries(archive):
    seen = set()
    output = []
    for member in archive.infolist():
        safe_name(member.filename, directory=member.is_dir())
        key = member.filename.rstrip("/").casefold()
        if key in seen:
            raise ValueError(f"Duplicate or case-conflicting archive path: {member.filename}")
        seen.add(key)
        mode = member.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Archive links/special files are not supported: {member.filename}")
        if member.flag_bits & 1:
            raise ValueError(f"Encrypted archive member is not supported: {member.filename}")
        output.append(member)
    return output


def stream_digest(source, output=None):
    digest, checksum, size = hashlib.sha256(), 0, 0
    for block in iter(lambda: source.read(BLOCK_SIZE), b""):
        size += len(block)
        digest.update(block)
        checksum = zlib.crc32(block, checksum)
        if output is not None:
            output.write(block)
    return {"size_bytes": size, "crc32": f"{checksum & 0xffffffff:08x}", "sha256": digest.hexdigest()}


def file_digest(path):
    with path.open("rb") as source:
        return stream_digest(source)


def checked_member_digest(archive, member, output=None):
    with archive.open(member) as source:
        digest = stream_digest(source, output)
    if digest["size_bytes"] != member.file_size or digest["crc32"] != f"{member.CRC:08x}":
        raise ValueError(f"Archive size/CRC mismatch: {member.filename}")
    return digest


def output_name(member):
    name = safe_name(member.filename, directory=member.is_dir())
    # Accept the release prefix once, while retaining speaker/suitcase directories.
    if name.parts[0] == CORPUS:
        if len(name.parts) == 1:
            return None
        name = PurePosixPath(*name.parts[1:])
    return name.as_posix()


def add_inventory(archive, archive_label, outputs, directories, claimed, destination):
    count, total = 0, 0
    for member in entries(archive):
        name = output_name(member)
        if name is None:
            if not member.is_dir():
                raise ValueError("Release root must be a directory")
            continue
        if not member.is_dir() and name.lower().endswith(".zip"):
            raise ValueError(f"Unexpected additional ZIP nesting inside {archive_label}: {name}")
        key = name.casefold()
        kind = "directory" if member.is_dir() else "file"
        prior = claimed.get(key)
        if prior is not None and not (prior == kind == "directory"):
            raise ValueError(f"Multiple archive members target the same output: {name}")
        for parent in PurePosixPath(name).parents:
            if parent == PurePosixPath("."):
                continue
            parent_key = parent.as_posix().casefold()
            if claimed.get(parent_key) == "file":
                raise ValueError(f"File/directory collision: {name}")
            claimed[parent_key] = "directory"
        claimed[key] = kind
        if member.is_dir():
            directories.add(name)
        else:
            outputs.append({"path": name, "archive": archive_label, "member": member.filename,
                            "size_bytes": member.file_size, "crc32": f"{member.CRC:08x}"})
            count += 1
            total += member.file_size
    return count, total


@contextmanager
def nested_archive(outer, member, source_directory, *, verified=False):
    loose = source_directory / member.filename
    if loose.is_file() and not loose.is_symlink():
        with zipfile.ZipFile(loose) as nested:
            yield nested
    elif verified:
        # Execution materializes only one missing/mismatching nested ZIP at a time.
        with tempfile.TemporaryFile() as temporary:
            checked_member_digest(outer, member, temporary)
            temporary.seek(0)
            with zipfile.ZipFile(temporary) as nested:
                yield nested
    else:
        # ZipExtFile is seekable; planning can inspect a missing nested ZIP without writing it.
        with outer.open(member) as source, zipfile.ZipFile(source) as nested:
            yield nested


def outer_members(outer):
    members = entries(outer)
    for member in members:
        if member.filename.lower().endswith(".zip") and len(PurePosixPath(member.filename).parts) != 1:
            raise ValueError("Nested speaker ZIPs must be at the release root")
    return members


def plan(archive_path, destination, *, source_directory=None, root=ROOT):
    destination = destination_path(destination, root=root)
    archive_path = Path(archive_path).resolve()
    source_directory = Path(source_directory or archive_path.parent).resolve()
    outputs, directories, claimed, archives = [], set(), {}, []
    with zipfile.ZipFile(archive_path) as outer:
        members = outer_members(outer)
        # Treat root metadata separately; nested ZIP files are containers, not dataset outputs.
        metadata = [member for member in members if not member.filename.lower().endswith(".zip")]
        class MetadataView:
            def infolist(self):
                return metadata
        count, size = add_inventory(MetadataView(), archive_path.name, outputs, directories, claimed, destination)
        archives.append({"archive": archive_path.name, "file_count": count, "output_bytes": size})
        for member in members:
            if member.is_dir() or not member.filename.lower().endswith(".zip"):
                continue
            with nested_archive(outer, member, source_directory) as nested:
                count, size = add_inventory(nested, member.filename, outputs, directories, claimed, destination)
            archives.append({"archive": member.filename, "file_count": count, "output_bytes": size})
    return {"schema_version": "1.0", "purpose": "full_local_dataset_organization",
            "mode": "dry_run", "completed": False, "destination": str(destination),
            "outer_archive": str(archive_path), "source_directory": str(source_directory),
            "file_count": len(outputs), "output_bytes": sum(item["size_bytes"] for item in outputs),
            "archives": archives, "directories": sorted(directories), "outputs": outputs,
            "verified_zip_deletable": [],
            "verification": "Inventory only; source bytes and destination files have not been verified."}


def extract_member(archive, member, destination):
    name = output_name(member)
    if name is None:
        return None
    target = target_path(destination, name)
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        remember_created_directories(target, destination)
        return None
    if target.exists():
        if not target.is_file():
            raise FileExistsError(f"Existing output is not a regular file: {target}")
        expected = checked_member_digest(archive, member)
        if file_digest(target) != expected:
            raise FileExistsError(f"Existing output differs; preserved without overwriting: {target}")
        return {"path": name, **expected, "action": "reused_verified"}
    target.parent.mkdir(parents=True, exist_ok=True)
    remember_created_directories(target.parent, destination)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".dataset-", suffix=".part", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            expected = checked_member_digest(archive, member, output)
        # Hard-link publication is atomic and fails rather than overwriting a concurrent file.
        os.link(temporary, target)
        return {"path": name, **expected, "action": "extracted"}
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".dataset-receipt-", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(archive_path, destination, *, source_directory=None, receipt_path=None, root=ROOT):
    destination = destination_path(destination, root=root)
    archive_path = Path(archive_path).resolve()
    source_directory = Path(source_directory or archive_path.parent).resolve()
    receipt_path = Path(receipt_path or Path(root) / "data/generated/setup/dataset_organization.json")
    receipt = {"schema_version": "1.0", "purpose": "full_local_dataset_organization", "mode": "extract",
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed": False,
               "destination": str(destination), "source_archives": [], "nested_archives": [],
               "outputs": [], "verified_zip_deletable": []}
    try:
        outer_spec = {"path": str(archive_path), **file_digest(archive_path)}
        outer_stat = archive_path.stat()
        outer_spec["mtime_ns"] = outer_stat.st_mtime_ns
        receipt["source_archives"].append(outer_spec)
        with zipfile.ZipFile(archive_path) as outer, tempfile.TemporaryDirectory(prefix="mdd-dataset-") as scratch:
            members = outer_members(outer)
            prepared = {}
            # Validate the outer compressed bytes and select byte-identical loose ZIPs first.
            for member in members:
                if member.is_dir() or not member.filename.lower().endswith(".zip"):
                    continue
                print(f"Verifying source {member.filename}", flush=True)
                expected = checked_member_digest(outer, member)
                receipt["nested_archives"].append({"member": member.filename, **expected})
                loose = source_directory / member.filename
                loose_valid = loose.is_file() and not loose.is_symlink() and file_digest(loose) == expected
                if loose_valid:
                    spec = {"path": str(loose), **expected, "mtime_ns": loose.stat().st_mtime_ns}
                    receipt["source_archives"].append(spec)
                    prepared[member.filename] = loose
                else:
                    prepared[member.filename] = Path(scratch) / member.filename
                    with prepared[member.filename].open("wb") as output:
                        checked_member_digest(outer, member, output)
            # Reject collisions/unsafe names in every archive before writing dataset outputs.
            inventory, directories, claimed = [], set(), {}
            metadata = [member for member in members if not member.filename.lower().endswith(".zip")]
            class MetadataView:
                def infolist(self):
                    return metadata
            add_inventory(MetadataView(), archive_path.name, inventory, directories, claimed, destination)
            for name, path in prepared.items():
                with zipfile.ZipFile(path) as nested:
                    add_inventory(nested, name, inventory, directories, claimed, destination)
            receipt["file_count"] = len(inventory)
            receipt["output_bytes"] = sum(item["size_bytes"] for item in inventory)
            for member in metadata:
                result = extract_member(outer, member, destination)
                if result is not None:
                    receipt["outputs"].append(result)
            for name, path in prepared.items():
                print(f"Extracting and checking {name}", flush=True)
                with zipfile.ZipFile(path) as nested:
                    for member in entries(nested):
                        result = extract_member(nested, member, destination)
                        if result is not None:
                            receipt["outputs"].append(result)
        if len(receipt["outputs"]) != receipt["file_count"]:
            raise ValueError("Extracted output inventory differs from the plan")
        # A separate readback catches incomplete writes before any archive is marked deletable.
        for output in receipt["outputs"]:
            expected = {key: output[key] for key in ("size_bytes", "crc32", "sha256")}
            if file_digest(target_path(destination, output["path"])) != expected:
                raise ValueError(f"Destination readback failed: {output['path']}")
        for source in receipt["source_archives"]:
            current = Path(source["path"]).stat()
            if current.st_size != source["size_bytes"] or current.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(f"Source archive changed during extraction: {source['path']}")
        receipt["completed"] = True
        receipt["verification"] = "All archive members passed CRC; all output files passed separate SHA-256 readback."
        receipt["verified_zip_deletable"] = list(receipt["source_archives"])
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_receipt(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Complete outer L2-ARCTIC release ZIP")
    parser.add_argument("--source-directory", type=Path, help="Existing loose speaker ZIPs (default: outer ZIP directory)")
    parser.add_argument("--destination", type=Path, default=ROOT / CORPUS)
    parser.add_argument("--receipt", type=Path, help="Save inventory/verification JSON; extraction has a default receipt")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Inspect all file inventories; do not extract (default)")
    modes.add_argument("--extract", action="store_true", help="Extract, verify readback and write receipt; never delete ZIPs")
    args = parser.parse_args()
    if args.extract:
        receipt = extract(args.archive, args.destination, source_directory=args.source_directory, receipt_path=args.receipt)
    else:
        receipt = plan(args.archive, args.destination, source_directory=args.source_directory)
        if args.receipt:
            write_receipt(args.receipt, receipt)
    summary = {key: value for key, value in receipt.items()
               if key not in {"outputs", "directories", "source_archives", "nested_archives"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
