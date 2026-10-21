"""Exercise complete nested ZIP organization using tiny local fixtures, without deleting sources."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import organize_local_dataset as organizer


def fixture(root, *, unsafe=None):
    payloads = {"README.md": b"dataset readme", "LICENSE": b"dataset license",
                "ABA/annotation/one.TextGrid": b"manual annotation",
                "ABA/wav/one.wav": b"audio", "ABA/wav/unannotated.wav": b"complete dataset",
                "suitcase_corpus/read.txt": b"suitcase must also be extracted"}
    archive = root / "release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        for name in ("README.md", "LICENSE"):
            outer.writestr(name, payloads[name])
        for prefix in ("ABA", "suitcase_corpus"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as nested:
                nested.writestr(prefix + "/", b"")
                for name, content in payloads.items():
                    if name.startswith(prefix + "/"):
                        nested.writestr(name, content)
                if prefix == "ABA" and unsafe == "inner":
                    nested.writestr("../escaped.txt", b"unsafe")
                if prefix == "ABA" and unsafe == "symlink":
                    link = zipfile.ZipInfo("ABA/link")
                    link.create_system = 3
                    link.external_attr = (stat.S_IFLNK | 0o777) << 16
                    nested.writestr(link, "../../outside")
            (root / (prefix + ".zip")).write_bytes(buffer.getvalue())
            outer.writestr(prefix + ".zip", buffer.getvalue())
        if unsafe == "outer":
            outer.writestr("../escaped.txt", b"unsafe")
    return archive, payloads


class OrganizeDatasetTests(unittest.TestCase):
    def extract(self, archive, destination, root):
        with redirect_stdout(io.StringIO()):
            return organizer.extract(archive, destination, root=root)

    def test_dry_run_lists_every_file_without_extraction_or_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, payloads = fixture(root)
            destination = root / organizer.CORPUS
            report = organizer.plan(archive, destination, root=root)
            self.assertEqual({item["path"] for item in report["outputs"]}, set(payloads))
            self.assertEqual(report["file_count"], 6)
            self.assertFalse(report["completed"])
            self.assertFalse(destination.exists())
            self.assertEqual(report["verified_zip_deletable"], [])
            self.assertEqual(len(list(root.glob("*.zip"))), 3)

    def test_extracts_all_members_readback_hashes_and_resumes_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, payloads = fixture(root)
            destination = root / organizer.CORPUS
            report = self.extract(archive, destination, root)
            self.assertTrue(report["completed"])
            self.assertEqual(len(report["verified_zip_deletable"]), 3)
            for name, expected in payloads.items():
                self.assertEqual((destination / name).read_bytes(), expected)
            output = {item["path"]: item for item in report["outputs"]}
            self.assertEqual(output["ABA/wav/one.wav"]["sha256"], hashlib.sha256(b"audio").hexdigest())
            times = {name: (destination / name).stat().st_mtime_ns for name in payloads}
            resumed = self.extract(archive, destination, root)
            self.assertTrue(all(item["action"] == "reused_verified" for item in resumed["outputs"]))
            self.assertEqual(times, {name: (destination / name).stat().st_mtime_ns for name in payloads})
            self.assertEqual(len(list(root.glob("*.zip"))), 3)
            self.assertFalse(list(destination.rglob("*.part")))

    def test_invalid_loose_archive_uses_verified_outer_copy_and_is_not_marked_deletable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, payloads = fixture(root)
            (root / "ABA.zip").write_bytes(b"invalid loose duplicate must be preserved")
            destination = root / organizer.CORPUS
            report = self.extract(archive, destination, root)
            self.assertTrue(report["completed"])
            self.assertEqual((destination / "ABA/wav/one.wav").read_bytes(), payloads["ABA/wav/one.wav"])
            deletable = {Path(item["path"]).name for item in report["verified_zip_deletable"]}
            self.assertNotIn("ABA.zip", deletable)
            self.assertEqual((root / "ABA.zip").read_bytes(), b"invalid loose duplicate must be preserved")

    def test_conflicting_file_is_preserved_and_no_archives_are_marked_deletable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _ = fixture(root)
            destination = root / organizer.CORPUS
            destination.mkdir()
            (destination / "README.md").write_bytes(b"existing content")
            with self.assertRaisesRegex(FileExistsError, "preserved"):
                self.extract(archive, destination, root)
            self.assertEqual((destination / "README.md").read_bytes(), b"existing content")
            receipt = json.loads((root / "data/generated/setup/dataset_organization.json").read_text())
            self.assertFalse(receipt["completed"])
            self.assertEqual(receipt["verified_zip_deletable"], [])

    def test_unsafe_outer_inner_and_symlink_members_are_rejected_before_dataset_writes(self):
        for unsafe in ("outer", "inner", "symlink"):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive, _ = fixture(root, unsafe=unsafe)
                destination = root / organizer.CORPUS
                with self.assertRaisesRegex(ValueError, "Unsafe|links/special"):
                    self.extract(archive, destination, root)
                self.assertFalse(destination.exists())
                self.assertFalse((root.parent / "escaped.txt").exists())

    def test_case_collisions_and_destinations_outside_project_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "inside the MDD project"):
                organizer.plan(archive, root.parent / "outside-dataset", root=root)
            with zipfile.ZipFile(root / "ABA.zip", "a") as nested:
                nested.writestr("ABA/WAV/ONE.WAV", b"case collision")
            with self.assertRaisesRegex(ValueError, "same output|case-conflicting"):
                organizer.plan(archive, root / organizer.CORPUS, root=root)

    def test_failed_disk_readback_never_marks_archive_deletable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _ = fixture(root)
            destination = root / organizer.CORPUS
            original = organizer.file_digest
            def corrupt_readback(path):
                result = original(path)
                if path == destination / "LICENSE":
                    result["sha256"] = "0" * 64
                return result
            with patch.object(organizer, "file_digest", side_effect=corrupt_readback):
                with self.assertRaisesRegex(ValueError, "readback failed"):
                    self.extract(archive, destination, root)
            receipt = json.loads((root / "data/generated/setup/dataset_organization.json").read_text())
            self.assertFalse(receipt["completed"])
            self.assertEqual(receipt["verified_zip_deletable"], [])

    def test_new_run_rechecks_parent_links_after_cache_was_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / organizer.CORPUS
            parent = destination / "ABA/wav"
            parent.mkdir(parents=True)
            destination = organizer.destination_path(destination, root=root)
            organizer.target_path(destination, "ABA/wav/one.wav")
            self.assertIn(parent, organizer._CHECKED_DIRECTORIES)
            organizer.destination_path(destination, root=root)
            original = Path.lstat
            def replaced_parent(path):
                if path == parent:
                    return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
                return original(path)
            with patch.object(Path, "lstat", replaced_parent):
                with self.assertRaisesRegex(ValueError, "link or junction"):
                    organizer.target_path(destination, "ABA/wav/one.wav")


if __name__ == "__main__":
    unittest.main()
