"""Validate CPU command selection without importing or training the speech model."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_local_training.sh"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"),
                     "The Bash CPU launcher requires a POSIX host and bash; native Windows uses the PowerShell launcher.")
class LocalLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.task_root = Path(cls.fixture.name)
        (cls.task_root / "scripts").mkdir()
        shutil.copyfile(LAUNCHER, cls.task_root / "scripts/run_local_training.sh")
        (cls.task_root / "configs").mkdir()
        for name in ("local_smoke", "local_source", "local_adapt_vi", "local_control"):
            (cls.task_root / f"configs/{name}.json").write_text("{}", encoding="utf-8")
        (cls.task_root / ".venv/bin").mkdir(parents=True)
        python = cls.task_root / ".venv/bin/python"
        python.write_text("#!/usr/bin/env bash\nprintf 'Python must not run during a dry run.' >&2\nexit 99\n", encoding="utf-8")
        python.chmod(0o755)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def invoke(self, *args, cwd=None):
        return subprocess.run(["bash", str(self.task_root / "scripts/run_local_training.sh"), *args], cwd=cwd or ROOT,
                              capture_output=True, text=True, check=False)

    def test_each_stage_chooses_cpu_config_and_never_runs_python_in_dry_run(self):
        for stage, name in (("smoke", "local_smoke"), ("A", "local_source"),
                            ("B", "local_adapt_vi"), ("C", "local_control")):
            with self.subTest(stage=stage):
                result = self.invoke(stage, "--run-id", "dry-check", "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(result.stdout)
                self.assertEqual(command[0], str(self.task_root / ".venv/bin/python"))
                self.assertEqual(command[1:3], ["-u", str(self.task_root / "scripts/train_mdd.py")])
                self.assertEqual(command[command.index("--config") + 1], str(self.task_root / f"configs/{name}.json"))
                self.assertEqual(command[-2:], ["--device", "cpu"])
                self.assertNotIn("--dry-run", command)
                self.assertNotIn("--max-steps", command)

    def test_passes_resume_and_paths_as_single_arguments_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke("A", "--run-id=local-source", "--runs-root", "runs with spaces",
                                 "--resume", "runs with spaces/local-source/checkpoint", "--dry-run", cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertIn("--run-id=local-source", command)
        self.assertEqual(command[command.index("--runs-root") + 1], "runs with spaces")
        self.assertEqual(command[command.index("--resume") + 1], "runs with spaces/local-source/checkpoint")

    def test_adaptation_passes_fold_and_initial_checkpoint(self):
        result = self.invoke("B", "--run-id", "B-fold2", "--fold", "2", "--init-checkpoint",
                             "runs/A/best_checkpoint", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[command.index("--fold") + 1], "2")
        self.assertEqual(command[command.index("--init-checkpoint") + 1], "runs/A/best_checkpoint")

    def test_explicit_run_id_is_required(self):
        for args in (("A", "--dry-run"), ("A", "--run-id"), ("A", "--run-id="),
                     ("A", "--run-id", "--dry-run")):
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--run-id", result.stderr)

    def test_rejects_device_or_config_overrides(self):
        for argument in ("--device=cuda", "--device", "--config=other.json", "--config"):
            with self.subTest(argument=argument):
                result = self.invoke("A", "--run-id", "check", argument, "--dry-run")
                self.assertEqual(result.returncode, 2)
                self.assertIn("local CPU", result.stderr)

    def test_help_does_not_require_run_id(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--resume", result.stdout)
        result = self.invoke("A", "--help", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--help", shlex.split(result.stdout))

    def test_unknown_stage_stops_before_any_python_execution(self):
        result = self.invoke("D", "--run-id", "invalid", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown stage", result.stderr)


if __name__ == "__main__":
    unittest.main()
