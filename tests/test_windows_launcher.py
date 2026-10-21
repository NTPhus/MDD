"""Exercise native Windows launcher argument handling without starting Python."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_local_training.ps1"
PWSH = shutil.which("pwsh")


@unittest.skipUnless(os.name == "nt" and PWSH,
                     "The native Windows CUDA launcher requires Windows and PowerShell 7 (pwsh).")
class WindowsLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No interpreter/model/data in this fixture: even a broken dry-run
        # branch cannot accidentally start real training.
        cls.fixture_parent = Path(tempfile.gettempdir()).resolve()
        cls.task_root = cls.fixture_parent / ("mdd Windows launcher O'Brien " + uuid.uuid4().hex)
        cls.task_root.mkdir()
        cls.addClassCleanup(cls.clean_fixture)
        (cls.task_root / "scripts").mkdir()
        shutil.copyfile(LAUNCHER, cls.task_root / "scripts/run_local_training.ps1")
        (cls.task_root / "configs").mkdir()
        for name in ("windows_smoke", "windows_source", "windows_adapt_vi", "windows_control"):
            (cls.task_root / f"configs/{name}.json").write_text("{}", encoding="utf-8")

    @classmethod
    def clean_fixture(cls):
        resolved = cls.task_root.resolve()
        if resolved.parent != cls.fixture_parent or not resolved.name.startswith("mdd Windows launcher O'Brien "):
            raise RuntimeError("Refusing cleanup outside the explicitly created launcher fixture")
        shutil.rmtree(resolved)

    def invoke(self, **arguments):
        environment = dict(os.environ)
        environment["MDD_TEST_LAUNCHER"] = str(self.task_root / "scripts/run_local_training.ps1")
        environment["MDD_TEST_LAUNCHER_ARGUMENTS"] = json.dumps(arguments)
        # A structured parameter map preserves arrays and paths containing spaces
        # or apostrophes without interpolating them into PowerShell source.
        command = r"""
$ErrorActionPreference = 'Stop'
try {
    $taskArguments = ConvertFrom-Json $env:MDD_TEST_LAUNCHER_ARGUMENTS -AsHashtable
    & $env:MDD_TEST_LAUNCHER @taskArguments
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""
        return subprocess.run([PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                              cwd=ROOT.parent, env=environment, capture_output=True,
                              text=True, encoding="utf-8", check=False, timeout=20)

    def printed_arguments(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout.strip()
        self.assertTrue(text.startswith("& "), text)
        quoted_argument = r"'((?:[^']|'')*)'"
        self.assertEqual(re.sub(quoted_argument, "", text[2:]).strip(), "", text)
        return [value.replace("''", "'") for value in re.findall(quoted_argument, text[2:])]

    def test_each_stage_selects_windows_config_and_cuda_in_dry_run(self):
        for stage, config in (("smoke", "windows_smoke"), ("A", "windows_source"),
                              ("B", "windows_adapt_vi"), ("C", "windows_control")):
            with self.subTest(stage=stage):
                command = self.printed_arguments(self.invoke(Stage=stage, RunId="dry-check", DryRun=True))
                self.assertEqual(Path(command[0]), self.task_root / ".venv/Scripts/python.exe")
                self.assertEqual(command[1], "-u")
                self.assertEqual(Path(command[2]), self.task_root / "scripts/train_mdd.py")
                self.assertEqual(Path(command[command.index("--config") + 1]), self.task_root / f"configs/{config}.json")
                self.assertEqual(command[-2:], ["--device", "cuda"])
                self.assertEqual(command[command.index("--run-id") + 1], "dry-check")
                self.assertNotIn("--dry-run", command)
                self.assertNotIn("--max-steps", command)

    def test_adaptation_and_control_preserve_fold_initialization_and_paths(self):
        for stage, fold in (("B", 2), ("C", 4)):
            with self.subTest(stage=stage):
                runs_root = "runs O'Brien with spaces"
                initialization = f"{runs_root}/A/best_checkpoint"
                command = self.printed_arguments(self.invoke(
                    Stage=stage, RunId=f"{stage}-fold{fold}", Fold=fold, InitCheckpoint=initialization,
                    RunsRoot=runs_root, Seed=7, DryRun=True))
                self.assertEqual(command[command.index("--runs-root") + 1], runs_root)
                self.assertEqual(command[command.index("--init-checkpoint") + 1], initialization)
                self.assertEqual(command[command.index("--fold") + 1], str(fold))
                self.assertEqual(command[command.index("--seed") + 1], "7")

    def test_resume_and_step_override_are_forwarded_as_separate_arguments(self):
        resume = "runs with spaces/A/checkpoint"
        command = self.printed_arguments(self.invoke(
            Stage="A", RunId="A", RunsRoot="runs with spaces", Resume=resume,
            Seed=17, MaxSteps=2100, DryRun=True))
        self.assertEqual(command[command.index("--resume") + 1], resume)
        self.assertEqual(command[command.index("--seed") + 1], "17")
        self.assertEqual(command[command.index("--max-steps") + 1], "2100")
        self.assertNotIn("--init-checkpoint", command)

    def test_rejects_forbidden_training_argument_overrides(self):
        for argument in ("--device", "--device=cpu", "--config", "--config=other.json",
                         "--run-id", "--run-id=other"):
            with self.subTest(argument=argument):
                result = self.invoke(Stage="A", RunId="check", DryRun=True, TrainingArguments=[argument])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CUDA device and Windows config are fixed", result.stderr)
                self.assertNotIn("& '", result.stdout)

    def test_explicit_run_id_is_required(self):
        for arguments in ({}, {"RunId": ""}, {"RunId": "   "}):
            with self.subTest(arguments=arguments):
                result = self.invoke(Stage="A", DryRun=True, **arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("explicit -RunId", result.stderr)

    def test_init_checkpoint_and_resume_are_mutually_exclusive(self):
        result = self.invoke(Stage="B", RunId="check", InitCheckpoint="runs/A/best_checkpoint",
                             Resume="runs/B/checkpoint", DryRun=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("either -InitCheckpoint or -Resume", result.stderr)

    def test_help_does_not_require_a_run_id(self):
        result = self.invoke(Help=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-Resume", result.stdout)
        self.assertIn("Windows config", result.stdout)


if __name__ == "__main__":
    unittest.main()
