"""Local setup must fail clearly without breaking the CUDA notebook's probe."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from scripts import check_training_environment


class LocalSetupReadinessTests(unittest.TestCase):
    def invoke(self, *, local_ready, strict=False):
        result = {"ready_for_cpu_pipeline_check": True,
                  "ready_for_local_cpu_training": local_ready}
        arguments = ["check_training_environment.py"]
        if strict:
            arguments.append("--require-local-cpu")
        with patch.object(check_training_environment, "report", return_value=result), \
                patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
            return check_training_environment.main()

    def test_local_setup_rejects_missing_cpu_presets_or_unverified_data(self):
        self.assertEqual(self.invoke(local_ready=False, strict=True), 1)

    def test_local_setup_accepts_complete_cpu_environment(self):
        self.assertEqual(self.invoke(local_ready=True, strict=True), 0)

    def test_default_probe_remains_usable_by_cuda_colab_notebook(self):
        self.assertEqual(self.invoke(local_ready=False), 0)


if __name__ == "__main__":
    unittest.main()
