"""Resume unchanged cascade recipe; retry transient Windows status-file locks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cascade_training as runner
from scripts.projector_data import read_json
from scripts.l2arctic_plus_data import rooted
from scripts.prepare_feedback_data import digest

original_save = runner.save


def retry_save(path, value):
    for attempt in range(61):
        try:
            return original_save(path, value)
        except PermissionError:
            if attempt == 60:
                raise
            time.sleep(min(.05 * (attempt + 1), 1.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=runner.CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    record = rooted(config["report_root"]) / "execution_io_patch.json"
    retry_save(record, {"wrapper": str(Path(__file__).relative_to(ROOT)), "wrapper_sha256": digest(Path(__file__)),
                        "scope": "Retry PermissionError during pipeline status/manifest JSON writes only. Training subprocess, data, recipe and frozen pipeline files unchanged.",
                        "reason": "Windows file sharing violation interrupted launcher at A step 1850; resume existing checkpoints."})
    runner.save = retry_save
    runner.run(config, "all", args.config.resolve())
