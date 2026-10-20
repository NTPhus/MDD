#!/usr/bin/env python3
"""Entry point for explicit held-out phone-CTC scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.runtime import eval_parser, run_evaluate

if __name__ == "__main__":
    try:
        run_evaluate(eval_parser().parse_args())
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
