#!/usr/bin/env python3
"""Entry point for audio-only wav2vec2-base phone-CTC training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.runtime import run_train, train_parser

if __name__ == "__main__":
    try:
        run_train(train_parser().parse_args())
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
