#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: bash scripts/setup_local_training.sh

Prepare the local CPU environment, verify the BASE checkpoint and raw data,
and write an environment report. Existing pinned dependencies are reused
without contacting a package index. This command never starts training.
HELP
  exit 0
fi
if (( $# )); then
  printf 'Unknown setup option: %s\n' "$1" >&2
  exit 2
fi

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

requirements_match() {
  .venv/bin/python - "$1" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    requirement = line.split("#", 1)[0].strip()
    if not requirement:
        continue
    name, separator, expected = requirement.partition("==")
    if not separator:
        raise SystemExit(f"Expected a pinned requirement: {requirement}")
    try:
        actual = importlib.metadata.version(name.strip())
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(1)
    if actual != expected.strip():
        raise SystemExit(1)
PY
}

if requirements_match requirements/cpu-torch.txt; then
  printf 'Reusing the pinned CPU PyTorch installation.\n'
else
  .venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir --retries 1 --timeout 30 \
    --index-url https://download.pytorch.org/whl/cpu -r requirements/cpu-torch.txt
fi
if requirements_match requirements/training.txt; then
  printf 'Reusing the pinned training dependencies.\n'
else
  .venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir --retries 1 --timeout 30 \
    --index-url https://pypi.org/simple -r requirements/training.txt
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -m pip --no-cache-dir check
.venv/bin/python -m pip --no-cache-dir freeze > requirements/local-cpu.lock.txt
.venv/bin/python scripts/download_models.py --models wav2vec2-base --verify-only
.venv/bin/python scripts/prepare_l2arctic.py --verify-audio
.venv/bin/python scripts/check_training_environment.py --require-local-cpu --output data/generated/setup/environment.json
printf '\nLocal setup is ready. No model training was started.\n'
printf 'Next: bash scripts/run_local_training.sh smoke --run-id local-smoke-01\n'
