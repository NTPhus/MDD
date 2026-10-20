#!/usr/bin/env bash
# Select explicit CPU settings without changing the research protocol or budget.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_local_training.sh smoke|A|B|C --run-id ID [options]

Stages:
  smoke  One technical check; its result is not a scientific benchmark.
  A      Source-language training, using configs/local_source.json.
  B      Vietnamese adaptation, using configs/local_adapt_vi.json.
  C      Matched source-only control, using configs/local_control.json.

Examples:
  bash scripts/run_local_training.sh smoke --run-id local-smoke-01
  bash scripts/run_local_training.sh A --run-id local-A-seed42
  bash scripts/run_local_training.sh B --run-id local-B-fold1 --fold 1 --init-checkpoint runs/local-A-seed42/best_checkpoint
  bash scripts/run_local_training.sh A --run-id local-A-seed42 --resume runs/local-A-seed42/checkpoint

Options are passed to scripts/train_mdd.py (including --runs-root, --fold,
--init-checkpoint, --resume, --seed and --max-steps). --dry-run prints the
command without running Python. --config and --device are fixed by this CPU
launcher; use scripts/train_mdd.py directly for other environments.
HELP
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

stage="$1"
shift
case "$stage" in
  smoke) config_name=local_smoke.json ;;
  A) config_name=local_source.json ;;
  B) config_name=local_adapt_vi.json ;;
  C) config_name=local_control.json ;;
  *) printf 'Unknown stage: %s (expected smoke, A, B or C).\n' "$stage" >&2; exit 2 ;;
esac

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
dry_run=false
has_run_id=false
wants_help=false
runtime_args=()
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --run-id)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        printf '%s\n' '--run-id requires a nonempty value.' >&2
        exit 2
      fi
      has_run_id=true
      runtime_args+=("$1" "$2")
      shift 2
      ;;
    --run-id=*)
      if [[ -z "${1#*=}" ]]; then
        printf '%s\n' '--run-id requires a nonempty value.' >&2
        exit 2
      fi
      has_run_id=true
      runtime_args+=("$1")
      shift
      ;;
    --config|--config=*|--device|--device=*)
      printf '%s\n' 'This launcher selects the local CPU config and device. Use scripts/train_mdd.py directly to override them.' >&2
      exit 2
      ;;
    --help|-h) wants_help=true; runtime_args+=("$1"); shift ;;
    *) runtime_args+=("$1"); shift ;;
  esac
done
if [[ "$has_run_id" == false && "$wants_help" == false ]]; then
  printf '%s\n' 'Choose an explicit --run-id; an existing run must be continued with --resume.' >&2
  exit 2
fi
if [[ ! -x "$TASK_ROOT/.venv/bin/python" ]]; then
  printf '%s\n' 'Missing .venv. Run: bash scripts/setup_local_training.sh' >&2
  exit 1
fi
if [[ ! -f "$TASK_ROOT/configs/$config_name" ]]; then
  printf 'Missing CPU config: configs/%s\n' "$config_name" >&2
  exit 1
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
command=("$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/scripts/train_mdd.py"
  "${runtime_args[@]}" --config "$TASK_ROOT/configs/$config_name" --device cpu)
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
printf 'Starting stage %s with configs/%s on CPU.\n' "$stage" "$config_name"
exec "${command[@]}"
