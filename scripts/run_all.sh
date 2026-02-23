#!/usr/bin/env bash
set -euo pipefail

COMMIT=$(git rev-parse --short HEAD)
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"

run_and_log() {
  local name=$1
  shift
  echo "[run_all] $name: $*"
  "$@" | tee "${LOGDIR}/${name}_${COMMIT}_${TS}.log"
}

usage() {
  echo "Usage: $0 [--train] [--rollout] [--grad]"
  echo "If no flags are provided, all are run."
}

RUN_TRAIN=0
RUN_ROLLOUT=0
RUN_GRAD=0

if [[ $# -eq 0 ]]; then
  RUN_TRAIN=1
  RUN_ROLLOUT=1
  RUN_GRAD=1
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train) RUN_TRAIN=1 ;;
      --rollout) RUN_ROLLOUT=1 ;;
      --grad) RUN_GRAD=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
  done
fi

if [[ $RUN_TRAIN -eq 1 ]]; then
  run_and_log train python scripts/train.py --config configs/train_fno.yaml
fi
if [[ $RUN_ROLLOUT -eq 1 ]]; then
  run_and_log rollout python scripts/eval_rollout.py --config configs/eval_rollout.yaml
fi
if [[ $RUN_GRAD -eq 1 ]]; then
  run_and_log grad python scripts/grad_fidelity.py --config configs/grad_fidelity.yaml
fi
