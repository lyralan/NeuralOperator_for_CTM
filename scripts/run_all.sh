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

run_and_log train python scripts/train.py --config configs/train_fno.yaml
run_and_log rollout python scripts/eval_rollout.py --config configs/eval_rollout.yaml
run_and_log grad python scripts/grad_fidelity.py --config configs/grad_fidelity.yaml
