#!/usr/bin/env bash
# Detached launcher for the EPLB time-imbalance sweep. Runs the EP-imbalance harness
# (TIME phase only) for the given scenario file, logging to claude-logs/artifacts/.
#   bash run_eplb_sweep.sh scenario_eplb_nomtp.yaml
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
SCEN="${1:?usage: run_eplb_sweep.sh <scenario.yaml>}"
mkdir -p claude-logs/artifacts
LOG="claude-logs/artifacts/eplb_$(basename "${SCEN%.yaml}")_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] scenario=$SCEN PHASES=time log=$LOG" | tee "$LOG"
SCENARIO_YAML="$SCEN" PHASES=time bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1
echo "[done] $(date)" >> "$LOG"
