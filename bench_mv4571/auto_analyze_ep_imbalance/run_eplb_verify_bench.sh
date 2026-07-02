#!/usr/bin/env bash
# Phase 3 — verification bench (3 runs, mean/std, NO profiler) for baseline vs best-EPLB
# (eplb-freq-pynccl), both presets, 10k @ conc36. Uses env_eplb_bench.yaml (runs=3).
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"
export ENV_YAML="$TICKET/env_eplb_bench.yaml"
export LOG_ROOT="$AAD/logs/eplb_verify_bench"
mkdir -p "$LOG_ROOT" "$AAD/claude-logs/artifacts"
LOG="$AAD/claude-logs/artifacts/eplb_verify_bench_$(date +%Y%m%d_%H%M%S).log"
PRESETS=( noMTP-bs64-dg noMTP-bs64-dg-eplb-freq-pynccl MTP5-bs64-dg MTP5-bs64-dg-eplb-freq-pynccl )
echo "[verify-bench] start $(date) log=$LOG" | tee "$LOG"
for p in "${PRESETS[@]}"; do
  echo "================= BENCH (runs=3): $p =================" | tee -a "$LOG"
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
    && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p (see log)" | tee -a "$LOG"
done
echo "[verify-bench] done $(date)" | tee -a "$LOG"
