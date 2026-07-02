#!/usr/bin/env bash
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"
export ENV_YAML="$TICKET/env_eplb_bench.yaml"
export LOG_ROOT="$AAD/logs/eplb_verify_bench"
LOG="$AAD/claude-logs/artifacts/eplb_async_nixl_bench_$(date +%Y%m%d_%H%M%S).log"
echo "[async-bench] start $(date) log=$LOG" | tee "$LOG"
PRESET="glm5.2/dp8ep8/noMTP-bs64-dg-eplb-async-nixl.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
  && echo "  OK" | tee -a "$LOG" || echo "  FAIL" | tee -a "$LOG"
echo "[async-bench] done $(date)" | tee -a "$LOG"
