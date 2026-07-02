#!/usr/bin/env bash
# MTP5 10k_c36 fill-ins: (1) imbalance freq-pynccl/sync-nixl/async-nixl; (2) bench 3x freq(nccl)/sync-nixl/async-nixl.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_fill_bench
LOG="$AAD/claude-logs/artifacts/mtp5_fill_$(date +%Y%m%d_%H%M%S).log"
echo "[mtp5-fill] start $(date) log=$LOG" | tee "$LOG"
echo "=== PHASE 1: imbalance (freq-pynccl, sync-nixl, async-nixl) ===" | tee -a "$LOG"
SCENARIO_YAML=scenario_mtp5_fill.yaml PHASES=time bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  (imbalance sweep done)" | tee -a "$LOG"
export ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_mtp5_fill_bench"
for p in MTP5-bs64-dg-eplb-freq MTP5-bs64-dg-eplb-sync-nixl MTP5-bs64-dg-eplb-async-nixl; do
  echo "=== PHASE 2: bench 3x $p ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
done
echo "[mtp5-fill] done $(date)" | tee -a "$LOG"
