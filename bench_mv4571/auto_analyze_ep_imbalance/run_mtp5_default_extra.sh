#!/usr/bin/env bash
# MTP5 10k_c36 default-eplb (window1000/step3000): pynccl + nixl. Profile (time_analyze) + bench 3x.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_default_bench
LOG="$AAD/claude-logs/artifacts/mtp5_default_$(date +%Y%m%d_%H%M%S).log"
echo "[mtp5-default] start $(date) log=$LOG" | tee "$LOG"
echo "=== PHASE 1: profile/time_analyze (default-pynccl, default-nixl) ===" | tee -a "$LOG"
SCENARIO_YAML=scenario_mtp5_default_extra.yaml PHASES=time bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  (profile done)" | tee -a "$LOG"
export ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_mtp5_default_bench"
for p in MTP5-bs64-dg-eplb-default-pynccl MTP5-bs64-dg-eplb-default-nixl; do
  echo "=== PHASE 2: bench 3x $p ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
done
echo "[mtp5-default] done $(date)" | tee -a "$LOG"
