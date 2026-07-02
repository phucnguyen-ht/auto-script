#!/usr/bin/env bash
# MTP5 10k_c36 async EPLB (pynccl + torch_nccl): (1) time-imbalance (profile), (2) bench 3x fair.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_async_bench
LOG="$AAD/claude-logs/artifacts/mtp5_async_$(date +%Y%m%d_%H%M%S).log"
echo "[mtp5-async] start $(date) log=$LOG" | tee "$LOG"
# (1) time-imbalance cho cả 2 preset (harness patch enforce_eager=false + EP_LOG=0 -> fair)
echo "=== PHASE 1: time-imbalance (pynccl + nccl) ===" | tee -a "$LOG"
SCENARIO_YAML=scenario_eplb_mtp5_async.yaml PHASES=time bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  (imbalance sweep kết thúc)" | tee -a "$LOG"
# (2) bench 3x fair (auto_bench serve preset as-is; MTP5 base fair: cudagraph on, no EP_LOG)
export ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_mtp5_async_bench"
for p in MTP5-bs64-dg-eplb-async-pynccl MTP5-bs64-dg-eplb-async-nccl; do
  echo "=== PHASE 2: bench 3x $p ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
done
echo "[mtp5-async] done $(date)" | tee -a "$LOG"
