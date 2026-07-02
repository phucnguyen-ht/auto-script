#!/usr/bin/env bash
# Batch 2 — noMTP FAIR bench (enforce_eager=false + EP_LOG=0), 3x, 10k_c36. Lấp cột bench noMTP (đang none).
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_nomtp_fair_bench
LOG="$AAD/claude-logs/artifacts/nomtp_fair_bench_$(date +%Y%m%d_%H%M%S).log"
export ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_nomtp_fair_bench"
echo "[nomtp-fair] start $(date) log=$LOG" | tee "$LOG"
for p in noMTP-bs64-dg-fair noMTP-bs64-dg-eplb-freq-pynccl-fair noMTP-bs64-dg-eplb-step500-pynccl-fair noMTP-bs64-dg-eplb-async-nixl-fair; do
  echo "=== bench 3x $p ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
done
echo "[nomtp-fair] done $(date)" | tee -a "$LOG"
