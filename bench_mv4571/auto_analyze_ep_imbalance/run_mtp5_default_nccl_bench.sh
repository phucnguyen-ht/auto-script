#!/usr/bin/env bash
# Fill MTP5 default nccl BENCH (profile đã có). Chờ run_mtp5_default_extra (pynccl+nixl) xong rồi bench 3x.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
LOG="$AAD/claude-logs/artifacts/mtp5_default_nccl_$(date +%Y%m%d_%H%M%S).log"
echo "[mtp5-default-nccl] wait for run_mtp5_default_extra ..." | tee "$LOG"
while pgrep -f run_mtp5_default_extra >/dev/null 2>&1; do sleep 30; done
echo "[mtp5-default-nccl] prior done, bench start $(date)" | tee -a "$LOG"
pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
export ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_mtp5_default_bench"
PRESET="glm5.2/dp8ep8/MTP5-bs64-dg-eplb-default.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
  && echo "  OK nccl-default" | tee -a "$LOG" || echo "  FAIL nccl-default" | tee -a "$LOG"
echo "[mtp5-default-nccl] done $(date)" | tee -a "$LOG"
