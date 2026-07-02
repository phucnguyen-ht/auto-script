#!/usr/bin/env bash
# Re-run the broken noMTP EPLB bench (per "broken case -> re-run") + test a less-frequent-rearrange
# config (step500) for stability. bench runs=3, 10k@conc36, no profiler.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"
export ENV_YAML="$TICKET/env_eplb_bench.yaml"
export LOG_ROOT="$AAD/logs/eplb_verify_bench"
mkdir -p "$AAD/claude-logs/artifacts"
LOG="$AAD/claude-logs/artifacts/eplb_rerun_bench_$(date +%Y%m%d_%H%M%S).log"
# delete the crashed noMTP freq-pynccl bench result (broken case, replace it)
rm -rf "$LOG_ROOT/glm5.2/dp8ep8/noMTP-bs64-dg-eplb-freq-pynccl"
PRESETS=( noMTP-bs64-dg-eplb-freq-pynccl noMTP-bs64-dg-eplb-step500-pynccl )
echo "[rerun-bench] start $(date) log=$LOG" | tee "$LOG"
for p in "${PRESETS[@]}"; do
  echo "================= BENCH (runs=3): $p =================" | tee -a "$LOG"
  PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
    && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
done
echo "[rerun-bench] done $(date)" | tee -a "$LOG"
