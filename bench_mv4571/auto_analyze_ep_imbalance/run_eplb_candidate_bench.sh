#!/usr/bin/env bash
# Compute-bound-candidate bench: baseline vs async-NIXL EPLB across noMTP 8k(c8-52)+100k(c8,31) and
# MTP5 8k(control). One serve per (preset,group), benches all its cases x3 runs. Detached.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"
export LOG_ROOT="$AAD/logs/eplb_candidate_bench"
mkdir -p "$AAD/claude-logs/artifacts"
LOG="$AAD/claude-logs/artifacts/eplb_candidate_bench_$(date +%Y%m%d_%H%M%S).log"
run1() {  # <env> <preset-stem>
  local env="$1" p="$2"
  echo "================= BENCH $p (env=$env) =================" | tee -a "$LOG"
  podman_kill
  ENV_YAML="$TICKET/$env" PRESET="glm5.2/dp8ep8/$p.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
    && echo "  OK $p" | tee -a "$LOG" || echo "  FAIL $p" | tee -a "$LOG"
}
podman_kill() { pkill -9 -f "[v]llm serve" 2>/dev/null || true; pkill -9 -f "[E]ngineCore" 2>/dev/null || true; sleep 6; }
echo "[cand-bench] start $(date) log=$LOG" | tee "$LOG"
# 8k group (prime compute-bound candidates + MTP5 control)
run1 env_bench_8k.yaml   noMTP-bs64-dg-base-8k
run1 env_bench_8k.yaml   noMTP-bs64-dg-async-8k
run1 env_bench_8k.yaml   MTP5-bs64-dg-base-8k
run1 env_bench_8k.yaml   MTP5-bs64-dg-async-8k
# 100k group (highest-imbalance noMTP; 100k_c8 comm-bound contrast, 100k_c31 compute-bound)
run1 env_bench_100k.yaml noMTP-bs64-dg-base-100k
run1 env_bench_100k.yaml noMTP-bs64-dg-async-100k
echo "[cand-bench] done $(date)" | tee -a "$LOG"
