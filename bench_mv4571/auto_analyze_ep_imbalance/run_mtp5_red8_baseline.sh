#!/usr/bin/env bash
# TASK1: baseline (EPLB off) profile scale=2 (np=72) -> để copy vào time_imbalance_2.
# TASK2: redundant=8 (async-default) cho nixl/nccl/pynccl -> mỗi cái BENCH 3x rồi PROFILE(time).
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_red8_bench
LOG="$AAD/claude-logs/artifacts/mtp5_red8_baseline_$(date +%Y%m%d_%H%M%S).log"
mkscen(){ cat > "$1" <<EOF
defaults: { dataset_dir: /remote/vast0/share-mv/longbenchv2-custom, prompts_per_concurrency: 2, num_prompts_floor: 1, num_prompts_cap: 256 }
models:
  glm5.2:
    model_path: /remote/vast0/share-mv/zai-org/GLM-5.2-FP8
    presets:
      - glm5.2/dp8ep8/$2.yaml
cases:
  - { model: glm5.2, name: 10k, osl: 500, rates: ["inf"], concurrencies: [36] }
EOF
}
echo "[red8+base] start $(date) log=$LOG" | tee "$LOG"

# ===== TASK1: baseline scale=2 profile =====
echo "=== TASK1: baseline (EPLB off) profile scale=2 ===" | tee -a "$LOG"
pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
mkscen scenario_base_scale2.yaml MTP5-bs64-dg
PROFILE_NP_SCALE=2 SCENARIO_YAML=scenario_base_scale2.yaml PHASES=time \
  bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  (base scale2 done/err)" | tee -a "$LOG"

# ===== TASK2: redundant=8, 3 communicator: bench 3x -> profile =====
export ENV_YAML="$TICKET/env_eplb_bench.yaml"
for c in nixl nccl pynccl; do
  P="MTP5-bs64-dg-eplb-red8-$c"
  echo "=== TASK2 [$c] BENCH 3x ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  LOG_ROOT="$AAD/logs/eplb_mtp5_red8_bench" PRESET="glm5.2/dp8ep8/$P.yaml" \
    bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK bench $c" | tee -a "$LOG" || echo "  FAIL bench $c" | tee -a "$LOG"
  echo "=== TASK2 [$c] PROFILE(time) ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  mkscen scenario_red8_$c.yaml "$P"
  SCENARIO_YAML=scenario_red8_$c.yaml PHASES=time \
    bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  ($c profile done/err)" | tee -a "$LOG"
done
echo "[red8+base] done $(date)" | tee -a "$LOG"
