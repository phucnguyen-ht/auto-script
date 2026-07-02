#!/usr/bin/env bash
# MTP5 async + window250/step250 (nhỏ hơn default 1000/3000 để rearrange THỰC SỰ nổ trong run ngắn 10k_c36).
# Chờ run_mtp5_red8_baseline xong. Mỗi communicator: BENCH 3x rồi PROFILE(time). Thứ tự nixl -> nccl -> pynccl.
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_s250_bench
LOG="$AAD/claude-logs/artifacts/mtp5_s250_$(date +%Y%m%d_%H%M%S).log"
echo "[s250] wait for run_mtp5_red8_baseline ..." | tee "$LOG"
while pgrep -f run_mtp5_red8_baseline >/dev/null 2>&1; do sleep 30; done
echo "[s250] prior done, start $(date)" | tee -a "$LOG"
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
export ENV_YAML="$TICKET/env_eplb_bench.yaml"
for c in nixl nccl pynccl; do
  P="MTP5-bs64-dg-eplb-async-s250-$c"
  echo "=== [$c] BENCH 3x ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  LOG_ROOT="$AAD/logs/eplb_mtp5_s250_bench" PRESET="glm5.2/dp8ep8/$P.yaml" \
    bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 && echo "  OK bench $c" | tee -a "$LOG" || echo "  FAIL bench $c" | tee -a "$LOG"
  echo "=== [$c] PROFILE(time) ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  mkscen scenario_s250_$c.yaml "$P"
  SCENARIO_YAML=scenario_s250_$c.yaml PHASES=time \
    bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  ($c profile done/err)" | tee -a "$LOG"
done
echo "[s250] done $(date)" | tee -a "$LOG"
