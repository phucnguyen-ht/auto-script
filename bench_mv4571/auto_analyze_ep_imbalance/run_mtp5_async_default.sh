#!/usr/bin/env bash
# MTP5 async-default (use_async:true, window/step mặc định 1000/3000). Ưu tiên nixl -> nccl -> pynccl.
# Mỗi config làm ĐẦY ĐỦ profile(time) + bench 3x rồi mới sang config sau (nixl không bị nccl/pynccl chặn).
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts logs/eplb_mtp5_async_default_bench
LOG="$AAD/claude-logs/artifacts/mtp5_async_default_$(date +%Y%m%d_%H%M%S).log"
echo "[async-default] start $(date) log=$LOG" | tee "$LOG"
for c in nixl nccl pynccl; do
  P="MTP5-bs64-dg-eplb-async-default-$c"
  SC="$AAD/scenario_asyncdef_$c.yaml"
  cat > "$SC" <<EOF
defaults: { dataset_dir: /remote/vast0/share-mv/longbenchv2-custom, prompts_per_concurrency: 2, num_prompts_floor: 1, num_prompts_cap: 256 }
models:
  glm5.2:
    model_path: /remote/vast0/share-mv/zai-org/GLM-5.2-FP8
    presets:
      - glm5.2/dp8ep8/$P.yaml
cases:
  - { model: glm5.2, name: 10k, osl: 500, rates: ["inf"], concurrencies: [36] }
EOF
  echo "=== [$c] PROFILE (time) ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  SCENARIO_YAML="$SC" PHASES=time bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  ($c profile done/err)" | tee -a "$LOG"
  echo "=== [$c] BENCH 3x ===" | tee -a "$LOG"
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
  ENV_YAML="$TICKET/env_eplb_bench.yaml" LOG_ROOT="$AAD/logs/eplb_mtp5_async_default_bench" \
    PRESET="glm5.2/dp8ep8/$P.yaml" bash "$TICKET/auto_bench.sh" >> "$LOG" 2>&1 \
    && echo "  OK $c" | tee -a "$LOG" || echo "  FAIL $c" | tee -a "$LOG"
done
echo "[async-default] done $(date)" | tee -a "$LOG"
