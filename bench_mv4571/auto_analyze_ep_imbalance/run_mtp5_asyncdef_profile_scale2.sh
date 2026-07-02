#!/usr/bin/env bash
# Profile(time) 3 async-default với PROFILE_NP_SCALE=2 (np=72 khớp bench) — kiểm chứng profile-tpot
# có đại diện khi np giống bench không. Chỉ PHASES=time (bench thật đã có).
set -uo pipefail
TICKET=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571
AAD="$TICKET/auto_analyze_ep_imbalance"; cd "$AAD"
mkdir -p claude-logs/artifacts
LOG="$AAD/claude-logs/artifacts/mtp5_asyncdef_scale2_$(date +%Y%m%d_%H%M%S).log"
echo "[asyncdef-scale2] start $(date) log=$LOG" | tee "$LOG"
pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 6
PROFILE_NP_SCALE=2 SCENARIO_YAML=scenario_asyncdef_scale2.yaml PHASES=time \
  bash auto_analyze_ep_imbalance.sh >> "$LOG" 2>&1 || echo "  (profile done/err)" | tee -a "$LOG"
echo "[asyncdef-scale2] done $(date)" | tee -a "$LOG"
