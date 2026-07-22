#!/usr/bin/env bash
# =============================================================================
# MV-4572 Task1.1 — CONFIRM V2 root-cause: full-grid sweep 4 config x conc{16,64,128,256}
#   x ds{8k,10k,100k}. CHẠY TRONG container phuc-nguyen-mv4572-analysis-0.24.0 (repo vllm-moreh),
#   tmux window confirm-v024. BENCH thuần (preset ablation KHÔNG profiler) -> tps.
#
# 4 config (preset trong dp8ep8/v0.24.0_ablation): base_v2, r0_v2 (V2 ON = 0.24 gốc)
#   + base_noV2, r0_noV2 (V2 OFF). Xác nhận: V2 OFF -> r0>base; V2 ON -> r0 regress.
#
# LƯU Ý: PHẢI export AITER_MOREH_ROOT_DIR (aiter JIT core.py assert) — sweep_presets.sh KHÔNG tự set.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/v0.24.0_ablation"
SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_0.24.0_confirm/$TS"

echo "======== [confirm] AITER_MOREH_ROOT_DIR=$AITER_MOREH_ROOT_DIR | TS=$TS ========"
PDIR="$PDIR" \
  PRESET_LIST="${PRESET_LIST:-base_v2.yaml r0_v2.yaml base_noV2.yaml r0_noV2.yaml}" \
  CONCS="${CONCS:-16 64 128 256}" DATASETS="${DATASETS:-8k 10k 100k}" \
  SWEEP_ROOT="$SWEEP_ROOT" \
  bash "$WD/auto-script/bench_mv4572/scripts/sweep_presets.sh"

echo ""; echo "======== [confirm] TÓM TẮT tps ========"
find "$SWEEP_ROOT" -name scenario_summary.csv 2>/dev/null | sort | while read -r f; do
  echo "--- $(basename "$(dirname "$f")") ---"; cat "$f"
done
