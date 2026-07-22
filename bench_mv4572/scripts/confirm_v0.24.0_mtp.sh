#!/usr/bin/env bash
# =============================================================================
# MV-4572 Task1.2 — CONFIRM V2 root-cause trên MTP: CHỈ arm no_v2 (base + r0),
#   full-grid conc{16,64,128,256} x ds{8k,10k,100k}. KHÔNG chạy v2 arm (đã có ở sheet).
#   CHẠY TRONG container phuc-nguyen-mv4572-analysis-0.24.0 (repo vllm-moreh),
#   tmux window confirm-mtp. BENCH thuần (preset MTP ablation KHÔNG profiler) -> tps.
#
# Mục tiêu: bỏ V2 (=0) trên MTP -> r0 EPLB có > base lại không (như 0.23). Confirm thêm cho non-mtp.
#
# LƯU Ý: PHẢI export AITER_MOREH_ROOT_DIR (aiter JIT core.py assert) — sweep_presets.sh KHÔNG tự set.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8_mtp/v0.24.0_ablation"
SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_0.24.0_confirm_mtp/$TS"

echo "======== [confirm-mtp] AITER_MOREH_ROOT_DIR=$AITER_MOREH_ROOT_DIR | TS=$TS ========"
PDIR="$PDIR" \
  PRESET_LIST="${PRESET_LIST:-base_noV2.yaml r0_noV2.yaml}" \
  CONCS="${CONCS:-16 64 128 256}" DATASETS="${DATASETS:-8k 10k 100k}" \
  SWEEP_ROOT="$SWEEP_ROOT" \
  bash "$WD/auto-script/bench_mv4572/scripts/sweep_presets.sh"

echo ""; echo "======== [confirm-mtp] TÓM TẮT tps ========"
find "$SWEEP_ROOT" -name scenario_summary.csv 2>/dev/null | sort | while read -r f; do
  echo "--- $(basename "$(dirname "$f")") ---"; cat "$f"
done
