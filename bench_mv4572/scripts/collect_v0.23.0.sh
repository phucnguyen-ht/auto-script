#!/usr/bin/env bash
# =============================================================================
# MV-4572 deep-analysis — COLLECT runner (Option A) cho v0.23.0.
# CHẠY TRONG container: phuc-nguyen-mv4572-pr-0.23.0  (repo retest-0.23.0/vllm-moreh).
#
# Preset r0_collect.yaml = EPLB r0 (nixl async, log_balancedness) nhưng KHÔNG torch profiler
#   -> KHÔNG có post-bench shm_broadcast gridlock -> POST /collect_eplb broadcast (DPLB) tới
#   ĐỦ 8 DP engine rảnh -> 8 file eplb_tokens_rank{0..7}.npz. (Time/trace đo ở profile_v0.23.0.sh.)
#
# ► ĐỔI conc/dataset: sửa CONCS/DATASETS bên dưới.
# ► THÊM/BỚT preset : sửa "SECTION PRESETS" ở CUỐI file.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
VER=v0.23.0                                          # <- version CỐ ĐỊNH của file này

# ---- knobs ----
CONCS="${CONCS:-16}"; DATASETS="${DATASETS:-8k}"     # <- ĐỔI conc / dataset tại đây
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"                   # folder datetime riêng mỗi lần chạy

PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8_deepanalysis/$VER"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/retest-0.23.0/vllm-moreh/src/aiter_moreh}"

echo "======== [collect] LOGS -> auto-script/bench_mv4572/logs_analysis/${TS}/  (TS=${TS}) ========"

run_preset() {
  local PRESET="$1"
  [ -f "$PDIR/$PRESET" ] || { echo "!!! [collect $VER] preset thiếu: $PDIR/$PRESET"; return 1; }
  local NAME="${PRESET%.yaml}"
  local SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_analysis/${TS}/${VER}_${NAME}_c${CONCS// /_}_${DATASETS// /_}"
  mkdir -p "$SWEEP_ROOT"
  export EPLB_COLLECT_DIR="$SWEEP_ROOT/$NAME/eplb_tokens"     # curl POST /collect_eplb vào đây
  echo "======== [collect $VER] PRESET=$PRESET conc=$CONCS ds=$DATASETS -> $SWEEP_ROOT ========"
  PDIR="$PDIR" PRESET_LIST="$PRESET" CONCS="$CONCS" DATASETS="$DATASETS" \
    SWEEP_ROOT="$SWEEP_ROOT" COLLECT_ONLY=1 \
    bash "$HERE/sweep_presets_profile.sh"
}

# ============================ SECTION PRESETS ============================
run_preset r0_collect.yaml      # EPLB ON, KHÔNG profiler -> /collect_eplb -> 8 npz
# run_preset <preset>.yaml
# ========================================================================
