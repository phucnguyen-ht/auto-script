#!/usr/bin/env bash
# =============================================================================
# MV-4572 deep-analysis — profile runner cho v0.23.0.
# CHẠY TRONG container: phuc-nguyen-mv4572-pr-0.23.0  (repo retest-0.23.0/vllm-moreh).
# Mỗi `run_preset <file.yaml>` = 1 serve -> warmup -> bench -> tự /collect_eplb -> stop.
# Log/trace/eplb_tokens -> logs_analysis/profile/v0.23.0_<name>_c<conc>_<ds>/<name>/.
#
# ► ĐỔI conc/dataset: sửa dòng CONCS/DATASETS ngay bên dưới.
# ► THÊM/BỚT preset : sửa "SECTION PRESETS" ở CUỐI file (mỗi preset 1 dòng run_preset ...).
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
VER=v0.23.0                                          # <- version CỐ ĐỊNH của file này

# ---- knobs (đổi ở đây) ----
CONCS="${CONCS:-16}"; DATASETS="${DATASETS:-8k}"     # <- ĐỔI conc / dataset tại đây
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"                   # folder datetime: mỗi lần chạy tách riêng (khỏi xoá run cũ)

# ---- env (giữ nếu container đã set; nếu chưa thì fallback) ----
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8_deepanalysis/$VER"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/retest-0.23.0/vllm-moreh/src/aiter_moreh}"

echo "======== [profile] LOGS -> auto-script/bench_mv4572/logs_analysis/${TS}/  (TS=${TS}) ========"

run_preset() {
  local PRESET="$1"
  [ -f "$PDIR/$PRESET" ] || { echo "!!! [profile $VER] preset thiếu: $PDIR/$PRESET"; return 1; }
  local NAME="${PRESET%.yaml}"
  local SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_analysis/${TS}/${VER}_${NAME}_c${CONCS// /_}_${DATASETS// /_}"
  mkdir -p "$SWEEP_ROOT"
  export EPLB_COLLECT_DIR="$SWEEP_ROOT/$NAME/eplb_tokens"     # driver POST /collect_eplb vào đây
  echo "======== [profile $VER] PRESET=$PRESET conc=$CONCS ds=$DATASETS -> $SWEEP_ROOT ========"
  PDIR="$PDIR" PRESET_LIST="$PRESET" CONCS="$CONCS" DATASETS="$DATASETS" \
    SWEEP_ROOT="$SWEEP_ROOT" EXPECT_TRACES="${EXPECT_TRACES:-8}" \
    TRACE_TIMEOUT="${TRACE_TIMEOUT:-10800}" \
    bash "$HERE/sweep_presets_profile.sh"
}

# ============================ SECTION PRESETS ============================
# Chạy tuần tự (mỗi dòng = 1 serve riêng). Append/comment tuỳ ý.
run_preset r0.yaml      # EPLB ON  -> #tokens + rearrange + time + #experts
run_preset base.yaml    # EPLB OFF -> time only (control)
# run_preset <preset>.yaml
# ========================================================================
