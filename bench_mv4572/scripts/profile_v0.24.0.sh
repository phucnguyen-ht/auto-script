#!/usr/bin/env bash
# =============================================================================
# MV-4572 deep-analysis — profile runner cho v0.24.0.
# CHẠY TRONG container: phuc-nguyen-mv4572-analysis-0.24.0  (repo vllm-moreh).
# Mỗi `run_preset <file.yaml>` = 1 serve -> warmup -> bench -> tự /collect_eplb -> stop.
# Log/trace/eplb_tokens -> logs_analysis/profile/v0.24.0_<name>_c<conc>_<ds>/<name>/.
#
# ► ĐỔI conc/dataset: sửa dòng CONCS/DATASETS ngay bên dưới.
# ► THÊM/BỚT preset : sửa "SECTION PRESETS" ở CUỐI file (mỗi preset 1 dòng run_preset ...).
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
VER=v0.24.0                                          # <- version CỐ ĐỊNH của file này

# ---- knobs (đổi ở đây) ----
CONCS="${CONCS:-16}"; DATASETS="${DATASETS:-8k}"     # <- ĐỔI conc / dataset tại đây
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"                   # folder datetime: mỗi lần chạy tách riêng (khỏi xoá run cũ)

# ---- env (giữ nếu container đã set; nếu chưa thì fallback) ----
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8_deepanalysis/$VER"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"

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
# [Task2 §4.9] profile GPU-KERNEL-TIME breakdown (tps quá noisy ~13% -> đo GPU-time robust).
# 2x2 {V2,V1} x {base(EPLB off), norearr(EPLB on)} -> breakdown per-kernel: EPLB-on có làm kernel
# NÀO chậm hơn TRONG graph (ngoài routing kernel) không? + sign-flip V2-vs-V1. V2 pair TRƯỚC (primary).
# DIFF-IN-DIFF: có r0-with-rearrange rồi (063732 BREAKDOWN_r0). Cần base pair -> so ncclDevKernel gap
# tại base(V2-V1) vs tại r0(V2-V1): nếu comm-gap ~0 ở base nhưng +4.6s ở r0 -> EPLB dưới V2 gây comm-wait.
run_preset base_v2_prof.yaml           # V2, EPLB OFF -> baseline comm/GEMM (diff-in-diff)
run_preset base_noV2_prof.yaml         # V1, EPLB OFF -> baseline comm/GEMM (diff-in-diff)
run_preset r0_v2_norearr_prof.yaml     # V2, EPLB ON norearr -> tách machinery vs rearrange (GPU-time)
run_preset r0_noV2_norearr_prof.yaml   # V1, EPLB ON norearr -> sign-flip check
# --- cặp cũ (V2 vs V1 EPLB-on): run_preset r0_v2_prof.yaml ; run_preset r0_noV2_prof.yaml
# ========================================================================
