#!/usr/bin/env bash
# =============================================================================
# MV-4572 phuc-ablation PROFILE — torch-profiler traces (8 rank) cho A0 + A8 để so per-kernel/
#   section tìm +8% (§11.5 fork GPU-vs-host). Chạy TRONG container phuc-nguyen-mv4572-analysis-0.24.0.
#   Preset: dp8ep8/phuc-ablation-profile/{A0,A8}.yaml (có profiler_config). Logs -> logs_phuc_ablation_profile/.
#   token_buffer/npz RỖNG cho A0/A8 (option-1: A0 eplb-off, A8 skip_step) -> KHÔNG set EPLB_COLLECT_DIR
#   (không chờ npz). Chỉ chờ 8 trace/rank rồi stop. profile()→dump() CHẬM — ĐỢI xong, KHÔNG kill ngang.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/phuc-ablation-profile"
LOGROOT="$WD/auto-script/bench_mv4572/logs_phuc_ablation_profile"
CONCS="${CONCS:-16}"; DATASETS="${DATASETS:-8k}"
PRESET_LIST="${PRESET_LIST:-A0.yaml A8.yaml}"
EXPECT_TRACES="${EXPECT_TRACES:-8}"
mkdir -p "$LOGROOT/$TS"

# ---- GPU-politeness: debounce free liên tục >= DEBOUNCE giây ----
DEBOUNCE="${DEBOUNCE:-120}"; POLL="${GPU_POLL_INTERVAL:-30}"; THR="${GPU_VRAM_BUSY_THRESHOLD:-10}"
gpu_busy_count() {
  local out n_lines
  out=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)')
  n_lines=$(printf '%s\n' "$out" | grep -c 'VRAM%')
  if [ "${n_lines}" -eq 0 ]; then echo 99; return; fi
  printf '%s\n' "$out" | awk -F': ' '{print $NF+0}' | awk -v t="${THR}" '$1 > t' | wc -l
}
echo "======== [phuc-profile] TS=$TS | presets='$PRESET_LIST' | conc=$CONCS ds=$DATASETS -> $LOGROOT/$TS ========"
echo "======== [phuc-profile] debounce GPU free >= ${DEBOUNCE}s ========"
free_since=0; last_beat=0
while :; do
  nb=$(gpu_busy_count); now=$(date +%s)
  if [ "${nb}" -eq 0 ]; then
    [ "${free_since}" -eq 0 ] && { free_since=${now}; echo "[phuc-profile] GPU free bắt đầu @$(date +%H:%M:%S)"; }
    elapsed=$(( now - free_since )); echo "[phuc-profile] GPU free ${elapsed}s / ${DEBOUNCE}s"
    [ "${elapsed}" -ge "${DEBOUNCE}" ] && { echo "[phuc-profile] debounce OK -> chạy"; break; }
  else
    [ "${free_since}" -ne 0 ] && echo "[phuc-profile] GPU busy lại (${nb}) -> reset @$(date +%H:%M:%S)"
    free_since=0
    if [ $(( now - last_beat )) -ge 300 ]; then echo "[phuc-profile] ...đợi GPU (busy=${nb})"; last_beat=${now}; fi
  fi
  sleep "${POLL}"
done

# ---- Profile TỪNG preset riêng (1 serve -> bench 1 scenario -> chờ 8 trace -> stop) ----
for PRESET in $PRESET_LIST; do
  [ -f "$PDIR/$PRESET" ] || { echo "!!! [phuc-profile] thiếu preset $PDIR/$PRESET"; continue; }
  NAME="${PRESET%.yaml}"
  SWEEP_ROOT="$LOGROOT/$TS/${NAME}_c${CONCS// /_}_${DATASETS// /_}"
  mkdir -p "$SWEEP_ROOT"
  echo "======== [phuc-profile] PRESET=$PRESET -> $SWEEP_ROOT ========"
  # KHÔNG export EPLB_COLLECT_DIR: token_buffer rỗng cho A0/A8 -> khỏi chờ npz (chỉ chờ trace).
  PDIR="$PDIR" PRESET_LIST="$PRESET" CONCS="$CONCS" DATASETS="$DATASETS" \
    SWEEP_ROOT="$SWEEP_ROOT" EXPECT_TRACES="$EXPECT_TRACES" \
    bash "$HERE/sweep_presets_profile.sh"
done

echo ""; echo "======== [phuc-profile] TÓM TẮT trace count ========"
for PRESET in $PRESET_LIST; do
  NAME="${PRESET%.yaml}"
  d=$(find "$LOGROOT/$TS/${NAME}_c"* -type d -name profiling_result 2>/dev/null | head -1)
  n=$(find "$d" -type f \( -name '*.pt.trace.json*' -o -name '*.json*' \) 2>/dev/null | wc -l)
  echo "  $NAME: ${n} trace file(s) @ ${d:-<none>}"
done
echo "======== [phuc-profile] DONE -> $LOGROOT/$TS ========"
