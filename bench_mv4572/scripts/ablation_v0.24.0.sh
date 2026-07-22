#!/usr/bin/env bash
# =============================================================================
# MV-4572 phuc-ablation — bisect ROOT-CAUSE của regression "enable_eplb=true × V2".
#   Chạy TRONG container phuc-nguyen-mv4572-analysis-0.24.0 (repo vllm-moreh), tmux window `analysis`.
#   BENCH thuần (KHÔNG profiler) @ conc16 / ISL 8k -> scenario_summary.csv (requests/mean_decode_tps/
#   mean_tpot/output_tps). Docs thiết kế: docs/regression/phuc-analyse.md §9.
#
# 3 preset (auto-script/presets/kimi2.6.mxfp4/dp8ep8/phuc-ablation), CHẠY THEO THỨ TỰ A0 -> A6 -> A5:
#   A0 = base (EPLB OFF, V2, 0.85)                         -> MỐC reference.
#   A6 = EPLB ON + no-rearrange + MV4572_EPLB_LAYER_OFF -> tắt TOÀN BỘ layer-effect (trục C construction).
#   A5 = EPLB ON + no-rearrange + SKIP_VALIDATE+REMAP+STEP  -> tắt mọi RUNTIME (L1+L2+R), giữ construction.
# Đọc: A6 recover về A0 nhiều hơn A5 => root-cause = construction; A5~A6 => runtime; cả 2 không recover => runner.
#
# CÁCH CHẠY (host mi355-gpu-58, tmux window `analysis`):
#   podman exec -i phuc-nguyen-mv4572-analysis-0.24.0 bash -lc \
#     'cd /shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572 && \
#      bash auto-script/bench_mv4572/scripts/ablation_v0.24.0.sh 2>&1 | tee auto-script/bench_mv4572/logs_phuc_ablation/console_$(date +%Y%m%d_%H%M%S).log'
#
# LƯU Ý:
#   - A5/A6 dùng SKIP_EPLB_REMAP / LAYER_OFF -> CHỈ đúng vì no-rearrange (identity). KHÔNG dùng khi rearrange thật.
#   - GPU-politeness: debounce GPU free liên tục >= DEBOUNCE giây (default 600) trước khi bắt đầu.
#   - PHẢI export AITER_MOREH_ROOT_DIR (aiter JIT core.py assert). Serve LẦN ĐẦU chậm (JIT .so), không phải hang.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/${PRESET_SUBDIR:-phuc-ablation}"
LOGROOT="${LOGROOT:-$WD/auto-script/bench_mv4572/logs_phuc_ablation}"
SWEEP_ROOT="$LOGROOT/$TS"
mkdir -p "$SWEEP_ROOT"

# THỨ TỰ chạy: A0 -> A6 -> A5 (theo yêu cầu).
PRESET_LIST="${PRESET_LIST:-A0.yaml A6.yaml A5.yaml}"
CONCS="${CONCS:-16}"
DATASETS="${DATASETS:-8k}"

# ---- GPU-politeness: debounce free liên tục >= DEBOUNCE giây (giống norearr_v0.24.0.sh) ----
DEBOUNCE="${DEBOUNCE:-0}"
POLL="${GPU_POLL_INTERVAL:-30}"
THR="${GPU_VRAM_BUSY_THRESHOLD:-10}"
gpu_busy_count() {  # rocm-smi trả rỗng (flaky) -> coi BUSY (an toàn politeness)
  local out n_lines
  out=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)')
  n_lines=$(printf '%s\n' "$out" | grep -c 'VRAM%')
  if [ "${n_lines}" -eq 0 ]; then echo 99; return; fi
  printf '%s\n' "$out" | awk -F': ' '{print $NF+0}' | awk -v t="${THR}" '$1 > t' | wc -l
}
echo "======== [phuc-ablation] TS=$TS | presets='$PRESET_LIST' | conc=$CONCS ds=$DATASETS -> $SWEEP_ROOT ========"
echo "======== [phuc-ablation] debounce GPU free >= ${DEBOUNCE}s (poll ${POLL}s, VRAM thr ${THR}%) ========"
free_since=0; last_beat=0
while :; do
  nb=$(gpu_busy_count); now=$(date +%s)
  if [ "${nb}" -eq 0 ]; then
    [ "${free_since}" -eq 0 ] && { free_since=${now}; echo "[phuc-ablation] GPU free bắt đầu @$(date +%H:%M:%S)"; }
    elapsed=$(( now - free_since )); echo "[phuc-ablation] GPU free ${elapsed}s / ${DEBOUNCE}s"
    [ "${elapsed}" -ge "${DEBOUNCE}" ] && { echo "[phuc-ablation] debounce OK -> chạy"; break; }
  else
    [ "${free_since}" -ne 0 ] && echo "[phuc-ablation] GPU busy lại (${nb} gpu) -> reset @$(date +%H:%M:%S)"
    free_since=0
    if [ $(( now - last_beat )) -ge 300 ]; then echo "[phuc-ablation] ...vẫn đợi GPU (busy=${nb}) @$(date +%H:%M:%S)"; last_beat=${now}; fi
  fi
  sleep "${POLL}"
done

echo "======== [phuc-ablation] AITER_MOREH_ROOT_DIR=$AITER_MOREH_ROOT_DIR ========"
PDIR="$PDIR" \
  PRESET_LIST="$PRESET_LIST" \
  CONCS="$CONCS" DATASETS="$DATASETS" \
  SWEEP_ROOT="$SWEEP_ROOT" \
  bash "$WD/auto-script/bench_mv4572/scripts/sweep_presets.sh"

# ---- TÓM TẮT: scenario_summary + verify no-rearrange + verify gate ăn ----
echo ""; echo "======== [phuc-ablation] TÓM TẮT (theo PRESET_LIST) ========"
for f in $PRESET_LIST; do
  name="${f%.yaml}"
  d="$SWEEP_ROOT/$name"
  [ -d "$d" ] || { echo "--- $name: (chưa có log) ---"; continue; }
  echo "--- $name ---"
  csv="$d/scenario_summary.csv"; [ -f "$csv" ] && cat "$csv" || echo "  (thiếu scenario_summary.csv)"
  sl="$d/serve.log"; [ -f "$sl" ] || sl="$(find "$d" -name serve.log 2>/dev/null | head -1)"
  if [ -n "${sl:-}" ] && [ -f "$sl" ]; then
    echo "  [rearrange count] $(grep -c 'Rearranging experts' "$sl" 2>/dev/null) (kỳ vọng: A0=0, A5/A6 chỉ 1 dòng '(profile)' = 0 rearrange thật)"
    grep -q "EPLB_LAYER_OFF=1" "$sl" 2>/dev/null && echo "  [gate] MV4572_EPLB_LAYER_OFF ĐÃ ăn (thấy log build EPLB-off)"
  fi
done
echo "======== [phuc-ablation] DONE -> $SWEEP_ROOT ========"
