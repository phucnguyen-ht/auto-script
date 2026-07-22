#!/usr/bin/env bash
# =============================================================================
# MV-4572 Task2 (WHY) — MECHANISM SPLIT: rearrange vs per-forward machinery.
#   CHẠY TRONG container phuc-nguyen-mv4572-analysis-0.24.0 (repo vllm-moreh), tmux window `analysis`.
#   BENCH thuần (preset ablation KHÔNG profiler) -> tps.
#
# Preset (dp8ep8/v0.24.0_ablation):
#   V2 triple : base_v2_gm85 (off,0.85) | r0_v2 (rearrange 3000) | r0_v2_norearr (step_interval 1e9)
#   noV2 mirror: base_noV2_gm85 (off,0.85) | r0_noV2 (rearrange) | r0_noV2_norearr
# Logic:
#   base_v2_gm85 -> r0_v2_norearr = chi phí MACHINERY per-forward (mapping kernel + EPLB buffer VRAM)
#   r0_v2_norearr -> r0_v2        = chi phí REARRANGE (all_reduce + copy main-stream + transfer + đổi routing)
#
# LƯU Ý: PHẢI export AITER_MOREH_ROOT_DIR (aiter JIT core.py assert).
# GPU-politeness: debounce GPU free LIÊN TỤC >= DEBOUNCE giây (default 600) trước khi bắt đầu.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/v0.24.0_ablation"
SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_0.24.0_norearr/$TS"

# ---- GPU-politeness: debounce free liên tục >= DEBOUNCE giây ----
DEBOUNCE="${DEBOUNCE:-600}"
POLL="${GPU_POLL_INTERVAL:-30}"
THR="${GPU_VRAM_BUSY_THRESHOLD:-10}"
# gpu_busy_count: số GPU có VRAM% > THR. HARDENED: nếu rocm-smi trả RỖNG (query fail/flaky)
# -> coi là BUSY (in ra 99) để KHÔNG tính nhầm "free" (an toàn politeness).
gpu_busy_count() {
  local out n_lines
  out=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)')
  n_lines=$(printf '%s\n' "$out" | grep -c 'VRAM%')
  if [ "${n_lines}" -eq 0 ]; then echo 99; return; fi   # rocm-smi flaky -> treat busy
  printf '%s\n' "$out" | awk -F': ' '{print $NF+0}' | awk -v t="${THR}" '$1 > t' | wc -l
}
echo "======== [norearr] debounce GPU free >= ${DEBOUNCE}s (poll ${POLL}s, VRAM thr ${THR}%, rocm-smi-empty=busy) ========"
free_since=0
last_beat=0
while :; do
  nb=$(gpu_busy_count)
  now=$(date +%s)
  if [ "${nb}" -eq 0 ]; then
    [ "${free_since}" -eq 0 ] && { free_since=${now}; echo "[norearr] GPU free bắt đầu @$(date +%H:%M:%S)"; }
    elapsed=$(( now - free_since ))
    echo "[norearr] GPU free ${elapsed}s / ${DEBOUNCE}s"
    [ "${elapsed}" -ge "${DEBOUNCE}" ] && { echo "[norearr] debounce OK -> chạy"; break; }
  else
    [ "${free_since}" -ne 0 ] && echo "[norearr] GPU busy lại (${nb} gpu) -> reset đồng hồ @$(date +%H:%M:%S)"
    free_since=0
    # heartbeat mỗi ~300s để biết loop còn sống khi busy kéo dài
    if [ $(( now - last_beat )) -ge 300 ]; then echo "[norearr] ...vẫn đợi GPU (busy=${nb}) @$(date +%H:%M:%S)"; last_beat=${now}; fi
  fi
  sleep "${POLL}"
done

echo "======== [norearr] AITER_MOREH_ROOT_DIR=$AITER_MOREH_ROOT_DIR | TS=$TS ========"
PDIR="$PDIR" \
  PRESET_LIST="${PRESET_LIST:-base_v2_gm85.yaml r0_v2.yaml r0_v2_norearr.yaml}" \
  CONCS="${CONCS:-16 128}" DATASETS="${DATASETS:-8k}" \
  SWEEP_ROOT="$SWEEP_ROOT" \
  bash "$WD/auto-script/bench_mv4572/scripts/sweep_presets.sh"

echo ""; echo "======== [norearr] TÓM TẮT tps + rearrange count ========"
find "$SWEEP_ROOT" -name scenario_summary.csv 2>/dev/null | sort | while read -r f; do
  d="$(dirname "$f")"
  echo "--- $(basename "$d") ---"; cat "$f"
  # verify-by-logging: đếm số lần rearrange thực tế (serve.log "Rearranging experts")
  sl="$d/serve.log"; [ -f "$sl" ] || sl="$(dirname "$d")/serve.log"
  if [ -f "$sl" ]; then
    echo "  [rearrange count] $(grep -c 'Rearranging experts' "$sl" 2>/dev/null) (serve.log)"
  fi
done
