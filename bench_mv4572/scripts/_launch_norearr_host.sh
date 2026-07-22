#!/usr/bin/env bash
# Host-side launcher: exec vào analysis container rồi chạy norearr_v0.24.0.sh (có debounce GPU).
# Chạy trong tmux window trên host mi355-gpu-58. Tee console ra logs_0.24.0_norearr/console_<TS>.log.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CT=phuc-nguyen-mv4572-analysis-0.24.0
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="$WD/auto-script/bench_mv4572/logs_0.24.0_norearr"
mkdir -p "$LOGDIR"
CONSOLE="$LOGDIR/console_$TS.log"
echo "[launch] TS=$TS  console=$CONSOLE  @$(date)" | tee "$CONSOLE"
PRESET_LIST="${PRESET_LIST:-base_v2_gm85.yaml r0_v2.yaml r0_v2_norearr.yaml}"
CONCS="${CONCS:-16 128}"
DATASETS="${DATASETS:-8k}"
DEBOUNCE="${DEBOUNCE:-600}"
podman exec -i "$CT" bash -lc "cd $WD && TS=$TS DEBOUNCE=$DEBOUNCE PRESET_LIST='$PRESET_LIST' CONCS='$CONCS' DATASETS='$DATASETS' bash auto-script/bench_mv4572/scripts/norearr_v0.24.0.sh" 2>&1 | tee -a "$CONSOLE"
echo "[launch] DONE TS=$TS @$(date)" | tee -a "$CONSOLE"
