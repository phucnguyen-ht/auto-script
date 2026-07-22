#!/usr/bin/env bash
# Host-side launcher (mi355-gpu-58): podman exec vào analysis container rồi chạy ablation_v0.24.0.sh
# với PRESET_SUBDIR=phuc-ablation-fwddev + LOGROOT=logs_phuc_ablation_fwddev (bench THƯỜNG + FWDDEV probe).
# Tee console -> logs_phuc_ablation_fwddev/console_<TS>.log. Chạy trong tmux.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CT=phuc-nguyen-mv4572-analysis-0.24.0
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="$WD/auto-script/bench_mv4572/logs_phuc_ablation_fwddev"
mkdir -p "$LOGDIR"
CONSOLE="$LOGDIR/console_$TS.log"
PRESET_LIST="${PRESET_LIST:-A0.yaml A8.yaml}"
CONCS="${CONCS:-16}"; DATASETS="${DATASETS:-8k}"; DEBOUNCE="${DEBOUNCE:-120}"
echo "[launch] TS=$TS console=$CONSOLE presets='$PRESET_LIST' conc=$CONCS ds=$DATASETS debounce=${DEBOUNCE}s @$(date)" | tee "$CONSOLE"
podman exec -i "$CT" bash -lc "cd $WD && TS=$TS DEBOUNCE=$DEBOUNCE PRESET_SUBDIR=phuc-ablation-fwddev LOGROOT=$LOGDIR PRESET_LIST='$PRESET_LIST' CONCS='$CONCS' DATASETS='$DATASETS' bash auto-script/bench_mv4572/scripts/ablation_v0.24.0.sh" 2>&1 | tee -a "$CONSOLE"
echo "[launch] DONE TS=$TS @$(date)" | tee -a "$CONSOLE"
