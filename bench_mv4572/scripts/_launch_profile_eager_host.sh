#!/usr/bin/env bash
# [MV-4572 Task 2] Host launcher: EAGER-mode profile (enforce_eager -> NO cudagraph ->
# NO capture-counting confound §4.9.10) for base_v2 + r0_v2 -> clean MoE sub-kernel breakdown
# to pin the exact op that EPLB inflates. Analysis container, tmux. conc16/8k.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CT=phuc-nguyen-mv4572-analysis-0.24.0
TS="${TS:-$(date +%Y%m%d_%H%M%S)_eager}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8_deepanalysis/v0.24.0"
LOGDIR="$WD/auto-script/bench_mv4572/logs_analysis"
CONSOLE="$LOGDIR/console_${TS}.log"
echo "[eager-prof] TS=$TS @$(date)" | tee "$CONSOLE"
export AITER_MOREH_ROOT_DIR="$WD/vllm-moreh/src/aiter_moreh"
for PRESET in base_v2_prof_eager.yaml r0_v2_norearr_prof_eager.yaml; do
  NAME="${PRESET%.yaml}"
  SWEEP_ROOT="$LOGDIR/${TS}/v0.24.0_${NAME}_c16_8k"
  echo "[eager-prof] PRESET=$PRESET -> $SWEEP_ROOT" | tee -a "$CONSOLE"
  podman exec -i "$CT" bash -lc \
    "cd $WD && export AITER_MOREH_ROOT_DIR=$WD/vllm-moreh/src/aiter_moreh && \
     PDIR='$PDIR' PRESET_LIST='$PRESET' CONCS='16' DATASETS='8k' SWEEP_ROOT='$SWEEP_ROOT' \
     bash auto-script/bench_mv4572/scripts/sweep_presets_profile.sh" 2>&1 | tee -a "$CONSOLE"
done
echo "[eager-prof] DONE TS=$TS @$(date)" | tee -a "$CONSOLE"
