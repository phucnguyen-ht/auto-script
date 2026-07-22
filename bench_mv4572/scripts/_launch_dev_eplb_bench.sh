#!/usr/bin/env bash
# [MV-4572 Task 4] Bench the SGLang-ported metric-gated rebalance skip in the DEV container.
# Compares gate OFF (baseline r0 EPLB, rearrange on) vs gate ON (skip rearrange when balanced).
# Runs INSIDE dev container (phuc-nguyen-mv-4572-dev-eplb) => uses dev repo vllm (with the patch).
# AITER_MOREH_ROOT_DIR -> dev repo. norearr_v0.24.0.sh gives debounce (auto-wait GPU) +
# rearrange-count logging (verify gate skips). Full 8k case per research §5.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CT=phuc-nguyen-mv-4572-dev-eplb
DEV_AITER="$WD/retest-v0.24.0-sglang-eplb-dev/vllm-moreh/src/aiter_moreh"
TS="${TS:-$(date +%Y%m%d_%H%M%S)_devgate}"
CONSOLE="$WD/auto-script/bench_mv4572/logs_0.24.0_norearr/console_${TS}.log"
CONCS="${CONCS:-16 64 128 256}"
DATASETS="${DATASETS:-8k}"
echo "[dev-gate] TS=$TS CT=$CT @$(date)" | tee "$CONSOLE"
podman exec -i "$CT" bash -lc \
  "cd $WD && git config --global --add safe.directory '*' >/dev/null 2>&1; \
   AITER_MOREH_ROOT_DIR='$DEV_AITER' TS='$TS' DEBOUNCE='${DEBOUNCE:-600}' \
   PRESET_LIST='r0_v2_gate_off.yaml r0_v2_gate_on.yaml' CONCS='$CONCS' DATASETS='$DATASETS' \
   bash auto-script/bench_mv4572/scripts/norearr_v0.24.0.sh" 2>&1 | tee -a "$CONSOLE"
echo "[dev-gate] DONE TS=$TS @$(date)" | tee -a "$CONSOLE"
