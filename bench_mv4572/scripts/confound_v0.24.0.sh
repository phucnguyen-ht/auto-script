#!/usr/bin/env bash
# =============================================================================
# MV-4572 Task2(a) Phase A — CONFOUND check + verify refutation (conc16/8k ONLY).
#  1) EPLB regress có SỐNG khi loại confound gpu_mem? -> base_v2_gm85 (V2,0.85,EPLB off)
#     vs r0_v2 (V2,0.85,EPLB on). So với confound run (base_v2@0.9 vs r0_v2@0.85 = -11.7% tps).
#  2) Verify (RULE1, điểm 3 user): serve.log bắt [MV-4572] EngineCore cfg -> xác nhận
#     max_concurrent_batches THẬT: V2 (base_v2_gm85/r0_v2) vs V1 (base_noV2).
# BENCH thuần (no profiler). Container analysis, tmux confound-v024.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
export AITER_MOREH_ROOT_DIR="${AITER_MOREH_ROOT_DIR:-$WD/vllm-moreh/src/aiter_moreh}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PDIR="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/v0.24.0_ablation"
SWEEP_ROOT="$WD/auto-script/bench_mv4572/logs_0.24.0_confound/$TS"

echo "======== [confound] TS=$TS | conc16/8k | base_v2_gm85 + r0_v2 + base_noV2 ========"
PDIR="$PDIR" \
  PRESET_LIST="${PRESET_LIST:-base_v2_gm85.yaml r0_v2.yaml base_noV2.yaml}" \
  CONCS="${CONCS:-16}" DATASETS="${DATASETS:-8k}" \
  SWEEP_ROOT="$SWEEP_ROOT" \
  bash "$WD/auto-script/bench_mv4572/scripts/sweep_presets.sh"

echo ""; echo "======== [confound] tps + [MV-4572] EngineCore cfg ========"
find "$SWEEP_ROOT" -name scenario_summary.csv 2>/dev/null | sort | while read -r f; do
  echo "--- $(basename "$(dirname "$f")") ---"; cat "$f"
done
echo "--- [MV-4572] runtime cfg (verify max_concurrent_batches V1 vs V2) ---"
find "$SWEEP_ROOT" -name serve.log 2>/dev/null | sort | while read -r f; do
  echo "[$(basename "$(dirname "$f")")] $(grep -am1 '\[MV-4572\] EngineCore cfg' "$f")"
done
