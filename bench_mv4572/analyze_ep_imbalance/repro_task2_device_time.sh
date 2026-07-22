#!/usr/bin/env bash
# =============================================================================
# [MV-4572 Task 2] REPRODUCE — per-forward GPU device-time (cuda-event, live) proving
# the V2×EPLB regression is real GPU device-time in the forward, + device-time ablation
# proving it is NOT routing-kernel / eplb.step but the enable_eplb-CONSTRUCTION dispatch.
#
# RULE 2 reproduce artifact. Two parts:
#   PART A (PARSE existing logs — no GPU): compute the 2×2 + ablation from serve.logs.
#   PART B (RE-RUN on GPU — optional): how to regenerate the logs from scratch.
#
# Evidence source (already run): logs_0.24.0_norearr/20260718_fwddev  (2x2: base/r0 x V2/V1)
#                                logs_0.24.0_norearr/20260718_fwdabl   (ablation on r0_v2)
# Probe: vllm-moreh/3rdparty/vllm/vllm/v1/worker/mv4572_fwddev.py (gate MV4572_FWDDEV=1);
#   wired at V2 gpu/model_runner.py:~1265, V1 gpu_model_runner.py:~4341.
# =============================================================================
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
cd "$WD"

parse() {  # $1 = run TS dir under logs_0.24.0_norearr ; $2.. = preset tags
  local dir="auto-script/bench_mv4572/logs_0.24.0_norearr/$1"; shift
  printf "%-38s %10s %10s\n" "config" "dev_ms" "tpot_ms"
  for tag in "$@"; do
    local log="$dir/$tag/serve.log"
    [ -f "$log" ] || { printf "%-38s %10s\n" "$tag" "MISSING"; continue; }
    # mean of last 8 stable [MV4572 FWDDEV] readings (skip warmup-heavy early)
    local dev; dev=$(grep "MV4572 FWDDEV" "$log" 2>/dev/null \
      | grep -oE "mean_fwd_dev_ms=[0-9.]+" | sed 's/.*=//' | tail -8 \
      | awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
    local ss; ss=$(find "$dir/$tag" -name scenario_summary.csv 2>/dev/null | head -1)
    local tpot; tpot=$([ -n "$ss" ] && sed -n '2p' "$ss" | cut -d, -f7 || echo NA)
    printf "%-38s %10s %10s\n" "$tag" "$dev" "$tpot"
  done
}

echo "################ PART A.1 — 2x2 per-forward device-time (run 20260718_fwddev) ################"
echo "# adjacent pairs = clock-stable: (base_v2,r0_v2) and (base_noV2,r0_noV2)"
parse 20260718_fwddev base_v2_gm85_fwddev r0_v2_norearr_fwddev base_noV2_gm85_fwddev r0_noV2_norearr_fwddev
echo
echo "# READ: EPLB effect per runner (adjacent, clock-clean): V2 base->r0 = +~19%; V1 base->r0 = -~7%."
echo "#       confound-free V2/V1 at each gpu_mem: base ~0.87 (V2 faster) -> r0 ~1.11 (V2 slower) = SIGN-FLIP."
echo
echo "################ PART A.2 — device-time ABLATION on r0_v2 (run 20260718_fwdabl) ################"
echo "# r0_v2 EPLB-on variants are adjacent = clock-stable; base_v2 = floor."
parse 20260718_fwdabl r0_v2_norearr_fwddev r0_v2_norearr_fwddev_noremap r0_v2_norearr_fwddev_nostep r0_v2_norearr_fwddev_noboth base_v2_gm85_fwddev
echo
echo "# READ: skip-remap ~ -1.4%, skip-step ~0, skip-BOTH ~0 recovery (all EPLB-on variants cluster)."
echo "#       => routing kernel + eplb.step are NOT the device-time cost."
echo "#       => cost = enable_eplb-CONSTRUCTION dispatch effect (survives skipping both)."
echo
cat <<'PARTB'
################ PART B — RE-RUN from scratch (GPU, analysis container, tmux) ################
# 1) Presets (already created, dp8ep8/v0.24.0_ablation/): *_fwddev.yaml add `MV4572_FWDDEV: "1"`;
#    ablation adds MV4572_SKIP_EPLB_REMAP/STEP. Recreate:
#      PD=auto-script/presets/kimi2.6.mxfp4/dp8ep8/v0.24.0_ablation
#      awk '/^env_vars:/{print;print "  MV4572_FWDDEV: \"1\"";next}{print}' $PD/base_v2_gm85.yaml > $PD/base_v2_gm85_fwddev.yaml   # etc
# 2) 2x2 run (tmux window on mi355-gpu-58):
#      TS=<ts> PRESET_LIST="base_v2_gm85_fwddev.yaml r0_v2_norearr_fwddev.yaml base_noV2_gm85_fwddev.yaml r0_noV2_norearr_fwddev.yaml" \
#        CONCS=16 DATASETS=8k DEBOUNCE=600 bash auto-script/bench_mv4572/scripts/_launch_norearr_host.sh
# 3) Ablation run:
#      TS=<ts> PRESET_LIST="r0_v2_norearr_fwddev.yaml r0_v2_norearr_fwddev_noremap.yaml r0_v2_norearr_fwddev_nostep.yaml r0_v2_norearr_fwddev_noboth.yaml base_v2_gm85_fwddev.yaml" \
#        CONCS=16 DATASETS=8k DEBOUNCE=600 bash auto-script/bench_mv4572/scripts/_launch_norearr_host.sh
# 4) Parse: bash auto-script/bench_mv4572/analyze_ep_imbalance/repro_task2_device_time.sh  (edit TS above)
PARTB
