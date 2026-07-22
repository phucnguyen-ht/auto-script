#!/usr/bin/env bash
# RUN #2 — verification: dp8ep8/default-r0 bench ALONE on the freshly-cleaned PR container,
# to determine whether Run #1's mid-bench coordinator death was transient (dirty-state
# cascade) or systemic. Bakes in the clean-state-before-serve fix. Runs in tmux pr- so the
# user can watch. death_forensics.sh runs alongside to LOG the kill cause if it recurs.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CONT=phuc-nguyen-mv4572-pr-0.24.0
AITER=$WD/retest/vllm-moreh/src/aiter_moreh
SCRIPTS=$WD/auto-script/bench_mv4572/scripts
PDIR=$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8
SR=$WD/auto-script/bench_mv4572/logs_v0.24.0/pr_decode/campaign_run2_r0verify
LOG=$WD/auto-script/bench_mv4572/logs_v0.24.0/pr_decode/RUN2_r0verify.log
mkdir -p "$SR"
echo "=== [$(date -u)] RUN2 r0verify START ===" | tee "$LOG"
# clean-state-before-serve (Run #1 cascade fix). NOTE: run pkill DIRECTLY via podman exec
# (NOT inside bash -lc whose cmdline contains the pattern) — a `bash -lc "...vllm..."` gets
# matched by `pkill -f vllm` and self-kills before cleaning. Direct pkill excludes its own PID.
podman exec "$CONT" pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 1
podman exec "$CONT" pkill -9 VLLM 2>/dev/null || true; sleep 3
podman exec "$CONT" bash -c 'rm -f /tmp/aiter_configs/*.lock 2>/dev/null; echo "[clean] leftover=$(pgrep -cf "vllm serve|VLLM::") locks=$(ls /tmp/aiter_configs/*.lock 2>/dev/null | wc -l)"' 2>&1 | tee -a "$LOG"
# r0 bench (full concs x datasets, 1 serve). check_nixl advisory only.
podman exec "$CONT" bash -lc "git config --global --add safe.directory '*'; export AITER_MOREH_ROOT_DIR='$AITER'; export VLLM_ENGINE_READY_TIMEOUT_S=1800; cd '$SCRIPTS' && PDIR='$PDIR' PRESET_LIST='base-eplb-nixl-async-default-r0.yaml' SWEEP_ROOT='$SR' bash sweep_presets.sh" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
ns=$(find "$SR" -name scenario_summary.csv 2>/dev/null | wc -l)
echo "=== [$(date -u)] RUN2 r0verify DONE rc=$rc summaries=$ns ===" | tee -a "$LOG"
touch "$SR/.done"; echo "$rc:$ns" > "$SR/.result"
