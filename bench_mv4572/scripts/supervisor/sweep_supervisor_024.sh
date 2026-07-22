#!/usr/bin/env bash
# Host-side supervisor for the MV-4572 campaign. Runs ON mi355-gpu-58 (NOT in the
# container) so it survives a `podman stop` of our container. A login-node watcher
# relaunches it if the host reaps it (campaign is fully resumable, so no data lost).
#
# Sequence (both phases gated on the GPUs being free of OTHER users' containers via
# police.sh, polled every POLL s -- we never stomp a colleague):
#   PHASE acc   : run_all_full_presets_vllm.sh (accuracy check). Run ONCE (ACC_DONE marker).
#   PHASE sweep : sweep_presets.sh (DATASETS 8k,10k,100k; 12 presets). Resumable
#                 per-preset via the scenario_summary.csv skip-guard.
#
# On a colleague `podman stop` of our container mid-run: the `podman exec` returns and
# the container is down -> the in-flight preset is UNFINISHED, its partial dir is MOVED
# out of the results tree (-> logs/sweep_interrupted/, never deleted), we back off
# BACKOFF s, wait for the GPUs to free, `podman restart` our container, and resume.
# NEVER recreate the container (only `podman restart`).
set -uo pipefail

CONTAINER="${CONTAINER:-phuc-nguyen-mv4572-rebench}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572}"
TICKET="${WORKING_DIR}/auto-script/bench_mv4572"
SCRIPTS="${TICKET}/scripts"
POLICE="${POLICE:-/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh}"
BACKOFF="${BACKOFF:-300}"
POLL="${POLL:-60}"
MAX_RELAUNCH="${MAX_RELAUNCH:-500}"
# vllm-moreh imports aiter at startup, which asserts AITER_MOREH_ROOT_DIR is set. Our
# non-interactive `podman exec bash -lc` lacks setup_dev.sh's export, so set it here.
AITER_MOREH_ROOT_DIR_VAL="${AITER_MOREH_ROOT_DIR_VAL:-${WORKING_DIR}/vllm-moreh/src/aiter_moreh}"

mkdir -p "${TICKET}/logs/sweep_supervisor"
LOCKF="${TICKET}/logs/sweep_supervisor_024/supervisor.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCKF}"; flock -n 9 || { echo "[sup] another supervisor running -> exit"; exit 0; }
fi

CUR="${TICKET}/logs/sweep_supervisor_024/CURRENT_CAMPAIGN"
if [ -f "${CUR}" ]; then CAMPAIGN_TS="$(cat "${CUR}")"; else CAMPAIGN_TS="$(date +%Y%m%d_%H%M%S)"; echo "${CAMPAIGN_TS}" >"${CUR}"; fi

CAMPAIGN="${TICKET}/logs/sweep_024_decode/campaign_${CAMPAIGN_TS}"
INTERRUPTED="${TICKET}/logs/sweep_interrupted/campaign_${CAMPAIGN_TS}"
STATE="${TICKET}/logs/sweep_supervisor_024/campaign_${CAMPAIGN_TS}"
LOG="${STATE}/supervisor.log"
mkdir -p "${CAMPAIGN}" "${INTERRUPTED}" "${STATE}"
rm -f "${STATE}/DONE" "${STATE}/ESCALATE"

log()    { echo "[sup $(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
status() { echo "$1" >"${STATE}/status"; }
# Heartbeat self-terminates when the main supervisor ($$) dies -- even on SIGKILL
# (reaper), where the EXIT trap can't run -- so it never orphans and holds the flock.
( while kill -0 "$$" 2>/dev/null; do : >"${STATE}/heartbeat"; sleep 30; done ) & HB_PID=$!
trap 'kill "${HB_PID}" 2>/dev/null' EXIT

foreign_list() { bash "${POLICE}" 2>/dev/null | awk '/^[[:space:]]+[^[:space:]].*id=/{print $1}' | grep -vx "${CONTAINER}" || true; }
foreign_busy() { [ -n "$(foreign_list)" ]; }
wait_free() {
  while foreign_busy; do
    status waiting_gpu
    log "GPU busy by: $(foreign_list | paste -sd,) -> wait ${POLL}s (being polite)"
    sleep "${POLL}"
  done
  log "GPU free (no foreign container) -> proceed"
}
container_running() { [ "$(podman inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" = "true" ]; }
ensure_container() {
  container_running && return 0
  if podman container exists "${CONTAINER}" 2>/dev/null; then
    log "container down -> podman restart ${CONTAINER}"
    podman restart "${CONTAINER}" >>"${LOG}" 2>&1 || { log "podman restart FAILED"; return 1; }
    sleep 5; container_running && { log "container back up"; return 0; }
    log "still not running after restart"; return 1
  fi
  log "ESCALATE: container ${CONTAINER} gone; NOT recreating per brief"
  status escalate_container_gone; : >"${STATE}/ESCALATE"; return 1
}
# All engine procs are named VLLM::* -> one pkill reaps them; container PID ns => only ours.
clean_container() { podman exec "${CONTAINER}" bash -lc "pkill -9 VLLM 2>/dev/null || true" >>"${LOG}" 2>&1 || true; sleep 3; }
move_partials() {
  local root="$1" d name dest; [ -d "${root}" ] || return 0; shopt -s nullglob
  for d in "${root}"/*/; do
    name="$(basename "${d}")"
    if [ ! -f "${d}scenario_summary.csv" ]; then
      dest="${INTERRUPTED}/${name}__$(date +%Y%m%d_%H%M%S)"; mkdir -p "${INTERRUPTED}"
      log "MOVE unfinished preset '${name}' -> ${dest}"; mv "${d%/}" "${dest}" 2>>"${LOG}" || log "  (move failed ${name})"
    fi
  done; shopt -u nullglob
}

relaunch=0
backoff_after_interrupt() {
  relaunch=$((relaunch+1))
  if [ "${relaunch}" -ge "${MAX_RELAUNCH}" ]; then log "MAX_RELAUNCH hit -> ESCALATE"; status escalate_maxrelaunch; : >"${STATE}/ESCALATE"; exit 1; fi
  status interrupted_backoff
  log "interrupted -> back off ${BACKOFF}s then resume (relaunch #${relaunch})"
  sleep "${BACKOFF}"
}

# ---- PHASE acc: accuracy check via run_all_full_presets_vllm.sh (run once) ----
run_acc() {
  [ -f "${STATE}/ACC_DONE" ] && { log "ACC already done -> skip"; return 0; }
  while :; do
    wait_free
    ensure_container || { sleep "${POLL}"; continue; }
    clean_container
    status "running:acc"
    log "LAUNCH acc: run_all_full_presets_vllm.sh"
    podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; cd '${TICKET}' && bash run_all_full_presets_vllm.sh" 2>&1 | tee -a "${LOG}"
    if container_running; then : >"${STATE}/ACC_DONE"; log "ACC phase complete"; return 0; fi
    log "acc INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

# ---- PHASE acc2: SECOND accuracy round, READABLE-ONLY (no eval), base + r0 ----
run_acc2() {
  [ -f "${STATE}/ACC2_DONE" ] && { log "ACC2 already done -> skip"; return 0; }
  while :; do
    wait_free
    ensure_container || { sleep "${POLL}"; continue; }
    clean_container
    status "running:acc2"
    log "LAUNCH acc2: run_all.sh RUN_EVAL=0 for base + eplb-nixl-async-default-r8 (readable only)"
    podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; cd '${TICKET}' && PRESET=kimi2.6.mxfp4/dp8ep8/base.yaml RUN_EVAL=0 bash run_all.sh; PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r8.yaml RUN_EVAL=0 bash run_all.sh" 2>&1 | tee -a "${LOG}"
    if container_running; then : >"${STATE}/ACC2_DONE"; log "ACC2 phase complete"; return 0; fi
    log "acc2 INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

# ---- PHASE sweep: sweep_presets.sh, resumable per-preset ----
run_sweep() {
  while :; do
    wait_free
    ensure_container || { sleep "${POLL}"; continue; }
    move_partials "${CAMPAIGN}"
    clean_container
    status "running:sweep"
    log "LAUNCH sweep: sweep_presets.sh (SWEEP_ROOT=${CAMPAIGN})"
    # VLLM_ENGINE_READY_TIMEOUT_S=1800: 0.24.0 1T-model cold start (fast_moe_cold_start=False)
    # exceeds the 600s default -> ApiServer TimeoutError. 30min gives headroom (a wait, not a
    # behaviour change; keeps 0.24.0 defaults).
    podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; export VLLM_ENGINE_READY_TIMEOUT_S=1800; cd '${SCRIPTS}' && SWEEP_ROOT='${CAMPAIGN}' bash sweep_presets.sh" 2>&1 | tee -a "${LOG}"
    if container_running; then log "sweep phase finished; summaries: $(find "${CAMPAIGN}" -name scenario_summary.csv 2>/dev/null | wc -l)"; return 0; fi
    log "sweep INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

log "================ supervisor start (campaign ${CAMPAIGN_TS}) ================"
log "CAMPAIGN=${CAMPAIGN}"
log "phases: acc (run_all_full_presets_vllm.sh) -> acc2 (base+r8 readable-only) -> sweep (sweep_presets.sh)"
# run_acc  # [0.24.0] skip accuracy phase
# run_acc2 # [0.24.0] skip
run_sweep
status done; : >"${STATE}/DONE"
log "================ CAMPAIGN DONE -> ${CAMPAIGN} ================"
find "${CAMPAIGN}" -name scenario_summary.csv 2>/dev/null | sort | tee -a "${LOG}"
