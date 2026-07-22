#!/usr/bin/env bash
# Host-side supervisor for the MV-4572 PREFILL sweep. Runs ON mi355-gpu-58 (NOT in the
# container) so it survives a `podman stop`; a login-node watcher relaunches it if the
# host reaps it (campaign is fully resumable). Prefill counterpart of
# scripts/sweep_supervisor.sh, but ONE phase (the prefill sweep) -- no acc/acc2.
#
# Gated on the GPUs being free of OTHER users' containers (police.sh, poll POLL s).
# On a colleague `podman stop`: exec returns, container down -> the in-flight preset is
# UNFINISHED, its partial dir is MOVED out of the results tree, back off BACKOFF s, wait
# for the GPUs to free, `podman restart` our container, resume. NEVER recreate.
#
# Uses its OWN campaign/state/lock namespace (…_prefill) so it never collides with the
# decode supervisor.
set -uo pipefail

CONTAINER="${CONTAINER:-phuc-nguyen-mv4572-rebench}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572}"
TICKET="${WORKING_DIR}/auto-script/bench_mv4572"
SCRIPTS="${TICKET}/scripts_prefill"
POLICE="${POLICE:-/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh}"
BACKOFF="${BACKOFF:-300}"
POLL="${POLL:-60}"
MAX_RELAUNCH="${MAX_RELAUNCH:-500}"
# vllm-moreh imports aiter at startup, which asserts AITER_MOREH_ROOT_DIR is set. Our
# non-interactive `podman exec bash -lc` lacks setup_dev.sh's export, so set it here.
AITER_MOREH_ROOT_DIR_VAL="${AITER_MOREH_ROOT_DIR_VAL:-${WORKING_DIR}/vllm-moreh/src/aiter_moreh}"

SUPDIR="${TICKET}/logs/sweep_supervisor_prefill"
mkdir -p "${SUPDIR}"
LOCKF="${SUPDIR}/supervisor.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCKF}"; flock -n 9 || { echo "[sup-prefill] another supervisor running -> exit"; exit 0; }
fi

CUR="${SUPDIR}/CURRENT_CAMPAIGN"
if [ -f "${CUR}" ]; then CAMPAIGN_TS="$(cat "${CUR}")"; else CAMPAIGN_TS="$(date +%Y%m%d_%H%M%S)"; echo "${CAMPAIGN_TS}" >"${CUR}"; fi

CAMPAIGN="${TICKET}/logs/sweep_prefill_tune_result/campaign_${CAMPAIGN_TS}"
INTERRUPTED="${TICKET}/logs/sweep_prefill_interrupted/campaign_${CAMPAIGN_TS}"
STATE="${SUPDIR}/campaign_${CAMPAIGN_TS}"
LOG="${STATE}/supervisor.log"
mkdir -p "${CAMPAIGN}" "${INTERRUPTED}" "${STATE}"
rm -f "${STATE}/DONE" "${STATE}/ESCALATE"

log()    { echo "[sup-prefill $(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
status() { echo "$1" >"${STATE}/status"; }
# Heartbeat self-terminates when the main supervisor ($$) dies -- even on SIGKILL --
# so it never orphans and holds the flock (see decode supervisor lesson).
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
clean_container() { podman exec "${CONTAINER}" bash -lc "pkill -9 VLLM 2>/dev/null; pkill -9 -f 'sweep_presets_[1].sh' 2>/dev/null; pkill -9 -f 'multi_proces[s]_test.py' 2>/dev/null; pkill -9 -f 'uni[t]_test.py' 2>/dev/null; true" >>"${LOG}" 2>&1 || true; sleep 3; }
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

run_sweep() {
  while :; do
    wait_free
    ensure_container || { sleep "${POLL}"; continue; }
    move_partials "${CAMPAIGN}"
    clean_container
    status "running:sweep"
    log "LAUNCH prefill sweep: sweep_presets_1.sh (SWEEP_ROOT=${CAMPAIGN})"
    # 9>&- : close the flock fd for the exec child. Otherwise the podman-exec client
    # inherits fd 9 and, if the supervisor is reaped while the sweep runs, the orphaned
    # exec keeps the lock held -> every relaunch hits "another supervisor running -> exit".
    podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; cd '${SCRIPTS}' && SWEEP_ROOT='${CAMPAIGN}' bash sweep_presets_1.sh" 9>&- 2>&1 | tee -a "${LOG}"
    if container_running; then log "prefill sweep finished; summaries: $(find "${CAMPAIGN}" -name scenario_summary.csv 2>/dev/null | wc -l)"; return 0; fi
    log "prefill sweep INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

log "================ prefill supervisor start (campaign ${CAMPAIGN_TS}) ================"
log "CAMPAIGN=${CAMPAIGN}"
run_sweep
status done; : >"${STATE}/DONE"
log "================ PREFILL CAMPAIGN DONE -> ${CAMPAIGN} ================"
find "${CAMPAIGN}" -name scenario_summary.csv 2>/dev/null | sort | tee -a "${LOG}"
