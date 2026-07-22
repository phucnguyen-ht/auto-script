#!/usr/bin/env bash
# Host-side supervisor for the MV-4572 EP-imbalance PROFILE run. Same resilience model as
# the decode/prefill sweep supervisors: politely wait while a FOREIGN container holds the
# GPU, podman-restart our container when it's down (NEVER recreate), and RETRY until every
# enabled profile preset has produced its 8 torch traces. A profile serve that dies mid
# CUDA-graph-capture (colleague reclaims GPU) just triggers a retry.
#
#   bash profile_supervisor.sh
# Done when: sweep_presets_profile.sh printed "[sweep] DONE" AND no "[profile] WARN"
# (trace-timeout) AND >=8 traces exist. State/lock in its own _profile namespace.
set -uo pipefail

CONTAINER="${CONTAINER:-phuc-nguyen-mv4572-rebench}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572}"
TICKET="${WORKING_DIR}/auto-script/bench_mv4572"
SCRIPTS="${TICKET}/scripts"
POLICE="${POLICE:-/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh}"
BACKOFF="${BACKOFF:-120}"
POLL="${POLL:-60}"
MAX_RELAUNCH="${MAX_RELAUNCH:-500}"
TRACE_TIMEOUT="${TRACE_TIMEOUT:-300}"   # short: a crashed serve should fail fast -> retry
AITER_MOREH_ROOT_DIR_VAL="${AITER_MOREH_ROOT_DIR_VAL:-${WORKING_DIR}/vllm-moreh/src/aiter_moreh}"

SUPDIR="${TICKET}/logs/profile_supervisor"
mkdir -p "${SUPDIR}"
LOCKF="${SUPDIR}/supervisor.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCKF}"; flock -n 9 || { echo "[prof-sup] another supervisor running -> exit"; exit 0; }
fi

CUR="${SUPDIR}/CURRENT_CAMPAIGN"
if [ -f "${CUR}" ]; then CAMPAIGN_TS="$(cat "${CUR}")"; else CAMPAIGN_TS="$(date +%Y%m%d_%H%M%S)"; echo "${CAMPAIGN_TS}" >"${CUR}"; fi
CAMPAIGN="${TICKET}/logs/sweep_profile/campaign_${CAMPAIGN_TS}"
INTERRUPTED="${TICKET}/logs/sweep_profile_interrupted/campaign_${CAMPAIGN_TS}"
STATE="${SUPDIR}/campaign_${CAMPAIGN_TS}"
LOG="${STATE}/supervisor.log"
mkdir -p "${CAMPAIGN}" "${INTERRUPTED}" "${STATE}"
rm -f "${STATE}/DONE" "${STATE}/ESCALATE"

log()    { echo "[prof-sup $(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
status() { echo "$1" >"${STATE}/status"; }
( while kill -0 "$$" 2>/dev/null; do : >"${STATE}/heartbeat"; sleep 30; done ) & HB_PID=$!

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
  log "ESCALATE: container gone; NOT recreating per brief"; status escalate_container_gone; : >"${STATE}/ESCALATE"; return 1
}
clean_container() { podman exec "${CONTAINER}" bash -lc "pkill -9 VLLM 2>/dev/null; pkill -9 -f 'sweep_presets_profil[e].sh' 2>/dev/null; pkill -9 -f 'multi_proces[s]_test.py' 2>/dev/null; pkill -9 -f 'uni[t]_test.py' 2>/dev/null; true" >>"${LOG}" 2>&1 || true; sleep 3; }

ntraces() { find "${CAMPAIGN}" -name '*.pt.trace.json*' 2>/dev/null | wc -l; }
# Move any preset dir that has <8 traces out of the campaign (partial/interrupted) so the
# retry starts that preset clean (its scenario_summary.csv would otherwise make the sweep
# skip-guard wrongly skip it).
move_partials() {
  shopt -s nullglob
  for d in "${CAMPAIGN}"/*/; do
    local name n; name="$(basename "${d%/}")"
    n=$(find "${d}profiling_result" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null | wc -l)
    if [ "${n}" -lt 8 ]; then
      log "MOVE partial preset '${name}' (${n}/8 traces) -> ${INTERRUPTED}/${name}__$(date +%Y%m%d_%H%M%S)"
      mv "${d%/}" "${INTERRUPTED}/${name}__$(date +%Y%m%d_%H%M%S)" 2>>"${LOG}" || log "  (move failed ${name})"
    fi
  done; shopt -u nullglob
}

relaunch=0
run_loop() {
  while :; do
    wait_free
    ensure_container || { sleep "${POLL}"; continue; }
    move_partials
    clean_container
    status "running:profile"
    log "LAUNCH profile: sweep_presets_profile.sh (SWEEP_ROOT=${CAMPAIGN}, TRACE_TIMEOUT=${TRACE_TIMEOUT})"
    # 9>&-: don't leak the flock fd into the exec child (else an orphaned exec holds the lock).
    podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; cd '${SCRIPTS}' && SWEEP_ROOT='${CAMPAIGN}' TRACE_TIMEOUT='${TRACE_TIMEOUT}' bash sweep_presets_profile.sh" 9>&- 2>&1 | tee -a "${LOG}"
    if ! container_running; then log "profile INTERRUPTED (container down)"; relaunch=$((relaunch+1)); [ "${relaunch}" -ge "${MAX_RELAUNCH}" ] && { log "MAX_RELAUNCH -> ESCALATE"; : >"${STATE}/ESCALATE"; exit 1; }; status interrupted_backoff; sleep "${BACKOFF}"; continue; fi
    # container alive: check every preset dir got its 8 traces.
    local incomplete
    incomplete=$(find "${CAMPAIGN}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | while read -r d; do
        n=$(find "${d}/profiling_result" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null | wc -l); [ "${n}" -lt 8 ] && echo "${d}"; done | wc -l)
    if [ "$(ntraces)" -ge 8 ] && [ "${incomplete}" -eq 0 ]; then return 0; fi
    log "profile finished but traces incomplete (total=$(ntraces), incomplete_dirs=${incomplete}) -> retry"
    relaunch=$((relaunch+1)); [ "${relaunch}" -ge "${MAX_RELAUNCH}" ] && { log "MAX_RELAUNCH -> ESCALATE"; : >"${STATE}/ESCALATE"; exit 1; }
    status interrupted_backoff; sleep "${BACKOFF}"
  done
}

log "================ profile supervisor start (campaign ${CAMPAIGN_TS}) ================"
log "CAMPAIGN=${CAMPAIGN}"
run_loop
status done; : >"${STATE}/DONE"
log "================ PROFILE DONE -> ${CAMPAIGN} (traces=$(ntraces)) ================"
find "${CAMPAIGN}" -name '*.pt.trace.json*' 2>/dev/null | sed -E 's#.*/([^/]+)/profiling_result/.*#\1#' | sort | uniq -c | tee -a "${LOG}"
