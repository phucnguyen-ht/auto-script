#!/usr/bin/env bash
# PR-container supervisor for MV-4572 @ v0.24.0 (clean retest build, NO timing/dev-log).
# Runs ON mi355-gpu-58 (host), polite to colleagues via police.sh, fully resumable.
#
# For each family {dp8ep8 (non-mtp), dp8ep8_mtp} and each decode config
# {default-r0, default-r8, default-r16} it runs, on container phuc-nguyen-mv4572-pr-0.24.0:
#   PHASE bench : sweep_presets.sh (PDIR=<family>) -> scenario_summary.csv per config.
#   PHASE acc   : run_all.sh (gsm8k eval + readable longbench2+pychat) per config.
# All logs under logs_v0.24.0/pr_decode/campaign_<ts>/. Resumable: bench skips configs
# with an existing scenario_summary.csv; acc skips configs with an ACC_DONE marker.
set -uo pipefail

CONTAINER="${CONTAINER:-phuc-nguyen-mv4572-pr-0.24.0}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572}"
TICKET="${WORKING_DIR}/auto-script/bench_mv4572"
SCRIPTS="${TICKET}/scripts"
PRESETS_ROOT="${WORKING_DIR}/auto-script/presets/kimi2.6.mxfp4"
POLICE="${POLICE:-/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh}"
# RULE: polite GPU wait / backoff for colleague contention = 10 phút (nâng từ 5). Khi bị
# stomp hoặc GPU bận, chờ 10 phút rồi mới resume để nhường đồng nghiệp cho chắc.
BACKOFF="${BACKOFF:-600}"; POLL="${POLL:-60}"; MAX_RELAUNCH="${MAX_RELAUNCH:-500}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# PR container's aiter (retest repo). vllm-moreh asserts AITER_MOREH_ROOT_DIR at import.
AITER_MOREH_ROOT_DIR_VAL="${AITER_MOREH_ROOT_DIR_VAL:-${WORKING_DIR}/retest/vllm-moreh/src/aiter_moreh}"

FAMILIES=(${FAMILIES:-dp8ep8 dp8ep8_mtp})
CONFIGS=(${CONFIGS:-base-eplb-nixl-async-default-r0 base-eplb-nixl-async-default-r8 base-eplb-nixl-async-default-r16})
# Bench sweep dims (passed through to sweep_presets.sh -> multi_process_test.py).
CONCS_ENV="${CONCS:-16 64 128 256}"; DATASETS_ENV="${DATASETS:-8k 10k 100k}"
# Optional: override the bench preset list (e.g. BENCH_PRESET_LIST="base.yaml" for the
# no-EPLB base reference). Empty => sweep_presets.sh uses its default r0/r8/r16 list.
BENCH_PRESET_LIST="${BENCH_PRESET_LIST:-}"

LOGROOT="${TICKET}/logs_v0.24.0"
mkdir -p "${LOGROOT}/sweep_supervisor_pr"
LOCKF="${LOGROOT}/sweep_supervisor_pr/supervisor.lock"
if command -v flock >/dev/null 2>&1; then exec 9>"${LOCKF}"; flock -n 9 || { echo "[sup-pr] another PR supervisor running -> exit"; exit 0; }; fi

CUR="${LOGROOT}/sweep_supervisor_pr/CURRENT_CAMPAIGN"
if [ -f "${CUR}" ]; then TS="$(cat "${CUR}")"; else TS="$(date +%Y%m%d_%H%M%S)"; echo "${TS}" >"${CUR}"; fi

CAMPAIGN="${LOGROOT}/pr_decode/campaign_${TS}"
STATE="${LOGROOT}/sweep_supervisor_pr/campaign_${TS}"
LOG="${STATE}/supervisor.log"
mkdir -p "${CAMPAIGN}" "${STATE}"; rm -f "${STATE}/DONE" "${STATE}/ESCALATE"

log()    { echo "[sup-pr $(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
status() { echo "$1" >"${STATE}/status"; }
( while kill -0 "$$" 2>/dev/null; do : >"${STATE}/heartbeat"; sleep 30; done ) & HB_PID=$!
trap 'kill "${HB_PID}" 2>/dev/null' EXIT

foreign_list() { bash "${POLICE}" 2>/dev/null | awk '/^[[:space:]]+[^[:space:]].*id=/{print $1}' | grep -vx "${CONTAINER}" || true; }
wait_free() {
  while [ -n "$(foreign_list)" ]; do
    status waiting_gpu
    log "GPU busy by: $(foreign_list | paste -sd,) -> wait ${POLL}s (polite)"
    sleep "${POLL}"
  done
  log "GPU free (no foreign container) -> proceed"
}
container_running() { [ "$(podman inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" = "true" ]; }
ensure_container() {
  container_running && return 0
  if podman container exists "${CONTAINER}" 2>/dev/null; then
    log "container down -> podman restart ${CONTAINER}"; podman restart "${CONTAINER}" >>"${LOG}" 2>&1 || { log "restart FAILED"; return 1; }
    sleep 5; container_running && { log "container back up"; return 0; }
    log "still not running after restart"; return 1
  fi
  log "ESCALATE: container ${CONTAINER} gone; NOT recreating"; status escalate_container_gone; : >"${STATE}/ESCALATE"; return 1
}
# Thorough clean between serves: a crashed/OOM'd serve leaves bench workers, a live
# `vllm serve`, and a stale /tmp/aiter_configs/*.lock baton -> next serve fails engine-core
# init (the Run#1 cascade). Kill all of it + clear locks, then let GPU VRAM settle.
clean_container() {
  podman exec "${CONTAINER}" bash -lc \
    "pkill -9 -f multi_process_test.py 2>/dev/null; pkill -9 -f unit_test.py 2>/dev/null; \
     pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 VLLM 2>/dev/null; \
     rm -f /tmp/aiter_configs/*.lock 2>/dev/null; true" >>"${LOG}" 2>&1 || true
  sleep 8
}

relaunch=0
backoff_after_interrupt() {
  relaunch=$((relaunch+1))
  [ "${relaunch}" -ge "${MAX_RELAUNCH}" ] && { log "MAX_RELAUNCH -> ESCALATE"; status escalate_maxrelaunch; : >"${STATE}/ESCALATE"; exit 1; }
  status interrupted_backoff; log "interrupted -> back off ${BACKOFF}s then resume (#${relaunch})"; sleep "${BACKOFF}"
}

exec_in_container() {  # $1 = shell snippet
  podman exec "${CONTAINER}" bash -lc \
    "export AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR_VAL}'; export VLLM_ENGINE_READY_TIMEOUT_S=${READY_TIMEOUT}; $1" 2>&1 | tee -a "${LOG}"
}

# ---- PHASE bench: sweep_presets.sh per family (resumable via scenario_summary.csv) ----
run_bench() {
  local fam="$1" broot="${CAMPAIGN}/bench/${fam}"
  mkdir -p "${broot}"
  while :; do
    wait_free; ensure_container || { sleep "${POLL}"; continue; }
    clean_container; status "running:bench:${fam}"
    log "LAUNCH bench family=${fam} -> ${broot}"
    exec_in_container "cd '${SCRIPTS}' && PDIR='${PRESETS_ROOT}/${fam}' SWEEP_ROOT='${broot}' CONCS='${CONCS_ENV}' DATASETS='${DATASETS_ENV}' PRESET_LIST='${BENCH_PRESET_LIST}' bash sweep_presets.sh"
    if container_running; then
      local ns; ns=$(find "${broot}" -name scenario_summary.csv 2>/dev/null | wc -l)
      if [ "${ns:-0}" -eq 0 ]; then log "!!! bench ${fam} produced 0 summaries -> FAILED (check serve.log for OOM / gpu_memory_utilization)"; \
      else log "bench ${fam} finished OK; summaries=${ns}"; fi
      return 0
    fi
    log "bench ${fam} INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

# ---- PHASE acc: run_all.sh (gsm8k + readable) per config (resumable via ACC_DONE marker) ----
run_acc_one() {
  local fam="$1" cfg="$2"
  local aroot="${CAMPAIGN}/accuracy/${fam}/${cfg}" marker="${STATE}/ACC_DONE__${fam}__${cfg}"
  [ -f "${marker}" ] && { log "acc ${fam}/${cfg} already done -> skip"; return 0; }
  mkdir -p "${aroot}"
  while :; do
    wait_free; ensure_container || { sleep "${POLL}"; continue; }
    clean_container; status "running:acc:${fam}/${cfg}"
    log "LAUNCH acc ${fam}/${cfg} -> ${aroot}"
    exec_in_container "cd '${TICKET}' && PRESET='kimi2.6.mxfp4/${fam}/${cfg}.yaml' LOG_ROOT='${aroot}' bash run_all.sh"
    if container_running; then : >"${marker}"; log "acc ${fam}/${cfg} complete"; return 0; fi
    log "acc ${fam}/${cfg} INTERRUPTED (container down)"; backoff_after_interrupt
  done
}

log "================ PR supervisor start (campaign ${TS}) ================"
log "CONTAINER=${CONTAINER}  CAMPAIGN=${CAMPAIGN}"
log "families=(${FAMILIES[*]})  configs=(${CONFIGS[*]})"
log "plan: per family -> bench(all configs) ; then per config -> acc(gsm8k+readable)"
for fam in "${FAMILIES[@]}"; do
  run_bench "${fam}"
  for cfg in "${CONFIGS[@]}"; do run_acc_one "${fam}" "${cfg}"; done
done
status done; : >"${STATE}/DONE"
log "================ PR CAMPAIGN DONE -> ${CAMPAIGN} ================"
find "${CAMPAIGN}" -name scenario_summary.csv 2>/dev/null | sort | tee -a "${LOG}"
