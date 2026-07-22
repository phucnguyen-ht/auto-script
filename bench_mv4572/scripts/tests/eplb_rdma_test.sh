#!/usr/bin/env bash
# Task-3 (MV-4572): does adding rc_x (RDMA, for the multi-node "global fix") to UCX_TLS
# change the current INTRA-node EPLB result? Also: does UNSET UCX_TLS hit the NIXL
# backend-fail we saw before? Runs a few short EPLB serves+benches with EPLB_TIMING=1 +
# UCX_PROTO_INFO=y, then checks: backend init, transport (rocm_ipc vs tcp via timing),
# rearrange count, and 'no peer failure handler' lines. Logs everything under OUT.
#
# Runs ON the host (drives podman exec), gated on GPU being free (police), in tmux.
set -uo pipefail

CONTAINER="${CONTAINER:-phuc-nguyen-mv4572-rebench}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572}"
TICKET="${WORKING_DIR}/auto-script/bench_mv4572"
SCRIPTS="${TICKET}/scripts"                       # decode driver (multi_process_test.py) triggers rearrange
PDIR="${WORKING_DIR}/auto-script/presets/kimi2.6.mxfp4/dp8ep8"
# Experiment presets live in a SEPARATE dir (never overwrite the real dp8ep8/ presets).
PDIR_EXP="${WORKING_DIR}/auto-script/presets/kimi2.6.mxfp4/dp8ep8_exp"
OUT="${TICKET}/logs/eplb-rdma-add"
POLICE="${POLICE:-/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh}"
AITER="${WORKING_DIR}/vllm-moreh/src/aiter_moreh"
MODEL="${MODEL:-/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/}"
PORT=8000
BENCH_CONC="${BENCH_CONC:-64}"          # one scenario, enough steps to trigger rearrange
BENCH_DS="${BENCH_DS:-8k}"
mkdir -p "${OUT}" "${PDIR_EXP}"
SUMMARY="${OUT}/SUMMARY.md"

log(){ echo "[rdma-test $(date '+%m-%d %H:%M:%S')] $*"; }

foreign_list(){ bash "${POLICE}" 2>/dev/null | awk '/^[[:space:]]+[^[:space:]].*id=/{print $1}' | grep -vx "${CONTAINER}" || true; }
wait_free(){ while [ -n "$(foreign_list)" ]; do log "GPU busy by: $(foreign_list|paste -sd,) -> wait 60s"; sleep 60; done; log "GPU free -> proceed"; }
clean(){ podman exec "${CONTAINER}" bash -lc "pkill -9 VLLM 2>/dev/null||true" >/dev/null 2>&1 || true; sleep 5; }

# make_variant <src_preset_basename> <tag> <ucx_tls-or-empty>
#   strips existing UCX_TLS/EPLB_TIMING/UCX_PROTO/UCX_LOG lines, injects our test env under env_vars.
make_variant(){
  local src="${PDIR}/$1" tag="$2" tls="$3" dst="${PDIR_EXP}/${2}.yaml"
  python3 - "$src" "$dst" "$tls" <<'PY'
import sys
src,dst,tls=sys.argv[1],sys.argv[2],sys.argv[3]
out=[]
for ln in open(src).read().splitlines():
    s=ln.strip()
    if s.split(':',1)[0] in ("UCX_TLS","VLLM_MOREH_EPLB_TIMING","UCX_PROTO_INFO","UCX_LOG_LEVEL"):
        continue
    out.append(ln)
    if s=="env_vars:":
        out.append("  VLLM_MOREH_EPLB_TIMING: '1'")
        out.append("  UCX_PROTO_INFO: 'y'")
        if tls:
            out.append(f"  UCX_TLS: {tls}")
open(dst,'w').write("\n".join(out)+"\n")
print(f"[make_variant] {dst}  UCX_TLS={tls or '(UNSET)'}")
PY
}

# run_case <tag> <src_preset_basename> <ucx_tls-or-empty>
run_case(){
  local tag="$1" src="$2" tls="$3" d="${OUT}/${1}"
  mkdir -p "${d}/results"
  make_variant "$src" "$tag" "$tls" | tee -a "${d}/run.log"
  wait_free
  clean
  log "[$tag] serve preset=$src UCX_TLS='${tls:-UNSET}'"
  podman exec "${CONTAINER}" bash -lc "export AITER_MOREH_ROOT_DIR='${AITER}'; nohup vllm-moreh serve '${MODEL}' '${PDIR_EXP}/${tag}.yaml' > '${d}/serve.log' 2>&1 & echo \$! > '${d}/serve.pid'"
  # wait health (or serve death) up to 20 min
  local t=0 ok=0
  while [ $t -lt 1200 ]; do
    if podman exec "${CONTAINER}" bash -lc "curl -sf http://localhost:${PORT}/health >/dev/null 2>&1"; then ok=1; break; fi
    if podman exec "${CONTAINER}" bash -lc "kill -0 \$(cat '${d}/serve.pid') 2>/dev/null"; then : ; else log "[$tag] SERVE DIED before healthy"; break; fi
    sleep 15; t=$((t+15))
  done
  if [ "$ok" = 1 ]; then
    log "[$tag] HEALTH OK -> short bench (conc ${BENCH_CONC}, ds ${BENCH_DS})"
    podman exec "${CONTAINER}" bash -lc "cd '${SCRIPTS}' && REBENCH_RESULTS_DIR='${d}/results' REBENCH_CONC='${BENCH_CONC}' REBENCH_DATASETS='${BENCH_DS}' python3 -u multi_process_test.py > '${d}/mpt.log' 2>&1" || true
  else
    log "[$tag] server not healthy -> capturing serve.log for backend-fail analysis"
  fi
  clean
  analyze "$tag" "$d" "$tls"
}

analyze(){
  local tag="$1" d="$2" tls="$3" sl="${d}/serve.log"
  {
    echo "## ${tag}  (UCX_TLS='${tls:-UNSET}')"
    echo "- serve.log: ${sl}"
    echo -n "- EPLB nixl backend init: "; grep -aoE "\[EPLB nixl\] UCX error-handling mode = [a-z]+" "$sl" 2>/dev/null | head -1 || true
    echo -n "- backend/NIXL errors: "; grep -acaiE "NIXL Error|backend.*fail|Failed to create backend|WorkerProc initialization failed|EngineCore.*fatal|create_backend.*rror" "$sl" 2>/dev/null
    echo -n "- 'no peer failure handler' (rocm_ipc dropped) lines: "; grep -aca "no peer failure handler" "$sl" 2>/dev/null
    echo -n "- health reached: "; grep -aca "Starting vLLM API server\|Application startup complete\|Uvicorn running" "$sl" 2>/dev/null
    echo -n "- rearrange (steady, non-profile): "; grep -aca "Rearranging experts" "$sl" 2>/dev/null
    echo "- [EPLB timing] transfer per layer (last 8):"
    grep -aoE "\[EPLB timing\] layer [0-9]+ transfer=[0-9.]+ .* total=[0-9.]+" "$sl" 2>/dev/null | tail -8 | sed 's/^/    /'
    echo "- transfer mean (s):"
    grep -aoE "\[EPLB timing\] layer [0-9]+ transfer=[0-9.]+" "$sl" 2>/dev/null \
      | grep -oE "transfer=[0-9.]+" | cut -d= -f2 \
      | awk '{s+=$1;n++} END{if(n)printf "    mean=%.4f  n=%d\n","",0; }' >/dev/null
    grep -aoE "transfer=[0-9.]+" "$sl" 2>/dev/null | cut -d= -f2 | awk '{s+=$1;n++} END{if(n)printf "    transfer mean=%.4f s over n=%d layers\n",s/n,n; else print "    (no timing lines)"}'
    echo -n "- UCX proto rocm_ipc mentions: "; grep -aca "rocm_ipc" "$sl" 2>/dev/null
    echo
  } | tee -a "${SUMMARY}"
  # rearrange/timing breakdown via the existing analyzer
  python3 "${WORKING_DIR}/eplb_timing_mean.py" --log-file "$sl" > "${d}/eplb_timing_mean.txt" 2>&1 || true
}

echo "# EPLB rc_x/RDMA + UCX_TLS experiment — $(date)" > "${SUMMARY}"
echo "container=${CONTAINER} node has RDMA NIC: (checked separately, expect none)" >> "${SUMMARY}"
echo >> "${SUMMARY}"

# --- case matrix ---
run_case "base_s250r0"  "base-eplb-nixl-async-s250-r0.yaml"  "tcp,self,sm,rocm_copy,rocm_ipc"
run_case "rcx_s250r0"   "base-eplb-nixl-async-s250-r0.yaml"  "tcp,self,sm,rocm_copy,rocm_ipc,rc_x"
run_case "rcx_s500r0"   "base-eplb-nixl-async-s500-r0.yaml"  "tcp,self,sm,rocm_copy,rocm_ipc,rc_x"
run_case "rcx_s1000r0"  "base-eplb-nixl-async-s1000-r0.yaml" "tcp,self,sm,rocm_copy,rocm_ipc,rc_x"
run_case "unset_s250r0" "base-eplb-nixl-async-s250-r0.yaml"  ""

log "ALL CASES DONE -> ${SUMMARY}"
: > "${OUT}/DONE"
