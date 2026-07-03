#!/usr/bin/env bash
# DEBUG run for mv-4571: 5 configs in order, all logs under logs_debugs/.
# Order: no-eplb baseline -> nixl(async) -> gloo(async) -> pynccl(sync) -> nccl(sync)
# Run INSIDE the pod (needs GPUs). Idempotent/resumable: a config whose
# scenario_summary.csv already exists is SKIPPED (so a re-launch resumes).
#   setsid bash run_debug5.sh > <ticket>/logs_debugs/debug5.log 2>&1 < /dev/null &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"     # bench_mv4571_rebench_auto
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"      # auto-script
PRESETS="${AUTO_ROOT}/presets/glm5.2.rebench"
OUT="${TICKET_DIR}/logs_debugs"
mkdir -p "${OUT}"

RUNS=(
  "1-baseline-noeplb:MTP5-bs64-dg.yaml"
  "2-nixl-async-default-r0:MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml"
  "3-gloo-async-default-r0:MTP5-bs64-dg-eplb-gloo-async-default-r0.yaml"
  "4-pynccl-sync-default-r0:MTP5-bs64-dg-eplb-pynccl-sync-default-r0.yaml"
  "5-nccl-sync-default-r0:MTP5-bs64-dg-eplb-nccl-sync-default-r0.yaml"
)

# Kill ALL serve/worker/coordinator procs. Patterns chosen so they can NEVER
# match this driver's own argv ("bash run_debug5.sh"). This is the crucial fix:
# the previous run's 8 DP workers (comm "VLLM::Worker_DP") outlive a bare
# `pkill vllm-moreh serve` and keep ~107 GiB pinned -> next serve OOM pre-check.
kill_all() {
  pkill -9 -f "vllm-moreh serve" 2>/dev/null
  pkill -9 -f "VLLM::"           2>/dev/null
  pkill -9 -f "EngineCore"       2>/dev/null
  pkill -9 -f "api_server"       2>/dev/null
  sleep 8
}

# Poll until every GPU < 5% VRAM (call AFTER kill_all so leftovers are gone).
# Long wait (up to ~60 min) so a colleague's job on this SHARED node drains
# before we serve — proceeding into contention would OOM the pre-flight check.
wait_gpu_free() {
  for _ in $(seq 1 240); do
    local busy
    busy=$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): [0-9]+' \
           | awk -F': ' '$2>5{c++} END{print c+0}')
    [ "${busy:-1}" -eq 0 ] && return 0
    echo "[debug5]   ${busy} GPU(s) still >5% VRAM; waiting 15s..."; sleep 15
  done
  echo "[debug5]   WARN: GPUs still busy after wait; proceeding anyway"
}

echo "[debug5] START $(date -u +%FT%TZ)  host=$(hostname)"
for spec in "${RUNS[@]}"; do
  name="${spec%%:*}"; preset="${spec#*:}"
  run="${OUT}/${name}"; mkdir -p "${run}"
  if [ -f "${run}/scenario_summary.csv" ]; then
    echo "[debug5] SKIP ${name} (already has scenario_summary.csv)"; continue
  fi
  echo "=================================================================="
  echo "[debug5] >>> START ${name}  preset=${preset}  $(date -u +%FT%TZ)"
  kill_all          # 1) kill leftovers FIRST
  wait_gpu_free     # 2) then wait for VRAM to actually free
  RUN="${run}" PRESET="${PRESETS}/${preset}" PORT=8000 SERVER_WAIT_TIMEOUT=3600 \
    bash "${SCRIPT_DIR}/run_and_bench.sh" > "${run}/console.log" 2>&1
  rc=$?
  if [ -f "${run}/scenario_summary.csv" ]; then
    echo "[debug5] <<< OK   ${name} exit=${rc}"
    sed -n '1,3p' "${run}/scenario_summary.csv" | sed 's/^/[debug5]     /'
  else
    echo "[debug5] <<< FAIL ${name} exit=${rc} (no scenario_summary.csv)"
    echo "[debug5]     tail serve.log:"; tail -12 "${run}/serve.log" 2>/dev/null | sed 's/^/[debug5]       /'
  fi
  kill_all          # 3) cleanup after each run so the next starts clean
done
kill_all
echo "[debug5] ALL DONE $(date -u +%FT%TZ)"
