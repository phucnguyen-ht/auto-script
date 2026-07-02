#!/usr/bin/env bash
set -uo pipefail
# Thin wrapper: serve GLM-5.2-FP8 -> run scripts/ python flow -> extract
# (scripts/RUN_AND_BENCH.md §4). The flow reads sweep + results-dir from env vars
# (REBENCH_CONC/DATASETS/RESULTS_DIR), so we run scripts/ directly (no copy) with
# results landing under RUN_DIR. Everything stays inside this ticket.

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${TICKET_DIR}/../common"
export ENV_YAML="${ENV_YAML:-${TICKET_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${TICKET_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${TICKET_DIR}}"
export PRESET="${PRESET:-${PRESET_YAML:-glm5.2.rebench/MTP5-bs64-dg.yaml}}"

METHOD=rebench

# Abort the server wait early on a fatal serve error (else it waits out
# SERVER_WAIT_TIMEOUT) so the trap kills the server and run_all_full_presets moves
# to the next preset. Death-detection is automatic; the patterns below are the
# real fatal signatures seen in the eplb bench runs (NIXL/UCX init, silent
# worker-death hang, NCCL heartbeat/peer crash, OOM).
export SERVER_ERROR_DETECT="${SERVER_ERROR_DETECT:-1}"
export SERVER_FATAL_RE="${SERVER_FATAL_RE:-WorkerProc failed to start|EngineCore.*(failed|exited|died)|Failed to initialize Nixl|NIXL (EPLB init failed|ERR_BACKEND)|NIXL_ERR_BACKEND|nixlBackendError|registerMem: registration failed|VRAM registration cannot proceed|UCX is likely not configured|out of memory|OutOfMemoryError|Did the remote server shutdown or crash|HeartbeatMonitor|\[shutdown\] Executor: waiting for worker exit|RuntimeError:.*(HIP|CUDA|NCCL|NIXL)|core dumped|Address already in use}"

# This ticket mirrors the manual serve_glm5.sh + bench.sh path by default:
#   AUTO_CLEAN=0 -> skip pre-serve hygiene (no gpu-free wait / stale kill / compile-cache clear)
#   SERVE_TEE=1  -> serve via tee (logging backpressure -> matches manual tpot tail)
# For a CLEAN sweep (frees GPUs between presets), run: AUTO_CLEAN=1 SERVE_TEE=0 ...
export AUTO_CLEAN="${AUTO_CLEAN:-0}"
export SERVE_TEE="${SERVE_TEE:-0}"

REBENCH_DIR="${REBENCH_DIR:-${TICKET_DIR}/scripts}"   # run the flow straight from here

declare -a CONC DS

load_scenarios() {
    local p=".${MODE}.rebench"
    read -r -a CONC <<< "$(yaml_list "${p}.concurrencies")"; [ "${#CONC[@]}" -gt 0 ] || CONC=(64)
    read -r -a DS   <<< "$(yaml_list "${p}.datasets")";      [ "${#DS[@]}"   -gt 0 ] || DS=("100k")
    echo "[rebench] concurrencies=(${CONC[*]}) datasets=(${DS[*]}) windows=$(( ${#CONC[@]} * ${#DS[@]} ))"
    SCENARIOS=(0)
}

# One driver invocation = the whole sweep (RUN_AND_BENCH.md §3). Sweep + results
# dir passed via env; results land in RUN_DIR/results. runs=1.
run_one() {
    local marker="${RUN_DIR}/bench.start"; : > "${marker}"
    mkdir -p "${RUN_DIR}/results"
    REBENCH_CONC="$(IFS=,; echo "${CONC[*]}")" \
    REBENCH_DATASETS="$(IFS=,; echo "${DS[*]}")" \
    REBENCH_RESULTS_DIR="${RUN_DIR}/results" \
        python3 -u "${REBENCH_DIR}/multi_process_test.py" 2>&1 | tee "${RUN_DIR}/mpt_run.log"
    [ "${MODE}" = "profile" ] && [ -n "${PROFILER_DIR:-}" ] && harvest_profiles "${marker}" "${PROFILER_DIR}/windows"
    return 0
}

# RUN_AND_BENCH.md §4: labels from driver log -> decode batch size -> summary.
aggregate() {
    local min_ts=0
    [ -f "${RUN_DIR}/bench.start" ] && min_ts="$(stat -c %Y "${RUN_DIR}/bench.start")"
    local -a LABELS
    mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "${RUN_DIR}/mpt_run.log" 2>/dev/null \
        | awk '{gsub(/,/,""); print $4"_p"$2}')
    mkdir -p "${RUN_DIR}/decode_running"
    python3 "${REBENCH_DIR}/extract_decode_running.py" \
        "${RUN_DIR}/serve.log" "${RUN_DIR}/decode_running/decode_running" \
        --scenarios "${LABELS[@]}" 2>&1 | tee "${RUN_DIR}/decode_running.log"
    python3 "${REBENCH_DIR}/summarize_scenarios.py" \
        "${RUN_DIR}/results" "${RUN_DIR}/scenario_summary.csv" \
        --totals-dir "${RUN_DIR}/decode_running" --min-ts "${min_ts}" 2>&1 | tee "${RUN_DIR}/summarize.log"
    [ -f "${RUN_DIR}/scenario_summary.csv" ] && column -t -s, "${RUN_DIR}/scenario_summary.csv"
}

source "${COMMON_DIR}/auto_bench_template.sh"
