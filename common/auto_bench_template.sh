#!/usr/bin/env bash
set -uo pipefail
# Template for bench/profile runners (run-major). A concrete sets METHOD and
# defines load_scenarios / run_one / aggregate, then sources this. The template
# owns the shared flow: read config (runs), serve (profiler injected when
# MODE=profile), the run-major loop (for each run: for each scenario), kill.
#
# Concrete must define (before sourcing):
#   METHOD=random|custom
#   load_scenarios   # populate SCENARIOS=(idx...) + per-scenario state arrays
#   run_one <scenario_idx> <run_idx>   # one bench/profile invocation
#   aggregate        # build result tables from RUN_DIR (after all runs)
#
# MODE=bench|profile. Reads .<MODE>.method (must equal METHOD) and
# .<MODE>.<METHOD>.runs. Logs: logs/<preset>/auto_<MODE>/<ts>/

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_DIR}/helper.sh"

MODE="${MODE:-bench}"
case "${MODE}" in bench|profile) ;; *) echo "[ERROR] MODE must be bench|profile" >&2; exit 1 ;; esac
: "${METHOD:?concrete runner must set METHOD before sourcing auto_bench_template.sh}"
for fn in load_scenarios run_one aggregate; do
    declare -F "${fn}" >/dev/null || { echo "[ERROR] ${fn}() not defined" >&2; exit 1; }
done

resolve_backend
resolve_preset
resolve_model_path
AUTO_SERVE="${AUTO_SERVE:-1}"

m="$(yaml_get ".${MODE}.method")"
[ "${m}" = "${METHOD}" ] || { echo "[${MODE}] method='${m:-<unset>}' != '${METHOD}'; skipping." >&2; exit 0; }

RUNS="$(yaml_get ".${MODE}.${METHOD}.runs" 1)"
SCENARIOS=()
load_scenarios
[ "${#SCENARIOS[@]}" -gt 0 ] || { echo "[ERROR] no scenarios for .${MODE}.${METHOD}" >&2; exit 1; }

setup_run_dir "auto_${MODE}"
cleanup() { is_enabled "${AUTO_SERVE}" && kill_server; }
trap cleanup EXIT

echo "=== ${MODE} (${METHOD}) at $(date) ==="
echo "preset=${PRESET_YAML} model=${MODEL_PATH} runs=${RUNS} scenarios=${#SCENARIOS[@]}"
echo "run dir=${RUN_DIR}"
[ -f "${PRESET_YAML}" ] || { echo "[ERROR] preset not found: ${PRESET_YAML}" >&2; exit 1; }

if is_enabled "${AUTO_SERVE}"; then
    served="${RUN_DIR}/preset.yaml"
    if [ "${MODE}" = "profile" ]; then
        PROFILER_DIR="${RUN_DIR}/profiling_result"; mkdir -p "${PROFILER_DIR}"
        PC="$(profiler_config_json "${PROFILER_DIR}")" \
            yq e ".engine_args.profiler_config = strenv(PC)" "${PRESET_YAML}" > "${served}"
    else
        cp -f "${PRESET_YAML}" "${served}"
    fi
    wait_for_gpu_free
    kill_server
    PRESET_YAML="${served}" serve_backend "${RUN_DIR}/serve.log"
    wait_for_server || { echo "[ERROR] server failed to start." >&2; exit 1; }
fi

# Run-major: each run sweeps all scenarios (one full table per run).
for r in $(seq 1 "${RUNS}"); do
    echo "================= run ${r}/${RUNS} ================="
    for s in "${SCENARIOS[@]}"; do
        run_one "${s}" "${r}"
    done
done

aggregate

echo "=== ${MODE} (${METHOD}) done at $(date). Results: ${RUN_DIR} ==="
