#!/usr/bin/env bash
set -euo pipefail
# Serve vLLM with the torch profiler enabled, then sweep
# bench_serving_glm4p5_65k.sh (PROFILE=1) over NUM_PROMPTS_LIST x NUM_ITERS.
#
#   PRESET_YAML=/path bash auto_profile.sh
#   AUTO_SERVE=0 bash auto_profile.sh
#   NUM_PROMPTS_LIST="8 16" NUM_ITERS=1 bash auto_profile.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LOG_ROOT:=${SCRIPT_DIR}/logs}"
source "${SCRIPT_DIR}/../common/helper.sh"

resolve_backend
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml}"
resolve_model_path
AUTO_SERVE="${AUTO_SERVE:-1}"
setup_run_dir auto_profile

TORCH_PROFILER_WITH_STACK="$(yaml_get '.profiler.TORCH_PROFILER_WITH_STACK')"
TORCH_PROFILER_RECORD_SHAPES="$(yaml_get '.profiler.TORCH_PROFILER_RECORD_SHAPES')"
TORCH_PROFILER_WITH_MEMORY="$(yaml_get '.profiler.TORCH_PROFILER_WITH_MEMORY')"
TORCH_PROFILER_WITH_FLOPS="$(yaml_get '.profiler.TORCH_PROFILER_WITH_FLOPS')"

PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST:-16}"
NUM_ITERS="${NUM_ITERS:-1}"
RESULT_TAG="${RESULT_TAG:-dp8ep8_mtp2_model_runner_v2}"

PROFILER_DIR="${RUN_DIR}/profiling_result"; mkdir -p "${PROFILER_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/results}"; mkdir -p "${OUTPUT_DIR}"

cleanup() { is_enabled "${AUTO_SERVE}" && kill_server; }
trap cleanup EXIT

echo "=== auto_profile.sh started at $(date) ==="
echo "Preset=${PRESET_YAML} model=${MODEL_PATH} activities=${PROFILE_ACTIVITIES}"
echo "Prompts=${NUM_PROMPTS_LIST} iters=${NUM_ITERS}"
echo "Run dir=${RUN_DIR}"

[ -f "${PRESET_YAML}" ] || { echo "[ERROR] Preset not found: ${PRESET_YAML}" >&2; exit 1; }

if is_enabled "${AUTO_SERVE}"; then
    profiler_config_json=$(printf \
        '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":"%s","torch_profiler_record_shapes":"%s","torch_profiler_with_memory":"%s","torch_profiler_with_flops":"%s"}' \
        "${PROFILER_DIR}" "${TORCH_PROFILER_WITH_STACK}" "${TORCH_PROFILER_RECORD_SHAPES}" \
        "${TORCH_PROFILER_WITH_MEMORY}" "${TORCH_PROFILER_WITH_FLOPS}")
    TMP_YAML="${RUN_DIR}/tmp_preset.yaml"
    PROFILER_CONFIG="${profiler_config_json}" \
        yq e ".engine_args.profiler_config = strenv(PROFILER_CONFIG)" "${PRESET_YAML}" > "${TMP_YAML}"
    echo "[yaml] Tmp preset: ${TMP_YAML}"

    wait_for_gpu_free
    kill_server
    # Serve with the profiler-injected preset copy (not the original).
    PRESET_YAML="${TMP_YAML}" serve_backend "${RUN_DIR}/serve.log"
    wait_for_server || { echo "[ERROR] Server failed to start." >&2; exit 1; }
fi

BENCH_LOG="${RUN_DIR}/bench.log"
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST//,/ }"
read -r -a prompts_arr <<< "${NUM_PROMPTS_LIST}"

{
    for np in "${prompts_arr[@]}"; do
        for i in $(seq 1 "${NUM_ITERS}"); do
            echo "========== profile NUM_PROMPTS=${np} iter ${i}/${NUM_ITERS} =========="
            MODEL_PATH="${MODEL_PATH}" PROFILE=1 PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES}" \
            NUM_PROMPTS="${np}" OUTPUT_FILE="${OUTPUT_DIR}/${RESULT_TAG}_${np}.jsonl" \
                bash "${SCRIPT_DIR}/bench_serving_glm4p5_65k.sh"
        done
    done
} 2>&1 | tee "${BENCH_LOG}"

echo "=== auto_profile.sh completed at $(date) ==="
echo "Results: ${OUTPUT_DIR}  Profiler traces: ${PROFILER_DIR}"
