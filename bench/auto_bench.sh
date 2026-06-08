#!/usr/bin/env bash
set -euo pipefail

# Auto-serve vLLM then sweep bench_serving_glm4p5_65k.sh (PROFILE=0) over a list
# of NUM_PROMPTS values, repeating each value NUM_ITERS times.
#
# This is the newbench2 variant of auto_bench.sh: instead of a single bench run,
# it mimics bench_serving_auto.sh and loops NUM_PROMPTS_LIST x NUM_ITERS.
#
# Usage:
#   bash auto_bench.sh
#
# Common overrides:
#   PRESET_YAML=/path/to/preset.yaml bash auto_bench.sh
#   AUTO_SERVE=0 bash auto_bench.sh                       # bring your own server
#   NUM_PROMPTS_LIST="16 18 20 25" NUM_ITERS=3 bash auto_bench.sh
#   NUM_PROMPTS_LIST="16,32,64" RESULT_TAG=dp8ep8_mtp2 bash auto_bench.sh
#   RESET_PREFIX_CACHE=0 bash auto_bench.sh
#
# All env-vars accepted by bench_serving_glm4p5_65k.sh are forwarded as-is,
# e.g. JSONL_START_INDEX, JSONL_OUTPUT_LEN, MAX_CONCURRENCY, DATASET, BASE_URL, …
# NUM_PROMPTS and OUTPUT_FILE are set per-iteration by this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="${SCRIPT_DIR}/../env.yaml"
PRESETS_DIR="${SCRIPT_DIR}/../presets"
SERVE_SH="${SCRIPT_DIR}/../serve.sh"

if ! command -v yq >/dev/null 2>&1; then
    echo "[ERROR] yq is required. Run serve.sh once to auto-install it."
    exit 1
fi

if [ -f "${ENV_YAML}" ]; then
    MODEL_PATH="${MODEL_PATH:-$(yq e '.model.path' "${ENV_YAML}")}"
else
    MODEL_PATH="${MODEL_PATH:-/remote/vast0/share-mv/zai-org/GLM-5-FP8}"
fi

# Preset used to start the server. Override with PRESET_YAML=…
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-profile.yaml}"

# Set AUTO_SERVE=0 to manage the server yourself.
AUTO_SERVE="${AUTO_SERVE:-1}"

# Sweep configuration (mirrors bench_serving_auto.sh).
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST:-25}"
NUM_ITERS="${NUM_ITERS:-3}"
RESULT_TAG="${RESULT_TAG:-dp8ep8_mtp2_model_runner_v2}"

AUTO_LOG_DIR="${SCRIPT_DIR}/logs/auto_bench"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${AUTO_LOG_DIR}/${ts}"
mkdir -p "${RUN_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/results}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
is_enabled() {
  case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------------------
wait_for_gpu_free() {
    local check_interval="${GPU_POLL_INTERVAL:-30}"
    local elapsed=0
    echo "[gpu-wait] Waiting for all GPUs to be free (poll every ${check_interval}s)..."
    while true; do
        local busy_count
        # A GPU holding a loaded model reports GPU use (%)=0 while idle, so we
        # gate on VRAM allocation instead: any GPU whose VRAM% exceeds the
        # threshold (default 10) counts as busy.
        busy_count=$(rocm-smi --showmemuse 2>/dev/null \
            | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' \
            | awk -v thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" '$1 > thr' \
            | wc -l)
        if [ "${busy_count}" -eq 0 ]; then
            echo "[gpu-wait] All GPUs are free after ${elapsed}s. Proceeding."
            return 0
        fi
        echo "[gpu-wait] ${busy_count} GPU(s) still busy (${elapsed}s elapsed). Retrying in ${check_interval}s..."
        sleep "${check_interval}"
        elapsed=$((elapsed + check_interval))
    done
}

wait_for_server() {
    local base_url="${BASE_URL:-http://localhost:8000}"
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}"
    local elapsed=0
    echo "[wait] Polling ${base_url}/health (timeout: ${max_wait}s)..."
    while ! curl -sf "${base_url}/health" >/dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "[ERROR] Server did not respond within ${max_wait}s"
            return 1
        fi
        echo "[wait] ${elapsed}s elapsed, still waiting..."
    done
    echo "[wait] Server ready after ${elapsed}s"
}

kill_vllm() {
    echo "[kill] Sending SIGKILL to VLLM processes..."
    pkill -9 VLLM 2>/dev/null || true
    sleep 15
    echo "[kill] Done."
}

cleanup() {
    if is_enabled "${AUTO_SERVE}"; then
        kill_vllm
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
echo "=== auto_bench.sh started at $(date) ==="
echo "Preset       : ${PRESET_YAML}"
echo "Run dir      : ${RUN_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Model        : ${MODEL_PATH}"
echo "Prompts list : ${NUM_PROMPTS_LIST}"
echo "Iters        : ${NUM_ITERS}"
echo "Result tag   : ${RESULT_TAG}"

if [ ! -f "${PRESET_YAML}" ]; then
    echo "[ERROR] Preset not found: ${PRESET_YAML}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Serve phase
# ---------------------------------------------------------------------------
if is_enabled "${AUTO_SERVE}"; then
    wait_for_gpu_free

    pkill -9 VLLM 2>/dev/null || true
    sleep 5

    SERVE_LOG="${RUN_DIR}/serve.log"
    echo "[serve] Starting server with preset: ${PRESET_YAML}"
    echo "[serve] Log: ${SERVE_LOG}"
    (bash "${SERVE_SH}" "${MODEL_PATH}" "${PRESET_YAML}") >"${SERVE_LOG}" 2>&1 &
    echo "[serve] PID: $!"

    if ! wait_for_server; then
        echo "[ERROR] Server failed to start. Aborting."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Bench sweep phase  (PROFILE=0)
# ---------------------------------------------------------------------------
BENCH_LOG="${RUN_DIR}/bench.log"

# Normalize commas to spaces, then iterate.
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST//,/ }"
read -r -a prompts_arr <<< "${NUM_PROMPTS_LIST}"

echo "[bench] Sweeping bench_serving_glm4p5_65k.sh (PROFILE=0)"
echo "[bench] Log: ${BENCH_LOG}"

{
    for np in "${prompts_arr[@]}"; do
        for i in $(seq 1 "${NUM_ITERS}"); do
            echo "=========================================================="
            echo "Running bench_serving_glm4p5_65k.sh NUM_PROMPTS=${np} iteration ${i}/${NUM_ITERS}"
            echo "=========================================================="
            MODEL_PATH="${MODEL_PATH}" \
            PROFILE=0 \
            NUM_PROMPTS="${np}" \
            OUTPUT_FILE="${OUTPUT_DIR}/${RESULT_TAG}_${np}.jsonl" \
                bash "${SCRIPT_DIR}/bench_serving_glm4p5_65k.sh"
        done
    done
} 2>&1 | tee "${BENCH_LOG}"

echo ""
echo "=== auto_bench.sh completed at $(date) ==="
echo "Bench log    : ${BENCH_LOG}"
echo "Results dir  : ${OUTPUT_DIR}"
echo "Run dir      : ${RUN_DIR}"
