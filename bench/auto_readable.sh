#!/usr/bin/env bash
set -uo pipefail

# Auto-serve vLLM then run a set of simple readable (smoke-test) requests.
#
# Usage:
#   bash auto_readable.sh
#
# Common overrides:
#   PRESET_YAML=/path/to/preset.yaml bash auto_readable.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="${SCRIPT_DIR}/../env.yaml"
PRESETS_DIR="${SCRIPT_DIR}/../presets"
SERVE_SH="${SCRIPT_DIR}/../serve.sh"

if ! command -v yq >/dev/null 2>&1; then
    echo "[ERROR] yq is required. Run serve.sh once to auto-install it."
    exit 1
fi

MODEL_PATH=$(yq e '.model.path' "${ENV_YAML}")

AUTO_LOG_DIR="${SCRIPT_DIR}/logs/auto_readable"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${AUTO_LOG_DIR}/${ts}"
mkdir -p "${RUN_DIR}"

CURL_URL="http://localhost:8000/v1/chat/completions"

echo "=== auto_readable.sh started at $(date) ==="

make_requests() {
    local prompt="$1"
    echo "Making requests for prompt: $prompt"
    echo "------------------------------------------"

    response=$(curl -s "${CURL_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL_PATH}"'",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "'"${prompt}"'"}
            ],
            "seed": 42,
            "max_tokens": 200
        }')
    python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"${response}"
    echo ""
    echo ""
}

run() {
    make_requests "Who won the world series in 2020?"
    make_requests "What are the main causes of climate change?"
    make_requests "Can you summarize the plot of Pride and Prejudice?"
    make_requests "What are the health benefits of regular exercise?"
    make_requests "How does photosynthesis work in plants?"
    make_requests "What are the key themes in Shakespeare Hamlet?"
    make_requests "What is the capital of France?"
    make_requests "Who painted the Mona Lisa?"
    make_requests "What is the largest planet in our solar system?"
    make_requests "What are the primary colors?"
    make_requests "What is the chemical symbol for water?"
    make_requests "Who wrote Romeo and Juliet?"
    make_requests "What is the speed of light?"
    make_requests "What is the tallest mountain in the world?"
    make_requests "What is the currency of Japan?"
    make_requests "What is the definition of a black hole?"
}

# ===========================================================
# Helper: poll until localhost:8000/health responds
# ===========================================================
wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}"
    local elapsed=0
    echo "  [wait] Polling localhost:8000/health (timeout: ${max_wait}s)..."
    while ! curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "  [ERROR] Server did not respond within ${max_wait}s"
            return 1
        fi
        echo "  [wait] ${elapsed}s elapsed, still waiting..."
    done
    echo "  [wait] Server ready after ${elapsed}s"
}

# ===========================================================
# Helper: kill all VLLM processes
# ===========================================================
kill_vllm() {
    echo "  [kill] Sending SIGKILL to VLLM processes..."
    pkill -9 VLLM 2>/dev/null || true
    sleep 15
    echo "  [kill] Done."
}

# ===========================================================
# run_readable IC_SUFFIX
#
#   Per IC: serve -> run -> kill
# ===========================================================
run_readable() {
    local ic_suffix="$1"
    # Honor an external PRESET_YAML override — same idea as auto_bench.sh.
    local ic_yaml
    if [ -n "${PRESET_YAML:-}" ]; then
        ic_yaml="${PRESET_YAML}"
    else
        ic_yaml="${PRESETS_DIR}/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-${ic_suffix}ic-profile.yaml"
    fi
    local result_dir="${RUN_DIR}/ic${ic_suffix}"
    mkdir -p "${result_dir}"

    echo ""
    echo "=========================================================="
    echo "READABLE: ic${ic_suffix}"
    echo "  preset  : ${ic_yaml}"
    echo "  results : ${result_dir}"
    echo "  Started : $(date)"
    echo "=========================================================="

    if [ ! -f "${ic_yaml}" ]; then
        echo "  [ERROR] preset not found: ${ic_yaml}"
        return 1
    fi

    pkill -9 VLLM 2>/dev/null || true
    sleep 5

    local serve_log="${result_dir}/serve.log"
    echo "  [serve] Starting server, log: ${serve_log}"
    (bash "${SERVE_SH}" "${MODEL_PATH}" "${ic_yaml}") >"${serve_log}" 2>&1 &
    echo "  [serve] PID: $!"

    if ! wait_for_server; then
        echo "  [ERROR] Aborting readable for ic${ic_suffix}."
        kill_vllm
        return 1
    fi

    local readable_log="${result_dir}/readable.log"
    echo "  [readable] log: ${readable_log}"
    run 2>&1 | tee "${readable_log}"

    kill_vllm

    echo "READABLE ic${ic_suffix} completed at $(date)"
}

# ===========================================================
# Kill any leftover VLLM before starting
# ===========================================================
pkill -9 VLLM 2>/dev/null || true

run_readable 0

echo ""
echo "=== All readable runs completed at $(date) ==="
