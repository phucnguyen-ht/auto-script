#!/usr/bin/env bash
set -uo pipefail

# Auto-serve vLLM then run a set of simple readable (smoke-test) requests.
#
# Usage:
#   bash auto_readable2.sh
#
# Common overrides:
#   PRESET_YAML=/path/to/preset.yaml bash auto_readable2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="${SCRIPT_DIR}/../env.yaml"
PRESETS_DIR="${SCRIPT_DIR}/../presets"
SERVE_SH="${SCRIPT_DIR}/../serve.sh"
SERVE_SGLANG_SH="${SCRIPT_DIR}/serve_sglang_ds3.2.sh"

# Backend: vllm (default) or sglang.
#   vllm   — serve.sh with a preset; logs under logs/<preset_name>/...
#   sglang — serve_sglang_ds3.2.sh (no preset, DeepSeek-V3.2); logs under
#            logs_sglang/...; serves on port 30000.
BACKEND="${BACKEND:-vllm}"
case "${BACKEND,,}" in
    vllm)   SERVER_PORT="${SERVER_PORT:-8000}" ;;
    sglang) SERVER_PORT="${SERVER_PORT:-${SGLANG_PORT:-30000}}" ;;
    *) echo "[ERROR] BACKEND must be 'vllm' or 'sglang' (got: ${BACKEND})" >&2; exit 1 ;;
esac

if ! command -v yq >/dev/null 2>&1; then
    echo "[ERROR] yq is required. Run serve.sh once to auto-install it."
    exit 1
fi

# Honor MODEL_PATH if exported by run_all.sh (per-family); else env.yaml
# default. SGLang serves DeepSeek-V3.2 — MODEL_PATH must match the id passed to
# serve_sglang_ds3.2.sh --model-path (also used as the client "model" field).
if [ "${BACKEND,,}" = "sglang" ]; then
    MODEL_PATH="${MODEL_PATH:-${SGLANG_MODEL_PATH:-/remote/vast0/share-mv/deepseek-ai/DeepSeek-V3.2}}"
else
    MODEL_PATH="${MODEL_PATH:-$(yq e '.model.path' "${ENV_YAML}")}"
fi

# Log dir layout depends on backend:
#   vllm   — logs/<preset_name>/auto_readable2/<ts>. run_all.sh exports
#            PRESET_NAME; standalone runs derive it from PRESET_YAML's path
#            relative to presets/ ("/" -> "_", .yaml stripped). Without
#            PRESET_YAML (per-ic preset mode below) logs land under "default".
#   sglang — logs_sglang/auto_readable2/<ts> (no preset).
if [ "${BACKEND,,}" = "sglang" ]; then
    AUTO_LOG_DIR="${SCRIPT_DIR}/logs_sglang/auto_readable2"
else
    if [ -z "${PRESET_NAME:-}" ]; then
        if [ -n "${PRESET_YAML:-}" ]; then
            abs_preset="$(cd "$(dirname "${PRESET_YAML}")" && pwd)/$(basename "${PRESET_YAML}")"
            preset_rel="${abs_preset#"$(cd "${PRESETS_DIR}" && pwd)/"}"
            [ "${preset_rel}" = "${abs_preset}" ] && preset_rel="$(basename "${abs_preset}")"
            preset_rel="${preset_rel%.yaml}"
            PRESET_NAME="${preset_rel//\//_}"
        else
            PRESET_NAME="default"
        fi
    fi
    AUTO_LOG_DIR="${SCRIPT_DIR}/logs/${PRESET_NAME}/auto_readable2"
fi
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${AUTO_LOG_DIR}/${ts}"
mkdir -p "${RUN_DIR}"

CURL_CMD="curl -s http://localhost:${SERVER_PORT}/v1/completions -H \"Content-Type: application/json\""

echo "=== auto_readable2.sh started at $(date) ==="

make_requests() {
    local prompt="$1"
    echo "Making requests for prompt: $prompt"
    echo "------------------------------------------"

    response=$(eval "$CURL_CMD -d '{
        \"model\": \"$MODEL_PATH\",
        \"prompt\": \"$prompt\",
        \"max_tokens\": 150,
        \"temperature\": 0
    }'")
    echo "$response" | sed -n 's/.*"text":\(.*\),"logprobs".*/\1/p'
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
# Helper: poll until the server health endpoint responds
# ===========================================================
wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}"
    local elapsed=0
    echo "  [wait] Polling localhost:${SERVER_PORT}/health (timeout: ${max_wait}s)..."
    while ! curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; do
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
# Helper: kill the running backend's server processes
# ===========================================================
kill_server() {
    if [ "${BACKEND,,}" = "sglang" ]; then
        echo "  [kill] Sending SIGKILL to SGLang processes..."
        # pkill WITHOUT -f matches the process name (comm), so it kills the
        # sglang server + its setproctitle'd workers ("sglang::...") but never
        # our bash/tee wrappers (whose comm isn't "sglang") — no self-kill.
        pkill -9 sglang 2>/dev/null || true
    else
        echo "  [kill] Sending SIGKILL to VLLM processes..."
        pkill -9 VLLM 2>/dev/null || true
    fi
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
    local result_dir="${RUN_DIR}/ic${ic_suffix}"
    mkdir -p "${result_dir}"

    # vLLM needs a preset; sglang has none. Honor an external PRESET_YAML
    # override — same idea as auto_bench.sh.
    local ic_yaml=""
    if [ "${BACKEND,,}" != "sglang" ]; then
        if [ -n "${PRESET_YAML:-}" ]; then
            ic_yaml="${PRESET_YAML}"
        else
            ic_yaml="${PRESETS_DIR}/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-${ic_suffix}ic.yaml"
        fi
    fi

    echo ""
    echo "=========================================================="
    echo "READABLE: ic${ic_suffix}"
    echo "  backend : ${BACKEND}"
    echo "  preset  : ${ic_yaml:-<none (sglang)>}"
    echo "  results : ${result_dir}"
    echo "  Started : $(date)"
    echo "=========================================================="

    if [ "${BACKEND,,}" != "sglang" ] && [ ! -f "${ic_yaml}" ]; then
        echo "  [ERROR] preset not found: ${ic_yaml}"
        return 1
    fi

    kill_server

    local serve_log="${result_dir}/serve.log"
    echo "  [serve] Starting server, log: ${serve_log}"
    if [ "${BACKEND,,}" = "sglang" ]; then
        (SGLANG_PORT="${SERVER_PORT}" bash "${SERVE_SGLANG_SH}" "${MODEL_PATH}") >"${serve_log}" 2>&1 &
    else
        (bash "${SERVE_SH}" "${MODEL_PATH}" "${ic_yaml}") >"${serve_log}" 2>&1 &
    fi
    echo "  [serve] PID: $!"

    if ! wait_for_server; then
        echo "  [ERROR] Aborting readable for ic${ic_suffix}."
        kill_server
        return 1
    fi

    local readable_log="${result_dir}/readable.log"
    echo "  [readable] log: ${readable_log}"
    run 2>&1 | tee "${readable_log}"

    kill_server

    echo "READABLE ic${ic_suffix} completed at $(date)"
}

# ===========================================================
# Kill any leftover server before starting
# ===========================================================
kill_server

run_readable 0

echo ""
echo "=== All readable runs completed at $(date) ==="
