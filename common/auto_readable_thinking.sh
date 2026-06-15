#!/usr/bin/env bash
set -uo pipefail
# Serve, then run smoke-test requests against /v1/chat/completions (thinking).
# Override: PRESET_YAML=/path bash auto_readable_thinking.sh  /  BACKEND=sglang ...

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_DIR}/helper.sh"

resolve_backend
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml}"
resolve_model_path
setup_run_dir auto_readable_thinking

CURL_URL="${BASE_URL}/v1/chat/completions"

make_requests() {
    echo "Making requests for prompt: $1"
    echo "------------------------------------------"
    response=$(curl -s "${CURL_URL}" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL_PATH}"'",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "'"$1"'"}
            ],
            "seed": 42,
            "max_tokens": 200
        }')
    python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"${response}"
    echo ""; echo ""
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

echo "=== auto_readable_thinking.sh started at $(date) (backend=${BACKEND}) ==="

if [ "${BACKEND,,}" != "sglang" ] && [ ! -f "${PRESET_YAML}" ]; then
    echo "[ERROR] preset not found: ${PRESET_YAML}" >&2; exit 1
fi

kill_server
serve_backend "${RUN_DIR}/serve.log"
if ! wait_for_server; then echo "[ERROR] server failed to start." >&2; kill_server; exit 1; fi

run 2>&1 | tee "${RUN_DIR}/readable.log"

kill_server
echo "=== auto_readable_thinking.sh done at $(date) ==="
