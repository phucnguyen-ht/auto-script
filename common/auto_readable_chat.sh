#!/usr/bin/env bash
set -uo pipefail
# readable method 'chat': /v1/chat/completions (server applies the chat template).
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READABLE=chat

run_readable() {
    local p response
    for p in "${PROMPTS[@]}"; do
        echo "Making requests for prompt: ${p}"
        echo "------------------------------------------"
        response=$(curl -s "${BASE_URL}/v1/chat/completions" -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"${p}\"}],\"seed\":42,\"max_tokens\":200}")
        python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"${response}"
        echo ""; echo ""
    done
}

source "${COMMON_DIR}/auto_readable_template.sh"
