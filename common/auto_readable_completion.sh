#!/usr/bin/env bash
set -uo pipefail
# readable method 'completion': raw /v1/completions, bare prompt (no chat template).
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READABLE=completion

run_readable() {
    local p response
    for p in "${PROMPTS[@]}"; do
        echo "Making requests for prompt: ${p}"
        echo "------------------------------------------"
        response=$(curl -s "${BASE_URL}/v1/completions" -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"${p}\",\"max_tokens\":300,\"temperature\":0}")
        echo "${response}" | sed -n 's/.*"text":\(.*\),"logprobs".*/\1/p'
        echo ""; echo ""
    done
}

source "${COMMON_DIR}/auto_readable_template.sh"
