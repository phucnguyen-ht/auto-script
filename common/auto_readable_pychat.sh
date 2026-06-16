#!/usr/bin/env bash
set -uo pipefail
# readable method 'pychat': apply the chat template CLIENT-SIDE then POST raw
# /v1/completions (sidesteps a wrong/mismatched server chat template).
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READABLE=pychat

run_readable() {
    python3 "${COMMON_DIR}/readable_pychat.py" \
        --base-url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --prompts-file "${PROMPTS_FILE}" \
        --max-tokens 300
}

source "${COMMON_DIR}/auto_readable_template.sh"
