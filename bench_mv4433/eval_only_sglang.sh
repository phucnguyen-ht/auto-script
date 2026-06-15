#!/usr/bin/env bash
set -euo pipefail

# Run eval ONLY (AUTO_SERVE=0) against a server you started yourself, e.g.:
#     bash serve_sglang_ds3.2.sh
# Datasets/run-counts come from env.yaml -> eval.datasets, same as run_all.
#
#   bash eval_only_sglang.sh
#   BACKEND=vllm BASE_URL=http://localhost:8000 bash eval_only_sglang.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Wire the shared eval script to this ticket.
export ENV_YAML="${SCRIPT_DIR}/env.yaml"
export LOG_ROOT="${SCRIPT_DIR}/logs"
export DATA_DIR="${SCRIPT_DIR}"
export SERVE_SGLANG_SH="${SCRIPT_DIR}/serve_sglang_ds3.2.sh"

export AUTO_SERVE=0
export BACKEND="${BACKEND:-sglang}"
[ "${BACKEND,,}" = "sglang" ] && _default_port="${SGLANG_PORT:-30000}" || _default_port=8000
export BASE_URL="${BASE_URL:-http://localhost:${_default_port}}"

if ! curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1; then
    echo "[eval-only] WARNING: no server at ${BASE_URL}/health" >&2
    echo "[eval-only] Start one first, e.g.: bash ${SCRIPT_DIR}/serve_sglang_ds3.2.sh" >&2
fi

exec bash "${SCRIPT_DIR}/../common/auto_eval.sh"
