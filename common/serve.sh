#!/usr/bin/env bash
set -uo pipefail
# Generic DEBUG serve-only (foreground). A ticket's serve.sh sets SCRIPT_DIR then
# sources this. Logs land like the auto_* phases:
#   logs/<preset>/serve/<ts>/{serve.log,preset.yaml}
# vLLM: scheduler-cls is stripped by default (eval-ready, matching auto_eval's
# lm_eval server). STRIP_SCHEDULER_CLS=0 to serve the preset verbatim. Ctrl-C to stop.
#
#   bash serve.sh
#   PRESET=glm5/dp8ep8/bs64-moreh.yaml bash serve.sh
#   BACKEND=sglang bash serve.sh
#   STRIP_SCHEDULER_CLS=0 bash serve.sh

: "${SCRIPT_DIR:?ticket serve.sh must set SCRIPT_DIR before sourcing}"
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_YAML="${ENV_YAML:-${SCRIPT_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}}"
STRIP_SCHEDULER_CLS="${STRIP_SCHEDULER_CLS:-1}"

source "${COMMON_DIR}/helper.sh"

resolve_backend
resolve_preset
resolve_model_path
setup_run_dir serve

echo "=== serve.sh (debug) at $(date) ==="
echo "backend=${BACKEND} port=${SERVER_PORT} model=${MODEL_PATH}"
echo "run dir=${RUN_DIR}"

if [ "${BACKEND,,}" = "sglang" ]; then
    [ -f "${SERVE_SGLANG_SH}" ] || { echo "[ERROR] SERVE_SGLANG_SH not set/found: '${SERVE_SGLANG_SH}'" >&2; exit 1; }
    SGLANG_PORT="${SERVER_PORT}" bash "${SERVE_SGLANG_SH}" "${MODEL_PATH}" 2>&1 | tee "${RUN_DIR}/serve.log"
else
    [ -f "${PRESET_YAML}" ] || { echo "[ERROR] preset not found: ${PRESET_YAML}" >&2; exit 1; }
    served="${RUN_DIR}/preset.yaml"
    if is_enabled "${STRIP_SCHEDULER_CLS}"; then
        yq 'del(.engine_args["scheduler-cls"])' "${PRESET_YAML}" > "${served}"
        echo "preset (scheduler-cls stripped): ${served}"
    else
        cp -f "${PRESET_YAML}" "${served}"
        echo "preset: ${served}"
    fi
    bash "${SERVE_SH}" "${MODEL_PATH}" "${served}" 2>&1 | tee "${RUN_DIR}/serve.log"
fi
