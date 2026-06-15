#!/usr/bin/env bash
set -uo pipefail
# DEBUG: run readable smoke tests against an already-running server (AUTO_SERVE=0),
# e.g. one started by `bash serve.sh`. Logs land like the auto_readable phases:
#   logs/<preset>/auto_readable{,_thinking}/<ts>/...
#
#   bash serve.sh        # (another terminal) start the server first
#   bash readable.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${SCRIPT_DIR}/../common"
export ENV_YAML="${SCRIPT_DIR}/env.yaml"
export LOG_ROOT="${SCRIPT_DIR}/logs"
export DATA_DIR="${SCRIPT_DIR}"

source "${COMMON_DIR}/helper.sh"
resolve_backend
resolve_preset
[ "${BACKEND,,}" = "sglang" ] || export MODEL_FAMILY="${PRESET_FAMILY}"
export PRESET_YAML PRESET_NAME
export AUTO_SERVE=0
bash "${COMMON_DIR}/auto_readable_thinking.sh"
bash "${COMMON_DIR}/auto_readable.sh"
