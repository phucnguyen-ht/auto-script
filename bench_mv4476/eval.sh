#!/usr/bin/env bash
set -uo pipefail
# DEBUG: run eval against an already-running server (AUTO_SERVE=0), e.g. one
# started by `bash serve.sh`. Which datasets/runs/method run is whatever you set
# in env.yaml. Logs/results land like the auto_eval phase:
#   logs/<preset>/auto_eval/<ts>/...
#
#   bash serve.sh        # (another terminal) start the server first
#   bash eval.sh
#   PRESET=glm5/dp8ep8/bs64-moreh.yaml bash eval.sh   # match serve's preset

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
exec bash "${COMMON_DIR}/auto_eval.sh"
