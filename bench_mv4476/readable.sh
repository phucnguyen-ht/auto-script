#!/usr/bin/env bash
set -uo pipefail
# DEBUG: run the readable smoke tests (methods from env.yaml .eval.readable)
# against an already-running server (AUTO_SERVE=0), e.g. one from `bash serve.sh`.
# Logs like the auto_readable phases: logs/<preset>/auto_readable_<method>/<ts>/.
#
#   bash serve.sh          # (another terminal) start the server first
#   bash readable.sh
#   PRESET=kimi2.6/dp8ep8/... bash readable.sh

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

for meth in $(readable_list); do
    bash "${COMMON_DIR}/auto_readable_${meth}.sh"
done
