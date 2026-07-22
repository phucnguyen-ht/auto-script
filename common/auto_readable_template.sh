#!/usr/bin/env bash
set -uo pipefail
# Readable smoke-test template. A concrete sets READABLE (method name) and
# defines run_readable, then sources this. Template owns: serve (AUTO_SERVE),
# load shared PROMPTS (readable_prompts.txt), run_readable -> log, kill.
# Logs: logs/<preset>/auto_readable_<method>/<ts>/readable.log
#
# Concrete must define (before sourcing):
#   READABLE=<method>     # e.g. completion | chat | pychat
#   run_readable          # send the prompts (uses the PROMPTS array / BASE_URL / MODEL_PATH)

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_DIR}/helper.sh"

: "${READABLE:?concrete must set READABLE before sourcing auto_readable_template.sh}"
declare -F run_readable >/dev/null || { echo "[ERROR] run_readable() not defined" >&2; exit 1; }

resolve_backend
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml}"
resolve_model_path
AUTO_SERVE="${AUTO_SERVE:-1}"

mapfile -t PROMPTS < "${COMMON_DIR}/readable_prompts.txt"
PROMPTS_FILE="${COMMON_DIR}/readable_prompts.txt"

setup_run_dir "auto_readable_${READABLE}"
echo "=== readable:${READABLE} at $(date) (backend=${BACKEND}) ==="

if is_enabled "${AUTO_SERVE}"; then
    [ "${BACKEND,,}" = "sglang" ] || [ -f "${PRESET_YAML}" ] || { echo "[ERROR] preset not found: ${PRESET_YAML}" >&2; exit 1; }
    kill_server
    # kill first (release our own/stale server), THEN wait for VRAM to actually
    # drain before serving -- otherwise the new server's memory profiling can OOM
    # on leftover allocations. Mirrors the eval path (auto_eval.sh).
    wait_for_gpu_free
    serve_backend "${RUN_DIR}/serve.log"
fi
if ! wait_for_server; then echo "[ERROR] server not reachable." >&2; is_enabled "${AUTO_SERVE}" && kill_server; exit 1; fi

run_readable 2>&1 | tee "${RUN_DIR}/readable.log"

is_enabled "${AUTO_SERVE}" && kill_server
echo "=== readable:${READABLE} done at $(date) ==="
