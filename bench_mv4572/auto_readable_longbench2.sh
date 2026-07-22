#!/usr/bin/env bash
set -uo pipefail
# readable method 'longbench2' (ticket-local, MV-4572): LongBench-v2 100k SPOT-CHECK.
# Concrete for ../common/auto_readable_template.sh — the template serves the model
# (AUTO_SERVE), waits for /health, then calls run_readable, then kills the server.
# We just point check_longbench.sh (the single source of truth) at that server.
# Template globals available in run_readable: BASE_URL, MODEL_PATH.
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READABLE=longbench2

run_readable() {
    EVAL_MODEL="${MODEL_PATH}" EVAL_BASE_URL="${BASE_URL}" \
        bash "${THIS_DIR}/check_longbench.sh"
}

source "$(cd "${THIS_DIR}/../common" && pwd)/auto_readable_template.sh"
