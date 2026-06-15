#!/usr/bin/env bash
# mv-4476 driver — just the shared readable/eval phases (no ticket-specific
# bench/profile). See ../common/run_all.sh for options.
#
#   bash run_all.sh
#   PRESET=glm5/dp8ep8/bs64-moreh.yaml bash run_all.sh
#   RUN_EVAL=0 bash run_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/run_all.sh"
