#!/usr/bin/env bash
# Standalone random-dataset bench (config from env.yaml .bench.random).
#   bash auto_bench.sh
#   PRESET=kimi2.6/dp8ep8/... bash auto_bench.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_YAML="${SCRIPT_DIR}/env.yaml"
export LOG_ROOT="${SCRIPT_DIR}/logs"
export DATA_DIR="${SCRIPT_DIR}"
MODE=bench exec bash "${SCRIPT_DIR}/../common/auto_bench_random.sh"
