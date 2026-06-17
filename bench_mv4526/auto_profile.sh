#!/usr/bin/env bash
# mv-4526 custom-dataset profile = auto_bench.sh with MODE=profile (adds --profile
# + torch profiler on the server).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=profile exec bash "${SCRIPT_DIR}/auto_bench.sh"
