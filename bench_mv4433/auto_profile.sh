#!/usr/bin/env bash
# mv-4433 custom profile = auto_bench.sh with MODE=profile (torch profiler on).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=profile exec bash "${SCRIPT_DIR}/auto_bench.sh"
