#!/usr/bin/env bash
# profile = auto_bench.sh with MODE=profile (torch profiler injected into preset).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=profile exec bash "${SCRIPT_DIR}/auto_bench.sh"
