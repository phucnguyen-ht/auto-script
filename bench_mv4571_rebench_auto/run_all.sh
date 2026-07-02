#!/usr/bin/env bash
# mv-4571 rebench-auto driver. bench/profile via this ticket's wrappers.
#   bash run_all.sh
#   PRESET=glm5.2.rebench/MTP5-bs64-dg.yaml bash run_all.sh
#   RUN_BENCH=0 RUN_PROFILE=0 bash run_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_BENCH="${RUN_BENCH:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

ticket_phase() {
    [[ "${BACKEND,,}" == "sglang" ]] && { echo "[run_all] skip $1 for sglang"; return 0; }
    case "$1" in
        bench)
            is_enabled "${RUN_BENCH}" || return 0
            # is_enabled "${AUTO_CLEAN:-0}" && rm -rf /root/.cache/vllm/torch_compile_cache/
            phase bench bash "${SCRIPT_DIR}/auto_bench.sh"
            ;;
        profile)
            is_enabled "${RUN_PROFILE}" || return 0
            # is_enabled "${AUTO_CLEAN:-0}" && rm -rf /root/.cache/vllm/torch_compile_cache/
            phase profile bash "${SCRIPT_DIR}/auto_profile.sh"
            ;;
    esac
}

source "${SCRIPT_DIR}/../common/run_all.sh"
