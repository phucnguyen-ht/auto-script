#!/usr/bin/env bash
# mv-4476 driver. profile/bench use the common random-dataset runner; readable/
# eval are the shared phases. Order/selection via env.yaml .phases.
#
#   bash run_all.sh
#   PRESET=kimi2.6/dp8ep8/... bash run_all.sh
#   RUN_BENCH=0 RUN_PROFILE=0 bash run_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_BENCH="${RUN_BENCH:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

# profile/bench dispatched by name from .phases -> this ticket's wrappers.
# vLLM-only (skipped for sglang); toggled by RUN_BENCH / RUN_PROFILE.
ticket_phase() {
    [[ "${BACKEND,,}" == "sglang" ]] && { echo "[run_all] skip $1 for sglang"; return 0; }
    case "$1" in
        bench)
            is_enabled "${RUN_BENCH}" || return 0
            rm -rf /root/.cache/vllm/torch_compile_cache/
            phase bench bash "${SCRIPT_DIR}/auto_bench.sh"
            ;;
        profile)
            is_enabled "${RUN_PROFILE}" || return 0
            rm -rf /root/.cache/vllm/torch_compile_cache/
            phase profile bash "${SCRIPT_DIR}/auto_profile.sh"
            ;;
    esac
}

source "${SCRIPT_DIR}/../common/run_all.sh"
