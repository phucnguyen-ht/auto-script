#!/usr/bin/env bash
# mv-4433 driver. Adds ticket-specific bench/profile phases on top of the shared
# readable/eval driver in ../common/run_all.sh.
#
#   bash run_all.sh
#   RUN_BENCH=0 RUN_PROFILE=0 bash run_all.sh
#   ENFORCE_EAGER_PROFILE=1 bash run_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ticket-specific toggles. sglang serve_script/model come from env.yaml.
RUN_BENCH="${RUN_BENCH:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
ENFORCE_EAGER_PROFILE="${ENFORCE_EAGER_PROFILE:-0}"

# Ticket-specific phases dispatched by name from env.yaml .phases. bench/profile
# are vLLM-only (skipped for sglang). Order is whatever env.yaml .phases lists.
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
            local profile_preset="${PRESET}"
            if is_enabled "${ENFORCE_EAGER_PROFILE}"; then
                profile_preset="${MASTER_LOG_DIR}/preset.profile_enforce_eager.yaml"
                yq '.engine_args.enforce_eager = true' "${PRESET}" > "${profile_preset}"
                echo "[run_all] ENFORCE_EAGER_PROFILE=1 -> profiling with enforce_eager:true"
            fi
            PRESET_YAML="${profile_preset}" phase profile bash "${SCRIPT_DIR}/auto_profile.sh"
            ;;
    esac
}

source "${SCRIPT_DIR}/../common/run_all.sh"
