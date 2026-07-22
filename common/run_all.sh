#!/usr/bin/env bash
# Shared master driver. A ticket's run_all.sh sets SCRIPT_DIR, optionally tweaks
# toggles / exports SERVE_SGLANG_SH / defines a ticket_phases() function, then
# sources this. ticket_phases <backend> is called inside the backend loop (after
# BACKEND/MODEL_FAMILY are set, before readable/eval) for ticket-specific phases.
#
#   bash run_all.sh
#   PRESET=glm5/dp8ep8/bs64-moreh.yaml bash run_all.sh
#   BACKEND=sglang bash run_all.sh
#   RUN_EVAL=0 RUN_READABLE=0 bash run_all.sh

set -euo pipefail

: "${SCRIPT_DIR:?ticket run_all.sh must set SCRIPT_DIR before sourcing}"
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_READABLE="${RUN_READABLE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
# SHARED_SERVE=on: when the phases include readable and/or eval, serve the model ONCE
# and run every readable method + eval against that single server (sub-scripts run with
# AUTO_SERVE=0 -> no kill/restart between them). off (default) = original behavior
# (each readable method and eval kills + restarts its own server).
SHARED_SERVE="${SHARED_SERVE:-off}"

# Wire the shared scripts to this ticket.
export ENV_YAML="${ENV_YAML:-${SCRIPT_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}}"

source "${COMMON_DIR}/helper.sh"   # also installs yq via ensure_yq

# Preset resolution (vLLM) — shared with the debug serve.sh / eval_*.sh helpers.
BACKEND=vllm resolve_preset
export PRESET_YAML PRESET_NAME

BACKENDS_TO_RUN="${BACKEND:-$(backends_list)}"

ts="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="${LOG_ROOT}/run_all/${ts}"
mkdir -p "${MASTER_LOG_DIR}"
cp "${PRESET}" "${MASTER_LOG_DIR}/preset.yaml"

cat <<EOF
========================================================================
run_all.sh
  backends    : ${BACKENDS_TO_RUN}
  preset      : ${PRESET}
  master log  : ${MASTER_LOG_DIR}
  phases      : $(phases_list)  (toggles: readable=${RUN_READABLE} eval=${RUN_EVAL})
  started at  : $(date)
========================================================================
EOF
exec > >(tee -a "${MASTER_LOG_DIR}/run_all.log") 2>&1

phase() {
    local name="$1"; shift
    echo; echo "------------------------------------------------------------------------"
    echo "[run_all] phase=${name} backend=${BACKEND} at $(date)"
    echo "------------------------------------------------------------------------"
    "$@"
    echo "[run_all] phase=${name} DONE at $(date)"
}

for be in ${BACKENDS_TO_RUN}; do
    export BACKEND="${be}"
    [[ "${be,,}" == "sglang" ]] && unset MODEL_FAMILY || export MODEL_FAMILY="${PRESET_FAMILY}"

    echo; echo "############### BACKEND=${be} ###############"

    # SHARED_SERVE=on: bring up ONE server here for all readable+eval phases, then run
    # the sub-scripts with AUTO_SERVE=0 so they reuse it (no kill/restart per method).
    _shared=0
    _phl=" $(phases_list) "
    if is_enabled "${SHARED_SERVE}" && { [[ "${_phl}" == *" readable "* ]] || [[ "${_phl}" == *" eval "* ]]; }; then
        _shared=1
        echo "[run_all] SHARED_SERVE=on -> one serve reused by all readable+eval phases"
        resolve_backend
        resolve_model_path
        kill_server
        # kill first (release stale/our server), THEN wait for VRAM to actually drain
        # before serving so the model load doesn't OOM on leftover allocations (mirrors
        # the per-phase serve paths in auto_readable_template.sh / auto_eval.sh).
        wait_for_gpu_free
        _shared_log="${MASTER_LOG_DIR}/serve_shared_${be}.log"
        if [ "${be,,}" = "sglang" ]; then
            serve_backend "${_shared_log}"
        else
            # scheduler-cls (PDSLoggingScheduler) is incompatible with lm_eval; strip it so
            # the single shared server works for BOTH readable and eval.
            _shared_preset="${MASTER_LOG_DIR}/preset_shared.yaml"
            yq 'del(.engine_args["scheduler-cls"])' "${PRESET_YAML}" > "${_shared_preset}"
            PRESET_YAML="${_shared_preset}"
            serve_backend "${_shared_log}"
        fi
        if ! wait_for_server; then
            echo "[run_all][ERROR] shared serve failed to become ready" >&2
            kill_server; exit 1
        fi
        export AUTO_SERVE=0
    fi

    # Run phases in the order given by env.yaml .phases (default: readable eval).
    # readable/eval are handled here; any other name is delegated to the ticket's
    # ticket_phase() (e.g. bench/profile), which should guard on $BACKEND.
    for ph in $(phases_list); do
        case "${ph}" in
            readable)
                is_enabled "${RUN_READABLE}" || continue
                # Run only the readable methods enabled in env.yaml .eval.readable.
                for meth in $(readable_list); do
                    rm -rf /root/.cache/vllm/torch_compile_cache/
                    # Prefer a ticket-local readable script over the shared one, so a
                    # ticket can add its own method (e.g. longbench2) without touching common/.
                    rscript="${SCRIPT_DIR}/auto_readable_${meth}.sh"
                    [ -f "${rscript}" ] || rscript="${COMMON_DIR}/auto_readable_${meth}.sh"
                    phase "readable:${meth}" bash "${rscript}"
                done
                ;;
            eval)
                is_enabled "${RUN_EVAL}" || continue
                rm -rf /root/.cache/vllm/torch_compile_cache/
                phase eval bash "${COMMON_DIR}/auto_eval.sh"
                ;;
            *)
                if declare -F ticket_phase >/dev/null; then ticket_phase "${ph}"
                else echo "[run_all] unknown phase '${ph}' (no ticket_phase); skipping." >&2; fi
                ;;
        esac
    done

    # Tear down the shared server (if any) after this backend's phases.
    if [ "${_shared}" -eq 1 ]; then
        echo "[run_all] SHARED_SERVE teardown for backend=${be}"
        unset AUTO_SERVE
        kill_server
    fi
done

cat <<EOF

========================================================================
run_all.sh DONE at $(date)
  master log  : ${MASTER_LOG_DIR}
========================================================================
EOF
