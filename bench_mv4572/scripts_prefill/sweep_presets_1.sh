#!/usr/bin/env bash
# PREFILL sweep over presets via run_and_bench.sh (prefill). Counterpart of
# scripts/sweep_presets_1.sh (decode). SWEEP_MODE=on: 1 serve/preset, the whole
# CONCS x DATASETS list is pushed once into multi_process_test.py (dataset-outer,
# conc-inner, each window auto-sized by window_params()). Server + EPLB stay warm
# across scenarios. Prefill = prefix cache OFF (set in run_and_bench.sh), no warmup.
#
#   bash sweep_presets_1.sh
#   CONCS="16 64" DATASETS="8k 100k" bash sweep_presets_1.sh
# Env passthrough to run_and_bench.sh: MODEL PORT SERVER_WAIT_TIMEOUT SWEEP_ROOT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"
PDIR="${AUTO_ROOT}/presets/kimi2.6.mxfp4/dp8ep8"
SWEEP_ROOT="${SWEEP_ROOT:-${TICKET_DIR}/logs/sweep_prefill_tune_result/$(date +%Y%m%d_%H%M%S)}"

# Same axes as the decode sweep.
CONCS=(${CONCS:-16 64 128 256})
DATASETS=(${DATASETS:-8k 10k 100k})   # buckets under /remote/vast0/share-mv/longbenchv2-custom

# Only nixl works with Kimi-MXFP4. Regenerate presets: bash ../gen_eplb_presets.sh
if bash "${SCRIPT_DIR}/check_nixl.sh"; then
    echo "[sweep-prefill] nixl VIABLE"
else
    echo "[sweep-prefill] WARNING: nixl NOT viable here -> nixl presets will fail."
fi

# --- PRESETS to sweep (mirror the decode list) ---
PRESETS=(
    base.yaml

    base-eplb-nixl-async-default-r0.yaml
    base-eplb-nixl-async-default-r8.yaml
    base-eplb-nixl-async-default-r16.yaml

    base-eplb-nixl-async-s100-r0.yaml
    base-eplb-nixl-async-s100-r8.yaml
    base-eplb-nixl-async-s100-r16.yaml
    base-eplb-nixl-async-s250-r0.yaml
    base-eplb-nixl-async-s250-r8.yaml
    base-eplb-nixl-async-s500-r0.yaml
    base-eplb-nixl-async-s500-r8.yaml
    base-eplb-nixl-async-s1000-r0.yaml
    base-eplb-nixl-async-s1000-r8.yaml
)

wait_gpu_free() {
    local iv="${GPU_POLL_INTERVAL:-30}" thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" n
    while :; do
        n=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' | awk -v t="${thr}" '$1 > t' | wc -l)
        [ "${n}" -eq 0 ] && { echo "[sweep-prefill] GPUs free"; return 0; }
        echo "[sweep-prefill] ${n} GPU(s) busy; retry in ${iv}s"; sleep "${iv}"
    done
}

CONCS_CSV="$(IFS=,; echo "${CONCS[*]}")"
DS_CSV="$(IFS=,; echo "${DATASETS[*]}")"

mkdir -p "${SWEEP_ROOT}"
echo "[sweep-prefill] ${#PRESETS[@]} presets x concs(${CONCS[*]}) x datasets(${DATASETS[*]}) -> ${SWEEP_ROOT}"

for p in "${PRESETS[@]}"; do
    [ -f "${PDIR}/${p}" ] || { echo "[sweep-prefill] SKIP missing ${p}"; continue; }
    name="${p%.yaml}"

    # [supervisor-resume] Idempotent resume: skip a preset already finished in this
    # SWEEP_ROOT (scenario_summary.csv present). No-op for the default fresh-timestamp
    # root; lets a restart after an interruption continue from the unfinished preset.
    if [ -f "${SWEEP_ROOT}/${name}/scenario_summary.csv" ]; then
        echo "[sweep-prefill] SKIP ${name} (already complete in ${SWEEP_ROOT})"; continue
    fi

    echo "=================== ${name} @ concs=${CONCS_CSV} ds=${DS_CSV} (prefill, 1 serve) ==================="
    wait_gpu_free
    RUN="${SWEEP_ROOT}/${name}" PRESET="${PDIR}/${p}" \
        REBENCH_CONC="${CONCS_CSV}" REBENCH_DATASETS="${DS_CSV}" \
        bash "${SCRIPT_DIR}/run_and_bench.sh" \
        || echo "[sweep-prefill] ${name} FAILED -> next"
done

echo "[sweep-prefill] DONE -> ${SWEEP_ROOT}"
find "${SWEEP_ROOT}" -name scenario_summary.csv 2>/dev/null | while read -r f; do echo "  ${f}"; done
