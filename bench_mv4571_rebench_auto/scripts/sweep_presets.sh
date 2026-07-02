#!/usr/bin/env bash
# Sweep presets by running RUN_AND_BENCH end-to-end (run_and_bench.sh) per preset,
# each in its own RUN dir. Waits for GPUs to free between presets; a preset that
# dies/times out just moves on. Comment out lines in PRESETS to trim the sweep.
#   bash sweep_presets.sh
# Env passthrough to run_and_bench.sh: MODEL PORT SERVE_SH SERVER_WAIT_TIMEOUT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"
PDIR="${AUTO_ROOT}/presets/glm5.2.rebench"
SWEEP_ROOT="${SWEEP_ROOT:-${TICKET_DIR}/logs/sweep/$(date +%Y%m%d_%H%M%S)}"

# VIABLE configs only (verified on gpu-5 / image 260626 / 1P1D; see
# scripts/DEBUG_ASYNC_HANG.md). nixl/gloo run ASYNC (transfer off the NCCL group);
# nccl/pynccl run SYNC because their ASYNC hangs "at collective communication
# calls" (vLLM eplb_state.py:242 -- unfixable framework limit). Regenerate the
# preset files with: bash ../gen_eplb_presets.sh . Comment out lines to trim.
PRESETS=(
    MTP5-bs64-dg.yaml                                    # baseline (no EPLB)
    # -- default interval, r0 --
    MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml
    MTP5-bs64-dg-eplb-gloo-async-default-r0.yaml
    MTP5-bs64-dg-eplb-nccl-sync-default-r0.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-default-r0.yaml
    # -- default interval, redundancy r8/r16 --
    MTP5-bs64-dg-eplb-nixl-async-default-r8.yaml
    MTP5-bs64-dg-eplb-gloo-async-default-r8.yaml
    MTP5-bs64-dg-eplb-nccl-sync-default-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-default-r8.yaml
    MTP5-bs64-dg-eplb-nixl-async-default-r16.yaml
    MTP5-bs64-dg-eplb-gloo-async-default-r16.yaml
    MTP5-bs64-dg-eplb-nccl-sync-default-r16.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-default-r16.yaml
    # -- s250 interval, r0/r8/r16 --
    MTP5-bs64-dg-eplb-nixl-async-s250-r0.yaml
    MTP5-bs64-dg-eplb-gloo-async-s250-r0.yaml
    MTP5-bs64-dg-eplb-nccl-sync-s250-r0.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s250-r0.yaml
    MTP5-bs64-dg-eplb-nixl-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-gloo-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-nccl-sync-s250-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s250-r8.yaml
    MTP5-bs64-dg-eplb-nixl-async-s250-r16.yaml
    MTP5-bs64-dg-eplb-gloo-async-s250-r16.yaml
    MTP5-bs64-dg-eplb-nccl-sync-s250-r16.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s250-r16.yaml
)

wait_gpu_free() {
    local iv="${GPU_POLL_INTERVAL:-30}" thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" n
    while :; do
        n=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' | awk -v t="${thr}" '$1 > t' | wc -l)
        [ "${n}" -eq 0 ] && { echo "[sweep] GPUs free"; return 0; }
        echo "[sweep] ${n} GPU(s) busy; retry in ${iv}s"; sleep "${iv}"
    done
}

mkdir -p "${SWEEP_ROOT}"
echo "[sweep] ${#PRESETS[@]} presets -> ${SWEEP_ROOT}"
for p in "${PRESETS[@]}"; do
    [ -f "${PDIR}/${p}" ] || { echo "[sweep] SKIP missing ${p}"; continue; }
    name="${p%.yaml}"
    echo "=================== ${name} ==================="
    wait_gpu_free
    RUN="${SWEEP_ROOT}/${name}" PRESET="${PDIR}/${p}" bash "${SCRIPT_DIR}/run_and_bench.sh" \
        || echo "[sweep] ${name} FAILED -> next"
done

echo "[sweep] DONE -> ${SWEEP_ROOT}"
echo "[sweep] summaries:"
for f in "${SWEEP_ROOT}"/*/scenario_summary.csv; do [ -f "${f}" ] && echo "  ${f}"; done
