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

# Ordered least->most hang-prone (evidence: ../mtp_default_compare.csv, async runs).
# default interval = safe for all; s250 makes nccl/pynccl worker-stall CRASH;
# nixl (UCX+kv fix) is stable at every interval. Comment out tiers you don't want.
PRESETS=(
    # -- Tier 1: default interval, r0 -- all completed 72/72/72 (safest) --
    MTP5-bs64-dg.yaml
    MTP5-bs64-dg-eplb-nccl-async-default-r0.yaml
    MTP5-bs64-dg-eplb-pynccl-async-default-r0.yaml
    MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml
    # -- Tier 2: default interval + redundancy -- interval safe; red8 ok (pynccl 1 slow run) --
    MTP5-bs64-dg-eplb-nccl-async-default-r8.yaml
    MTP5-bs64-dg-eplb-nixl-async-default-r8.yaml
    MTP5-bs64-dg-eplb-nccl-async-default-r16.yaml
    MTP5-bs64-dg-eplb-nixl-async-default-r16.yaml
    MTP5-bs64-dg-eplb-pynccl-async-default-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-async-default-r16.yaml
    # -- Tier 3: s250 + nixl -- nixl handles dense rearrange OK (72/72/72) --
    MTP5-bs64-dg-eplb-nixl-async-s250-r0.yaml
    MTP5-bs64-dg-eplb-nixl-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-nixl-async-s250-r16.yaml
    # -- Tier 4: s250 + nccl/pynccl -- HANG-PRONE (compare.csv: worker-stall CRASH); last --
    MTP5-bs64-dg-eplb-nccl-async-s250-r0.yaml
    MTP5-bs64-dg-eplb-nccl-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-nccl-async-s250-r16.yaml
    MTP5-bs64-dg-eplb-pynccl-async-s250-r0.yaml
    MTP5-bs64-dg-eplb-pynccl-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-async-s250-r16.yaml
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
