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
SWEEP_ROOT="${SWEEP_ROOT:-${TICKET_DIR}/logs/sweep_results/$(date +%Y%m%d_%H%M%S)}"

# VIABLE configs on MI300 (192 GiB/GPU, node tw031, image 260626, 1P1D; see
# scripts/progress.md). Differences vs MI325:
#   - nixl DROPPED: node has no RDMA NIC (ibv_devices empty) -> UCX inits over tcp
#     but the async expert-weight transfer (GPU-mem RDMA read) CRASHES the engine on
#     the first rearrange (progress.md §6). Hardware limit, not fixable by preset.
#   - gloo runs ASYNC (CPU-staged, off the NCCL group -> no deadlock, no RDMA need).
#   - nccl/pynccl run SYNC (their ASYNC hangs "at collective communication calls",
#     vLLM eplb_state.py:242 -- framework limit, same as MI325).
#   - ALL EPLB presets cap max_model_len=512K: EPLB eats ~10 GiB of KV so 1M no longer
#     fits (needs 54.62 GiB, only ~43-49 GiB left). Workload is 100K ISL so this does
#     not affect the benched requests. (baseline keeps 1M.)
# Regenerate the preset files with: bash ../gen_eplb_presets.sh . Comment out to trim.
PRESETS=(
    # MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml
    # MTP5-bs64-dg.yaml                                       # baseline (no EPLB, 1M)
    # # -- default interval, r0 --
    # MTP5-bs64-dg-eplb-gloo-async-default-r0.yaml
    # MTP5-bs64-dg-eplb-nccl-sync-default-r0.yaml
    # MTP5-bs64-dg-eplb-pynccl-sync-default-r0.yaml
    # # -- default interval, redundancy r8/r16 --
    # MTP5-bs64-dg-eplb-gloo-async-default-r8.yaml
    # MTP5-bs64-dg-eplb-nccl-sync-default-r8.yaml
    # MTP5-bs64-dg-eplb-pynccl-sync-default-r8.yaml
    # # MTP5-bs64-dg-eplb-gloo-async-default-r16.yaml
    # MTP5-bs64-dg-eplb-nccl-sync-default-r16.yaml
    # MTP5-bs64-dg-eplb-pynccl-sync-default-r16.yaml
    # # -- s250 interval, r0/r8/r16 --
    # MTP5-bs64-dg-eplb-nixl-async-s250-r0.yaml
    # MTP5-bs64-dg-eplb-gloo-async-s250-r0.yaml
    # MTP5-bs64-dg-eplb-nccl-sync-s250-r0.yaml

    MTP5-bs64-dg-eplb-nixl-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-gloo-async-s250-r8.yaml
    MTP5-bs64-dg-eplb-nixl-async-s500-r0.yaml
    MTP5-bs64-dg-eplb-nixl-async-s500-r8.yaml
    MTP5-bs64-dg-eplb-gloo-async-s500-r0.yaml
    MTP5-bs64-dg-eplb-gloo-async-s500-r8.yaml

    MTP5-bs64-dg-eplb-nccl-sync-s250-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s250-r8.yaml

    # -- s500 interval, r0/r8/r16 --
    MTP5-bs64-dg-eplb-nccl-sync-s500-r0.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s500-r0.yaml

    MTP5-bs64-dg-eplb-nccl-sync-s500-r8.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s500-r8.yexpoaml

    # -- s250 interval, r16 --
    MTP5-bs64-dg-eplb-nccl-sync-s250-r16.yaml
    MTP5-bs64-dg-eplb-pynccl-sync-s250-r16.yaml

    MTP5-bs64-dg-eplb-pynccl-sync-s250-r0.yaml
    MTP5-bs64-dg-eplb-gloo-async-s250-r16.yaml
)

# nixl is async but only works on nodes WITH an RDMA NIC (see check_nixl.sh /
# progress.md §6). Auto-include the 6 nixl presets iff this node has RDMA; otherwise
# skip (this MI300 node has none). Override with NIXL_FORCE=1/0.
if bash "${SCRIPT_DIR}/check_nixl.sh"; then
    PRESETS+=(
        MTP5-bs64-dg-eplb-nixl-async-default-r8.yaml
        MTP5-bs64-dg-eplb-nixl-async-default-r16.yaml
        MTP5-bs64-dg-eplb-nixl-async-s250-r0.yaml
        MTP5-bs64-dg-eplb-nixl-async-s250-r8.yaml
        MTP5-bs64-dg-eplb-nixl-async-s250-r16.yaml
    )
    echo "[sweep] nixl VIABLE on this node -> added 6 nixl presets"
else
    echo "[sweep] nixl NOT viable on this node -> skipped (gloo covers async)"
fi

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
