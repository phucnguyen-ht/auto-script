#!/usr/bin/env bash
# Decide whether the nixl EPLB communicator is VIABLE on THIS node.
#
# nixl does receiver-initiated RDMA-READ of remote GPUs' expert weights via UCX.
# "RDMA" here means "read a peer's memory without bothering its CPU" -- it does NOT
# require an InfiniBand/RoCE NIC:
#   - INTRA-NODE (this deployment: 1 node, 8 GPU, dp8/ep8): UCX picks `rocm_ipc` and
#     reads peer VRAM straight over xGMI (Infinity Fabric). No NIC involved. So on a
#     single node nixl is viable as long as `rocm_ipc` + GPU P2P exist -- which they
#     do on these 8-GPU MI300/MI325 nodes. (See EPLB_DEEPDIVE §A.5.)
#   - CROSS-NODE (EP spread over >1 node): UCX then needs an RDMA NIC (rc_x/mlx5) to
#     read VRAM on another host.
# So: viable if `rocm_ipc` is available (intra-node) OR an RDMA NIC is present
# (cross-node). On this single-node setup the rocm_ipc check makes nixl viable.
#
# Exit 0 = nixl viable  |  Exit 1 = not viable (skip nixl).
# Env: NIXL_FORCE=1 -> force viable (exit 0); NIXL_FORCE=0 -> force not viable (exit 1).
set -uo pipefail

case "${NIXL_FORCE:-}" in
    1|true|yes|on)  echo "[check_nixl] NIXL_FORCE=1 -> forcing VIABLE"; exit 0 ;;
    0|false|no|off) echo "[check_nixl] NIXL_FORCE=0 -> forcing NOT viable"; exit 1 ;;
esac

# Intra-node signal: UCX rocm_ipc transport (peer VRAM read over xGMI). This is what
# makes nixl work on a single node WITHOUT any NIC.
rocm_ipc=0
if command -v ucx_info >/dev/null 2>&1; then
    rocm_ipc=$(ucx_info -d 2>/dev/null | grep -Ec 'Transport:[[:space:]]*rocm_ipc' || true)
fi

# Cross-node signal: an RDMA-capable NIC/transport (rc/rc_x/dc/ud) bound to a device.
ndev=0
if command -v ibv_devices >/dev/null 2>&1; then
    ndev=$(ibv_devices 2>/dev/null | awk 'NR>2 && NF>0' | wc -l)
fi
rdma_tls=0
if command -v ucx_info >/dev/null 2>&1; then
    rdma_tls=$(ucx_info -d 2>/dev/null \
        | grep -Ec 'Transport:[[:space:]]*(rc|rc_x|rc_verbs|rc_mlx5|dc|dc_x|dc_mlx5|ud|ud_x)' || true)
fi

echo "[check_nixl] rocm_ipc=${rocm_ipc} ibv_devices=${ndev} ucx_rdma_transports=${rdma_tls}"
if [ "${rocm_ipc}" -gt 0 ]; then
    echo "[check_nixl] rocm_ipc present -> nixl VIABLE intra-node (xGMI peer read; no NIC needed)"
    exit 0
fi
if [ "${ndev}" -gt 0 ] || [ "${rdma_tls}" -gt 0 ]; then
    echo "[check_nixl] RDMA NIC present -> nixl VIABLE (cross-node capable)"
    exit 0
fi
echo "[check_nixl] no rocm_ipc and no RDMA NIC -> nixl NOT viable -> skip (use gloo for async)"
exit 1
