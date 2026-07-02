#!/usr/bin/env bash
# Decide whether the nixl EPLB communicator is VIABLE on THIS node.
#
# Why: nixl does receiver-initiated RDMA READ of remote GPUs' expert weights.
#   - On a node WITH an RDMA NIC (e.g. MI325): nixl works (init + transfer OK).
#   - On a node WITHOUT RDMA (MI300 tw031: `ibv_devices` empty): UCX still *inits*
#     over tcp, but the async expert-weight transfer CRASHES the engine on the first
#     rearrange (hard GPU fault, no Python traceback). See scripts/progress.md §6.
# So nixl viability is a HARDWARE property -> gate on RDMA-device presence. This lets
# sweep_presets.sh auto-include nixl on RDMA nodes and auto-skip it on non-RDMA nodes.
#
# Exit 0 = nixl viable (RDMA present)  |  Exit 1 = not viable (skip nixl).
# Env: NIXL_FORCE=1 -> force viable (exit 0); NIXL_FORCE=0 -> force not viable (exit 1).
set -uo pipefail

case "${NIXL_FORCE:-}" in
    1|true|yes|on)  echo "[check_nixl] NIXL_FORCE=1 -> forcing VIABLE"; exit 0 ;;
    0|false|no|off) echo "[check_nixl] NIXL_FORCE=0 -> forcing NOT viable"; exit 1 ;;
esac

ndev=0
if command -v ibv_devices >/dev/null 2>&1; then
    # ibv_devices prints a 2-line header then one line per device.
    ndev=$(ibv_devices 2>/dev/null | awk 'NR>2 && NF>0' | wc -l)
fi

# Fallback signal: a UCX RDMA-capable transport (rc/rc_x/dc/ud) bound to a real device.
rdma_tls=0
if command -v ucx_info >/dev/null 2>&1; then
    rdma_tls=$(ucx_info -d 2>/dev/null \
        | grep -Ec 'Transport:[[:space:]]*(rc|rc_x|rc_verbs|dc|dc_x|ud|ud_x)' || true)
fi

echo "[check_nixl] ibv_devices=${ndev} ucx_rdma_transports=${rdma_tls}"
if [ "${ndev}" -gt 0 ] || [ "${rdma_tls}" -gt 0 ]; then
    echo "[check_nixl] RDMA present -> nixl VIABLE (async nixl can be swept)"
    exit 0
fi
echo "[check_nixl] no RDMA NIC -> nixl NOT viable on this node -> skip (use gloo for async)"
exit 1
