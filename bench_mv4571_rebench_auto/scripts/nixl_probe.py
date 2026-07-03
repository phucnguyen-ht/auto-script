#!/usr/bin/env python3
"""Minimal NIXL VRAM-registration probe (repro of eplb_communicator
_init_registered_buffers). Tests whether NIXL can register a ~768 MiB ROCm
buffer under the current UCX_TLS. Run inside the pod with UCX_TLS set:

  UCX_TLS=tcp,self,sm,rocm_copy,rocm_ipc UCX_MEMTYPE_CACHE=n \
    python3 scripts/nixl_probe.py
"""
import os
import sys

import torch

import vllm.distributed.nixl_utils as nu

print("UCX_TLS       =", os.environ.get("UCX_TLS"))
print("UCX_MEMTYPE   =", os.environ.get("UCX_MEMTYPE_CACHE"))
print("NixlWrapper   =", nu.NixlWrapper)
print("agent_config  =", nu.nixl_agent_config)

cfg = (
    nu.nixl_agent_config(capture_telemetry=False)
    if nu.nixl_agent_config is not None
    else None
)
w = nu.NixlWrapper("probe-0", cfg)

# ~768 MiB ROCm tensor — matches the failing length (805306368) in serve.log.
n = 805306368
t = torch.empty(n, dtype=torch.uint8, device="cuda:0")
print(f"tensor: device={t.device} bytes={t.nbytes}")

descs = w.get_reg_descs([t])
try:
    w.register_memory(descs)
except Exception as exc:  # noqa: BLE001
    print("REGISTER_FAIL:", type(exc).__name__, exc)
    sys.exit(2)
print("REGISTER_OK")
sys.exit(0)
