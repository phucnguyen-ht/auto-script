#!/usr/bin/env bash
# Generate the EPLB sweep presets from the base (async only):
#   communicator {pynccl, torch_nccl, nixl} x step {default, s250} x num_redundant {r0, r8, r16}.
# nixl needs: UCX env (else "VRAM detected as host by UCX") + a fixed
# kv_cache_memory_bytes (else NIXL registerMem fails). 40 GiB is MI300-safe
# (model 107.6 + 40 + overhead < 192 GiB). No max_model_len (keep 1M).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../presets/glm5.2.rebench" && pwd)"
BASE="${DIR}/MTP5-bs64-dg.yaml"
NIXL_KV_CACHE_BYTES="${NIXL_KV_CACHE_BYTES:-42949672960}"   # 40 GiB (MI300)

build() {  # <label> <communicator> <step_suffix> <window_json> <red_suffix> <red_json>
    local out="${DIR}/MTP5-bs64-dg-eplb-$1-async-$3-$5.yaml"
    local cfg="{\"use_async\": true, \"communicator\": \"$2\"$4$6}"
    local ucx=""
    [ "$2" = nixl ] && ucx="  UCX_TLS: self,sm,rc_x,rocm_copy,rocm_ipc\n  UCX_MEMTYPE_CACHE: n\n"
    awk -v ins="${ucx}" '/^parallelism_args:/ && !d {printf "%s", ins; d=1} {print}' "${BASE}" > "${out}"
    printf '  enable_eplb: true\n  eplb_config: %s\n' "'${cfg}'" >> "${out}"
    [ "$2" = nixl ] && printf '  kv_cache_memory_bytes: %s\n' "${NIXL_KV_CACHE_BYTES}" >> "${out}"
    echo "${out##*/}"
}

for spec in "pynccl:pynccl" "nccl:torch_nccl" "nixl:nixl"; do
    label="${spec%%:*}"; comm="${spec#*:}"
    for step in "default:" 's250:, "window_size": 250, "step_interval": 250'; do
        for red in "r0:" 'r8:, "num_redundant_experts": 8' 'r16:, "num_redundant_experts": 16'; do
            build "${label}" "${comm}" "${step%%:*}" "${step#*:}" "${red%%:*}" "${red#*:}"
        done
    done
done