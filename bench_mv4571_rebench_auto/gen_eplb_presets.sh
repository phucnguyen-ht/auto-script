#!/usr/bin/env bash
# Generate the VIABLE EPLB sweep presets (all verified on gpu-5 / image 260626 /
# 1P1D; see scripts/DEBUG_ASYNC_HANG.md):
#   nixl, torch_gloo   -> ASYNC  (transfer off the GPU NCCL group -> no deadlock)
#   torch_nccl, pynccl -> SYNC   (their ASYNC hangs "at collective communication
#                                 calls" per vLLM eplb_state.py:242 -- vLLM limit)
# grid: communicator x step{default,s250} x num_redundant{r0,r8,r16} = 24 presets.
# nixl also gets UCX env + kv_cache_memory_bytes (>=54.62 GiB for 1M ctx, else
# engine init fails). No max_model_len (keep 1M).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../presets/glm5.2.rebench" && pwd)"
BASE="${DIR}/MTP5-bs64-dg.yaml"
# KV cache for nixl (FIXED, needed to register UCX). MI300 (192 GiB/GPU): with EPLB
# ON, auto-KV is only ~49 GiB (r0) .. ~43 GiB (r16) -- rearrange buffers + redundant
# experts eat the budget. A FIXED KV must fit the HEAVIEST case (r16, ~43 GiB avail)
# -> 40 GiB (fits r0/r8/r16; >= 27.3 GiB needed for the 512K max_model_len). Measured:
# r16 model=114.17 GiB, avail KV=43.1 GiB. MI325 (256 GiB) can override higher via env.
NIXL_KV_CACHE_BYTES="${NIXL_KV_CACHE_BYTES:-42949672960}"   # 40 GiB (MI300-safe, fits r16)

# max_model_len for ALL EPLB presets. On MI300 (192 GiB/GPU) enabling EPLB costs
# ~10 GiB of the KV budget (rearrange buffers), dropping available KV below the
# 54.62 GiB needed for a 1M-token request -> engine init fails. So EPLB presets
# cap max_model_len (workload is 100K ISL, so this does NOT affect the benched
# requests). Value must fit the heaviest case (r16, least KV). Empty -> keep 1M.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"   # 512K (fits r0/r8/r16 on MI300; >=5x the 100K workload)

build() {  # <label> <communicator> <mode:async|sync> <step_sfx> <win_json> <red_sfx> <red_json>
    local out="${DIR}/MTP5-bs64-dg-eplb-$1-$3-$4-$6.yaml"
    local ua="false"; [ "$3" = async ] && ua="true"
    local cfg="{\"use_async\": ${ua}, \"communicator\": \"$2\"$5$7}"
    local ucx=""
    # tcp is REQUIRED on nodes without RDMA NICs (MI300 tw031: ibv_devices empty ->
    # no rc_x -> UCX has no active-message transport -> NIXL_ERR_BACKEND). tcp is a
    # harmless universal fallback on RDMA nodes too (UCX still prefers rc_x there).
    [ "$2" = nixl ] && ucx="  UCX_TLS: tcp,self,sm,rc_x,rocm_copy,rocm_ipc\n  UCX_MEMTYPE_CACHE: n\n"
    awk -v ins="${ucx}" '/^parallelism_args:/ && !d {printf "%s", ins; d=1} {print}' "${BASE}" > "${out}"
    printf '  enable_eplb: true\n  eplb_config: %s\n' "'${cfg}'" >> "${out}"
    [ -n "${MAX_MODEL_LEN}" ] && printf '  max_model_len: %s\n' "${MAX_MODEL_LEN}" >> "${out}"
    [ "$2" = nixl ] && printf '  kv_cache_memory_bytes: %s\n' "${NIXL_KV_CACHE_BYTES}" >> "${out}"
    echo "${out##*/}"
}

for spec in "nixl:nixl:async" "gloo:torch_gloo:async" "nccl:torch_nccl:sync" "pynccl:pynccl:sync"; do
    IFS=: read -r label comm mode <<< "${spec}"
    for step in "default:" 's250:, "window_size": 250, "step_interval": 250'; do
        for red in "r0:" 'r8:, "num_redundant_experts": 8' 'r16:, "num_redundant_experts": 16'; do
            build "${label}" "${comm}" "${mode}" "${step%%:*}" "${step#*:}" "${red%%:*}" "${red#*:}"
        done
    done
done