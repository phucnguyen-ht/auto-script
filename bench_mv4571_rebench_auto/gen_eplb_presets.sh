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
NIXL_KV_CACHE_BYTES="${NIXL_KV_CACHE_BYTES:-64424509440}"   # 60 GiB

build() {  # <label> <communicator> <mode:async|sync> <step_sfx> <win_json> <red_sfx> <red_json>
    local out="${DIR}/MTP5-bs64-dg-eplb-$1-$3-$4-$6.yaml"
    local ua="false"; [ "$3" = async ] && ua="true"
    local cfg="{\"use_async\": ${ua}, \"communicator\": \"$2\"$5$7}"
    local ucx=""
    [ "$2" = nixl ] && ucx="  UCX_TLS: self,sm,rc_x,rocm_copy,rocm_ipc\n  UCX_MEMTYPE_CACHE: n\n"
    awk -v ins="${ucx}" '/^parallelism_args:/ && !d {printf "%s", ins; d=1} {print}' "${BASE}" > "${out}"
    printf '  enable_eplb: true\n  eplb_config: %s\n' "'${cfg}'" >> "${out}"
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