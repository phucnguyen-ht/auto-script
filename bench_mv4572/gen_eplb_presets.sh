#!/usr/bin/env bash
# Generate the EPLB sweep presets for Kimi-K2.6-MXFP4 (dp8-ep8).
#
# IMPORTANT (MXFP4): only the *nixl* communicator can transfer the expert weights.
# torch_nccl / pynccl / torch_gloo all raise
#   TypeError: Input tensor data type is not supported for NCCL process group: Float4_e2m1fn_x2
# because the packed 4-bit MXFP4 weights (torch dtype Float4_e2m1fn_x2) have no NCCL/Gloo
# wire-datatype mapping, whereas nixl moves raw BYTES (rocm_ipc/xGMI, RDMA read) and is
# dtype-agnostic. See eplb_src/mxfp4-communicator-support.md (task 1 / 1b).
#
# => The sweep is NIXL-ONLY. For torch_nccl / pynccl / torch_gloo we still emit ONE
# <comm>-default-r0 preset each, purely as a RECORD of the (failing) config -- NOT swept.
#
# Grid generated (17 files, next to base.yaml, as base-eplb-<comm>-<mode>-<step>-<red>.yaml):
#   records : gloo(async) / nccl(sync) / pynccl(sync)  x default x r0            (3)
#   nixl    : async | sync                             x default x r0            (2)
#   nixl    : async  x step{s100,s250,s500,s1000} x num_redundant{r0,r8,r16}     (12)
# No max_model_len is set: KV auto-sizes to the model default.
#
# NOTE: nixl needs only UCX_TLS (rocm_ipc covers intra-node EP over xGMI -- no NIC).
# We do NOT emit kv_cache_memory_bytes / UCX_MEMTYPE_CACHE: the working Kimi-MXFP4 nixl
# runs on this node don't use them (the fixed 40 GiB KV region was a GLM/MI300-FP8 relic
# and can mis-size Kimi's KV).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../presets/kimi2.6.mxfp4/dp8ep8" && pwd)"
BASE="${DIR}/base.yaml"

# Balancedness logging (rank 0 logs avg/max tokens + balancedness + the cross-rank
# dim=1 metrics every N steps). LOG_BALANCEDNESS=1 -> it is the ACTIVE eplb_config;
# =0 -> emitted as a commented alternative. LOG_BALANCEDNESS_INTERVAL = every-N-steps.
# Default OFF while benching: plain eplb_config is active, and the log_balancedness variant
# is emitted as a COMMENTED line (with interval 1) so you can flip it on later without regen.
# LOG_BALANCEDNESS=1 to make it the active config instead.
LOG_BALANCEDNESS="${LOG_BALANCEDNESS:-0}"
LOG_BALANCEDNESS_INTERVAL="${LOG_BALANCEDNESS_INTERVAL:-1}"

# Reset EPLB (counter/window/layout) to cold-start at each scenario's bench_start
# (needs the patched eplb_state.reset_state + reset_eplb worker). base.yaml ships
# 'false' everywhere by default (benching). RESET_EPLB=true to reset EPLB per scenario.
RESET_EPLB="${RESET_EPLB:-false}"

# UCX transports for nixl. rocm_ipc over xGMI covers intra-node EP (no RDMA NIC needed).
UCX_TLS_VAL="${UCX_TLS_VAL:-tcp,self,sm,rocm_copy,rocm_ipc}"

build() {  # <label> <communicator> <mode:async|sync> <step_sfx> <win_json> <red_sfx> <red_json>
    local out="${DIR}/base-eplb-$1-$3-$4-$6.yaml"
    local ua="false"; [ "$3" = async ] && ua="true"
    # base eplb_config fields (shared); the balancedness variant just appends 2 fields.
    local fields="\"use_async\": ${ua}, \"communicator\": \"$2\"$5$7"
    local lb_fields=", \"log_balancedness\": true, \"log_balancedness_interval\": ${LOG_BALANCEDNESS_INTERVAL}"
    local cfg_plain="{${fields}}"
    local cfg_lb="{${fields}${lb_fields}}"
    # LOG_BALANCEDNESS=1 -> balancedness active, plain commented; =0 -> the reverse.
    local active="${cfg_lb}" commented="${cfg_plain}"
    [ "${LOG_BALANCEDNESS}" = 1 ] || { active="${cfg_plain}"; commented="${cfg_lb}"; }
    # nixl-only: inject UCX_TLS as the last env var (before parallelism_args).
    local ucx=""
    [ "$2" = nixl ] && ucx="  UCX_TLS: ${UCX_TLS_VAL}\n"
    awk -v ins="${ucx}" '/^parallelism_args:/ && !d {printf "%s", ins; d=1} {print}' "${BASE}" > "${out}"
    # EPLB presets reset by default: flip the base 'false' to ${RESET_EPLB}.
    sed -i "s/VLLM_MOREH_RESET_EPLB: 'false'/VLLM_MOREH_RESET_EPLB: '${RESET_EPLB}'/" "${out}"
    printf '  enable_eplb: true\n  eplb_config: %s\n' "'${active}'" >> "${out}"
    printf '  # toggle balancedness logging: uncomment next line + comment eplb_config above\n' >> "${out}"
    printf '  # eplb_config: %s\n' "'${commented}'" >> "${out}"
    echo "${out##*/}"
}

# Clean slate: this generator is the SINGLE source of truth for the EPLB preset set.
# (base.yaml is hand-maintained and is NOT matched by base-eplb-*.)
rm -f "${DIR}"/base-eplb-*.yaml

# --- RECORDS of the failing communicators (kept for the record, NOT swept) ---
build "gloo"   "torch_gloo" async default "" r0 ""
build "nccl"   "torch_nccl" sync  default "" r0 ""
build "pynccl" "pynccl"     sync  default "" r0 ""

# --- NIXL default: async-r0 (already ran) + sync-r0 (task 3, run first) + async-r32 (kept per request) ---
build "nixl" "nixl" async default "" r0 ""
build "nixl" "nixl" sync  default "" r0 ""
build "nixl" "nixl" async default "" r32 ', "num_redundant_experts": 32'

# --- NIXL async sweep (task 4): step_interval x num_redundant. window_size is tied to
#     step_interval (matches the earlier s250/s500 convention). r32 dropped. ---
for step in 's100:, "window_size": 100, "step_interval": 100' \
            's250:, "window_size": 250, "step_interval": 250' \
            's500:, "window_size": 500, "step_interval": 500' \
            's1000:, "window_size": 1000, "step_interval": 1000'; do
    for red in "r0:" 'r8:, "num_redundant_experts": 8' 'r16:, "num_redundant_experts": 16'; do
        build "nixl" "nixl" async "${step%%:*}" "${step#*:}" "${red%%:*}" "${red#*:}"
    done
done
