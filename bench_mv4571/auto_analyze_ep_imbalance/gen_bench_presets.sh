#!/usr/bin/env bash
# Generate matched baseline vs async-NIXL EPLB presets for the compute-bound-candidate bench.
# Per (family, ISL-group): baseline and async-nixl share the SAME max_model_len + kv_cache_memory_bytes
# (only enable_eplb/UCX differ) -> fair apples-to-apples. kv_cache_memory_bytes pins KV + skips the
# memory-profiling snapshot (needed because NIXL init trips it; applied to baseline too for parity).
set -euo pipefail
PDIR=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/presets/glm5.2/dp8ep8
cd "$PDIR"
ASYNC_EC='{"use_async": true, "communicator": "nixl", "window_size": 100, "step_interval": 100}'
gen() {  # <family> <grp> <mml> <kv_bytes>
  local fam="$1" grp="$2" mml="$3" kv="$4" base="$1.yaml"
  # baseline (matched memory config, no EPLB)
  yq ".engine_args.max_model_len = $mml | .engine_args.kv_cache_memory_bytes = $kv" \
    "$base" > "${fam}-base-${grp}.yaml"
  # async-NIXL EPLB (matched memory + UCX rocm fix + async nixl)
  UCX="self,sm,rc_x,rocm_copy,rocm_ipc" EC="$ASYNC_EC" \
  yq ".engine_args.max_model_len = $mml
    | .engine_args.kv_cache_memory_bytes = $kv
    | .env_vars.UCX_TLS = strenv(UCX)
    | .env_vars.UCX_MEMTYPE_CACHE = \"n\"
    | .engine_args.enable_eplb = true
    | .engine_args.eplb_config = strenv(EC)" \
    "$base" > "${fam}-async-${grp}.yaml"
  echo "  ${fam}-{base,async}-${grp}.yaml (mml=$mml kv=$kv)"
}
# 8k group: max_model_len 16384, kv 40 GiB
gen noMTP-bs64-dg 8k  16384 42949672960
gen MTP5-bs64-dg  8k  16384 42949672960
# 100k group: max_model_len 131072, kv 80 GiB
gen noMTP-bs64-dg 100k 131072 85899345920
echo "done gen presets"
