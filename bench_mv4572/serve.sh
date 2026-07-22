#!/usr/bin/env bash
# DEBUG: serve vLLM only (foreground) for MV-4572 (Kimi-K2.6-MXFP4 EPLB).
# Thin wrapper over ../common/serve.sh (options: PRESET=, STRIP_SCHEDULER_CLS=,
# BACKEND=). Logs land under logs/<preset>/serve/<ts>/{serve.log,preset.yaml}.
# Ctrl-C to stop. The GPU set / parallelism come from the preset itself
# (env_vars.HIP_VISIBLE_DEVICES + parallelism_args), which serve.sh exports
# verbatim before `vllm serve`.
#
#   bash serve.sh                                                                  # base.yaml (no EPLB)
#   PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml     bash serve.sh   # DP8 EPLB
#   PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0-exp.yaml bash serve.sh   # TP4 on GPU 4-7 (see that preset)
#   STRIP_SCHEDULER_CLS=0 bash serve.sh                                            # serve preset verbatim
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRESET="${PRESET:-kimi2.6.mxfp4/dp8ep8/base.yaml}"
source "${SCRIPT_DIR}/../common/serve.sh"
