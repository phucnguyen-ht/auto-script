#!/usr/bin/env bash
# mv-4572 eval/readable driver for Kimi-K2.6-MXFP4: runs the readable smoke tests
# + gsm8k accuracy eval (custom script) via common/run_all.sh. Phases + datasets
# come from ./env.yaml. Defaults to the no-EPLB base preset (clean accuracy
# baseline); pass PRESET=... to eval a specific EPLB config (e.g. nixl).
#
#   bash run_all.sh
#   PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml bash run_all.sh
#   RUN_READABLE=0 bash run_all.sh    # eval (gsm8k) only
#   RUN_EVAL=0     bash run_all.sh    # readable only
#   SHARED_SERVE=off bash run_all.sh  # per-phase serve/restart (default here is on)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Preset (resolved under presets/ by helper.resolve_preset). base = no EPLB.
export PRESET="${PRESET:-kimi2.6.mxfp4/dp8ep8/base.yaml}"
# Default this ticket to ONE shared server for all readable+eval phases (no
# kill/restart between methods). Override with SHARED_SERVE=off for the old behavior.
export SHARED_SERVE="${SHARED_SERVE:-on}"
source "$(cd "${SCRIPT_DIR}/../common" && pwd)/run_all.sh"
