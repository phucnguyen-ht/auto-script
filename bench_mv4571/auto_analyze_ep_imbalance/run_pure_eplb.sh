#!/usr/bin/env bash
# Pure EPLB (chỉ enable_eplb: true) qua flow time-imbalance. Kỳ vọng crash NIXL init.
# Kết quả: logs/run_<ts>/glm5.2/noMTP-bs64-dg-eplb-pure/10k_rinf_c36/time/serve.log
set -uo pipefail
cd "$(dirname "$0")"
SCENARIO_YAML=scenario_eplb_pure.yaml PHASES=time bash auto_analyze_ep_imbalance.sh
