#!/usr/bin/env bash
# Serve GLM-5.2-FP8 via repo serve.sh, tee-ing stdout to a serve.log under this
# ticket (§4 needs it).  bash serve_glm5.sh
set -euo pipefail

AUTO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/}"
PRESET="${PRESET:-${AUTO_ROOT}/presets/glm5.2.rebench/MTP5-bs64-dg.yaml}"
SERVE_LOG="${SERVE_LOG:-${TICKET_DIR}/logs/manual/serve_$(date +%Y%m%d_%H%M%S)/serve.log}"

mkdir -p "$(dirname "${SERVE_LOG}")"
echo "[serve] log -> ${SERVE_LOG}"
exec bash "${AUTO_ROOT}/serve.sh" "${MODEL}" "${PRESET}" > >(tee "${SERVE_LOG}") 2>&1
