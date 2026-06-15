#!/usr/bin/env bash
set -euo pipefail

# Run eval ONLY — RepoBench (generate predictions + metrics) and lm_eval —
# against a server you started yourself. It does NOT start or kill any server,
# so it never touches the SGLang process you launched with:
#
#     bash serve_sglang_ds3.2.sh
#
# Which datasets run is still driven by env.yaml -> eval.datasets
# (repobench / mmlu / gsm8k / longbench / longbench2), exactly like auto_eval.sh.
#
# Under the hood this is just auto_eval.sh with AUTO_SERVE=0 and SGLang defaults
# (endpoint localhost:30000, model DeepSeek-V3.2, logs under logs_sglang/).
# Everything auto_eval.sh accepts still works — override as needed:
#
#   BASE_URL=http://localhost:30000   # server endpoint (default per BACKEND)
#   MODEL_PATH=/remote/.../DeepSeek-V3.2   # MUST match the served model id
#   BACKEND=sglang|vllm               # picks port/model defaults + log dir
#   LEVELS=2k,8k  NUM_PROMPTS=50      # RepoBench subsetting
#   LMEVAL_TASKS="mmlu gsm8k"          # override env.yaml task selection
#
# Examples:
#   bash eval_only_sglang.sh                          # repobench+gsm8k (env.yaml)
#   GENERATE=0 RUN_EVAL=1 bash eval_only_sglang.sh    # only recompute metrics
#   BACKEND=vllm BASE_URL=http://localhost:8000 bash eval_only_sglang.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Connect to an already-running server: never serve, never kill.
export AUTO_SERVE=0
# Default target is the SGLang server; auto_eval derives port 30000, the
# DeepSeek-V3.2 model id, and the logs_sglang/ log dir from this.
export BACKEND="${BACKEND:-sglang}"

# Resolve the endpoint the same way auto_eval would, so we can pre-flight it.
if [ "${BACKEND,,}" = "sglang" ]; then
    _default_port="${SGLANG_PORT:-30000}"
else
    _default_port=8000
fi
export BASE_URL="${BASE_URL:-http://localhost:${_default_port}}"

# Pre-flight: we don't start a server, so warn loudly if none is up yet.
if ! curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1; then
    echo "[eval-only] WARNING: no server responding at ${BASE_URL}/health" >&2
    echo "[eval-only] Start one first, e.g.:  bash ${SCRIPT_DIR}/serve_sglang_ds3.2.sh" >&2
    echo "[eval-only] Continuing anyway (requests will retry, then fail)..." >&2
fi

exec bash "${SCRIPT_DIR}/auto_eval.sh"
