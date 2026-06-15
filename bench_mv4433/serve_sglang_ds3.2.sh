#!/usr/bin/env bash
set -euo pipefail

# Serve DeepSeek-V3.2 with SGLang.
#
# Optional overrides (bare invocation behaves exactly as before):
#   $1           model path     (default: /remote/vast0/share-mv/deepseek-ai/DeepSeek-V3.2)
#   SGLANG_PORT  server port     (default: 30000 — SGLang's own default)
#   extra args after $1 are forwarded verbatim to `sglang serve`.
#
# NOTE: the model id SGLang registers is whatever --model-path is, so any
# client (auto_eval / auto_readable) must send the SAME value in its
# "model" field. The auto_* scripts pass MODEL_PATH here and reuse it.

MODEL_PATH="${1:-/remote/vast0/share-mv/deepseek-ai/DeepSeek-V3.2}"
[ "$#" -gt 0 ] && shift

exec sglang serve \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --nsa-prefill-backend tilelang \
  --nsa-decode-backend tilelang \
  --cuda-graph-max-bs 64 \
  --tp 8 \
  --tool-call-parser deepseekv32 \
  --reasoning-parser deepseek-v3 \
  --port "${SGLANG_PORT:-30000}" \
  "$@"
