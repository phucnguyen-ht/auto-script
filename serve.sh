#!/usr/bin/env bash

# Usage: serve.sh /path/to/local/model/ckpt /path/to/model/preset [--arg val]

# TODO(loctran): replace this script with proper vllm-moreh cli entrypoint

set -euo pipefail

MODEL_PATH="$1"
PRESET="$2"
shift 2

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

if [ ! -f "$PRESET" ]; then
  echo "Preset not found: $PRESET" >&2
  exit 1
fi

install_yq() {
  YQ_VERSION="v4.50.1"
  YQ_SHA256="c7a1278e6bbc4924f41b56db838086c39d13ee25dcb22089e7fbf16ac901f0d4"
  YQ_BIN="/usr/local/bin/yq"

  echo "Installing yq ${YQ_VERSION}..."

  curl -fsSL -o "${YQ_BIN}" \
    "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64"

  echo "${YQ_SHA256}  ${YQ_BIN}" | sha256sum -c -

  chmod 0755 "${YQ_BIN}"

  echo "yq installed successfully"
}

if ! command -v yq >/dev/null 2>&1; then
  install_yq
fi


TMP_CONFIG="$(mktemp /tmp/engine-config.XXXXXX.yaml)"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

# 0. Extract env vars to be set
ENV_KEYS="$(
  yq -r '
    .env_vars // {} |
    keys |
    .[]
  ' "$PRESET"
)"

# 1. Export env vars (faithful, no interpretation)
# Produces lines like: export KEY="VALUE"
eval "$(
  yq -r '
    .env_vars // {} |
    to_entries |
    .[] |
    "export \(.key)=\(.value|@sh)"
  ' "$PRESET"
)"

# 2. Write args section as-is to temp config
yq '(.parallelism_args // {}) + (.engine_args // {})' "$PRESET" > "$TMP_CONFIG"

# 3. Being verbose
for k in $ENV_KEYS; do
  printf '%s=%q\n' "$k" "${!k}"
done
cat $TMP_CONFIG
echo $@

# 4. Exec engine
exec vllm serve "$MODEL_PATH" --config "$TMP_CONFIG" $@
