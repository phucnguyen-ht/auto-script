#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash bench_serving_glm4p5_65k.sh
#   PROFILE=1 bash bench_serving_glm4p5_65k.sh
#   PROFILE=1 PROFILE_PREFILL_URLS="http://localhost:8100" PROFILE_DECODE_URLS="http://localhost:8200" bash bench_serving_glm4p5_65k.sh
#   RESET_PREFIX_CACHE=0 bash bench_serving_glm4p5_65k.sh
#   RESET_PREFIX_CACHE_URLS="http://localhost:8100 http://localhost:8200" bash bench_serving_glm4p5_65k.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model path is read from the shared env.yaml (single source of truth), same as
# auto_bench / auto_profile / auto_eval. Fallback only if env.yaml/yq are absent.
ENV_YAML="${ROOT_DIR}/../env.yaml"
if [ -f "${ENV_YAML}" ] && command -v yq >/dev/null 2>&1; then
    MODEL_PATH="${MODEL_PATH:-$(yq e '.model.path' "${ENV_YAML}")}"
else
    MODEL_PATH="${MODEL_PATH:-/share-mv/zai-org/GLM-5-FP8}"
fi
# DATASET="${DATASET:-$ROOT_DIR/benchmark_root/data/input/glm_4p5_testcase_65536.jsonl}"
DATASET="${DATASET:-$ROOT_DIR/part_shard0.jsonl}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
RESET_PREFIX_CACHE="${RESET_PREFIX_CACHE:-1}"
RESET_PREFIX_CACHE_URLS="${RESET_PREFIX_CACHE_URLS:-}"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
JSONL_START_INDEX="${JSONL_START_INDEX:-0}"
JSONL_OUTPUT_LEN="${JSONL_OUTPUT_LEN:-256}"
JSONL_CONTEXT_LEN="${JSONL_CONTEXT_LEN:-}"
JSONL_TEXT_FIELD="${JSONL_TEXT_FIELD:-text}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-${NUM_PROMPTS}}"
# WARMUP_REQUESTS="${WARMUP_REQUESTS:-27}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-${NUM_PROMPTS}}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/results/tp8_mtp2_${NUM_PROMPTS}.jsonl}"

PD_SEPARATED="${PD_SEPARATED:-1}"
PROFILE="${PROFILE:-0}"
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"
PROFILE_PREFILL_URLS="${PROFILE_PREFILL_URLS:-}"
PROFILE_DECODE_URLS="${PROFILE_DECODE_URLS:-}"
DRY_RUN="${DRY_RUN:-0}"

is_enabled() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

append_words_arg() {
  local flag="$1"
  local value="${2//,/ }"
  local words=()

  if [[ -n "$value" ]]; then
    read -r -a words <<< "$value"
    if ((${#words[@]})); then
      BENCH_ARGS+=("$flag" "${words[@]}")
    fi
  fi
}

RESET_PREFIX_CACHE_TARGETS=()

append_reset_prefix_cache_urls() {
  local value="${1//,/ }"
  local words=()
  local url

  if [[ -n "$value" ]]; then
    read -r -a words <<< "$value"
    for url in "${words[@]}"; do
      if [[ -n "$url" ]]; then
        RESET_PREFIX_CACHE_TARGETS+=("${url%/}")
      fi
    done
  fi
}

build_reset_prefix_cache_targets() {
  RESET_PREFIX_CACHE_TARGETS=()

  if [[ -n "$RESET_PREFIX_CACHE_URLS" ]]; then
    append_reset_prefix_cache_urls "$RESET_PREFIX_CACHE_URLS"
    return
  fi

  append_reset_prefix_cache_urls "$BASE_URL"

  if is_enabled "$PROFILE" && is_enabled "$PD_SEPARATED"; then
    append_reset_prefix_cache_urls "$PROFILE_PREFILL_URLS"
    append_reset_prefix_cache_urls "$PROFILE_DECODE_URLS"
  fi
}

reset_prefix_cache() {
  local -A seen=()
  local url
  local endpoint
  local -a curl_args=(-fsS -X POST -o /dev/null)

  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${OPENAI_API_KEY}")
  elif [[ -n "${API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: ${API_KEY}")
  fi

  build_reset_prefix_cache_targets

  for url in "${RESET_PREFIX_CACHE_TARGETS[@]}"; do
    if [[ -n "${seen[$url]:-}" ]]; then
      continue
    fi
    seen["$url"]=1

    endpoint="${url}/reset_prefix_cache"
    echo "Resetting prefix cache: $endpoint"
    curl "${curl_args[@]}" "$endpoint"
  done
}

BENCH_ARGS=(
  python3 "$ROOT_DIR/bench_serving_glm4p5_65k.py"
  --backend vllm 
  --model "$MODEL_PATH"
  --base-url "$BASE_URL"
  --dataset-name jsonl-text
  --dataset-path "$DATASET"
  --num-prompts "$NUM_PROMPTS"
  --jsonl-start-index "$JSONL_START_INDEX"
  --jsonl-output-len "$JSONL_OUTPUT_LEN"
  --jsonl-text-field "$JSONL_TEXT_FIELD"
  --warmup-requests "$WARMUP_REQUESTS"
)

if [[ -n "$JSONL_CONTEXT_LEN" ]]; then
  BENCH_ARGS+=(--jsonl-context-len "$JSONL_CONTEXT_LEN")
fi

if is_enabled "$PD_SEPARATED"; then
  BENCH_ARGS+=(--pd-separated)
fi

if [[ -n "$MAX_CONCURRENCY" ]]; then
  BENCH_ARGS+=(--max-concurrency "$MAX_CONCURRENCY")
fi

if [[ -n "$OUTPUT_FILE" ]]; then
  BENCH_ARGS+=(--output-file "$OUTPUT_FILE")
fi

if is_enabled "$PROFILE"; then
  BENCH_ARGS+=(--profile)
  # append_words_arg --profile-activities "$PROFILE_ACTIVITIES"

  if is_enabled "$PD_SEPARATED"; then
    append_words_arg --profile-prefill-url "$PROFILE_PREFILL_URLS"
    append_words_arg --profile-decode-url "$PROFILE_DECODE_URLS"
  fi
fi

echo "Model: $MODEL_PATH"
echo "Benchmark URL: $BASE_URL"
echo "Dataset: $DATASET"
echo "Output file: ${OUTPUT_FILE:-stdout}"
echo "Enabled profiling: $(is_enabled "$PROFILE" && echo "Yes" || echo "No")"
echo "NUM_PROMPTS=$NUM_PROMPTS JSONL_START_INDEX=$JSONL_START_INDEX JSONL_OUTPUT_LEN=$JSONL_OUTPUT_LEN JSONL_CONTEXT_LEN=$JSONL_CONTEXT_LEN"
echo "RESET_PREFIX_CACHE=$RESET_PREFIX_CACHE"

wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}"
    local elapsed=0
    echo "  [wait] Polling localhost:8000/health (timeout: ${max_wait}s)..."
    while ! curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "  [ERROR] Server did not respond within ${max_wait}s"
            return 1
        fi
        echo "  [wait] ${elapsed}s elapsed, still waiting..."
    done
    echo "  [wait] Server ready after ${elapsed}s"
}


if ! wait_for_server; then
  return 1
fi

if is_enabled "$DRY_RUN"; then
  if is_enabled "$RESET_PREFIX_CACHE"; then
    build_reset_prefix_cache_targets
    printf 'Reset prefix cache URLs:'
    printf ' %q' "${RESET_PREFIX_CACHE_TARGETS[@]}"
    printf '\n'
  fi
  printf 'Command:'
  printf ' %q' "${BENCH_ARGS[@]}" "$@"
  printf '\n'
  exit 0
fi

if is_enabled "$RESET_PREFIX_CACHE"; then
  reset_prefix_cache
fi

"${BENCH_ARGS[@]}" "$@"
