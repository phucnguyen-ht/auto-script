#!/usr/bin/env bash
# Auto-benchmark one or more models across parallelism x dataset x request-rate
# x concurrency scenarios, driven by the built-in `vllm bench serve` client
# (instead of the in-tree bench_serving.py used by auto_benchmark.sh).
#
# Runs INSIDE the serving container. Iterates over MODELS (outermost). For each
# model the server is started ONCE per parallelism: launch `vllm-moreh serve`
# with the matching preset, wait for health, then run every
# dataset x request-rate x concurrency scenario against it, then kill the
# server (pkill -9 VLLM) before moving to the next parallelism / model.
#
# The new sweep axis vs auto_benchmark.sh: for every REQUEST_RATE we replay the
# whole CONCURRENCIES list. request_rate is the Poisson arrival rate (req/s);
# max_concurrency caps in-flight requests. request_rate=inf is the closed-loop
# case where concurrency alone limits load (the classic latency-vs-throughput
# curve). A finite rate below what a given concurrency could sustain makes the
# higher concurrency values in the list redundant for that rate -- that's
# expected; the cross product is intentional so you can see where each rate
# saturates.
#
# Request rates are sized to bracket each dataset's saturation knee (calibrated
# from 10k @ cc16/inf -> ~30s e2e -> ~0.5 req/s), so sweeping concurrency at a
# fixed rate actually shows a curve. Defaults (override with REQUEST_RATES):
#   8k   (OSL=1024): 0.5 1 2 4 8 inf
#   10k  (OSL=500):  1 2 4 8 16 inf
#   100k (OSL=500):  0.25 0.5 1 2 4 inf
#
# Examples:
#   bash auto_benchmark_serve.sh
#   MODELS="glm-5.1" bash auto_benchmark_serve.sh
#   MODELS="glm-5.1" PARALLELISMS="tp8-moe-tp8" DATASETS="8k" bash auto_benchmark_serve.sh
#   REQUEST_RATES="inf" CONCURRENCIES="1 8 16 32 64" bash auto_benchmark_serve.sh
#   DEVICE=mi308x bash auto_benchmark_serve.sh
#   DRY_RUN=1 bash auto_benchmark_serve.sh   # print scenarios, run nothing
#
# Completed scenarios (existing non-empty result file) are skipped, so the
# script is safe to re-run / resume.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-/share-mv/longbenchv2-custom}"
PRESET_DIR="${PRESET_DIR:-$REPO_DIR/presets/full}"
DEVICE="${DEVICE:-mi300x}"

# Models to benchmark (outermost loop). Each key must resolve in the registry
# functions below.
MODELS="${MODELS:-kimi-k2.6 glm-5.1}"

# --- Model registry -------------------------------------------------------
# Maps a model key to its checkpoint path and preset filename prefix. A model
# whose preset is missing for a given parallelism/device is skipped (recorded
# as "no_preset"), so listing a not-yet-supported model here is harmless.
model_path_for() {
  case "$1" in
    kimi-k2.6) echo "/share-mv/moonshotai/Kimi-K2.6/" ;;
    glm-5.1)   echo "/share-mv/zai-org/GLM-5.1-FP8/" ;;
    *)         echo "" ;;
  esac
}

model_prefix_for() {
  case "$1" in
    kimi-k2.6) echo "moonshotai-kimi-k2.6" ;;
    glm-5.1)   echo "zai-org-glm-5.1-fp8-mtp" ;;
    *)         echo "" ;;
  esac
}
# --------------------------------------------------------------------------

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-3600}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

PARALLELISMS="${PARALLELISMS:-dp8-moe-ep8}"
DATASETS="${DATASETS:-8k 10k 100k}"

# Concurrency sweep (replayed for every request rate). Per-dataset like the
# request rates: long ISL (100k) sweeps lower than short ISL (8k, 10k). An
# explicit CONCURRENCIES override wins for every dataset.
CONCURRENCIES="${CONCURRENCIES:-}"
SHORT_CONCURRENCIES="${SHORT_CONCURRENCIES:-8 16 32 48 64}"
LONG_CONCURRENCIES="${LONG_CONCURRENCIES:-4 8 12 16}"

# Request-rate sweep (outer of the two new axes), per dataset. The point is to
# bracket the saturation knee so that sweeping concurrency AT a fixed rate shows
# a real curve: below saturation the concurrency cap never binds (flat), near
# saturation low concurrency queues up while high concurrency keeps up (the
# divergence we want), and inf is the closed-loop case.
#
# Rates are calibrated from two measured 10k points (OSL=500):
#   cc16/inf -> e2e ~30s  -> ~0.53 req/s
#   cc64/inf -> e2e ~110s -> ~0.58 req/s
# Throughput barely rises (16->64 conc, +9%) while latency ~4x: 10k is already
# saturated by cc16, peak ~0.58 req/s. These rate lists deliberately sit at/above
# that knee: most rates saturate and behave like inf (each cell finishes in ~the
# inf time, not 30+ min), trading the below-knee detail for far shorter runs.
# Drop the low end further (e.g. RATES_10K="0.5 ...") if you want sub-knee points.
# 8k (OSL=1024) is ~2x slower in decode -> ~half the rates; 100k is prefill-
# bound -> lower still (estimate, no measured point). 8k/100k are extrapolations.
# An explicit REQUEST_RATES override wins for every dataset.
REQUEST_RATES="${REQUEST_RATES:-}"
RATES_8K="${RATES_8K:-0.1 0.25 0.5 1 2 inf}"
RATES_10K="${RATES_10K:-0.25 0.5 1 2 4 inf}"
RATES_100K="${RATES_100K:-0.05 0.1 0.25 0.5 1 inf}"

# num-prompts per scenario: a simple multiple of concurrency for every rate
# (~PROMPTS_PER_CONCURRENCY waves at inf; enough samples at finite rates),
# clamped to [NUM_PROMPTS_FLOOR, NUM_PROMPTS_CAP]. NUM_PROMPTS, if set,
# hard-overrides for every run.
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-3}"
NUM_PROMPTS_FLOOR="${NUM_PROMPTS_FLOOR:-32}"
NUM_PROMPTS_CAP="${NUM_PROMPTS_CAP:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-}"

# Dataset prompts in datasets/*.jsonl carry a pre-templated `text` field; vllm
# bench serve's `custom` dataset wants a `prompt` field, so we rename on a
# cached copy and skip the chat template (prompts are already formatted).
JSONL_TEXT_FIELD="${JSONL_TEXT_FIELD:-text}"
SKIP_CHAT_TEMPLATE="${SKIP_CHAT_TEMPLATE:-1}"
# Force the full OSL (don't stop early on EOS) so output lengths are comparable.
IGNORE_EOS="${IGNORE_EOS:-1}"
# Percentiles to record (p90 + p99 alongside mean/median).
METRIC_PERCENTILES="${METRIC_PERCENTILES:-90}"

# Reset the server prefix cache before each scenario for clean, comparable runs.
RESET_PREFIX_CACHE="${RESET_PREFIX_CACHE:-1}"

RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/serve_results}"
DRY_RUN="${DRY_RUN:-0}"

is_enabled() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# OSL per dataset (8k -> 1024, 10k/100k -> 500)
output_len_for() {
  case "$1" in
    8k) echo 1024 ;;
    *)  echo 500 ;;
  esac
}

# Request-rate sweep per dataset (see knob block above for calibration).
# An explicit REQUEST_RATES override wins for every dataset.
request_rates_for() {
  if [[ -n "$REQUEST_RATES" ]]; then
    echo "$REQUEST_RATES"
    return
  fi
  case "$1" in
    8k)   echo "$RATES_8K" ;;
    10k)  echo "$RATES_10K" ;;
    *)    echo "$RATES_100K" ;;
  esac
}

# Concurrency sweep per dataset: short ISL (8k, 10k) vs long ISL (100k, ...).
# An explicit CONCURRENCIES override wins for every dataset.
concurrencies_for() {
  if [[ -n "$CONCURRENCIES" ]]; then
    echo "$CONCURRENCIES"
    return
  fi
  case "$1" in
    8k|10k) echo "$SHORT_CONCURRENCIES" ;;
    *)      echo "$LONG_CONCURRENCIES" ;;
  esac
}

# num-prompts for a given concurrency (see knob block above).
num_prompts_for() {
  local cc="$1"
  if [[ -n "$NUM_PROMPTS" ]]; then
    echo "$NUM_PROMPTS"
    return
  fi
  local n=$(( cc * PROMPTS_PER_CONCURRENCY ))
  (( n < NUM_PROMPTS_FLOOR )) && n="$NUM_PROMPTS_FLOOR"
  (( n > NUM_PROMPTS_CAP )) && n="$NUM_PROMPTS_CAP"
  echo "$n"
}

# Filesystem-safe token for a request rate (inf -> rinf, 0.5 -> r0p5, 8 -> r8).
rate_tag() {
  local r="${1//./p}"
  echo "r${r}"
}

preset_for() {
  # preset_for <preset_prefix> <parallelism>
  echo "$PRESET_DIR/$1-amd-${DEVICE}-$2.yaml"
}

# Ensure a `prompt`-keyed JSONL exists for vllm bench serve's custom dataset.
# If the dataset already has `prompt`, use it as-is; otherwise rename the
# configured text field into a cached sibling under .vllm_prompt/.
prompt_dataset_for() {
  local src="$1"
  if [[ ! -f "$src" ]]; then
    echo ""
    return
  fi
  local first
  first="$(head -n1 "$src")"
  if echo "$first" | python3 -c "import json,sys; sys.exit(0 if 'prompt' in json.loads(sys.stdin.readline()) else 1)" 2>/dev/null; then
    echo "$src"
    return
  fi
  local cache_dir="$DATASET_DIR/.vllm_prompt"
  local dst="$cache_dir/$(basename "$src")"
  mkdir -p "$cache_dir"
  if [[ -s "$dst" && "$dst" -nt "$src" ]]; then
    echo "$dst"
    return
  fi
  if ! TEXT_FIELD="$JSONL_TEXT_FIELD" python3 - "$src" "$dst" <<'EOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
field = os.environ.get("TEXT_FIELD", "text")
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "prompt" not in d:
            if field not in d:
                raise SystemExit(f"line missing both 'prompt' and '{field}': keys={list(d)}")
            d["prompt"] = d.pop(field)
        fout.write(json.dumps({"prompt": d["prompt"]}) + "\n")
EOF
  then
    rm -f "$dst"
    echo ""
    return
  fi
  echo "$dst"
}

reset_prefix_cache() {
  is_enabled "$RESET_PREFIX_CACHE" || return 0
  local -a curl_args=(-fsS -X POST -o /dev/null)
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${OPENAI_API_KEY}")
  elif [[ -n "${API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: ${API_KEY}")
  fi
  echo "[bench] Resetting prefix cache: ${BASE_URL}/reset_prefix_cache"
  curl "${curl_args[@]}" "${BASE_URL}/reset_prefix_cache" || \
    echo "[bench] WARN: prefix cache reset failed (continuing)"
}

SERVER_PID=""

start_server() {
  local preset="$1" log_file="$2"
  echo "[server] Launching: vllm-moreh serve $MODEL_PATH $preset --port $PORT"
  echo "[server] Log: $log_file"
  # shellcheck disable=SC2086
  setsid vllm-moreh serve "$MODEL_PATH" "$preset" --port "$PORT" $SERVER_EXTRA_ARGS \
    &> "$log_file" &
  SERVER_PID=$!
}

wait_for_server() {
  local elapsed=0
  echo "[server] Waiting for ${BASE_URL}/health (timeout: ${SERVER_WAIT_TIMEOUT}s)..."
  while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[server] ERROR: server process died during startup (see log)"
      return 1
    fi
    if (( elapsed >= SERVER_WAIT_TIMEOUT )); then
      echo "[server] ERROR: not healthy within ${SERVER_WAIT_TIMEOUT}s"
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    (( elapsed % 60 == 0 )) && echo "[server] ${elapsed}s elapsed, still waiting..."
  done
  echo "[server] Ready after ${elapsed}s"
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  echo "[server] Killing vLLM processes (pkill -9 VLLM)..."
  pkill -9 VLLM 2>/dev/null
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep 10   # let GPU memory release before next launch
}

trap stop_server EXIT INT TERM

record() {
  # record <parallelism> <dataset> <request_rate> <concurrency> <num_prompts> <osl> <status> [metrics]
  echo "$MODEL,$1,$2,$3,$4,$5,$6,$7,${8:--,-,-,-,-,-,-}" >> "$SUMMARY_FILE"
}

# Print "mean_ttft_ms,p90_ttft_ms,mean_tpot_ms,p90_tpot_ms,request_throughput,output_throughput,completed"
# from a vllm bench serve result JSON.
extract_metrics() {
  python3 - "$1" <<'EOF'
import json, sys
KEYS = ["mean_ttft_ms", "p90_ttft_ms", "mean_tpot_ms", "p90_tpot_ms",
        "request_throughput", "output_throughput", "completed"]
try:
    d = json.load(open(sys.argv[1]))
    vals = [d.get(k, "-") for k in KEYS]
    print(",".join(str(round(v, 2)) if isinstance(v, float) else str(v) for v in vals))
except Exception:
    print(",".join("-" for _ in KEYS))
EOF
}

run_scenario() {
  # run_scenario <dataset_file> <num_prompts> <max_concurrency> <request_rate> <osl> <result_dir> <result_filename>
  local dataset_file="$1" num_prompts="$2" cc="$3" rate="$4" osl="$5" result_dir="$6" result_filename="$7"

  local -a args=(
    vllm bench serve
    --model "$MODEL_PATH"
    --base-url "$BASE_URL"
    --dataset-name custom
    --dataset-path "$dataset_file"
    --custom-output-len "$osl"
    --num-prompts "$num_prompts"
    --max-concurrency "$cc"
    --request-rate "$rate"
    --metric-percentiles "$METRIC_PERCENTILES"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --save-result
    --result-dir "$result_dir"
    --result-filename "$result_filename"
    --seed 0
  )
  is_enabled "$SKIP_CHAT_TEMPLATE" && args+=(--skip-chat-template)
  is_enabled "$IGNORE_EOS" && args+=(--ignore-eos)

  reset_prefix_cache
  "${args[@]}"
}

SUMMARY_FILES=()

for MODEL in $MODELS; do
  MODEL_PATH="$(model_path_for "$MODEL")"
  MODEL_PRESET_PREFIX="$(model_prefix_for "$MODEL")"

  RESULTS_DIR="$RESULTS_BASE/$MODEL"
  LOG_DIR="$RESULTS_DIR/logs"
  mkdir -p "$RESULTS_DIR" "$LOG_DIR"
  SUMMARY_FILE="$RESULTS_DIR/summary.csv"
  SUMMARY_FILES+=("$SUMMARY_FILE")
  [[ -f "$SUMMARY_FILE" ]] || echo "model,parallelism,dataset,request_rate,concurrency,num_prompts,output_len,status,mean_ttft_ms,p90_ttft_ms,mean_tpot_ms,p90_tpot_ms,request_throughput,output_throughput,completed" > "$SUMMARY_FILE"

  echo ""
  echo "########## MODEL: $MODEL ($MODEL_PATH) ##########"

  if [[ -z "$MODEL_PATH" || -z "$MODEL_PRESET_PREFIX" ]]; then
    echo "[skip] Unknown model '$MODEL' (not in registry)"
    for parallelism in $PARALLELISMS; do for ds in $DATASETS; do for rate in $(request_rates_for "$ds"); do for cc in $(concurrencies_for "$ds"); do
      record "$parallelism" "$ds" "$rate" "$cc" - - "unknown_model"
    done; done; done; done
    continue
  fi

  for parallelism in $PARALLELISMS; do
    preset="$(preset_for "$MODEL_PRESET_PREFIX" "$parallelism")"
    if [[ ! -f "$preset" ]]; then
      echo "[skip] No preset for '$MODEL' parallelism '$parallelism': $preset"
      for ds in $DATASETS; do for rate in $(request_rates_for "$ds"); do for cc in $(concurrencies_for "$ds"); do
        record "$parallelism" "$ds" "$rate" "$cc" - - "no_preset"
      done; done; done
      continue
    fi

    # Collect scenarios that still need running, so we don't launch the server
    # when everything is already done. Scenario token: "ds rate cc".
    pending=()
    for ds in $DATASETS; do
      for rate in $(request_rates_for "$ds"); do
        for cc in $(concurrencies_for "$ds"); do
          out="$RESULTS_DIR/${parallelism}_${ds}_$(rate_tag "$rate")_c${cc}.json"
          if [[ -s "$out" ]]; then
            echo "[skip] Done already: $out"
            continue
          fi
          pending+=("$ds $rate $cc")
        done
      done
    done

    if (( ${#pending[@]} == 0 )); then
      echo "[skip] All scenarios already done for '$MODEL' '$parallelism'"
      continue
    fi

    if is_enabled "$DRY_RUN"; then
      for scenario in "${pending[@]}"; do
        read -r ds rate cc <<< "$scenario"
        osl="$(output_len_for "$ds")"
        num_prompts="$(num_prompts_for "$cc")"
        echo "=== $MODEL | $parallelism | $ds (OSL=$osl) | rate=$rate | concurrency=$cc | prompts=$num_prompts ==="
        record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "dry_run"
      done
      continue
    fi

    # One server per parallelism, shared across all scenarios.
    start_server "$preset" "$LOG_DIR/server_${parallelism}.log"
    if ! wait_for_server; then
      stop_server
      for scenario in "${pending[@]}"; do
        read -r ds rate cc <<< "$scenario"
        osl="$(output_len_for "$ds")"
        num_prompts="$(num_prompts_for "$cc")"
        record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "server_failed"
      done
      continue
    fi

    for scenario in "${pending[@]}"; do
      read -r ds rate cc <<< "$scenario"
      osl="$(output_len_for "$ds")"
      num_prompts="$(num_prompts_for "$cc")"
      raw_dataset="$DATASET_DIR/longbenchv2-${ds}.jsonl"
      dataset_file="$(prompt_dataset_for "$raw_dataset")"

      if [[ -z "$dataset_file" ]]; then
        echo "[skip] Dataset not found / unusable: $raw_dataset"
        record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "no_dataset"
        continue
      fi

      result_filename="${parallelism}_${ds}_$(rate_tag "$rate")_c${cc}.json"
      out="$RESULTS_DIR/$result_filename"
      log="$LOG_DIR/bench_${parallelism}_${ds}_$(rate_tag "$rate")_c${cc}.log"
      echo ""
      echo "=== $MODEL | $parallelism | $ds (OSL=$osl) | rate=$rate | concurrency=$cc | prompts=$num_prompts ==="

      if run_scenario "$dataset_file" "$num_prompts" "$cc" "$rate" "$osl" \
           "$RESULTS_DIR" "$result_filename" 2>&1 | tee "$log"; then
        if [[ -s "$out" ]]; then
          record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "ok" "$(extract_metrics "$out")"
        else
          record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "no_result"
        fi
      else
        record "$parallelism" "$ds" "$rate" "$cc" "$num_prompts" "$osl" "bench_failed"
      fi
    done

    stop_server
  done
done

echo ""
echo "All done. Summaries:"
for f in "${SUMMARY_FILES[@]}"; do
  echo ""
  echo "### $f"
  column -s, -t "$f" 2>/dev/null || cat "$f"
done
