#!/usr/bin/env bash
# bench_kimi.sh — vllm-moreh serving benchmark for Kimi-K2.5 (custom jsonl, replay-as-is).
#
# Usage:
#   bash bench_kimi.sh [label]
#
# Configure via env (defaults shown):
#   MODEL=/share-mv/moonshotai/Kimi-K2.5      # must EXACTLY match the name the server was served with
#   PORT=8000
#   DATASET_PATH=/workspace/small_dataset_shard0.jsonl
#   OSL=256        # output len cap, passed as --custom-output-len
#   CONC=16        # --max-concurrency
#   NUM_PROMPTS=16 # --num-prompts (dataset has 25 rows; 16 are sampled)
#   RUNS=3
#   SEED=0         # custom dataset shuffles; fix the seed for reproducible sampling
#   PROFILE=0      # 1 -> add --profile (server must be launched with VLLM_TORCH_PROFILER_DIR set)
#   OUTDIR=logs/bench_<label>_<ts>
#
# DATASET NOTES:
#   - The custom dataset loader REQUIRES a "prompt" column. This shard uses "text",
#     so the script normalizes text->prompt into $OUTDIR/dataset.prompt.jsonl first.
#   - Prompts are already chat-templated ([gMASK]<sop>...), so --skip-chat-template is
#     used and they are sent verbatim to /v1/completions (no double-templating).
#   - "Replay as-is": ISL is the natural prompt length (NOT 256, and not controllable
#     for the custom dataset). Only OSL is capped, via --custom-output-len.
#
# AGENT KERNEL ON/OFF is set on the SERVER (not here):
#   ON : touch /tmp/agent_wna16_moe.on   (or serve with AGENT_WNA16_MOE=1) then start server
#   OFF: rm -f /tmp/agent_wna16_moe.on   then start server
# Confirm it engaged:  grep -E '\[agent_wna16\] ENGAGED' <serve_log>
set -euo pipefail

MODEL="${MODEL:-/remote/vast0/share-mv/zai-org/GLM-5.2-FP8}"
PORT="${PORT:-8000}"
# DATASET_PATH="${DATASET_PATH:-/remote/vast0/share-mv/longbenchv2-custom/longbenchv2-8k.jsonl}"
DATASET_PATH="${DATASET_PATH:-/remote/vast0/share-mv/longbenchv2-custom/longbenchv2-8k.jsonl}"
OSL="${OSL:-1024}"
# OSL="${OSL:-500}"
CONC="${CONC:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-36}"
RUNS="${RUNS:-1}"
SEED="${SEED:-0}"
LABEL="${1:-bench}"
OUTDIR="${OUTDIR:-logs/bench_${LABEL}_$(date +%Y%m%d-%H%M%S)}"

PROFILE_ARG=()
[ "${PROFILE:-0}" = "1" ] && PROFILE_ARG=(--profile)

[ -f "$DATASET_PATH" ] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "model=$MODEL port=$PORT dataset=$DATASET_PATH osl=$OSL conc=$CONC prompts=$NUM_PROMPTS runs=$RUNS seed=$SEED profile=${PROFILE:-0} -> $OUTDIR"

# server must be up
curl -sf "http://0.0.0.0:${PORT}/health" >/dev/null \
  || { echo "ERROR: server not healthy on :$PORT (start it first)"; exit 1; }

# # normalize dataset -> a jsonl with a "prompt" column (custom dataset requirement)
# NORM_DATASET="$OUTDIR/dataset.prompt.jsonl"
# python3 - "$DATASET_PATH" "$NORM_DATASET" <<'PY'
# import json, sys
# src, dst = sys.argv[1], sys.argv[2]
# n = 0
# with open(src) as f, open(dst, "w") as g:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         row = json.loads(line)
#         if "prompt" in row:
#             out = row
#         elif "text" in row:
#             out = dict(row)
#             out["prompt"] = out.pop("text")
#         else:
#             sys.exit(f"ERROR: row {n} has neither 'prompt' nor 'text': keys={list(row.keys())}")
#         g.write(json.dumps(out, ensure_ascii=False) + "\n")
#         n += 1
# print(f"normalized {n} rows -> {dst}")
# PY

for i in $(seq 1 "$RUNS"); do
  log_file_name="$OUTDIR/run${i}.log"
  echo "=== run $i/$RUNS  $(date +%T)  -> $log_file_name ==="
  # clean slate each run
  curl -X POST -s "http://0.0.0.0:${PORT}/reset_prefix_cache" -H "Content-Type: application/json" >/dev/null || true
  vllm bench serve \
      --port "$PORT" \
      --backend vllm \
      --model "$MODEL" \
      --trust-remote-code \
      --dataset-name custom \
      --dataset-path "$DATASET_PATH" \
      --custom-output-len "$OSL" \
      --skip-chat-template \
      --seed "$SEED" \
      # --temperature 0 \
      --max-concurrency "$CONC" \
      --num-prompts "$NUM_PROMPTS" \
      --percentile-metrics ttft,tpot,e2el,itl \
      --metric-percentiles 0,75,90,99,100 \
      --goodput ttft:1000 tpot:100 \
      --save-result --result-filename "$OUTDIR/run${i}.json" \
      "${PROFILE_ARG[@]}" \
      > "$log_file_name" 2>&1
  grep -iE "Successful requests|input tokens|generated tokens|throughput|Mean TTFT|Mean TPOT|Mean E2EL|Goodput" "$log_file_name" || tail -5 "$log_file_name"
done
echo "DONE -> $OUTDIR  (raw run JSONs + logs + normalized dataset)"