#!/usr/bin/env bash
# PREFILL RUN_AND_BENCH end-to-end for ONE preset: serve (prefix-cache OFF) -> bench
# -> extract -> stop. Prefill counterpart of scripts/run_and_bench.sh (decode).
# Differences vs decode (see scripts_prefill/RUN_AND_BENCH.md):
#   - serve with --no-enable-prefix-caching (preset-yaml value is silently ignored)
#   - NO warmup, NO KV-adaptive prompt trick; the driver sizes each window itself
#   - headline metric = prefill_throughput; extract via extract_prefill_running.py
#   - verify prefix caching is actually OFF (0% hit) or the numbers are meaningless
# Env: MODEL PRESET PORT RUN KEEP_SERVER SERVER_WAIT_TIMEOUT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"

MODEL="${MODEL:-/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/}"
PRESET="${PRESET:-${AUTO_ROOT}/presets/kimi2.6.mxfp4/dp8ep8/base.yaml}"
PORT="${PORT:-8000}"
KEEP_SERVER="${KEEP_SERVER:-0}"
WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-3600}"
RUN="${RUN:-${TICKET_DIR}/logs/runbench_prefill/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN}/results" "${RUN}/prefill_running"
echo "[run_and_bench_prefill] preset=${PRESET}"
echo "[run_and_bench_prefill] RUN=${RUN}"

# --- serve: prefix cache DISABLED on the CLI (the yaml value is ignored by vLLM) ---
cp -f "${PRESET}" "${RUN}/preset.yaml" 2>/dev/null || true
nohup vllm-moreh serve "${MODEL}" "${PRESET}" --no-enable-prefix-caching > "${RUN}/serve.log" 2>&1 &
echo $! > "${RUN}/serve.pid.txt"
echo "[run_and_bench_prefill] serve pid=$(cat "${RUN}/serve.pid.txt"); waiting for health..."
t=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    kill -0 "$(cat "${RUN}/serve.pid.txt")" 2>/dev/null \
        || { echo "[run_and_bench_prefill] server died"; tail -40 "${RUN}/serve.log"; exit 1; }
    sleep 15; t=$((t + 15))
    [ "${t}" -ge "${WAIT_TIMEOUT}" ] && { echo "[run_and_bench_prefill] health timeout ${WAIT_TIMEOUT}s"; exit 1; }
done
echo "[run_and_bench_prefill] HEALTH OK"
sleep 20

# --- verify prefix caching is really OFF (else prefill numbers are inflated) ---
EPC=$(grep -aoE "enable_prefix_caching=(True|False)" "${RUN}/serve.log" | head -1)
echo "[run_and_bench_prefill] effective ${EPC:-enable_prefix_caching=?}"
case "${EPC}" in
    *False) echo "[run_and_bench_prefill] prefix caching OFF (good)" ;;
    *) echo "[run_and_bench_prefill][WARN] prefix caching may be ON -> prefill numbers could be inflated" ;;
esac

# --- bench (no warmup; multi_process_test.py sizes each window). writes RUN/results ---
REBENCH_RESULTS_DIR="${RUN}/results" \
    nohup python3 -u "${SCRIPT_DIR}/multi_process_test.py" > "${RUN}/mpt_run.log" 2>&1 &
echo $! > "${RUN}/bench.pid.txt"
echo "[run_and_bench_prefill] bench pid=$(cat "${RUN}/bench.pid.txt"); running..."
while kill -0 "$(cat "${RUN}/bench.pid.txt")" 2>/dev/null; do sleep 30; done
echo "[run_and_bench_prefill] bench DONE"

# --- extract & summarize (prefill: per-step cluster prompt throughput) ---
MIN_TS=$(stat -c %Y "${RUN}/bench.pid.txt")
mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "${RUN}/mpt_run.log" \
    | awk '{gsub(/,/,""); print $4"_p"$2}')
python3 "${SCRIPT_DIR}/extract_prefill_running.py" \
    "${RUN}/serve.log" "${RUN}/prefill_running/prefill_running" --scenarios "${LABELS[@]}"
python3 "${SCRIPT_DIR}/summarize_scenarios.py" \
    "${RUN}/results" "${RUN}/scenario_summary.csv" \
    --totals-dir "${RUN}/prefill_running" --min-ts "${MIN_TS}"
command -v column >/dev/null 2>&1 && column -t -s, "${RUN}/scenario_summary.csv" || cat "${RUN}/scenario_summary.csv"

# --- sanity: no non-zero prefix-cache hit during the benches ---
HITS=$(grep -aoE "Prefix cache hit: [0-9.]+%" "${RUN}/serve.log" 2>/dev/null | grep -v "0.0%" | head -3)
[ -n "${HITS}" ] && echo "[run_and_bench_prefill][WARN] non-zero prefix-cache hit seen: ${HITS}" \
                  || echo "[run_and_bench_prefill] prefix-cache hit 0% across benches (good)"

# --- stop ---
case "${KEEP_SERVER}" in
    1|true|yes|on) echo "[run_and_bench_prefill] KEEP_SERVER -> server left running" ;;
    *) kill -TERM "$(cat "${RUN}/serve.pid.txt")" 2>/dev/null; sleep 5; pkill -9 VLLM 2>/dev/null; sleep 3
       echo "[run_and_bench_prefill] server stopped" ;;
esac
echo "[run_and_bench_prefill] -> ${RUN}/scenario_summary.csv"
