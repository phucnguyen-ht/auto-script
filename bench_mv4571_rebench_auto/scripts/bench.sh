#!/usr/bin/env bash
# Bench + extract per RUN_AND_BENCH.md §3+§4, straight from scripts/ (no copy).
# Results land under $RUN (via REBENCH_RESULTS_DIR). Server must already be up.
#   bash bench.sh [SERVE_LOG]     # SERVE_LOG defaults to $RUN/serve.log
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN="${RUN:-${TICKET_DIR}/logs/manual/$(date +%Y%m%d_%H%M%S)}"
# serve.log: arg1 > $SERVE_LOG > newest serve_glm5.sh log under the ticket.
SERVE_LOG="${1:-${SERVE_LOG:-}}"
[ -z "${SERVE_LOG}" ] && SERVE_LOG="$(ls -1dt "${TICKET_DIR}"/logs/manual/serve_*/serve.log 2>/dev/null | head -1)"
mkdir -p "${RUN}/results" "${RUN}/decode_running"

# §3
MIN_TS_FILE="${RUN}/bench.start"; : > "${MIN_TS_FILE}"
REBENCH_RESULTS_DIR="${RUN}/results" \
    python3 -u "${SCRIPT_DIR}/multi_process_test.py" 2>&1 | tee "${RUN}/mpt_run.log"

# §4
MIN_TS="$(stat -c %Y "${MIN_TS_FILE}")"
mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "${RUN}/mpt_run.log" \
    | awk '{gsub(/,/,""); print $4"_p"$2}')
if [ -f "${SERVE_LOG}" ]; then
    python3 "${SCRIPT_DIR}/extract_decode_running.py" \
        "${SERVE_LOG}" "${RUN}/decode_running/decode_running" --scenarios "${LABELS[@]}"
else
    echo "[bench] serve.log not found: ${SERVE_LOG} -> skip decode-batch extract (pass it as arg 1)"
fi
python3 "${SCRIPT_DIR}/summarize_scenarios.py" \
    "${RUN}/results" "${RUN}/scenario_summary.csv" \
    --totals-dir "${RUN}/decode_running" --min-ts "${MIN_TS}"
column -t -s, "${RUN}/scenario_summary.csv"
echo "[bench] -> ${RUN}/scenario_summary.csv"
