#!/usr/bin/env bash
# RUN_AND_BENCH.md end-to-end for ONE preset (§1 serve -> §3 bench -> §4 extract
# -> §6 stop), verbatim flow: nohup serve + the driver + extract. No template /
# AUTO_CLEAN / SERVE_TEE; no sweep/time_limit edits. Self-contained: results land
# under RUN. sweep_presets.sh calls this once per preset.
#   bash run_and_bench.sh
#   PRESET=/abs/x.yaml RUN=/abs/dir bash run_and_bench.sh
# Env: MODEL PRESET PORT RUN SERVE_SH(=1 -> serve.sh) KEEP_SERVER SERVER_WAIT_TIMEOUT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"

MODEL="${MODEL:-/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/}"
PRESET="${PRESET:-${AUTO_ROOT}/presets/glm5.2.rebench/MTP5-bs64-dg.yaml}"
PORT="${PORT:-8000}"
KEEP_SERVER="${KEEP_SERVER:-0}"
SERVE_SH="${SERVE_SH:-0}"
WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-3600}"
RUN="${RUN:-${TICKET_DIR}/logs/runbench/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN}/results" "${RUN}/decode_running"
echo "[run_and_bench] preset=${PRESET}"
echo "[run_and_bench] RUN=${RUN}"

# --- §1 serve (nohup vllm-moreh serve; SERVE_SH=1 -> repo serve.sh) ---
cp -f "${PRESET}" "${RUN}/preset.yaml" 2>/dev/null || true
case "${SERVE_SH}" in
    1|true|yes|on) nohup bash "${AUTO_ROOT}/serve.sh" "${MODEL}" "${PRESET}" > "${RUN}/serve.log" 2>&1 & ;;
    *)             nohup vllm-moreh serve "${MODEL}" "${PRESET}"             > "${RUN}/serve.log" 2>&1 & ;;
esac
echo $! > "${RUN}/serve.pid.txt"
echo "[run_and_bench] serve pid=$(cat "${RUN}/serve.pid.txt"); waiting for health..."
t=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    kill -0 "$(cat "${RUN}/serve.pid.txt")" 2>/dev/null \
        || { echo "[run_and_bench] server died"; tail -40 "${RUN}/serve.log"; exit 1; }
    sleep 15; t=$((t + 15))
    [ "${t}" -ge "${WAIT_TIMEOUT}" ] && { echo "[run_and_bench] health timeout ${WAIT_TIMEOUT}s"; exit 1; }
done
echo "[run_and_bench] HEALTH OK"

# --- §3 bench (driver writes into RUN/results) ---
REBENCH_RESULTS_DIR="${RUN}/results" \
    nohup python3 -u "${SCRIPT_DIR}/multi_process_test.py" > "${RUN}/mpt_run.log" 2>&1 &
echo $! > "${RUN}/bench.pid.txt"
echo "[run_and_bench] bench pid=$(cat "${RUN}/bench.pid.txt"); running..."
while kill -0 "$(cat "${RUN}/bench.pid.txt")" 2>/dev/null; do sleep 30; done
echo "[run_and_bench] bench DONE"

# --- §4 extract & summarize ---
MIN_TS=$(stat -c %Y "${RUN}/bench.pid.txt")
mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "${RUN}/mpt_run.log" \
    | awk '{gsub(/,/,""); print $4"_p"$2}')
python3 "${SCRIPT_DIR}/extract_decode_running.py" \
    "${RUN}/serve.log" "${RUN}/decode_running/decode_running" --scenarios "${LABELS[@]}"
python3 "${SCRIPT_DIR}/summarize_scenarios.py" \
    "${RUN}/results" "${RUN}/scenario_summary.csv" \
    --totals-dir "${RUN}/decode_running" --min-ts "${MIN_TS}"
column -t -s, "${RUN}/scenario_summary.csv"

# --- §5 validate prefix cache (per-scenario hit% / full_miss; important on dp8) ---
python3 - "${RUN}/serve.log" "${RUN}/mpt_run.log" <<'PY' | tee "${RUN}/prefix_cache_hit.txt"
import re,sys
L=open(sys.argv[1],errors="replace").readlines()
ansi=re.compile(r"\x1b\[[0-9;]*m"); pc=re.compile(r"Prefix cache hit:\s*[0-9.]+%\s*\((\d+)/(\d+)\)")
st=[i for i,l in enumerate(L) if 'bench_start' in l]; en=[i for i,l in enumerate(L) if 'bench_end' in l]
labs=[f"{m.group(2)}_p{m.group(1)}" for l in open(sys.argv[2],errors="replace")
      for m in [re.search(r'concurrency (\d+), dataset (\S+)',l)] if m]
for i,(s,e) in enumerate(zip(st,en)):
    H=Q=full=0
    for ln in L[s+1:e]:
        m=pc.search(ansi.sub("",ln))
        if not m: continue
        h,q=int(m.group(1)),int(m.group(2))
        if q==0: continue
        H+=h; Q+=q; full+= (h==0)
    lab=labs[i] if i<len(labs) else f"w{i+1}"
    print(f"{lab:>10} hit={100*H/Q if Q else 0:.2f}%  full_miss={full}")
PY

# --- §6 stop ---
case "${KEEP_SERVER}" in
    1|true|yes|on) echo "[run_and_bench] KEEP_SERVER -> server left running" ;;
    *) kill -TERM "$(cat "${RUN}/serve.pid.txt")" 2>/dev/null; sleep 5; pkill -9 VLLM 2>/dev/null; sleep 3
       echo "[run_and_bench] server stopped" ;;
esac
echo "[run_and_bench] -> ${RUN}/scenario_summary.csv"
