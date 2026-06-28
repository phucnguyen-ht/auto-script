#!/usr/bin/env bash
# =============================================================================
# MV-4571 : GLM-5.2 (DP8/EP8) EP-imbalance analysis -- ONE script, all steps.
# Methodology ported from MV-3382 (see process_ep_logs.ipynb / split_jobs.py):
#   1. Serve GLM-5.2 with EP routing logging on (VLLM_MOREH_EP_LOG=1 -> the
#      base_router patch prints "[TOPK] Expert selection counts: [...]" per
#      MoE-layer forward, from DP rank 0 only).
#   2. For each (ISL, conc) scenario, run `vllm bench serve` (ticket knobs) and
#      slice this scenario's [TOPK] lines out of the shared serve.log by byte
#      offset -> per-scenario routing log.
#   3. analyze_ep_imbalance.py -> per-scenario rank-imbalance % (max/avg),
#      split prefill vs decode. Compare MTP 0 vs 3, ISL 10k vs 100k.
#
# To extend the sweep later: just edit the CASES / CONCS arrays below.
# Re-running is incremental: a scenario whose routing log already exists and is
# non-empty is skipped (delete it to force a re-run).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO="${HERE}/../.."                       # auto-script/
SERVE_SH="${AUTO}/serve.sh"
MODEL=/remote/vast0/share-mv/zai-org/GLM-5.2-FP8
DATASET_DIR=/remote/vast0/share-mv/longbenchv2-custom
PORT="${PORT:-8100}"
BASE_URL="http://localhost:${PORT}"

OUTDIR="${HERE}/runs"; mkdir -p "${OUTDIR}"

# ---- EDIT HERE: which cases to run ------------------------------------------
# preset label -> preset yaml (MTP direction).
declare -A PRESETS=(
  [noMTP]="${AUTO}/presets/glm5.2/dp8ep8/noMTP-bs64-dg.yaml"
  [MTP3]="${AUTO}/presets/glm5.2/dp8ep8/MTP3-bs64-dg.yaml"
)
PRESET_ORDER=(noMTP MTP3)
# ISL -> output seq len (from env.yaml).
declare -A OSL=( [10k]=500 [100k]=500 [8k]=1024 [1M]=500 )
ISLS=(10k 100k)
CONCS=(36)
RUNS="${RUNS:-1}"                          # bench repeats per scenario (1 is enough for imbalance)
# MoE topology (GLM-5.2): 256 routed experts, top-8, EP8 -> 32 experts/rank.
NUM_EXPERTS=256; NUM_RANKS=8; TOPK=8
DECODE_MAX_TOKENS=256                       # steps with <= this many tokens = decode
# -----------------------------------------------------------------------------

log() { echo "[ep-analysis $(date +%H:%M:%S)] $*"; }

wait_for_gpu_free() {
  local thr_gb="${GPU_FREE_GB:-150}" iv="${GPU_POLL:-60}" elapsed=0
  log "waiting for >=8 GPUs each with >=${thr_gb}GB free (poll ${iv}s)..."
  while true; do
    local freecnt
    freecnt=$(python3 - "$thr_gb" <<'PY'
import sys, torch
thr=float(sys.argv[1])*2**30
print(sum(1 for i in range(torch.cuda.device_count())
          if torch.cuda.mem_get_info(i)[0] >= thr))
PY
)
    [ "${freecnt:-0}" -ge 8 ] && { log "GPUs free (${freecnt} ok) after ${elapsed}s."; return 0; }
    log "only ${freecnt} GPU(s) free (${elapsed}s); retry in ${iv}s..."
    sleep "${iv}"; elapsed=$((elapsed+iv))
  done
}

wait_for_server() {
  local to="${SERVER_WAIT:-3600}" e=0
  log "polling ${BASE_URL}/health (timeout ${to}s)..."
  while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
    sleep 10; e=$((e+10))
    [ "$e" -ge "$to" ] && { log "ERROR server not ready in ${to}s"; return 1; }
  done
  log "server ready after ${e}s."
}

kill_server() { log "SIGKILL vLLM..."; pkill -9 VLLM 2>/dev/null || true; pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 15; }

# trust-remote-code mirrors the preset (GLM tokenizer needs it).
bench_one() {  # <preset_label> <isl> <conc> <serve_log> <run_idx>
  local plabel="$1" isl="$2" conc="$3" serve_log="$4" ridx="$5"
  local osl="${OSL[$isl]}" np=$(( conc*2 )); (( np>256 )) && np=256
  local ds="${DATASET_DIR}/longbenchv2-${isl}.jsonl"
  local tag="${plabel}_${isl}_c${conc}_r${ridx}"
  local toplog="${OUTDIR}/${plabel}/topk_${isl}_c${conc}_r${ridx}.log"
  local benchlog="${OUTDIR}/${plabel}/bench_${isl}_c${conc}_r${ridx}.log"
  if [ -s "${toplog}" ]; then log "skip ${tag} (exists)"; return 0; fi

  local off0; off0=$(wc -c < "${serve_log}")
  log "bench ${tag} (osl=${osl} num_prompts=${np})"
  vllm bench serve \
      --backend vllm --model "${MODEL}" --trust-remote-code \
      --base-url "${BASE_URL}" --endpoint /v1/completions \
      --dataset-name custom --dataset-path "${ds}" \
      --custom-output-len "${osl}" --num-prompts "${np}" \
      --max-concurrency "${conc}" --request-rate inf \
      --percentile-metrics "ttft,tpot,e2el,itl" --metric-percentiles "75,90,99" \
      --skip-chat-template --ignore-eos --seed 0 \
      --metadata isl="${isl}" osl="${osl}" conc="${conc}" mtp="${plabel}" \
      > "${benchlog}" 2>&1
  sleep 3   # let async [TOPK] prints flush to serve.log
  # slice this scenario's new bytes -> keep only [TOPK] lines
  tail -c +$((off0+1)) "${serve_log}" | grep -F '[TOPK] Expert selection counts' > "${toplog}" || true
  log "  -> $(wc -l < "${toplog}") routing steps captured"
}

serve_preset() {  # <preset_label> <preset_yaml>
  local plabel="$1" preset="$2"
  local pdir="${OUTDIR}/${plabel}"; mkdir -p "${pdir}"
  local serve_log="${pdir}/serve.log"
  : > "${serve_log}"
  log "serving GLM-5.2 [${plabel}] preset=${preset} -> ${serve_log}"
  VLLM_MOREH_EP_LOG=1 VLLM_MOREH_EP_LOG_RANK=0 \
    bash "${SERVE_SH}" "${MODEL}" "${preset}" --port "${PORT}" > "${serve_log}" 2>&1 &
  if ! wait_for_server; then
    log "ERROR: ${plabel} server failed (tail serve.log):"; tail -30 "${serve_log}"; kill_server; return 1
  fi
  for isl in "${ISLS[@]}"; do
    for conc in "${CONCS[@]}"; do
      for r in $(seq 1 "${RUNS}"); do
        bench_one "${plabel}" "${isl}" "${conc}" "${serve_log}" "${r}"
      done
    done
  done
  kill_server
}

analyze_all() {
  log "=== analysis ==="
  local summary="${OUTDIR}/summary.txt"; : > "${summary}"
  for plabel in "${PRESET_ORDER[@]}"; do
    for isl in "${ISLS[@]}"; do
      for conc in "${CONCS[@]}"; do
        # merge all runs' routing logs for this scenario
        local logs=( "${OUTDIR}/${plabel}"/topk_${isl}_c${conc}_r*.log )
        [ -s "${logs[0]}" ] || { echo "MISSING ${plabel} ${isl} c${conc}" | tee -a "${summary}"; continue; }
        local lbl="${plabel}/${isl}/c${conc}"
        python3 "${HERE}/analyze_ep_imbalance.py" "${logs[@]}" \
            --experts "${NUM_EXPERTS}" --ranks "${NUM_RANKS}" --topk "${TOPK}" \
            --decode-max-tokens "${DECODE_MAX_TOKENS}" \
            --label "${lbl}" --json "${OUTDIR}/${plabel}/result_${isl}_c${conc}.json" \
            | tee -a "${summary}"
      done
    done
  done
  log "summary -> ${summary}"
}

main() {
  case "${1:-all}" in
    analyze) analyze_all; exit 0 ;;     # re-analyze existing logs only
  esac
  wait_for_gpu_free
  for plabel in "${PRESET_ORDER[@]}"; do
    serve_preset "${plabel}" "${PRESETS[$plabel]}" || log "WARN ${plabel} had errors"
  done
  analyze_all
  log "DONE."
}
main "$@"
