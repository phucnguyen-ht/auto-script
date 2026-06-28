#!/usr/bin/env bash
set -euo pipefail
# MV-4571: BENCH-ONLY (KHÔNG profile) lên server đã bật sẵn bởi serve_profile.sh.
# Dựa trên bench_only.sh nhưng:
#   * đọc .profile_session.env -> BASE_URL/MODEL_PATH của server đang chạy
#   * knobs khớp auto_bench (.bench.custom): --ignore-eos, --skip-chat-template,
#     custom dataset, --metric-percentiles 75,90,99, --request-rate inf,
#     num_prompts = 2*conc, KHÔNG reset prefix cache (engine đã tắt cache).
#
# Dùng:
#   bash bench_mv4571/serve_profile.sh      # cửa sổ 1 (server giữ chạy)
#   bash bench_mv4571/bench.sh              # cửa sổ 2
#
# Override qua env: DATASET OSL CONC NUM_PROMPTS RATE RUNS DATASET_DIR DATASET_PATH

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- session (BASE_URL/MODEL_PATH do serve_profile.sh ghi) ---
SESSION="${PROFILE_SESSION:-${TICKET_DIR}/.profile_session.env}"
# shellcheck disable=SC1090
[ -f "${SESSION}" ] && source "${SESSION}"

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-${MODEL_PATH:-/remote/vast0/share-mv/zai-org/GLM-5.2-FP8}}"

# --- knobs (khớp auto_bench .bench.custom) ---
DATASET="${DATASET:-8k}"
DATASET_DIR="${DATASET_DIR:-/remote/vast0/share-mv/longbenchv2-custom}"
DATASET_PATH="${DATASET_PATH:-${DATASET_DIR}/longbenchv2-${DATASET}.jsonl}"
OSL="${OSL:-1024}"
CONC="${CONC:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-$(( CONC * 2 ))}"   # ppc=2 như auto_bench
RATE="${RATE:-inf}"
RUNS="${RUNS:-1}"
SEED="${SEED:-0}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-75,90,99}"
LABEL="${DATASET}_r${RATE//./p}_c${CONC}"
# Ghi vào RUN_DIR của serve (cùng folder serve.log) nếu có session; else folder riêng.
if [ -n "${OUTDIR:-}" ]; then :
elif [ -n "${RUN_DIR:-}" ]; then OUTDIR="${RUN_DIR}/bench"
else OUTDIR="${TICKET_DIR}/logs/bench_only/${DATASET}_c${CONC}_$(date +%Y%m%d-%H%M%S)"
fi

[ -f "${DATASET_PATH}" ] || { echo "ERROR: dataset không tồn tại: ${DATASET_PATH}" >&2; exit 1; }
mkdir -p "${OUTDIR}"
echo "base_url=${BASE_URL} model=${MODEL}"
echo "dataset=${DATASET_PATH} osl=${OSL} conc=${CONC} num_prompts=${NUM_PROMPTS} rate=${RATE} runs=${RUNS} -> ${OUTDIR}"

# server phải up (do serve_profile.sh khởi động)
curl -sf "${BASE_URL%/}/health" >/dev/null \
  || { echo "ERROR: server chưa sẵn sàng ở ${BASE_URL} — chạy serve_profile.sh trước." >&2; exit 1; }

for i in $(seq 1 "${RUNS}"); do
  log="${OUTDIR}/run${i}.log"
  echo "=== run ${i}/${RUNS}  $(date +%T)  -> ${log} ==="
  # KHÔNG reset_prefix_cache: engine đã tắt prefix caching (mỗi request tự cold).
  vllm bench serve \
      --backend vllm \
      --model "${MODEL}" \
      --trust-remote-code \
      --base-url "${BASE_URL}" \
      --endpoint /v1/completions \
      --dataset-name custom \
      --dataset-path "${DATASET_PATH}" \
      --custom-output-len "${OSL}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${CONC}" \
      --request-rate "${RATE}" \
      --skip-chat-template \
      --ignore-eos \
      --percentile-metrics ttft,tpot,e2el,itl \
      --metric-percentiles "${METRIC_PERCENTILES}" \
      --metadata isl="${DATASET}" osl="${OSL}" conc="${CONC}" num_prompts="${NUM_PROMPTS}" request_rate="${RATE}" \
      --save-result --result-dir "${OUTDIR}" --result-filename "run${i}_${LABEL}.json" \
      --temperature 0 \
      --seed "${SEED}" \
      2>&1 | tee "${log}"
  grep -iE "Successful requests|input tokens|generated tokens|throughput|Mean TTFT|Mean TPOT|Mean E2EL" "${log}" || tail -5 "${log}"
done
echo "DONE -> ${OUTDIR}"
