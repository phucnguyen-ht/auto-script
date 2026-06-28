#!/usr/bin/env bash
set -uo pipefail
# =============================================================================
# MV-4571 — AUTO analyze EP8 imbalance (token + time) cho mọi testcase.
# Gộp toàn bộ quy trình thủ công thành 1 lệnh sweep. Với MỖI case (scenario.yaml),
# chạy 2 phase, mỗi phase serve RIÊNG (kill -> serve -> bench -> analyze -> kill):
#
#   [TOKEN]  preset: enforce_eager=true, VLLM_MOREH_EP_LOG='1', KHÔNG profiler.
#            -> serve -> bench (drive forward) -> serve.log đầy [EP_COLLECT]
#            -> analyze_tokens.py  (hist max/min, phân phối tải, per-layer...).
#
#   [TIME]   preset: BỎ enforce_eager (cudagraph FULL_DECODE_ONLY), EP_LOG='0',
#            inject profiler_config (torch), ÉP api_server_count=1.
#            -> serve -> bench --profile -> đợi 8 trace -> analyze_time.py.
#
# Vì sao tách 2 lần serve: token cần eager (để side-effect log [EP_COLLECT] chạy);
# time cần cudagraph (timing thực tế) — không thể cùng 1 server. Mỗi case cũng
# serve lại từ đầu để data sạch (không dồn vào 1 server chạy hết sweep).
#
# FIX GỐC cho auto_profile.sh cũ (chạy xong KHÔNG có trace, không ổn định):
#   Nguyên nhân: dp8 -> api_server_count mặc định = dp = 8 (serve.py:105-109).
#   `vllm bench serve --profile` gửi /start_profile rồi /stop_profile, bị load-
#   balance sang 2 frontend KHÁC nhau -> start/stop rơi vào engine khác nhau ->
#   worker không nhận đủ cặp start+stop -> trace KHÔNG được ghi (lúc có lúc không).
#   Cách sửa: ÉP api_server_count=1 (patch_time_preset) -> 1 frontend nhận cả
#   start lẫn stop, broadcast nhất quán tới 8 DP engine -> 8 trace tin cậy & sạch.
#   Bồi thêm: CHỜ đủ 8 file dp*_rank0.*.pt.trace.json.gz (flush bất đồng bộ sau
#   /stop_profile) RỒI MỚI kill server (KHÔNG kill ngay sau bench như bản cũ).
#
# Chạy TRONG docker `phuc-nguyen-mv-4571`:
#   docker exec -ti phuc-nguyen-mv-4571 bash -lc \
#     'bash /home/phuc-nguyen/workspaces/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/auto_analyze_ep_imbalance.sh'
#
# Env override: SCENARIO_YAML, PHASES=token,time (chọn phase), ONLY=<idx case>,
#   ANALYZE=0 (chỉ thu data, không chạy python), GANTT_MAX_FIGS, REL_THR, THR_US,
#   DROP_HEAD, SKIP_GPU_WAIT=1, TRACE_WAIT_TIMEOUT.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_DIR="${TICKET_DIR}/../common"
export ENV_YAML="${ENV_YAML:-${TICKET_DIR}/env.yaml}"   # cho resolve_* (model paths, backend)
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${TICKET_DIR}}"
# shellcheck source=../../common/helper.sh
source "${COMMON_DIR}/helper.sh"
ensure_yq

SCENARIO_YAML="${SCENARIO_YAML:-${SCRIPT_DIR}/scenario.yaml}"
[ -f "${SCENARIO_YAML}" ] || { echo "[ERROR] không thấy scenario: ${SCENARIO_YAML}" >&2; exit 1; }
PHASES="${PHASES:-token,time}"
ANALYZE="${ANALYZE:-1}"
ONLY="${ONLY:-}"                          # chỉ chạy 1 case theo index (0-based); rỗng = tất cả
DROP_HEAD="${DROP_HEAD:-0}"
REL_THR="${REL_THR:-0.2}"
THR_US="${THR_US:-50}"
GANTT_MAX_FIGS="${GANTT_MAX_FIGS:-200}"
TRACE_WAIT_TIMEOUT="${TRACE_WAIT_TIMEOUT:-600}"   # giây chờ đủ 8 trace flush
EXPECT_RANKS="${EXPECT_RANKS:-8}"                 # số worker trace mong đợi (= dp size)
API_SERVER_COUNT_PROFILE="${API_SERVER_COUNT_PROFILE:-1}"  # =1 để /start+/stop cùng 1 frontend (trace tin cậy)
PYTHON="${PYTHON:-python3}"

# helper đọc scenario.yaml.
yqs() { yq e "$1" "${SCENARIO_YAML}"; }                       # raw query
defv() {  # defv <key> <default>  — đọc .defaults.<key>
    local v; v="$(yq e ".defaults.$1 // \"__NULL__\"" "${SCENARIO_YAML}")"
    [ "${v}" = "__NULL__" ] && v="$2"; printf '%s' "${v}"
}
clamp() { local v="$1" lo="$2" hi="$3"; (( v < lo )) && v="${lo}"; (( v > hi )) && v="${hi}"; printf '%s' "${v}"; }

# defaults dùng chung cho mọi case
DSDIR="$(defv dataset_dir /remote/vast0/share-mv/longbenchv2-custom)"
PPC="$(defv prompts_per_concurrency 2)"
NP_FLOOR="$(defv num_prompts_floor 1)"
NP_CAP="$(defv num_prompts_cap 256)"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${LOG_ROOT}/${RUN_TS}"
mkdir -p "${RESULTS_ROOT}"

resolve_backend
[ "${BACKEND,,}" = "vllm" ] || { echo "[ERROR] chỉ hỗ trợ vllm." >&2; exit 1; }
trap 'is_enabled "${DRY_RUN:-0}" || kill_server' EXIT

N_CASES="$(yq e '.cases | length' "${SCENARIO_YAML}")"
echo "=============================================================="
echo " MV-4571 auto EP-imbalance analyze @ ${RUN_TS}"
echo " scenario = ${SCENARIO_YAML}  (cases=${N_CASES})  phases=${PHASES}"
echo " results  = ${RESULTS_ROOT}"
echo "=============================================================="

# --- profiler_config JSON (GIỮ frontend ON: KHÔNG thêm ignore_frontend) ---
profiler_pc() {  # profiler_pc <trace_dir>
    printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":"%s","torch_profiler_record_shapes":"%s","torch_profiler_with_memory":"%s","torch_profiler_with_flops":"%s"}' \
        "$1" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_STACK' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_RECORD_SHAPES' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_MEMORY' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_FLOPS' False)"
}

# --- patch preset cho từng phase ---
patch_token_preset() {  # <base> <out>
    yq e '.engine_args.enforce_eager = true
        | .env_vars.VLLM_MOREH_EP_LOG = "1"
        | .env_vars.VLLM_MOREH_EP_LOG_DEBUG = "0"
        | del(.engine_args.profiler_config)' "$1" > "$2"
}
patch_time_preset() {  # <base> <out> <trace_dir>
    # api_server_count=1 (QUAN TRỌNG): với dp8 mặc định api_server_count=dp=8
    # (serve.py:105-109) -> /start_profile & /stop_profile bị LB sang 2 frontend
    # KHÁC nhau -> start/stop lệch engine -> worker không nhận đủ cặp start+stop ->
    # KHÔNG sinh trace (đúng ca bạn gặp). Ép =1 -> 1 frontend nhận cả start lẫn stop,
    # broadcast nhất quán tới 8 DP engine -> 8 trace tin cậy, lại sạch /stop (no 500).
    local pc; pc="$(profiler_pc "$3")"
    PC="${pc}" yq e '.engine_args.enforce_eager = false
        | .env_vars.VLLM_MOREH_EP_LOG = "0"
        | .engine_args.api_server_count = '"${API_SERVER_COUNT_PROFILE}"'
        | .engine_args.profiler_config = strenv(PC)' "$1" > "$2"
}

# --- serve / kill ---
start_server() {  # <preset_yaml> <serve_log>
    PRESET_YAML="$1"
    kill_server
    is_enabled "${SKIP_GPU_WAIT:-0}" || wait_for_gpu_free
    PRESET_YAML="$1" serve_backend "$2"
    wait_for_server || { echo "[ERROR] server không lên (xem $2)" >&2; return 1; }
}

# --- bench (drive forward passes); profile=1 thêm --profile ---
run_bench() {  # <profile 0|1> <outdir> <label> <model> <dpath> <osl> <conc> <np> <rate>
    local profile="$1" outdir="$2" label="$3" model="$4" dpath="$5" osl="$6" conc="$7" np="$8" rate="$9"
    local prof=(); [ "${profile}" = "1" ] && prof=(--profile)
    mkdir -p "${outdir}"
    echo "  [bench${profile:+/profile}] np=${np} conc=${conc} osl=${osl} rate=${rate} -> ${outdir}"
    vllm bench serve \
        --backend vllm --model "${model}" --trust-remote-code \
        --base-url "${BASE_URL}" --endpoint /v1/completions \
        --dataset-name custom --dataset-path "${dpath}" \
        --custom-output-len "${osl}" --num-prompts "${np}" \
        --max-concurrency "${conc}" --request-rate "${rate}" \
        --skip-chat-template --ignore-eos \
        --percentile-metrics ttft,tpot,e2el,itl --metric-percentiles 75,90,99 \
        --metadata isl="${label}" osl="${osl}" conc="${conc}" num_prompts="${np}" request_rate="${rate}" \
        --save-result --result-dir "${outdir}" --result-filename "bench_${label}.json" \
        --temperature 0 --seed 0 "${prof[@]}" 2>&1 | tee "${outdir}/bench.log"
    return 0   # --profile có thể exit !=0 do /stop_profile 500 (vô hại) -> đừng abort
}

# --- chờ đủ N worker trace flush xong (KHÔNG kill server trước khi đủ) ---
wait_for_traces() {  # <trace_dir> <need> <timeout_s>
    local dir="$1" need="$2" timeout="$3" t=0 prev=-1 stable=0 cur
    echo "  [trace] chờ >=${need} file dp*_rank0.*.pt.trace.json.gz trong ${dir} (timeout ${timeout}s)..."
    while (( t < timeout )); do
        cur=$(find "${dir}" -maxdepth 1 -name 'dp*_rank0.*.pt.trace.json.gz' 2>/dev/null | wc -l)
        if (( cur >= need )); then
            if (( cur == prev )); then (( ++stable >= 3 )) && break    # ổn định ~6s
            else stable=0; fi
        fi
        prev=${cur}; sleep 2; (( t += 2 ))
    done
    cur=$(find "${dir}" -maxdepth 1 -name 'dp*_rank0.*.pt.trace.json.gz' 2>/dev/null | wc -l)
    if (( cur >= need )); then echo "  [trace] OK: ${cur}/${need} trace."; return 0
    else echo "  [trace][WARN] chỉ thấy ${cur}/${need} trace sau ${t}s (vẫn tiếp tục)." >&2; return 1; fi
}

# ---------- 1 phase TOKEN: serve(eager,EP_LOG=1) -> bench -> analyze ----------
do_token_phase() {  # <out_dir> <base_preset> <label> <model_path> <dpath> <osl> <conc> <rate>
    local out="$1" base="$2" label="$3" model="$4" dpath="$5" osl="$6" conc="$7" rate="$8"
    local np; np="$(clamp $(( PPC * conc )) "${NP_FLOOR}" "${NP_CAP}")"   # bench: ppc>=1
    echo "  --- [TOKEN] serve(eager,EP_LOG=1) -> bench(np=${np}) -> analyze ---"
    mkdir -p "${out}"
    local preset="${out}/preset.yaml"; patch_token_preset "${base}" "${preset}"
    local serve_log="${out}/serve.log"
    is_enabled "${DRY_RUN:-0}" && { echo "  [DRY] token preset -> ${preset}"; return 0; }
    start_server "${preset}" "${serve_log}" || { kill_server; echo "  [TOKEN][SKIP] serve fail." >&2; return 1; }
    run_bench 0 "${out}/bench" "${label}" "${model}" "${dpath}" "${osl}" "${conc}" "${np}" "${rate}"
    kill_server
    is_enabled "${ANALYZE}" || return 0
    echo "  [analyze] tokens -> ${out}/analysis"
    "${PYTHON}" "${SCRIPT_DIR}/analyze_tokens.py" --log "${serve_log}" --out "${out}/analysis" \
        --concurrency "${conc}" --layers all \
        2>&1 | tee "${out}/analyze_tokens.log" || echo "  [analyze][WARN] tokens lỗi." >&2
}

# ---------- 1 phase TIME: serve(cudagraph,profiler,frontend ON) -> bench --profile -> wait traces -> analyze ----------
do_time_phase() {  # <out_dir> <base_preset> <label> <model_path> <dpath> <osl> <conc> <rate>
    local out="$1" base="$2" label="$3" model="$4" dpath="$5" osl="$6" conc="$7" rate="$8"
    local np; np="$(clamp "${conc}" "${NP_FLOOR}" "${NP_CAP}")"           # profile: ppc LUÔN = 1
    echo "  --- [TIME] serve(cudagraph,profiler,api_server_count=${API_SERVER_COUNT_PROFILE}) -> bench --profile(np=${np}) -> wait ${EXPECT_RANKS} trace -> analyze ---"
    local trace="${out}/traces"; mkdir -p "${trace}"
    local preset="${out}/preset.yaml"; patch_time_preset "${base}" "${preset}" "${trace}"
    local serve_log="${out}/serve.log"
    is_enabled "${DRY_RUN:-0}" && { echo "  [DRY] time preset -> ${preset}"; return 0; }
    start_server "${preset}" "${serve_log}" || { kill_server; echo "  [TIME][SKIP] serve fail." >&2; return 1; }
    run_bench 1 "${out}/bench" "${label}" "${model}" "${dpath}" "${osl}" "${conc}" "${np}" "${rate}"
    wait_for_traces "${trace}" "${EXPECT_RANKS}" "${TRACE_WAIT_TIMEOUT}"  # CHỜ đủ trace trước khi kill
    kill_server
    is_enabled "${ANALYZE}" || return 0
    echo "  [analyze] time -> ${out}/analysis"
    "${PYTHON}" "${SCRIPT_DIR}/analyze_time.py" --trace-dir "${trace}" --out "${out}/analysis" \
        --drop-head "${DROP_HEAD}" --rel-thr "${REL_THR}" --thr-us "${THR_US}" --gantt-max-figs "${GANTT_MAX_FIGS}" \
        2>&1 | tee "${out}/analyze_time.log" || echo "  [analyze][WARN] time lỗi." >&2
}

# Cổng GPU ban đầu (TÔN TRỌNG job đang chạy của người khác): nếu GPU đang bận,
# ĐỢI & retry mỗi 30s tới khi rảnh RỒI mới bắt đầu — tránh kill nhầm server đang
# chạy. Sau cổng này GPU đã rảnh nên kill_server (per-phase) chỉ là no-op vô hại.
# SKIP_GPU_WAIT=1 để bỏ qua; GPU_POLL_INTERVAL=<giây> để đổi nhịp retry (mặc định 30).
if ! is_enabled "${DRY_RUN:-0}" && ! is_enabled "${SKIP_GPU_WAIT:-0}"; then
    export GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-30}"
    echo "[gpu-gate] kiểm tra GPU rảnh trước khi bắt đầu (retry mỗi ${GPU_POLL_INTERVAL}s)..."
    wait_for_gpu_free
fi

# ====================== VÒNG LẶP: case -> preset -> rate -> conc ======================
run_idx=0                       # đếm tổ hợp để lọc bằng ONLY
for (( i=0; i<N_CASES; i++ )); do
    MODEL="$(yqs ".cases[${i}].model")"
    NAME="$(yqs ".cases[${i}].name")"
    OSL="$(yqs ".cases[${i}].osl")"
    DPATH="${DSDIR}/longbenchv2-${NAME}.jsonl"
    MODEL_PATH_CFG="$(yqs ".models.\"${MODEL}\".model_path")"
    mapfile -t PRESETS < <(yqs ".models.\"${MODEL}\".presets[]")
    mapfile -t RATES   < <(yqs ".cases[${i}].rates[]")
    mapfile -t CONCS   < <(yqs ".cases[${i}].concurrencies[]")
    if [ -z "${MODEL_PATH_CFG}" ] || [ "${MODEL_PATH_CFG}" = "null" ]; then
        echo "[SKIP] model '${MODEL}' không có model_path trong scenario.yaml" >&2; continue
    fi

    for PRESET in "${PRESETS[@]}"; do
        # resolve preset -> đường tuyệt đối + tên ngắn (PRESET_NAME) cho thư mục.
        unset PRESET_NAME; PRESET="${PRESET}" resolve_preset
        BASE_PRESET="${PRESET_YAML}"
        PSHORT="$(basename "${PRESET_NAME}")"          # vd noMTP-bs64-dg
        export MODEL_PATH="${MODEL_PATH_CFG}"          # dùng thẳng model_path từ registry

        for RATE in "${RATES[@]}"; do
            for CONC in "${CONCS[@]}"; do
                if [ -n "${ONLY}" ] && [ "${ONLY}" != "${run_idx}" ]; then run_idx=$((run_idx+1)); continue; fi
                LABEL="${MODEL}_${PSHORT}_${NAME}_r${RATE//./p}_c${CONC}"
                CASE_DIR="${RESULTS_ROOT}/${MODEL}/${PSHORT}/${NAME}_r${RATE//./p}_c${CONC}"
                echo; echo "############ [${run_idx}] ${LABEL} ############"
                echo "  preset=${BASE_PRESET}"
                echo "  model =${MODEL_PATH}"
                echo "  dataset=${DPATH}  osl=${OSL} conc=${CONC} rate=${RATE}"
                run_idx=$((run_idx+1))
                if [ ! -f "${DPATH}" ]; then
                    echo "  [SKIP] dataset không tồn tại: ${DPATH}" >&2; continue
                fi
                [[ ",${PHASES}," == *",token,"* ]] && \
                    do_token_phase "${CASE_DIR}/tokens" "${BASE_PRESET}" "${LABEL}" "${MODEL_PATH}" "${DPATH}" "${OSL}" "${CONC}" "${RATE}"
                [[ ",${PHASES}," == *",time,"* ]] && \
                    do_time_phase  "${CASE_DIR}/time"   "${BASE_PRESET}" "${LABEL}" "${MODEL_PATH}" "${DPATH}" "${OSL}" "${CONC}" "${RATE}"
                echo "  ==> xong -> ${CASE_DIR}"
            done
        done
    done
done

echo; echo "=============================================================="
echo " DONE. Tất cả kết quả: ${RESULTS_ROOT}  (tổng ${run_idx} tổ hợp preset×rate×conc)"
echo "=============================================================="
