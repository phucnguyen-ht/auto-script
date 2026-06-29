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
#   [TIME]   preset: enforce_eager=false (cudagraph -> timing thực) + EP_LOG='0' + inject
#            profiler_config (torch). KHÔNG ép api_server_count (mặc định = dp).
#            -> serve -> bench --profile (NỀN) -> wait_for_traces -> kill -> analyze_time.py.
#
# Vì sao tách 2 lần serve: token cần eager (để side-effect log [EP_COLLECT] chạy); time chỉ cần
# tắt EP_LOG (tránh overhead log) + bật profiler — không thể cùng 1 server. Mỗi case cũng serve
# lại từ đầu để data sạch (không dồn vào 1 server chạy hết sweep).
#
# FIX GỐC "auto_profile chạy xong KHÔNG có trace / server treo" (KHÔNG sửa preset):
#   Nguyên nhân THẬT (đã xác nhận trên GPU): dp8 -> api_server_count mặc định = dp = 8
#   (serve.py:105-109). `vllm bench serve --profile` gửi /start_profile rồi /stop_profile
#   qua SO_REUSEPORT nên rơi vào 2 frontend KHÁC nhau. 8 file trace VẪN được worker ghi
#   ra trace_dir, NHƯNG sau đó các DP engine DEADLOCK (/stop_profile không trả về, worker
#   spin 100% CPU) -> `vllm bench` TREO mãi -> bản cũ block vào bench nên không bao giờ
#   harvest/kill được -> nhìn như "không có trace / server chết".
#   Cách sửa (đúng ý: KHÔNG đụng preset, mặc kệ deadlock): chạy bench Ở NỀN, POLL trace_dir
#   tới khi đủ 8 file & bytes ổn định (wait_for_traces) RỒI kill cả bench treo lẫn server.
#   (Nếu muốn log sạch, không deadlock: API_SERVER_COUNT_PROFILE=1 -> start+stop cùng 1 frontend.)
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

# AITER_MOREH_ROOT_DIR (do docker.sh set) có thể trỏ vào dev checkout KHÔNG tồn tại trong
# container build mới -> mọi EngineCore chết lúc startup (thiếu a8w8_*_bruteforce.csv). aiter.jit
# assert biến này phải set; aiter_moreh đọc configs từ đây. Nếu rỗng/không tồn tại -> fallback về
# package aiter_moreh đã cài (export để serve.sh + vllm serve con kế thừa). Chỉ sửa trong script này.
if [ -z "${AITER_MOREH_ROOT_DIR:-}" ] || [ ! -d "${AITER_MOREH_ROOT_DIR}" ]; then
    _aiter_pkg="$(${PYTHON:-python3} -c 'import os, aiter_moreh; print(os.path.dirname(aiter_moreh.__file__))' 2>/dev/null || true)"
    if [ -n "${_aiter_pkg}" ] && [ -d "${_aiter_pkg}" ]; then
        echo "[aiter] AITER_MOREH_ROOT_DIR='${AITER_MOREH_ROOT_DIR:-}' không hợp lệ -> dùng package đã cài: ${_aiter_pkg}" >&2
        export AITER_MOREH_ROOT_DIR="${_aiter_pkg}"
    fi
fi

SCENARIO_YAML="${SCENARIO_YAML:-${SCRIPT_DIR}/scenario.yaml}"
[ -f "${SCENARIO_YAML}" ] || { echo "[ERROR] không thấy scenario: ${SCENARIO_YAML}" >&2; exit 1; }
PHASES="${PHASES:-token,time}"
ANALYZE="${ANALYZE:-1}"
ONLY="${ONLY:-}"                          # chỉ chạy 1 case theo index (0-based); rỗng = tất cả
DROP_HEAD="${DROP_HEAD:-0}"
REL_THR="${REL_THR:-0.2}"
THR_US="${THR_US:-50}"
GANTT_MAX_FIGS="${GANTT_MAX_FIGS:-200}"
TRACE_WAIT_TIMEOUT="${TRACE_WAIT_TIMEOUT:-900}"   # giây chờ đủ 8 trace flush (export 1 cửa sổ to có thể ~8 phút)
EXPECT_RANKS="${EXPECT_RANKS:-8}"                 # số worker trace mong đợi (= dp size)
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-3600}"  # giây chờ /health TỐI ĐA (giảm 7200->3600)
SERVE_HEARTBEAT="${SERVE_HEARTBEAT:-600}"         # giây giữa 2 dòng log "đang chờ" (heartbeat)
SERVE_ATTEMPTS="${SERVE_ATTEMPTS:-2}"             # serve tối đa N lần (1 lần đầu + retry); fail-fast nếu serve.log có lỗi
API_SERVER_COUNT_PROFILE="${API_SERVER_COUNT_PROFILE:-}"  # RỖNG = KHÔNG đụng preset (api_server_count mặc định = dp).
                                                          # Đặt =1 nếu muốn /start+/stop về cùng 1 frontend (tránh deadlock /stop_profile, log sạch).
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

# 1 lần chạy = 1 folder run_<date>_<time>. TẤT CẢ (mọi model/preset/scenario, cả 2 phase
# token+time) đều ghi vào RESULTS_ROOT này -> CASE_DIR/{tokens,time} cùng 1 chỗ, dễ quản lý.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${LOG_ROOT}/run_${RUN_TS}"
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
    # GIỮ NGUYÊN preset (KHÔNG ép api_server_count): với dp8 mặc định api_server_count=dp=8
    # (serve.py:105-109) -> /start_profile & /stop_profile bị LB sang 2 frontend KHÁC nhau
    # -> sau khi đã GHI XONG 8 trace, các DP engine DEADLOCK (/stop_profile không trả về,
    # worker spin 100% CPU). Nhưng 8 file trace VẪN được flush ra trace_dir. Vì vậy ta
    # KHÔNG sửa preset để né deadlock; thay vào đó do_time_phase chạy bench NỀN rồi
    # wait_for_traces (poll trace_dir) -> đủ trace thì kill (xem do_time_phase).
    # Đặt API_SERVER_COUNT_PROFILE=1 nếu muốn né deadlock + log sạch (start+stop cùng frontend).
    # profile/time: enforce_eager=false (cudagraph -> timing thực), VLLM_MOREH_EP_LOG='0'
    # (tắt log token), inject profiler_config. (token/bench thì ngược lại: eager=true, EP_LOG=1.)
    local pc; pc="$(profiler_pc "$3")"
    PC="${pc}" yq e '.engine_args.enforce_eager = false
        | .env_vars.VLLM_MOREH_EP_LOG = "0"
        | .engine_args.profiler_config = strenv(PC)' "$1" > "$2"
    if [ -n "${API_SERVER_COUNT_PROFILE}" ]; then
        yq e -i '.engine_args.api_server_count = '"${API_SERVER_COUNT_PROFILE}" "$2"
    fi
}

# --- chờ server lên HOẶC fail-fast nếu serve.log có lỗi ---
# Poll /health; mỗi nhịp QUÉT serve.log tìm dấu hiệu crash (Traceback/assertion/NCCL broken
# pipe/worker chết...) -> thấy là DỪNG NGAY (return 1) thay vì chờ hết timeout. Heartbeat mỗi
# SERVE_HEARTBEAT giây. Tối đa SERVER_WAIT_TIMEOUT giây.
SERVE_FATAL_RE='Engine core initialization failed|Worker failed with error|BrokenPipeError|Traceback \(most recent call last\)|AssertionError|RuntimeError:|c10::Error|CUDA error|HIP error|Address already in use|EngineDeadError|EngineCore failed to start|ProcessGroupNCCL.*(Broken pipe|shut down)|multiproc_executor\.py.*\] ERROR'
wait_serve_or_fail() {  # <serve_log> ; 0=healthy, 1=fail/timeout
    local log="$1" timeout="${SERVER_WAIT_TIMEOUT:-3600}" hb="${SERVE_HEARTBEAT:-600}" t=0
    echo "  [wait] poll ${BASE_URL}/health (timeout ${timeout}s, heartbeat ${hb}s)..."
    while (( t < timeout )); do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            echo "  [wait] server READY sau ${t}s."; return 0
        fi
        if [ -f "${log}" ] && grep -qE "${SERVE_FATAL_RE}" "${log}" 2>/dev/null; then
            echo "  [wait][FAIL] phát hiện lỗi serve trong ${log} (sau ${t}s):" >&2
            grep -nE "${SERVE_FATAL_RE}" "${log}" 2>/dev/null | tail -3 >&2
            return 1
        fi
        # engine chết hẳn (sau 90s mà không còn EngineCore/vllm serve nào) -> fail luôn
        if (( t >= 90 )) && ! pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null 2>&1; then
            echo "  [wait][FAIL] không còn process vllm serve/EngineCore (sau ${t}s)." >&2
            return 1
        fi
        (( t % hb == 0 )) && echo "  [wait] ${t}s/${timeout}s..."
        sleep 10; (( t += 10 ))
    done
    echo "  [wait][FAIL] timeout ${timeout}s chờ ${BASE_URL}/health." >&2
    return 1
}

# --- serve / kill (retry SERVE_ATTEMPTS lần, fail-fast) ---
start_server() {  # <preset_yaml> <serve_log>
    local preset="$1" log="$2" attempts="${SERVE_ATTEMPTS:-2}" i
    for (( i=1; i<=attempts; i++ )); do
        kill_server
        # dọn thêm tàn dư serve/bench cũ (tránh đụng TCPStore/NCCL của lần serve trước treo/crash).
        pkill -9 -f "bin/vllm serve" 2>/dev/null || true
        pkill -9 -f "bin/vllm bench" 2>/dev/null || true
        sleep 3
        is_enabled "${SKIP_GPU_WAIT:-0}" || wait_for_gpu_free
        echo "  [serve] attempt ${i}/${attempts} -> ${log}"
        PRESET_YAML="${preset}" serve_backend "${log}"
        if wait_serve_or_fail "${log}"; then return 0; fi
        if (( i < attempts )); then
            echo "  [serve][WARN] attempt ${i} FAIL -> kill + retry." >&2
        else
            echo "  [serve][ERROR] server không lên sau ${attempts} lần (xem ${log})." >&2
        fi
        kill_server
    done
    return 1
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

# --- chờ đủ N worker trace flush XONG HẲN (KHÔNG kill server trước khi đủ + ổn định) ---
# Trace file rất to (~600MB/rank) và flush BẤT ĐỒNG BỘ nhiều phút sau /stop_profile;
# /stop_profile có thể deadlock nhưng file vẫn được ghi. Chờ tới khi đủ ${need} file VÀ
# tổng bytes KHÔNG đổi (đã đóng file) — nếu chỉ đếm file sẽ kill khi file đang ghi dở -> trace hỏng.
wait_for_traces() {  # <trace_dir> <need> <timeout_s>
    local dir="$1" need="$2" timeout="$3" t=0 prevcnt=-1 prevsize=-1 stable=0 cur cursize
    local find_gz=(find "${dir}" -maxdepth 1 -name 'dp*_rank0.*.pt.trace.json.gz')
    echo "  [trace] chờ >=${need} file dp*_rank0.*.pt.trace.json.gz trong ${dir} (timeout ${timeout}s)..."
    while (( t < timeout )); do
        cur=$("${find_gz[@]}" 2>/dev/null | wc -l)
        cursize=$("${find_gz[@]}" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s}')
        if (( cur >= need )); then
            if (( cur == prevcnt && cursize == prevsize )); then
                (( ++stable >= 3 )) && break    # đủ file + bytes đứng yên ~6s -> đã flush xong
            else stable=0; fi
        fi
        prevcnt=${cur}; prevsize=${cursize}; sleep 2; (( t += 2 ))
    done
    cur=$("${find_gz[@]}" 2>/dev/null | wc -l)
    if (( cur >= need )); then echo "  [trace] OK: ${cur}/${need} trace (đã ổn định)."; return 0
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

# ---------- 1 phase TIME: serve(cudagraph/enforce_eager=false,profiler,frontend ON) -> bench --profile -> wait traces -> analyze ----------
do_time_phase() {  # <out_dir> <base_preset> <label> <model_path> <dpath> <osl> <conc> <rate>
    local out="$1" base="$2" label="$3" model="$4" dpath="$5" osl="$6" conc="$7" rate="$8"
    local np; np="$(clamp "${conc}" "${NP_FLOOR}" "${NP_CAP}")"           # profile: ppc LUÔN = 1
    echo "  --- [TIME] serve(enforce_eager=false,profiler,api_server_count=${API_SERVER_COUNT_PROFILE:-preset}) -> bench --profile(np=${np}) -> wait ${EXPECT_RANKS} trace -> analyze ---"
    local trace="${out}/traces"; mkdir -p "${trace}"
    local preset="${out}/preset.yaml"; patch_time_preset "${base}" "${preset}" "${trace}"
    local serve_log="${out}/serve.log"
    is_enabled "${DRY_RUN:-0}" && { echo "  [DRY] time preset -> ${preset}"; return 0; }
    start_server "${preset}" "${serve_log}" || { kill_server; echo "  [TIME][SKIP] serve fail." >&2; return 1; }
    # Chạy bench --profile Ở NỀN: với api_server_count mặc định (=dp) thì /stop_profile có
    # thể DEADLOCK (client treo, worker spin) DÙ 8 trace đã được ghi. Nên KHÔNG block vào
    # bench; poll trace_dir bằng wait_for_traces rồi kill cả client treo lẫn server.
    run_bench 1 "${out}/bench" "${label}" "${model}" "${dpath}" "${osl}" "${conc}" "${np}" "${rate}" &
    local bench_pid=$!
    wait_for_traces "${trace}" "${EXPECT_RANKS}" "${TRACE_WAIT_TIMEOUT}"  # CHỜ đủ trace (đã flush xong) trước khi kill
    kill "${bench_pid}" 2>/dev/null; wait "${bench_pid}" 2>/dev/null      # bench treo trên /stop_profile -> kill
    kill_server
    is_enabled "${ANALYZE}" || return 0
    echo "  [analyze] time -> ${out}/analysis"
    "${PYTHON}" "${SCRIPT_DIR}/analyze_time.py" --trace-dir "${trace}" --out "${out}/analysis" \
        --drop-head "${DROP_HEAD}" --rel-thr "${REL_THR}" --thr-us "${THR_US}" --gantt-max-figs "${GANTT_MAX_FIGS}" \
        2>&1 | tee "${out}/analyze_time.log" || echo "  [analyze][WARN] time lỗi." >&2
}

# PREFLIGHT (chỉ khi chạy phase token): phase TOKEN cần dòng [EP_COLLECT] do moreh patch
# trong vllm/.../fp8.py sinh ra (gated VLLM_MOREH_EP_LOG). Container build mới KHÔNG có patch
# này -> analyze_tokens.py báo "No [EP_COLLECT] lines found". Tự áp patch (idempotent) cho chắc.
if [[ ",${PHASES}," == *",token,"* ]] && ! is_enabled "${DRY_RUN:-0}"; then
    echo "[preflight] đảm bảo [EP_COLLECT] patch trong installed vllm (cho phase token)..."
    "${PYTHON}" "${SCRIPT_DIR}/apply_ep_collect_patch.py" || \
        echo "[preflight][WARN] áp [EP_COLLECT] patch thất bại — phase token có thể thiếu log." >&2
fi

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
