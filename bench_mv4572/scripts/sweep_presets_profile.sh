#!/usr/bin/env bash
# Sweep presets qua run_and_bench.sh. Có 2 CHẾ ĐỘ (flag SWEEP_MODE):
#
#   SWEEP_MODE=on  (mặc định) — TUNE: MỖI PRESET chỉ 1 serve->bench->stop. Toàn bộ list
#       (CONCS x DATASETS) được đẩy 1 lần vào multi_process_test.py (nó tự duyệt
#       for dataset: for conc: warmup->bench). => 1 lần khởi động server (model load +
#       CUDA graph) cho cả loạt scenario; server + EPLB GIỮ ẤM xuyên suốt (amortize startup,
#       nhanh để tune). LƯU Ý: EPLB không cold mỗi scenario -> số đo phản ánh trạng thái ấm.
#
#   SWEEP_MODE=off — mỗi (preset, conc, dataset) là 1 serve->bench->stop RIÊNG trên server
#       MỚI => EPLB luôn cold, không rò state chéo scenario. Chậm (khởi động lại mỗi lần).
#
#   bash sweep_presets.sh                          # mode=on
#   SWEEP_MODE=off bash sweep_presets.sh           # mode cũ (fresh mỗi scenario)
#   CONCS="16 64 128" DATASETS="8k 10k" bash sweep_presets.sh
# Env passthrough tới run_and_bench.sh: MODEL PORT SERVE_SH SERVER_WAIT_TIMEOUT
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TICKET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_ROOT="$(cd "${TICKET_DIR}/.." && pwd)"
PDIR="${PDIR:-${AUTO_ROOT}/presets/kimi2.6.mxfp4/dp8ep8_profile}"
SWEEP_ROOT="${SWEEP_ROOT:-${TICKET_DIR}/logs/sweep_profile/$(date +%Y%m%d_%H%M%S)}"

SWEEP_MODE="${SWEEP_MODE:-on}"   # on = 1 serve/preset (gộp list) | off = fresh mỗi (conc,ds)

# Concurrencies (parallel_threads) + datasets (ISL buckets). Ở mode=on, cả 2 list được
# nối bằng dấu phẩy và đẩy 1 lần vào multi_process_test.py (REBENCH_CONC/REBENCH_DATASETS).
# PROFILE: mặc định 1 scenario/preset -> start_profile..stop_profile bọc ĐÚNG 1 cửa sổ ->
# đúng 8 file trace (1/DP rank) trong profiling_result. Nhiều scenario sẽ dump nhiều bộ
# trace lẫn lộn -> phân tích sai; nếu cần thì override CONCS/DATASETS (và EXPECT_TRACES=8*N).
CONCS=(${CONCS:-64})
DATASETS=(${DATASETS:-10k})              # buckets dưới /remote/vast0/share-mv/longbenchv2-custom (8k|10k|100k|1M)

# Chỉ nixl chạy được với Kimi-MXFP4 (gloo/nccl/pynccl chết ở profile all_gather float4). Chi tiết:
# eplb_src/mxfp4-communicator-support.md. Regenerate presets: bash ../gen_eplb_presets.sh
if bash "${SCRIPT_DIR}/check_nixl.sh"; then
    echo "[sweep] nixl VIABLE"
else
    echo "[sweep] WARNING: nixl NOT viable here -> preset nixl sẽ fail."
fi

# --- PRESET để TUNE (mỗi dòng 1 preset -> dễ comment để bật/tắt) ---
# Tune num_redundant_experts {0,8,16,32} + base (no-EPLB reference). Comment dòng để bỏ.
# PRESET_LIST env (space-separated) overrides — vd chọn r0 profile:
#   PRESET_LIST="base-eplb-nixl-async-default-r0.yaml" CONCS="256" DATASETS="100k" bash sweep_presets_profile.sh
if [ -n "${PRESET_LIST:-}" ]; then
    PRESETS=(${PRESET_LIST})
else
PRESETS=(
    base.yaml                                 # no-EPLB reference (uncomment to include)

    # base-eplb-nixl-async-default-r0.yaml        # default interval, r0
    # base-eplb-nixl-async-default-r8.yaml        # default interval, r8
    # base-eplb-nixl-async-default-r16.yaml       # default interval, r16
)
fi

wait_gpu_free() {
    local iv="${GPU_POLL_INTERVAL:-30}" thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" n
    while :; do
        n=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' | awk -v t="${thr}" '$1 > t' | wc -l)
        [ "${n}" -eq 0 ] && { echo "[sweep] GPUs free"; return 0; }
        echo "[sweep] ${n} GPU(s) busy; retry in ${iv}s"; sleep "${iv}"
    done
}

# [profile] Chờ profiler dump XONG đủ trace file. dp8 /stop_profile có thể ném AssertionError
# (500 "Profiler must be initialized") nhưng các worker VẪN dump trace -> ta KỆ lỗi đó, chỉ
# chờ tới khi đủ ${want} file VÀ tổng size ổn định (ngừng tăng) => trace đã ghi xong. Phải
# chờ TRƯỚC khi kill server (kill sớm sẽ cắt cụt file trace).
wait_for_traces() {
    local dir="$1" want="${2:-8}" iv="${TRACE_POLL:-15}" need_stable="${TRACE_STABLE:-2}"
    local max="${TRACE_TIMEOUT:-1800}" prev=-1 same=0 t=0 n sz
    echo "[profile] chờ ${want} trace file trong ${dir} (KỆ AssertionError của /stop_profile) ..."
    while :; do
        n=$(find "${dir}" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null | wc -l)
        sz=$(find "${dir}" -maxdepth 1 -name '*.pt.trace.json*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
        if [ "${n}" -ge "${want}" ]; then
            if [ "${sz}" = "${prev}" ]; then same=$((same+1)); else same=0; fi
            [ "${same}" -ge "${need_stable}" ] && { echo "[profile] OK: ${n} trace file, size ổn định ${sz}B"; return 0; }
        fi
        prev="${sz}"; sleep "${iv}"; t=$((t+iv))
        echo "[profile]   traces=${n}/${want} size=${sz}B (đã chờ ${t}s)"
        [ "${t}" -ge "${max}" ] && { echo "[profile] WARN: hết ${max}s mà chỉ có ${n}/${want} trace"; return 1; }
    done
}

# [profile] Serve 1 preset -> bench -> chờ trace -> stop. profiler_path (torch_profiler_dir)
# nằm ĐÚNG trong logs của run này: ${RUN}/profiling_result. Preset đã có sẵn profiler_config
# với placeholder __RUN_PROFILER_DIR__; ở đây ta sed thay bằng path thật (yêu cầu #2). Giữ
# server sống qua run_and_bench (KEEP_SERVER=1) để trace không bị cắt cụt; chờ đủ trace
# (KỆ AssertionError của /stop_profile, yêu cầu #3) rồi mới kill.
run_profile() {
    local RUN="$1" src_preset="$2" conc_csv="$3" ds_csv="$4"
    local PROFILER_DIR="${RUN}/profiling_result"
    mkdir -p "${PROFILER_DIR}"
    local served="${RUN}/preset.profile.yaml"
    sed "s#__RUN_PROFILER_DIR__#${PROFILER_DIR}#g" "${src_preset}" > "${served}"
    echo "[profile] profiler dir = ${PROFILER_DIR}"
    RUN="${RUN}" PRESET="${served}" KEEP_SERVER=1 \
        REBENCH_CONC="${conc_csv}" REBENCH_DATASETS="${ds_csv}" \
        EPLB_COLLECT_DIR="${EPLB_COLLECT_DIR:-}" \
        bash "${SCRIPT_DIR}/run_and_bench.sh" \
        || echo "[profile] run_and_bench exited nonzero (AssertionError KỆ - vẫn chờ trace)"
    if [ "${BENCH_ONLY:-0}" = "1" ]; then
        # [ABLATION] tps thuần: bench đã chạy trong run_and_bench (Summary ở mpt_run.log), preset
        # KHÔNG profiler -> không trace/collect/gridlock -> chỉ kill. So r0 vs base bằng
        # mean_decode_tps / mean_tpot trong [Summary].
        echo "[bench] BENCH_ONLY done -> Summary trong ${RUN}/mpt_run.log ; results -> ${RUN}/results/"
        grep -hE "\[Summary\]" "${RUN}/mpt_run.log" 2>/dev/null | tail -1 || true
    elif [ "${COLLECT_ONLY:-0}" = "1" ]; then
        # [MV-4572] OPTION A (preset r0_collect, KHÔNG profiler): sau bench 8 DP engine RẢNH
        # (không export trace -> KHÔNG có shm_broadcast gridlock). POST /collect_eplb 1 lần ->
        # internal-LB DPLBAsyncMPClient broadcast collective_rpc tới CẢ 8 engine -> 8 npz.
        # (Đây là cách reset_eplb đã chạy được 8 rank; blocker duy nhất trước đây là profiler.)
        if [ -n "${EPLB_COLLECT_DIR:-}" ]; then
            echo "[collect] POST /collect_eplb (8 rank) -> ${EPLB_COLLECT_DIR}"
            curl -sS -m "${EPLB_COLLECT_TIMEOUT:-1800}" -X POST \
                "http://localhost:${PORT:-8000}/collect_eplb" \
                -H 'Content-Type: application/json' \
                -d "{\"out_dir\": \"${EPLB_COLLECT_DIR}\"}" \
                && echo || echo "[collect] curl FAILED (xem serve.log: collect_eplb_buffer CALLED/WROTE)"
            nnpz=$(find "${EPLB_COLLECT_DIR}" -name '*.npz' 2>/dev/null | wc -l)
            echo "[collect] npz -> ${nnpz}/8 tại ${EPLB_COLLECT_DIR}"
            [ "${nnpz}" -lt 8 ] && echo "[collect] WARN: thiếu npz (xem serve.log: [EPLB][MV-4572] collect_eplb_buffer WROTE)"
        fi
    else
        wait_for_traces "${PROFILER_DIR}" "${EXPECT_TRACES:-8}"
        # [MV-4572] EPLB token buffer được dump TỰ ĐỘNG trong từng worker ngay sau khi profiler ghi
        # trace (gpu_worker.profile(is_start=False) -> collect_eplb_buffer), KHÔNG qua /collect_eplb
        # (collective_rpc bị gridlock shm_broadcast hậu-profiling -> treo, chỉ 1 rank kịp). npz nằm
        # cạnh profiling_result. Dump chạy NGAY SAU trace-write nên hơi trễ hơn trace -> chờ đủ 8 npz
        # rồi mới kill (kill sớm sẽ cắt cụt file npz đang ghi).
        if [ -n "${EPLB_COLLECT_DIR:-}" ]; then
            want="${EXPECT_TRACES:-8}"; nnpz=0
            for _i in $(seq 1 "${NPZ_WAIT_POLLS:-40}"); do
                nnpz=$(find "${EPLB_COLLECT_DIR}" -name '*.npz' 2>/dev/null | wc -l)
                [ "${nnpz}" -ge "${want}" ] && break
                sleep "${NPZ_WAIT_IV:-5}"
            done
            echo "[profile] EPLB npz (auto per-worker) -> ${nnpz}/${want} tại ${EPLB_COLLECT_DIR}"
            [ "${nnpz}" -lt "${want}" ] && echo "[profile] WARN: thiếu npz (xem serve.log: [EPLB][MV-4572] collect_eplb_buffer WROTE)"
        fi
    fi
    echo "[profile] stop server (${RUN})"; pkill -9 VLLM 2>/dev/null; sleep 3
}

# nối array bằng dấu phẩy (multi_process_test.py split(",")).
CONCS_CSV="$(IFS=,; echo "${CONCS[*]}")"
DS_CSV="$(IFS=,; echo "${DATASETS[*]}")"

mkdir -p "${SWEEP_ROOT}"
echo "[sweep] MODE=${SWEEP_MODE} | ${#PRESETS[@]} presets x concs(${CONCS[*]}) x datasets(${DATASETS[*]}) -> ${SWEEP_ROOT}"

for p in "${PRESETS[@]}"; do
    [ -f "${PDIR}/${p}" ] || { echo "[sweep] SKIP missing ${p}"; continue; }
    name="${p%.yaml}"

    # [supervisor-resume] Idempotent resume: skip a preset that already produced a
    # scenario_summary.csv under SWEEP_ROOT (i.e. it finished). No-op for the default
    # fresh-timestamp SWEEP_ROOT; lets a restart after an interruption continue from
    # the unfinished preset instead of redoing the whole list.
    if [ -f "${SWEEP_ROOT}/${name}/scenario_summary.csv" ]; then
        echo "[sweep] SKIP ${name} (already complete in ${SWEEP_ROOT})"; continue
    fi

    if [ "${SWEEP_MODE}" = "on" ]; then
        # 1 serve/preset, đẩy list -> multi_process_test.py bench trong 1 phiên (PROFILE).
        echo "=================== ${name} @ concs=${CONCS_CSV} ds=${DS_CSV} (1 serve, PROFILE) ==================="
        wait_gpu_free
        run_profile "${SWEEP_ROOT}/${name}" "${PDIR}/${p}" "${CONCS_CSV}" "${DS_CSV}"
    else
        # off: mỗi (conc, dataset) = 1 serve->bench->stop RIÊNG (fresh, cold EPLB).
        for c in "${CONCS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                echo "=================== ${name} @ conc ${c} ds ${ds} (fresh, PROFILE) ==================="
                wait_gpu_free
                run_profile "${SWEEP_ROOT}/${name}/conc${c}/${ds}" "${PDIR}/${p}" "${c}" "${ds}"
            done
        done
    fi
done

echo "[sweep] DONE -> ${SWEEP_ROOT}"
echo "[sweep] summaries:"
find "${SWEEP_ROOT}" -name scenario_summary.csv 2>/dev/null | while read -r f; do echo "  ${f}"; done

