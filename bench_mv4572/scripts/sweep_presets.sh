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
PDIR="${PDIR:-${AUTO_ROOT}/presets/kimi2.6.mxfp4/dp8ep8}"   # override for mtp family etc.
SWEEP_ROOT="${SWEEP_ROOT:-${TICKET_DIR}/logs/sweep_fixed_ipc_tune_result/$(date +%Y%m%d_%H%M%S)}"

SWEEP_MODE="${SWEEP_MODE:-on}"   # on = 1 serve/preset (gộp list) | off = fresh mỗi (conc,ds)

# Concurrencies (parallel_threads) + datasets (ISL buckets). Ở mode=on, cả 2 list được
# nối bằng dấu phẩy và đẩy 1 lần vào multi_process_test.py (REBENCH_CONC/REBENCH_DATASETS).
CONCS=(${CONCS:-16 64 128 256})
DATASETS=(${DATASETS:-8k 10k 100k})     # buckets dưới /remote/vast0/share-mv/longbenchv2-custom (8k|10k|100k|1M)

# Chỉ nixl chạy được với Kimi-MXFP4 (gloo/nccl/pynccl chết ở profile all_gather float4). Chi tiết:
# eplb_src/mxfp4-communicator-support.md. Regenerate presets: bash ../gen_eplb_presets.sh
if bash "${SCRIPT_DIR}/check_nixl.sh"; then
    echo "[sweep] nixl VIABLE"
else
    echo "[sweep] WARNING: nixl NOT viable here -> preset nixl sẽ fail."
fi

# --- PRESET để TUNE (mỗi dòng 1 preset -> dễ comment để bật/tắt) ---
# Tune num_redundant_experts {0,8,16,32} + base (no-EPLB reference). Comment dòng để bỏ.
# PRESET_LIST env (space-separated) overrides the tuned list below — e.g. run only the
# no-EPLB base reference: PRESET_LIST="base.yaml" bash sweep_presets.sh
if [ -n "${PRESET_LIST:-}" ]; then
    PRESETS=(${PRESET_LIST})
else
PRESETS=(
    # base.yaml [0.24.0: base chờ PR #446]                                 # no-EPLB reference (uncomment to include)

    base-eplb-nixl-async-default-r0.yaml        # default interval, r0
    base-eplb-nixl-async-default-r8.yaml        # default interval, r8
    base-eplb-nixl-async-default-r16.yaml       # default interval, r16

    # base-eplb-nixl-async-s250-r0.yaml           # step 250,  r0
    # base-eplb-nixl-async-s250-r8.yaml           # step 250,  r8
    # base-eplb-nixl-async-s500-r0.yaml           # step 500,  r0
    # base-eplb-nixl-async-s500-r8.yaml           # step 500,  r8
    # base-eplb-nixl-async-s1000-r0.yaml          # step 1000, r0
    # base-eplb-nixl-async-s1000-r8.yaml          # step 1000, r8
    # base-eplb-nixl-async-s2000-r0.yaml          # step 2000, r0
    # base-eplb-nixl-async-s2000-r8.yaml          # step 2000, r8
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
        # 1 serve/preset, đẩy TOÀN BỘ list -> multi_process_test.py bench hết trong 1 phiên.
        echo "=================== ${name} @ concs=${CONCS_CSV} ds=${DS_CSV} (1 serve) ==================="
        wait_gpu_free
        RUN="${SWEEP_ROOT}/${name}" PRESET="${PDIR}/${p}" \
            REBENCH_CONC="${CONCS_CSV}" REBENCH_DATASETS="${DS_CSV}" \
            bash "${SCRIPT_DIR}/run_and_bench.sh" \
            || echo "[sweep] ${name} FAILED -> next"
    else
        # off: mỗi (conc, dataset) = 1 serve->bench->stop RIÊNG (fresh, cold EPLB).
        for c in "${CONCS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                echo "=================== ${name} @ conc ${c} ds ${ds} (fresh) ==================="
                wait_gpu_free
                RUN="${SWEEP_ROOT}/${name}/conc${c}/${ds}" PRESET="${PDIR}/${p}" \
                    REBENCH_CONC="${c}" REBENCH_DATASETS="${ds}" \
                    bash "${SCRIPT_DIR}/run_and_bench.sh" \
                    || echo "[sweep] ${name} conc${c} ds${ds} FAILED -> next"
            done
        done
    fi
done

echo "[sweep] DONE -> ${SWEEP_ROOT}"
echo "[sweep] summaries:"
find "${SWEEP_ROOT}" -name scenario_summary.csv 2>/dev/null | while read -r f; do echo "  ${f}"; done

