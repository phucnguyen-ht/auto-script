#!/usr/bin/env bash
set -uo pipefail
# MV-4571: PROFILE-ONLY (tách khỏi serve).
# = auto_bench.sh chạy ở MODE=profile nhưng KHÔNG tự dựng/kill server (AUTO_SERVE=0),
# nên dùng Y NGUYÊN load_scenarios/run_one/aggregate + harvest traces của auto_bench
# -> knobs giống hệt phase profile (custom dataset, --ignore-eos, --skip-chat-template,
# --metric-percentiles 75,90,99, --request-rate inf, --profile). Chỉ khác: bench lên
# server ĐÃ chạy sẵn (bật bằng serve_profile.sh).
#
# Dùng:
#   bash bench_mv4571/serve_profile.sh      # cửa sổ 1: bật server (giữ chạy)
#   bash bench_mv4571/profile_only.sh       # cửa sổ 2: chạy profile + harvest
#
# Scenarios/runs đọc từ env.yaml .profile.custom (giống auto_bench). Trace được dump
# vào PROFILER_DIR mà server đã cấu hình lúc serve (đọc từ .profile_session.env).

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${PROFILE_SESSION:-${TICKET_DIR}/.profile_session.env}"
if [ -f "${SESSION}" ]; then
    # nạp RUN_DIR / PROFILER_DIR / BASE_URL / MODEL_PATH do serve_profile.sh ghi
    # shellcheck disable=SC1090
    source "${SESSION}"
    echo "[profile_only] session: ${SESSION}"
    echo "[profile_only]   BASE_URL=${BASE_URL:-} PROFILER_DIR=${PROFILER_DIR:-}"
else
    echo "[profile_only][WARN] không thấy ${SESSION}." >&2
    echo "  -> chạy serve_profile.sh trước, hoặc tự set BASE_URL/PROFILER_DIR/MODEL_PATH." >&2
fi

# Bench server đang chạy, KHÔNG serve/kill. PROFILER_DIR (từ session) để run_one harvest đúng chỗ.
export MODE=profile
export AUTO_SERVE=0
export PRESET="${PRESET:-glm5.2/dp8ep8/noMTP-bs64-dg.yaml}"
export BASE_URL="${BASE_URL:-}"          # rỗng -> auto_bench tự dùng http://localhost:8000
export MODEL_PATH="${MODEL_PATH:-}"      # rỗng -> auto_bench tự resolve theo preset family
export PROFILER_DIR="${PROFILER_DIR:-}"  # rỗng -> không harvest (trace vẫn nằm ở dir server cấu hình)
# Ghi kết quả bench/agg VÀO ĐÚNG RUN_DIR của serve (cùng folder serve.log + traces).
export RUN_DIR_OVERRIDE="${RUN_DIR:-}"

if [ -z "${PROFILER_DIR}" ]; then
    echo "[profile_only][WARN] PROFILER_DIR rỗng -> traces sẽ KHÔNG được gom về run dir." >&2
fi

exec bash "${TICKET_DIR}/auto_bench.sh" "$@"
