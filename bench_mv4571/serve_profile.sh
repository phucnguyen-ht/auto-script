#!/usr/bin/env bash
set -uo pipefail
# MV-4571: SERVE-ONLY cho profiling (tách khỏi auto_profile.sh).
# Khởi động server vLLM với torch `profiler_config` được inject y hệt
# auto_bench_template.sh (MODE=profile), rồi GIỮ SERVER CHẠY. Sau đó chạy
# profile_only.sh để capture trace (giống cặp serve-up + bench_only.sh).
#
# Dùng:
#   PRESET=glm5.2/dp8ep8/noMTP-bs64-dg.yaml bash bench_mv4571/serve_profile.sh
#   # dừng server:  pkill -9 VLLM
#
# Ghi 1 session file (.profile_session.env) chứa RUN_DIR / PROFILER_DIR /
# BASE_URL / MODEL_PATH để profile_only.sh biết trace được dump vào đâu.
#
# Env tuỳ chọn: PRESET, SKIP_GPU_WAIT=1 (bỏ chờ GPU free), PROFILE_SESSION.

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${TICKET_DIR}/../common"
export ENV_YAML="${ENV_YAML:-${TICKET_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${TICKET_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${TICKET_DIR}}"
export PRESET="${PRESET:-glm5.2/dp8ep8/noMTP-bs64-dg.yaml}"
export MODE=profile      # để profiler_config_json đọc .profile.config.* đúng nhánh

source "${COMMON_DIR}/helper.sh"

resolve_backend
resolve_preset
resolve_model_path
setup_run_dir "auto_profile"          # RUN_DIR = logs/<preset>/auto_profile/<ts>

# --- inject profiler_config vào preset ---
# CHỦ Ý cho luồng _only: chỉ giữ 4 cờ torch profiler gốc, KHÔNG thêm
# "ignore_frontend":true (khác với profiler_config_json của auto_profile). Nghĩa là
# frontend (AsyncLLM) profiler vẫn bật -> nếu chạy với api_server_count>1 + dp>1 thì
# /stop_profile có thể 500 "Profiler must be initialized"; khi đó giảm api_server_count=1.
PROFILER_DIR="${RUN_DIR}/profiling_result"; mkdir -p "${PROFILER_DIR}"
served="${RUN_DIR}/preset.yaml"
PC="$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":"%s","torch_profiler_record_shapes":"%s","torch_profiler_with_memory":"%s","torch_profiler_with_flops":"%s"}' \
    "${PROFILER_DIR}" \
    "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_STACK' False)" \
    "$(yaml_get '.profile.config.TORCH_PROFILER_RECORD_SHAPES' False)" \
    "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_MEMORY' False)" \
    "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_FLOPS' False)")"
PC="${PC}" yq e ".engine_args.profiler_config = strenv(PC)" "${PRESET_YAML}" > "${served}"

# --- session file cho profile_only.sh (ghi TRƯỚC khi serve, mọi path đã biết) ---
SESSION="${PROFILE_SESSION:-${TICKET_DIR}/.profile_session.env}"
cat > "${SESSION}" <<EOF
RUN_DIR="${RUN_DIR}"
PROFILER_DIR="${PROFILER_DIR}"
BASE_URL="${BASE_URL}"
MODEL_PATH="${MODEL_PATH}"
PRESET_YAML="${served}"
SERVE_LOG="${RUN_DIR}/serve.log"
EOF

echo "=== serve-profile @ $(date) ==="
echo "preset       = ${PRESET_YAML}"
echo "model        = ${MODEL_PATH}"
echo "run dir      = ${RUN_DIR}"
echo "profiler dir = ${PROFILER_DIR}"
echo "serve log    = ${RUN_DIR}/serve.log"
echo "session      = ${SESSION}"
echo "profiler_config = $(yq e '.engine_args.profiler_config' "${served}")"
echo
echo ">>> Server chạy FOREGROUND ở cửa sổ này."
echo ">>> Khi log hiện 'Application startup complete.' -> mở cửa sổ khác chạy:"
echo ">>>     bash bench_mv4571/profile_only.sh"
echo ">>> Tắt server: Ctrl+C ở cửa sổ này (hoặc 'pkill -9 VLLM' từ cửa sổ khác)."
echo

# kill server cũ trước, rồi chờ GPU free (đảo thứ tự so với template để re-run an toàn).
kill_server
is_enabled "${SKIP_GPU_WAIT:-0}" || wait_for_gpu_free

# Chạy server Ở FOREGROUND (không '&'): tee ra serve.log để vừa xem trực tiếp vừa
# có file cho EP_COLLECT/parse. Ctrl+C ở đây sẽ tắt server.
bash "${SERVE_SH}" "${MODEL_PATH}" "${served}" 2>&1 | tee "${RUN_DIR}/serve.log"
