#!/usr/bin/env bash
# Shared, ticket-agnostic helpers. Source this from an auto_*.sh script.
#
# Resolves common paths and provides backend/model/preset/log helpers plus
# serve/kill/wait functions. Callers (run_all.sh) may pre-set ENV_YAML,
# LOG_ROOT, DATA_DIR, BACKEND, PRESET_YAML, MODEL_FAMILY to steer resolution.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"
ENV_YAML="${ENV_YAML:-${AUTO_ROOT}/env.yaml}"
PRESETS_DIR="${PRESETS_DIR:-${AUTO_ROOT}/presets}"
SERVE_SH="${SERVE_SH:-${AUTO_ROOT}/serve.sh}"
# sglang serve script is ticket-specific; the ticket's run_all.sh exports it.
SERVE_SGLANG_SH="${SERVE_SGLANG_SH:-}"
LOG_ROOT="${LOG_ROOT:-${AUTO_ROOT}/logs}"
DATA_DIR="${DATA_DIR:-${AUTO_ROOT}}"
# Define (empty) so `${BACKEND,,}` is safe under `set -u` before resolve_backend.
: "${BACKEND:=}"

ensure_yq() {
    command -v yq >/dev/null 2>&1 && return 0
    local ver="v4.50.1" sha="c7a1278e6bbc4924f41b56db838086c39d13ee25dcb22089e7fbf16ac901f0d4"
    echo "Installing yq ${ver}..."
    curl -fsSL -o /usr/local/bin/yq \
        "https://github.com/mikefarah/yq/releases/download/${ver}/yq_linux_amd64"
    echo "${sha}  /usr/local/bin/yq" | sha256sum -c -
    chmod 0755 /usr/local/bin/yq
}
ensure_yq

# Deep-merge the global env.yaml (base) with the ticket's ENV_YAML (override):
# arrays replace, nested maps merge, base values are inherited. Reads then use
# the merged result, so a ticket env.yaml only needs the keys it changes.
BASE_ENV_YAML="${BASE_ENV_YAML:-${AUTO_ROOT}/env.yaml}"
if [ -n "${ENV_YAML}" ] && [ "${ENV_YAML}" != "${BASE_ENV_YAML}" ] \
   && [ -f "${ENV_YAML}" ] && [ -f "${BASE_ENV_YAML}" ]; then
    _merged="$(mktemp /tmp/env-merged.XXXXXX.yaml)"
    yq ea '. as $i ireduce ({}; . * $i)' "${BASE_ENV_YAML}" "${ENV_YAML}" > "${_merged}"
    ENV_YAML="${_merged}"
fi

# abspath <path> — leave absolute/empty as-is, else resolve relative to AUTO_ROOT.
abspath() { case "$1" in /*|"") printf '%s' "$1" ;; *) printf '%s' "${AUTO_ROOT}/$1" ;; esac; }

is_enabled() { case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac; }

# yaml_get <expr> [default] — empty/null falls back to default.
yaml_get() {
    local v; v="$(yq e "$1" "${ENV_YAML}" 2>/dev/null)"
    { [ -z "$v" ] || [ "$v" = "null" ]; } && v="${2:-}"
    printf '%s' "$v"
}

# eval_runs <dataset> — integer run count from eval.datasets.<k>.runs (0 if unset).
eval_runs() {
    local v; v="$(yaml_get ".eval.datasets.$1.runs" 0)"
    [[ "$v" =~ ^[0-9]+$ ]] || v=0
    printf '%s' "$v"
}

# backends_list — space-separated backend names from env.yaml (default: vllm).
backends_list() {
    local b; b="$(yq e '.backends[].name' "${ENV_YAML}" 2>/dev/null | tr '\n' ' ')"
    [ -z "${b// /}" ] && b="vllm"
    printf '%s' "$b"
}

# backend_field <name> <field> — a field of the .backends[] entry, or "".
backend_field() {
    yq e ".backends[] | select(.name == \"$1\") | .$2 // \"\"" "${ENV_YAML}" 2>/dev/null | head -n1
}

# preset_family — top-level presets/ subfolder of PRESET_YAML (e.g. glm5).
preset_family() {
    [ -z "${PRESET_YAML:-}" ] && return 0
    local abs rel pdir
    pdir="$(cd "${PRESETS_DIR}" && pwd)"
    abs="$(cd "$(dirname "${PRESET_YAML}")" && pwd)/$(basename "${PRESET_YAML}")"
    rel="${abs#"${pdir}/"}"
    [ "$rel" = "$abs" ] && return 0
    [[ "$rel" == */* ]] && printf '%s' "${rel%%/*}"
}

resolve_backend() {
    BACKEND="${BACKEND:-$(backends_list | awk '{print $1}')}"
    case "${BACKEND,,}" in
        vllm)   SERVER_PORT="${SERVER_PORT:-8000}" ;;
        sglang)
            SERVER_PORT="${SERVER_PORT:-${SGLANG_PORT:-30000}}"
            # sglang has no preset: take its serve script + model family from
            # the env.yaml backends entry (serve_script relative to auto-script/).
            SERVE_SGLANG_SH="${SERVE_SGLANG_SH:-$(abspath "$(backend_field sglang serve_script)")}"
            MODEL_FAMILY="${MODEL_FAMILY:-$(backend_field sglang model)}"
            ;;
        *) echo "[ERROR] BACKEND must be 'vllm' or 'sglang' (got: ${BACKEND})" >&2; exit 1 ;;
    esac
    BASE_URL="${BASE_URL:-http://localhost:${SERVER_PORT}}"
}

resolve_model_path() {
    [ -n "${MODEL_PATH:-}" ] && return 0
    local fam="${MODEL_FAMILY:-}"
    if [ -z "$fam" ]; then
        if [ "${BACKEND,,}" = "sglang" ]; then
            echo "[ERROR] sglang: set the backend's 'model' in env.yaml (or MODEL_PATH)" >&2
            exit 1
        fi
        fam="$(preset_family)"
    fi
    # Bracket-index so family keys with dots (e.g. kimi2.6) resolve correctly.
    MODEL_PATH="$(yaml_get ".model.paths[\"${fam}\"]")"
    if [ -z "${MODEL_PATH}" ]; then
        echo "[ERROR] no model.paths.${fam:-<unknown>} in ${ENV_YAML}" >&2
        exit 1
    fi
}

# resolve_preset_name — sets PRESET_NAME (keys the log dir), mirroring the
# preset's path under presets/ (e.g. deepseek/dp8ep8/foo), .yaml stripped.
resolve_preset_name() {
    [ -n "${PRESET_NAME:-}" ] && return 0
    if [ "${BACKEND,,}" = "sglang" ]; then PRESET_NAME="sglang"; return 0; fi
    if [ -n "${PRESET_YAML:-}" ]; then
        local abs rel pdir
        pdir="$(cd "${PRESETS_DIR}" && pwd)"
        abs="$(cd "$(dirname "${PRESET_YAML}")" && pwd)/$(basename "${PRESET_YAML}")"
        rel="${abs#"${pdir}/"}"
        [ "$rel" = "$abs" ] && rel="$(basename "$abs")"
        PRESET_NAME="${rel%.yaml}"
    else
        PRESET_NAME="default"
    fi
}

# setup_run_dir <phase> — sets RUN_DIR=<LOG_ROOT>/<preset>/<phase>/<ts> (sglang
# variant under <LOG_ROOT>_sglang) and creates it.
setup_run_dir() {
    local phase="$1" base
    if [ "${BACKEND,,}" = "sglang" ]; then
        base="${LOG_ROOT}_sglang/${phase}"
    else
        resolve_preset_name
        base="${LOG_ROOT}/${PRESET_NAME}/${phase}"
    fi
    RUN_DIR="${base}/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_DIR}"
}

wait_for_gpu_free() {
    local check_interval="${GPU_POLL_INTERVAL:-30}" elapsed=0
    echo "[gpu-wait] Waiting for all GPUs to be free (poll every ${check_interval}s)..."
    while true; do
        local busy_count
        # Idle GPUs holding a model report 0% util, so gate on VRAM allocation.
        busy_count=$(rocm-smi --showmemuse 2>/dev/null \
            | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' \
            | awk -v thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" '$1 > thr' | wc -l)
        if [ "${busy_count}" -eq 0 ]; then
            echo "[gpu-wait] All GPUs free after ${elapsed}s."; return 0
        fi
        echo "[gpu-wait] ${busy_count} GPU(s) busy (${elapsed}s). Retrying in ${check_interval}s..."
        sleep "${check_interval}"; elapsed=$((elapsed + check_interval))
    done
}

wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}" elapsed=0
    echo "[wait] Polling ${BASE_URL}/health (timeout: ${max_wait}s)..."
    while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
        sleep 10; elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "[ERROR] Server did not respond within ${max_wait}s"; return 1
        fi
        echo "[wait] ${elapsed}s elapsed, still waiting..."
    done
    echo "[wait] Server ready after ${elapsed}s"
}

kill_server() {
    if [ "${BACKEND,,}" = "sglang" ]; then
        # pkill without -f matches comm, killing sglang server+workers but not
        # our bash/tee wrappers.
        echo "[kill] SIGKILL SGLang..."; pkill -9 sglang 2>/dev/null || true
    else
        echo "[kill] SIGKILL VLLM..."; pkill -9 VLLM 2>/dev/null || true
    fi
    sleep 15; echo "[kill] Done."
}

# serve_backend <log> — start the backend in the background. vllm uses
# PRESET_YAML; sglang uses serve_sglang_ds3.2.sh.
serve_backend() {
    local log="$1"
    if [ "${BACKEND,,}" = "sglang" ]; then
        [ -f "${SERVE_SGLANG_SH}" ] || { echo "[ERROR] SERVE_SGLANG_SH not set/found: '${SERVE_SGLANG_SH}'" >&2; exit 1; }
        echo "[serve] SGLang (model: ${MODEL_PATH}, port: ${SERVER_PORT}) -> ${log}"
        (SGLANG_PORT="${SERVER_PORT}" bash "${SERVE_SGLANG_SH}" "${MODEL_PATH}") >"${log}" 2>&1 &
    else
        echo "[serve] vLLM (preset: ${PRESET_YAML}) -> ${log}"
        (bash "${SERVE_SH}" "${MODEL_PATH}" "${PRESET_YAML}") >"${log}" 2>&1 &
    fi
    echo "[serve] PID: $!"
}
