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

# yaml_list <expr> — space-separated items of a yaml sequence (empty if none).
yaml_list() { yq e "${1}[]" "${ENV_YAML}" 2>/dev/null | tr '\n' ' '; }

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

# readable_list — readable methods from env.yaml .eval.readable (default: all).
readable_list() {
    local r; r="$(yq e '.eval.readable[]' "${ENV_YAML}" 2>/dev/null | tr '\n' ' ')"
    [ -z "${r// /}" ] && r="completion chat pychat"
    printf '%s' "$r"
}

# phases_list — ordered phase names from env.yaml .phases (default: readable eval).
phases_list() {
    local p; p="$(yq e '.phases[]' "${ENV_YAML}" 2>/dev/null | tr '\n' ' ')"
    [ -z "${p// /}" ] && p="readable eval"
    printf '%s' "$p"
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

# resolve_preset — resolve PRESET (env or default) to an absolute path and set
# PRESET_YAML, PRESET_NAME, PRESET_FAMILY. Lookup order: $PRESET as-is, under
# presets/, then under DATA_DIR (the ticket dir). Skipped for sglang (no preset).
resolve_preset() {
    [ "${BACKEND,,}" = "sglang" ] && { PRESET_YAML=""; PRESET_NAME="${PRESET_NAME:-sglang}"; PRESET_FAMILY=""; return 0; }
    local def="${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml"
    local p="${PRESET:-${PRESET_YAML:-${def}}}"
    if [ -f "${p}" ]; then :
    elif [ -f "${PRESETS_DIR}/${p}" ]; then p="${PRESETS_DIR}/${p}"
    elif [ -f "${DATA_DIR}/${p}" ]; then p="${DATA_DIR}/${p}"
    else echo "[ERROR] preset not found: ${p}" >&2; exit 1; fi
    PRESET_YAML="$(cd "$(dirname "${p}")" && pwd)/$(basename "${p}")"
    PRESET="${PRESET_YAML}"
    PRESET_NAME=""; resolve_preset_name
    PRESET_FAMILY="$(preset_family)"
}

# setup_run_dir <phase> — sets RUN_DIR=<LOG_ROOT>/<preset>/<phase>/<ts> (sglang
# variant under <LOG_ROOT>_sglang) and creates it.
setup_run_dir() {
    local phase="$1" base
    # Cho phép tái dùng 1 RUN_DIR có sẵn (vd serve_profile.sh đã tạo) để serve +
    # bench/profile cùng ghi vào MỘT folder. Bỏ trống -> hành vi mặc định (tạo mới).
    if [ -n "${RUN_DIR_OVERRIDE:-}" ]; then
        RUN_DIR="${RUN_DIR_OVERRIDE}"; mkdir -p "${RUN_DIR}"; return 0
    fi
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

# wait_for_server — poll /health until ready or timeout. With SERVER_ERROR_DETECT=1
# (default) also aborts early (return 1) if the server process died (SERVER_PID) or
# its log (SERVER_LOG) matches SERVER_FATAL_RE — so a doomed preset frees the GPUs
# fast instead of waiting out SERVER_WAIT_TIMEOUT.
wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}" elapsed=0
    local detect="${SERVER_ERROR_DETECT:-1}" re="${SERVER_FATAL_RE:-}"
    echo "[wait] Polling ${BASE_URL}/health (timeout: ${max_wait}s)..."
    while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
        if is_enabled "${detect}"; then
            if [ -n "${SERVER_PID:-}" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                echo "[wait][ERROR] server process ${SERVER_PID} exited before healthy"
                [ -f "${SERVER_LOG:-}" ] && tail -n "${SERVER_ERROR_TAIL:-40}" "${SERVER_LOG}"
                return 1
            fi
            if [ -n "${re}" ] && [ -f "${SERVER_LOG:-}" ] && grep -qEi "${re}" "${SERVER_LOG}"; then
                echo "[wait][ERROR] fatal pattern in ${SERVER_LOG}:"
                grep -EiInm3 "${re}" "${SERVER_LOG}"
                return 1
            fi
        fi
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

# reset_prefix_cache — best-effort reset of the server's prefix/radix cache.
#   vllm   -> POST /reset_prefix_cache (server must run with VLLM_SERVER_DEV_MODE=1,
#             else the route is not mounted; auto_bench exports it before serving)
#   sglang -> POST /flush_cache
reset_prefix_cache() {
    local url="${BASE_URL%/}" ep
    [ "${BACKEND,,}" = "sglang" ] && ep="/flush_cache" || ep="/reset_prefix_cache"
    if curl -sf -X POST "${url}${ep}" -o /dev/null; then
        echo "  [reset_prefix_cache] POST ${url}${ep}"
    else
        echo "  [reset_prefix_cache][WARN] POST ${url}${ep} failed" >&2
    fi
}

# profiler_config_json <trace_dir> — moreh torch profiler_config (engine arg),
# reading flags from .profile.config.*. Inject into a preset before serving.
profiler_config_json() {
    local cfg
    cfg=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":"%s","torch_profiler_record_shapes":"%s","torch_profiler_with_memory":"%s","torch_profiler_with_flops":"%s"' \
        "$1" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_STACK' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_RECORD_SHAPES' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_MEMORY' False)" \
        "$(yaml_get '.profile.config.TORCH_PROFILER_WITH_FLOPS' False)")
    # We do NOT inject "ignore_frontend":true by default. With data_parallel + multiple
    # API servers, /start_profile and /stop_profile are load-balanced to different
    # front-ends, so the per-process front-end profiler's stop() raises a harmless
    # "Profiler must be initialized" 500. That 500 does NOT prevent the per-rank GPU
    # traces from being written -- and ignore_frontend does NOT prevent the (separate)
    # post-stop DP deadlock either -- so suppressing it only hides a cosmetic error
    # while changing the served config away from the real preset. The reliable fix is
    # to harvest by POLLING the trace dir (harvest_profiles) and kill the wedged client,
    # not to touch the preset. Set PROFILE_IGNORE_FRONTEND=1 to opt into the quiet
    # behavior (no front-end CPU trace, no 500).
    is_enabled "${PROFILE_IGNORE_FRONTEND:-0}" && cfg+=',"ignore_frontend":true'
    printf '%s}' "${cfg}"
}

# harvest_profiles <marker_file> <dest_dir> — torch profiler writes every capture
# to one fixed dir (set at server start), so scenarios/runs mix together. Call this
# right after a profiled bench invocation to move that scenario's freshly-written
# artifacts into <dest>. <marker_file> must be a file created just BEFORE the
# invocation (its mtime delimits "new" files). Waits for the async trace flush that
# happens on /stop_profile. No-op outside profile mode or when PROFILER_DIR is unset.
harvest_profiles() {
    local marker="$1" dest="$2"
    [ "${MODE:-bench}" = "profile" ] || return 0
    [ -n "${PROFILER_DIR:-}" ] && [ -d "${PROFILER_DIR}" ] || return 0
    # Per-rank trace files flush asynchronously AFTER /stop_profile. The export can take
    # MANY MINUTES for a large profiling window (a 4-min eager window -> ~645MB per rank,
    # ~8 min to write), and under DP + multiple API servers the bench client can WEDGE on
    # /stop_profile while the workers still write traces. So we POLL the dir instead of
    # assuming the client returned:
    #   phase 1: wait (up to TRACE_APPEAR_TIMEOUT) for the first *.pt.trace.json.gz;
    #   phase 2: wait until BOTH the file count AND total bytes stop changing (fully
    #            flushed) -- count-only "settle" would move/truncate a file mid-write.
    local appear_to="${TRACE_APPEAR_TIMEOUT:-900}" settle_s="${TRACE_SETTLE_S:-10}"
    local t cur=0 cursize prevcnt=-1 prevsize=-1 stable=0
    local find_gz=(find "${PROFILER_DIR}" -maxdepth 1 -type f -newer "${marker}" -name '*.pt.trace.json.gz')
    for (( t=0; t<appear_to; t+=2 )); do
        cur=$("${find_gz[@]}" 2>/dev/null | wc -l)
        (( cur > 0 )) && break
        sleep 2
    done
    if (( cur == 0 )); then
        echo "  [profile] no trace captured for ${dest##*/} after ${appear_to}s (stop_profile may have failed)" >&2
        return 0
    fi
    for (( t=0; t<appear_to; t+=2 )); do
        cur=$("${find_gz[@]}" 2>/dev/null | wc -l)
        # printf %.0f (NOT print s+0): big sums else print in sci-notation -> (( )) errors.
        cursize=$("${find_gz[@]}" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s+0}')
        if (( cur == prevcnt && cursize == prevsize )); then
            (( (++stable) * 2 >= settle_s )) && break   # count+size unchanged for settle_s
        else
            stable=0
        fi
        prevcnt=${cur}; prevsize=${cursize}; sleep 2
    done
    local -a new
    mapfile -t new < <(find "${PROFILER_DIR}" -maxdepth 1 -type f -newer "${marker}" \
        \( -name '*.pt.trace.json.gz' -o -name 'profiler_out_*.txt' \) 2>/dev/null)
    (( ${#new[@]} )) || { echo "  [profile] no trace captured for ${dest##*/} (stop_profile may have failed)" >&2; return 0; }
    mkdir -p "${dest}"
    mv -t "${dest}" "${new[@]}"
    echo "  [profile] ${#new[@]} artifact(s) -> ${dest#"${RUN_DIR}/"}"
}

# serve_backend <log> — start the backend in the background. vllm uses
# PRESET_YAML; sglang uses serve_sglang_ds3.2.sh.
serve_backend() {
    local log="$1"
    local -a launch
    if [ "${BACKEND,,}" = "sglang" ]; then
        [ -f "${SERVE_SGLANG_SH}" ] || { echo "[ERROR] SERVE_SGLANG_SH not set/found: '${SERVE_SGLANG_SH}'" >&2; exit 1; }
        echo "[serve] SGLang (model: ${MODEL_PATH}, port: ${SERVER_PORT}) -> ${log}"
        launch=(env "SGLANG_PORT=${SERVER_PORT}" bash "${SERVE_SGLANG_SH}" "${MODEL_PATH}")
    else
        echo "[serve] vLLM (preset: ${PRESET_YAML}) -> ${log}"
        # Snapshot the exact preset served (incl. profiler/eager-injected copies)
        # beside the serve log so each run records its config.
        cp -f "${PRESET_YAML}" "$(dirname "${log}")/preset.yaml" 2>/dev/null || true
        launch=(bash "${SERVE_SH}" "${MODEL_PATH}" "${PRESET_YAML}")
    fi
    # SERVE_TEE=1 pipes stdout through tee (to this shell's stdout + the log),
    # mirroring serve_glm5.sh -- adds logging backpressure (fatter tpot tail when
    # stdout is a slow terminal). Default 0 = direct file redirect (original).
    if is_enabled "${SERVE_TEE:-0}"; then
        "${launch[@]}" > >(tee "${log}") 2>&1 &
    else
        "${launch[@]}" >"${log}" 2>&1 &
    fi
    export SERVER_PID="$!" SERVER_LOG="${log}"   # for wait_for_server early-abort
    echo "[serve] PID: ${SERVER_PID}"
}
