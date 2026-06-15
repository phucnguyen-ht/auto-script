#!/usr/bin/env bash
# Master driver — chains every auto_*.sh against a single preset so you can
# leave it running overnight and come back to one tagged log dir.
#
# Usage:
#   bash run_all.sh
#   PRESET=glm5/dp8ep8/bs64-dg.yaml bash run_all.sh
#   PRESET=/absolute/path/to/preset.yaml bash run_all.sh
#
# Compare two presets back-to-back (A/B):
#   bash run_all.sh
#   PRESET=glm5/dp8ep8/bs64-moreh.yaml bash run_all.sh
#
# PRESET resolution order (first hit wins):
#   1. $PRESET (or fallback $PRESET_YAML) if it already points to an existing file
#   2. ../presets/<value>               — pass just the filename, no path needed
#   3. ./<value>                        — relative to this script's dir
#   4. ../presets/glm5/dp8ep8/bs64-dg.yaml  (default)
#
# Exported: PRESET_YAML — all child scripts (auto_bench / auto_profile /
# auto_eval / auto_readable) read this env-var, so one assignment here drives
# the whole chain with no per-line plumbing.
#
# Skip individual phases with:
#   RUN_BENCH=0    bash run_all.sh
#   RUN_PROFILE=0  bash run_all.sh
#   RUN_EVAL=0     bash run_all.sh
#   RUN_READABLE=0 bash run_all.sh
#
# Accurate profiling needs eager mode (cudagraph/compile off). Set
# ENFORCE_EAGER_PROFILE=1 (default 0) to run ONLY the profile phase against a
# generated copy of the preset with engine_args.enforce_eager=true. The other
# phases (bench/eval/readable) keep the original preset so their perf numbers
# are unaffected.
#   ENFORCE_EAGER_PROFILE=1 bash run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESETS_DIR="${SCRIPT_DIR}/../presets"

# ---------------------------------------------------------------------------
# yq bootstrap (needed for PRESET resolution below if it's a bare filename)
# ---------------------------------------------------------------------------
install_yq() {
  YQ_VERSION="v4.50.1"
  YQ_SHA256="c7a1278e6bbc4924f41b56db838086c39d13ee25dcb22089e7fbf16ac901f0d4"
  YQ_BIN="/usr/local/bin/yq"

  echo "Installing yq ${YQ_VERSION}..."
  curl -fsSL -o "${YQ_BIN}" \
    "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64"
  echo "${YQ_SHA256}  ${YQ_BIN}" | sha256sum -c -
  chmod 0755 "${YQ_BIN}"
  echo "yq installed successfully"
}

if ! command -v yq >/dev/null 2>&1; then
  install_yq
fi

pip install --no-cache-dir --force-reinstall \
  codebleu==0.7.0 \
  tree-sitter==0.22.3 \
  tree-sitter-python==0.21.0 fuzzywuzzy fire

# ---------------------------------------------------------------------------
# Backend: vllm (default) or sglang. Exported so every child phase agrees.
#   vllm   — resolve a preset, serve via serve.sh; logs under logs/<preset_name>
#   sglang — no preset (DeepSeek-V3.2 via serve_sglang_ds3.2.sh); logs under
#            logs_sglang/. Only the eval/readable phases support sglang;
#            bench/profile are skipped below.
# ---------------------------------------------------------------------------
BACKEND="${BACKEND:-vllm}"
case "${BACKEND,,}" in
    vllm|sglang) ;;
    *) echo "[run_all] ERROR: BACKEND must be 'vllm' or 'sglang' (got: ${BACKEND})" >&2; exit 1 ;;
esac
export BACKEND

ENV_YAML="${SCRIPT_DIR}/../env.yaml"

if [[ "${BACKEND,,}" != "sglang" ]]; then
    # -----------------------------------------------------------------------
    # Preset resolution (vLLM only)
    # -----------------------------------------------------------------------
    DEFAULT_PRESET="${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml"
    PRESET="${PRESET:-${PRESET_YAML:-${DEFAULT_PRESET}}}"

    if [[ -f "${PRESET}" ]]; then
        :  # already an absolute / resolvable path
    elif [[ -f "${PRESETS_DIR}/${PRESET}" ]]; then
        PRESET="${PRESETS_DIR}/${PRESET}"
    elif [[ -f "${SCRIPT_DIR}/${PRESET}" ]]; then
        PRESET="${SCRIPT_DIR}/${PRESET}"
    else
        echo "[run_all] ERROR: preset not found." >&2
        echo "[run_all]   tried: ${PRESET}"                >&2
        echo "[run_all]          ${PRESETS_DIR}/${PRESET}" >&2
        echo "[run_all]          ${SCRIPT_DIR}/${PRESET}"  >&2
        exit 1
    fi
    # Canonicalize to absolute path.
    PRESET="$(cd "$(dirname "${PRESET}")" && pwd)/$(basename "${PRESET}")"

    # Export once — every child script picks this up automatically.
    export PRESET_YAML="${PRESET}"

    # -----------------------------------------------------------------------
    # Model path resolution by preset family.
    #
    # The family is the top-level presets/ subfolder of the chosen preset
    # (presets/<family>/...). We look up model.paths.<family> in env.yaml and
    # export MODEL_PATH so every child script uses the right checkpoint. If the
    # preset is not under presets/ or the family has no entry, MODEL_PATH is
    # left unset and children fall back to env.yaml's model.path default.
    # -----------------------------------------------------------------------
    abs_presets="$(cd "${PRESETS_DIR}" && pwd)"
    preset_rel="${PRESET#"${abs_presets}/"}"

    # PRESET_NAME keys all log dirs (logs/<preset_name>/<phase>/<ts>): the
    # preset's path relative to presets/ with "/" joined by "_" and .yaml
    # stripped, e.g. glm5/dp8ep8/MTP-bs64-dg.yaml -> glm5_dp8ep8_MTP-bs64-dg.
    # Exported so every phase lands under the same subdir even when run_all
    # swaps in a generated preset copy (ENFORCE_EAGER_PROFILE).
    if [[ "${preset_rel}" != "${PRESET}" ]]; then
        PRESET_NAME="${preset_rel%.yaml}"
        PRESET_NAME="${PRESET_NAME//\//_}"
    else
        PRESET_NAME="$(basename "${PRESET}" .yaml)"
    fi
    export PRESET_NAME

    if [[ "${preset_rel}" != "${PRESET}" && "${preset_rel}" == */* ]]; then
        preset_family="${preset_rel%%/*}"
        fam_path="$(yq e ".model.paths.${preset_family} // \"\"" "${ENV_YAML}")"
        if [[ -n "${fam_path}" && "${fam_path}" != "null" ]]; then
            export MODEL_PATH="${fam_path}"
            echo "[run_all] preset family=${preset_family} -> MODEL_PATH=${MODEL_PATH}"
        else
            echo "[run_all] preset family=${preset_family} has no model.paths entry; using env.yaml model.path default" >&2
        fi
    fi
else
    # sglang: no preset, no MODEL_PATH-by-family. Children default MODEL_PATH to
    # SGLANG_MODEL_PATH (DeepSeek-V3.2) and write logs under logs_sglang/.
    PRESET=""
    echo "[run_all] BACKEND=sglang -> no preset; serving DeepSeek-V3.2 via serve_sglang_ds3.2.sh"
fi

# ---------------------------------------------------------------------------
# Phase toggles
# ---------------------------------------------------------------------------
RUN_BENCH="${RUN_BENCH:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_READABLE="${RUN_READABLE:-1}"

# When 1, the profile phase gets a preset variant with enforce_eager:true.
ENFORCE_EAGER_PROFILE="${ENFORCE_EAGER_PROFILE:-0}"

is_enabled() {
  case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------------------
# Master log dir.
#   vllm   — logs/<preset_name>/run_all/<ts> so every run of the same preset is
#            grouped together and easy to diff across runs.
#   sglang — logs_sglang/run_all/<ts> (no preset).
# ---------------------------------------------------------------------------
ts="$(date +%Y%m%d_%H%M%S)"
if [[ "${BACKEND,,}" == "sglang" ]]; then
    MASTER_LOG_DIR="${SCRIPT_DIR}/logs_sglang/run_all/${ts}"
else
    MASTER_LOG_DIR="${SCRIPT_DIR}/logs/${PRESET_NAME}/run_all/${ts}"
fi
mkdir -p "${MASTER_LOG_DIR}"

# Snapshot the exact preset used so you can audit it weeks later (vLLM only).
if [[ "${BACKEND,,}" != "sglang" ]]; then
    cp "${PRESET}" "${MASTER_LOG_DIR}/preset.yaml"
fi

cat <<EOF
========================================================================
run_all.sh (newbench2)
  backend     : ${BACKEND}
  preset      : ${PRESET:-<none (sglang)>}
  master log  : ${MASTER_LOG_DIR}
  phases      : bench=${RUN_BENCH} profile=${RUN_PROFILE} eval=${RUN_EVAL} readable=${RUN_READABLE}
  enforce_eager(profile): ${ENFORCE_EAGER_PROFILE}
  started at  : $(date)
========================================================================
EOF

# Tee combined stdout/stderr into a master summary log.
exec > >(tee -a "${MASTER_LOG_DIR}/run_all.log") 2>&1

# ---------------------------------------------------------------------------
# phase <name> <cmd…>  — wrapper for clean logging
# ---------------------------------------------------------------------------
phase() {
    local name="$1"; shift
    echo
    echo "------------------------------------------------------------------------"
    echo "[run_all] phase=${name}  preset=${PRESET}"
    echo "[run_all] starting at $(date)"
    echo "------------------------------------------------------------------------"
    "$@"
    echo "[run_all] phase=${name} DONE at $(date)"
}

# ---------------------------------------------------------------------------
# Chain
#
# bench/profile are vLLM-only (they parse presets and serve via serve.sh). Under
# BACKEND=sglang they are force-skipped with a warning — only eval/readable
# support sglang.
# ---------------------------------------------------------------------------
if [[ "${BACKEND,,}" == "sglang" ]] && { is_enabled "${RUN_BENCH}" || is_enabled "${RUN_PROFILE}"; }; then
    echo "[run_all] BACKEND=sglang: skipping bench/profile (vLLM-only phases)." >&2
    RUN_BENCH=0
    RUN_PROFILE=0
fi

if is_enabled "${RUN_BENCH}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    phase bench bash "${SCRIPT_DIR}/auto_bench.sh"
fi

if is_enabled "${RUN_PROFILE}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    # Accurate profiling needs eager mode. When ENFORCE_EAGER_PROFILE=1, profile
    # against a generated preset copy with enforce_eager:true; other phases are
    # untouched. PRESET_YAML is overridden only for this child invocation.
    profile_preset="${PRESET}"
    if is_enabled "${ENFORCE_EAGER_PROFILE}"; then
        profile_preset="${MASTER_LOG_DIR}/preset.profile_enforce_eager.yaml"
        yq '.engine_args.enforce_eager = true' "${PRESET}" > "${profile_preset}"
        echo "[run_all] ENFORCE_EAGER_PROFILE=1 -> profiling with enforce_eager:true"
        echo "[run_all]   preset: ${profile_preset}"
    fi
    PRESET_YAML="${profile_preset}" phase profile bash "${SCRIPT_DIR}/auto_profile.sh"
fi

if is_enabled "${RUN_READABLE}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    phase readable bash "${SCRIPT_DIR}/auto_readable.sh"
    phase readable bash "${SCRIPT_DIR}/auto_readable2.sh"
fi

if is_enabled "${RUN_EVAL}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    phase eval bash "${SCRIPT_DIR}/auto_eval.sh"
fi


cat <<EOF

========================================================================
run_all.sh (newbench2) — DONE at $(date)
  preset      : ${PRESET}
  master log  : ${MASTER_LOG_DIR}
========================================================================
EOF

# 1 1 2 2 3 3 4 4
# 5 4 4 4 3