#!/usr/bin/env bash
# Master driver — chains every auto_*.sh against a single preset so you can
# leave it running overnight and come back to one tagged log dir.
#
# Usage:
#   bash run_all.sh
#   PRESET=zai-org-glm-5-fp8-amd-mi325x-tp8-moe-tp8-0ic-ar.yaml bash run_all.sh
#   PRESET=/absolute/path/to/preset.yaml bash run_all.sh
#
# Compare two presets back-to-back (A/B):
#   bash run_all.sh
#   PRESET=zai-org-glm-5-fp8-amd-mi325x-tp8-moe-tp8-50ic.yaml bash run_all.sh
#
# PRESET resolution order (first hit wins):
#   1. $PRESET (or fallback $PRESET_YAML) if it already points to an existing file
#   2. ../presets/<value>               — pass just the filename, no path needed
#   3. ./<value>                        — relative to this script's dir
#   4. ../presets/zai-org-glm-5-fp8-amd-mi325x-tp8-moe-tp8-0ic.yaml  (default)
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

# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------
DEFAULT_PRESET="${PRESETS_DIR}/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml"
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
# Master log dir — keyed by timestamp + preset basename so two consecutive
# runs (default vs. -ar.yaml) land in distinct, easy-to-diff folders.
# ---------------------------------------------------------------------------
ts="$(date +%Y%m%d_%H%M%S)"
preset_tag="$(basename "${PRESET}" .yaml)"
MASTER_LOG_DIR="${SCRIPT_DIR}/logs/run_all/${ts}_${preset_tag}"
mkdir -p "${MASTER_LOG_DIR}"

# Snapshot the exact preset used so you can audit it weeks later.
cp "${PRESET}" "${MASTER_LOG_DIR}/preset.yaml"

cat <<EOF
========================================================================
run_all.sh (newbench2)
  preset      : ${PRESET}
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
# ---------------------------------------------------------------------------
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

if is_enabled "${RUN_EVAL}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    phase eval bash "${SCRIPT_DIR}/auto_eval.sh"
fi

if is_enabled "${RUN_READABLE}"; then
    rm -rf /root/.cache/vllm/torch_compile_cache/
    phase readable bash "${SCRIPT_DIR}/auto_readable.sh"
    phase readable bash "${SCRIPT_DIR}/auto_readable2.sh"
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