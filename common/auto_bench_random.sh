#!/usr/bin/env bash
set -uo pipefail
# Random-dataset bench/profile concrete (vllm bench serve). MODE=bench|profile.
# Config: env.yaml .<MODE>.random: {runs, scenarios: [{isl, osl, conc, prompt}, ...]}.
# Per run -> one table over scenarios; agg_bench.py writes run<i>.csv + mean/std.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=random
declare -a SC_ISL SC_OSL SC_CONC SC_NP SC_LABEL

# populate SCENARIOS (indices) + parallel per-scenario state from the dict list.
load_scenarios() {
    local n j
    n="$(yq e ".${MODE}.random.scenarios | length" "${ENV_YAML}" 2>/dev/null)"
    [[ "${n}" =~ ^[0-9]+$ ]] || n=0
    for (( j=0; j<n; j++ )); do
        SC_ISL[j]="$(yq e ".${MODE}.random.scenarios[$j].isl" "${ENV_YAML}")"
        SC_OSL[j]="$(yq e ".${MODE}.random.scenarios[$j].osl" "${ENV_YAML}")"
        SC_CONC[j]="$(yq e ".${MODE}.random.scenarios[$j].conc" "${ENV_YAML}")"
        SC_NP[j]="$(yq e ".${MODE}.random.scenarios[$j].prompt" "${ENV_YAML}")"
        SC_LABEL[j]="isl${SC_ISL[j]}_osl${SC_OSL[j]}_c${SC_CONC[j]}_p${SC_NP[j]}"
        SCENARIOS+=("$j")
    done
}

# one vllm bench serve invocation -> <RUN_DIR>/run<r>/<label>.json
run_one() {
    local j="$1" r="$2" prof=() trc=()
    [ "${MODE}" = "profile" ] && prof=(--profile)
    # Mirror the server's trust_remote_code (from the preset) so the client
    # tokenizer matches the server's (needed for custom tokenizers, e.g. Kimi).
    [ "$(yq e '.engine_args.trust_remote_code // false' "${PRESET_YAML}")" = "true" ] && trc=(--trust-remote-code)
    local dir="${RUN_DIR}/run${r}"; mkdir -p "${dir}"
    echo "  scenario ${SC_LABEL[j]}"
    vllm bench serve \
        --backend vllm \
        --model "${MODEL_PATH}" \
        "${trc[@]}" \
        --base-url "${BASE_URL}" \
        --dataset-name random \
        --random-input-len "${SC_ISL[j]}" \
        --random-output-len "${SC_OSL[j]}" \
        --num-prompts "${SC_NP[j]}" \
        --max-concurrency "${SC_CONC[j]}" \
        --ignore-eos \
        "${prof[@]}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result --result-dir "${dir}" --result-filename "${SC_LABEL[j]}.json" \
        2>&1 | tee "${dir}/${SC_LABEL[j]}.log"
}

# build run<i>.csv (scenario x metric) per run + mean.csv/std.csv across runs.
aggregate() { python3 "${COMMON_DIR}/agg_bench.py" "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/agg.log"; }

source "${COMMON_DIR}/auto_bench_template.sh"
