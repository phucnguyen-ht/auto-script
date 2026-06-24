#!/usr/bin/env bash
set -uo pipefail
# MV-4571 [GLM5.2] Analyze EP balancing impact. Custom-dataset bench/profile
# (vllm bench serve --dataset-name custom) on top of ../common/auto_bench_template.sh.
# Same custom-dataset flow as bench_mv4526, but with this ticket's bench METHOD
# (per bench_sweep.sh, the current reference) baked into the knobs below:
#   * NO warmup            -> vanilla `vllm bench serve` (we never pass --num-warmups).
#   * prefix caching OFF   -> presets carry engine_args.enable_prefix_caching: false
#                            (== --no-enable-prefix-caching). With the cache disabled
#                            every request recomputes, so each conc is measured cold
#                            WITHOUT any /reset_prefix_cache dance (RESET_PREFIX_CACHE=0).
#   * --metric-percentiles 75,90,99   (reference uses p75/p90/p99, not p50)
#   * --trust-remote-code  (GLM tokenizer needs it; gated on the preset's flag)
#   * num-prompts = 2 x concurrency   (ppc=2; no cap bites since conc<=52)
# MODE=bench (default) | profile (auto_profile.sh -> MODE=profile, adds --profile).
# Config: env.yaml .<MODE>.custom: { runs, dataset_dir, datasets:[{name,osl,rates,
# concurrencies}], prompts_per_concurrency, num_prompts_floor, num_prompts_cap }.
# Scenarios = per-dataset rates x concurrencies (run-major). Per run -> run<i>.csv
# (scenario x metric) + mean/std (agg_bench.py).

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="${TICKET_DIR}/../common"
export ENV_YAML="${ENV_YAML:-${TICKET_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${TICKET_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${TICKET_DIR}}"

METHOD=custom
JSONL_TEXT_FIELD="${JSONL_TEXT_FIELD:-text}"   # field renamed to `prompt` if the dataset lacks `prompt`
METRIC_PERCENTILES="${METRIC_PERCENTILES:-75,90,99}"  # reference: p75/p90/p99
SKIP_CHAT_TEMPLATE="${SKIP_CHAT_TEMPLATE:-1}"  # longbench prompts are pre-templated -> prompt_len == ISL
IGNORE_EOS="${IGNORE_EOS:-1}"                  # force full OSL for comparable lengths
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"    # GLM needs it; passed when the preset sets trust_remote_code: true
SAVE_DETAILED="${SAVE_DETAILED:-0}"            # 1 -> --save-detailed (per-request/per-token ITL,TTFT json; bigger files)

# Prefix caching is DISABLED at the engine (preset enable_prefix_caching: false),
# so there is nothing to reset -- skip the reset dance by default (and thus never
# need VLLM_SERVER_DEV_MODE). Set RESET_PREFIX_CACHE=1 only if you re-enable caching.
RESET_PREFIX_CACHE="${RESET_PREFIX_CACHE:-0}"
if [ "${MODE:-bench}" = "bench" ] && [ "${RESET_PREFIX_CACHE}" != "0" ]; then
    export VLLM_SERVER_DEV_MODE=1
fi

declare -a SC_DS SC_OSL SC_CONC SC_RATE SC_NP SC_LABEL
DATASET_DIR=""

# Each dataset lists its own rates x concurrencies -> SCENARIOS + state arrays.
load_scenarios() {
    local p=".${MODE}.custom" n j ppc floor cap rates concs dname dosl rate cc np idx=0
    DATASET_DIR="$(yaml_get "${p}.dataset_dir")"
    ppc="$(yaml_get "${p}.prompts_per_concurrency" 2)"
    floor="$(yaml_get "${p}.num_prompts_floor" 1)"
    cap="$(yaml_get "${p}.num_prompts_cap" 256)"
    n="$(yq e "${p}.datasets | length" "${ENV_YAML}" 2>/dev/null)"; [[ "${n}" =~ ^[0-9]+$ ]] || n=0
    for (( j=0; j<n; j++ )); do
        dname="$(yq e "${p}.datasets[$j].name" "${ENV_YAML}")"
        dosl="$(yq e "${p}.datasets[$j].osl" "${ENV_YAML}")"
        read -r -a rates <<< "$(yaml_list "${p}.datasets[$j].rates")"
        read -r -a concs <<< "$(yaml_list "${p}.datasets[$j].concurrencies")"
        for rate in "${rates[@]}"; do
            for cc in "${concs[@]}"; do
                np=$(( cc * ppc )); (( np < floor )) && np="${floor}"; (( np > cap )) && np="${cap}"
                SC_DS[idx]="${dname}"; SC_OSL[idx]="${dosl}"; SC_CONC[idx]="${cc}"
                SC_RATE[idx]="${rate}"; SC_NP[idx]="${np}"
                SC_LABEL[idx]="${dname}_r${rate//./p}_c${cc}"
                SCENARIOS+=("${idx}"); idx=$(( idx + 1 ))
            done
        done
    done
}

# Ensure a `prompt`-keyed JSONL exists (vllm custom dataset wants `prompt`). The
# longbenchv2-custom files already carry `prompt`, so this is a passthrough; kept
# for files that store the text under JSONL_TEXT_FIELD. Cached under the ticket
# dir. Echoes the usable path (empty if missing/failed).
prompt_dataset_for() {
    local src="$1"
    [ -f "${src}" ] || { echo ""; return; }
    if head -n1 "${src}" | python3 -c "import json,sys; sys.exit(0 if 'prompt' in json.loads(sys.stdin.readline()) else 1)" 2>/dev/null; then
        echo "${src}"; return
    fi
    local cache_dir="${DATA_DIR}/.prompt_cache" dst
    dst="${cache_dir}/$(basename "${src}")"; mkdir -p "${cache_dir}"
    if [ -s "${dst}" ] && [ "${dst}" -nt "${src}" ]; then echo "${dst}"; return; fi
    if ! TEXT_FIELD="${JSONL_TEXT_FIELD}" python3 - "${src}" "${dst}" <<'EOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
field = os.environ.get("TEXT_FIELD", "text")
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "prompt" not in d:
            if field not in d:
                raise SystemExit(f"line missing both 'prompt' and '{field}': keys={list(d)}")
            d["prompt"] = d.pop(field)
        fout.write(json.dumps({"prompt": d["prompt"]}) + "\n")
EOF
    then rm -f "${dst}"; echo ""; return; fi
    echo "${dst}"
}

# one vllm bench serve invocation -> <RUN_DIR>/run<r>/<label>.json
run_one() {
    local i="$1" r="$2" prof=() trc=()
    [ "${MODE}" = "profile" ] && prof=(--profile)
    # GLM needs --trust-remote-code (client-side tokenizer). 0 -> never pass it;
    # otherwise mirror the preset's engine_args.trust_remote_code.
    if [ "${TRUST_REMOTE_CODE}" != "0" ] \
       && [ "$(yq e '.engine_args.trust_remote_code // false' "${PRESET_YAML}")" = "true" ]; then
        trc=(--trust-remote-code)
    fi
    local dir="${RUN_DIR}/run${r}"; mkdir -p "${dir}"
    local ds="${SC_DS[i]}" osl="${SC_OSL[i]}" conc="${SC_CONC[i]}" rate="${SC_RATE[i]}" np="${SC_NP[i]}" label="${SC_LABEL[i]}"
    local raw="${DATASET_DIR}/longbenchv2-${ds}.jsonl" dataset_file
    dataset_file="$(prompt_dataset_for "${raw}")"
    if [ -z "${dataset_file}" ]; then echo "  [skip] dataset unusable: ${raw}" >&2; return 0; fi
    echo "  scenario ${label} (osl=${osl} num_prompts=${np})"
    local -a args=(
        vllm bench serve
        --backend vllm
        --model "${MODEL_PATH}" "${trc[@]}"
        --base-url "${BASE_URL}"
        --endpoint "/v1/completions"
        --dataset-name custom
        --dataset-path "${dataset_file}"
        --custom-output-len "${osl}"
        --num-prompts "${np}"
        --max-concurrency "${conc}"
        --request-rate "${rate}"
        --percentile-metrics "ttft,tpot,e2el,itl"
        --metric-percentiles "${METRIC_PERCENTILES}"
        --metadata isl="${ds}" osl="${osl}" conc="${conc}" num_prompts="${np}" request_rate="${rate}"
        --save-result --result-dir "${dir}" --result-filename "${label}.json"
        --seed 0 "${prof[@]}"
    )
    is_enabled "${SKIP_CHAT_TEMPLATE}" && args+=(--skip-chat-template)
    is_enabled "${IGNORE_EOS}" && args+=(--ignore-eos)
    is_enabled "${SAVE_DETAILED}" && args+=(--save-detailed)
    # prefix caching disabled at the engine -> no reset needed (default off).
    [ "${MODE}" = "bench" ] && is_enabled "${RESET_PREFIX_CACHE}" && reset_prefix_cache
    local marker=""; [ "${MODE}" = "profile" ] && marker="$(mktemp)"
    "${args[@]}" 2>&1 | tee "${dir}/${label}.log"
    # profile: move this scenario's freshly-written traces into a per-run/per-scenario
    # folder so each capture is traceable (torch profiler dumps them all into one dir).
    if [ -n "${marker}" ]; then
        [ -n "${PROFILER_DIR:-}" ] && harvest_profiles "${marker}" "${PROFILER_DIR}/run${r}/${label}"
        rm -f "${marker}"
    fi
}

# build run<i>.csv (scenario x metric) per run + mean.csv/std.csv across runs.
# Metrics/keys/order ported from bench_sweep.sh's summarize(): throughput, mean &
# p99 TTFT/TPOT, mean ITL, duration, completed.
AGG_COLS="completed,request_throughput,output_throughput,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms,mean_itl_ms,duration"
# Scenario labels are "<X>k_r<rate>_c<c>" (rate=inf here). Sort priority: request_rate
# -> ISL(k) -> concurrency. With a single rate all tie -> sorts by ISL then conc.
AGG_SORT='r(\d+);(\d+)k;c(\d+)'
aggregate() { python3 "${COMMON_DIR}/agg_bench.py" "${RUN_DIR}" "${AGG_COLS}" "${AGG_SORT}" 2>&1 | tee "${RUN_DIR}/agg.log"; }

source "${COMMON_DIR}/auto_bench_template.sh"
