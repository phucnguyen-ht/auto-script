#!/usr/bin/env bash
set -euo pipefail
# Serve vLLM then sweep bench_serving_glm4p5_65k.sh (PROFILE=0) over
# NUM_PROMPTS_LIST x NUM_ITERS. After each NUM_PROMPTS, mean+std rows are
# appended to its JSONL result file.
#
#   PRESET_YAML=/path bash auto_bench.sh
#   AUTO_SERVE=0 bash auto_bench.sh
#   NUM_PROMPTS_LIST="16 18 20 25" NUM_ITERS=3 bash auto_bench.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LOG_ROOT:=${SCRIPT_DIR}/logs}"
source "${SCRIPT_DIR}/../common/helper.sh"

resolve_backend
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml}"
resolve_model_path
AUTO_SERVE="${AUTO_SERVE:-1}"
setup_run_dir auto_bench

NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST:-16 18 20 25}"
NUM_ITERS="${NUM_ITERS:-3}"
RESULT_TAG="${RESULT_TAG:-dp8ep8_mtp2_model_runner_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/results}"
mkdir -p "${OUTPUT_DIR}"

cleanup() { is_enabled "${AUTO_SERVE}" && kill_server; }
trap cleanup EXIT

# append_aggregates <jsonl> — append mean ("agg":"mean") + sample std
# ("agg":"std") rows over all data rows. Numeric scalars averaged; equal-length
# numeric lists averaged element-wise; other fields copied from the first row.
# Previous aggregate rows are ignored (idempotent).
append_aggregates() {
    python3 - "$1" <<'PY'
import json
import statistics
import sys

path = sys.argv[1]
rows = []
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("agg") in ("mean", "std"):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
except FileNotFoundError:
    print(f"[aggregate] file not found, skipping: {path}")
    sys.exit(0)

if not rows:
    print(f"[aggregate] no data rows, skipping: {path}")
    sys.exit(0)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def stat(values, how):
    if how == "mean":
        return statistics.fmean(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


keys, seen = [], set()
for r in rows:
    for k in r:
        if k not in seen:
            seen.add(k)
            keys.append(k)


def build(how):
    out = {}
    for k in keys:
        present = [r[k] for r in rows if k in r]
        if not present:
            continue
        nums = [v for v in present if is_num(v)]
        first = present[0]
        if len(nums) == len(present):
            out[k] = stat(nums, how)
        elif (
            isinstance(first, list)
            and len(first) > 0
            and all(
                isinstance(v, list)
                and len(v) == len(first)
                and all(is_num(x) for x in v)
                for v in present
            )
        ):
            out[k] = [stat([v[i] for v in present], how) for i in range(len(first))]
        else:
            out[k] = first
    out["agg"] = how
    out["agg_n"] = len(rows)
    return out


with open(path, "a") as f:
    f.write(json.dumps(build("mean")) + "\n")
    f.write(json.dumps(build("std")) + "\n")

print(f"[aggregate] appended mean + std over {len(rows)} run(s) to {path}")
PY
}

echo "=== auto_bench.sh started at $(date) ==="
echo "Preset=${PRESET_YAML} model=${MODEL_PATH}"
echo "Prompts=${NUM_PROMPTS_LIST} iters=${NUM_ITERS} tag=${RESULT_TAG}"
echo "Run dir=${RUN_DIR}"

[ -f "${PRESET_YAML}" ] || { echo "[ERROR] Preset not found: ${PRESET_YAML}" >&2; exit 1; }

if is_enabled "${AUTO_SERVE}"; then
    wait_for_gpu_free
    kill_server
    serve_backend "${RUN_DIR}/serve.log"
    wait_for_server || { echo "[ERROR] Server failed to start." >&2; exit 1; }
fi

BENCH_LOG="${RUN_DIR}/bench.log"
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST//,/ }"
read -r -a prompts_arr <<< "${NUM_PROMPTS_LIST}"

{
    for np in "${prompts_arr[@]}"; do
        np_out_file="${OUTPUT_DIR}/${RESULT_TAG}_${np}.jsonl"
        for i in $(seq 1 "${NUM_ITERS}"); do
            echo "========== bench NUM_PROMPTS=${np} iter ${i}/${NUM_ITERS} =========="
            MODEL_PATH="${MODEL_PATH}" PROFILE=0 NUM_PROMPTS="${np}" \
            OUTPUT_FILE="${np_out_file}" \
                bash "${SCRIPT_DIR}/bench_serving_glm4p5_65k.sh"
        done
        append_aggregates "${np_out_file}" || echo "[aggregate][WARN] failed for ${np_out_file}"
    done
} 2>&1 | tee "${BENCH_LOG}"

echo "=== auto_bench.sh completed at $(date). Results: ${OUTPUT_DIR} ==="
