#!/usr/bin/env bash
set -uo pipefail
# mv-4433 custom bench/profile concrete (bench_serving_glm4p5_65k.sh) on top of
# ../common/auto_bench_template.sh. MODE=bench (default) | profile (auto_profile.sh
# calls this with MODE=profile). Config: env.yaml .<MODE>.custom: {runs, prompts[]}.
# Run-major: each scenario (prompt) accumulates one jsonl across runs; agg_one
# (append_aggregates) then appends mean/std rows.

TICKET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_YAML="${ENV_YAML:-${TICKET_DIR}/env.yaml}"
export LOG_ROOT="${LOG_ROOT:-${TICKET_DIR}/logs}"
export DATA_DIR="${DATA_DIR:-${TICKET_DIR}}"

METHOD=custom
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"
declare -a SC_NP SC_LABEL

# populate SCENARIOS (indices) + per-scenario prompt from .<MODE>.custom.prompts.
load_scenarios() {
    local prompts j
    read -r -a prompts <<< "$(yaml_list ".${MODE}.custom.prompts")"
    for j in "${!prompts[@]}"; do
        SC_NP[j]="${prompts[$j]}"
        SC_LABEL[j]="p${prompts[$j]}"
        SCENARIOS+=("$j")
    done
}

# one bench_serving invocation -> appends a row to <RUN_DIR>/<label>.jsonl.
run_one() {
    local j="$1" r="$2" prof=0
    [ "${MODE}" = "profile" ] && prof=1
    echo "  scenario ${SC_LABEL[j]} (run ${r})"
    MODEL_PATH="${MODEL_PATH}" PROFILE="${prof}" PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES}" \
    NUM_PROMPTS="${SC_NP[j]}" OUTPUT_FILE="${RUN_DIR}/${SC_LABEL[j]}.jsonl" \
        bash "${TICKET_DIR}/bench_serving_glm4p5_65k.sh"
}

# append_aggregates <jsonl> — append mean ("agg":"mean") + sample std ("agg":"std")
# rows over all data rows. Numeric scalars averaged; equal-length numeric lists
# element-wise; other fields copied from the first row. Idempotent.
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

# aggregate each scenario's accumulated jsonl.
aggregate() {
    local j
    for j in "${!SC_LABEL[@]}"; do
        append_aggregates "${RUN_DIR}/${SC_LABEL[j]}.jsonl" \
            || echo "[aggregate][WARN] failed for ${SC_LABEL[j]}.jsonl" >&2
    done
}

source "${TICKET_DIR}/../common/auto_bench_template.sh"
