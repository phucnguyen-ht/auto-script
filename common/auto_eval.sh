#!/usr/bin/env bash
set -uo pipefail
# Serve, then run the eval datasets enabled in env.yaml -> eval.datasets.
# Each dataset's `runs` is how many times it runs (0 = skip):
#   repobench  -> generate predictions + compute metrics, wrapped as ONE run
#   mmlu/gsm8k/longbench/longbench2 -> lm_eval task, run N times
#
# Override examples:
#   PRESET_YAML=/path bash auto_eval.sh
#   AUTO_SERVE=0 bash auto_eval.sh          # bring your own server
#   NUM_PROMPTS=8 CONCURRENCY=8 bash auto_eval.sh

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_DIR}/helper.sh"

resolve_backend
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/glm5/dp8ep8/bs64-dg.yaml}"
resolve_model_path
AUTO_SERVE="${AUTO_SERVE:-1}"
setup_run_dir auto_eval

# Run counts from env.yaml.
N_REPOBENCH="$(eval_runs repobench)"
declare -A LMEVAL_TASK=( [mmlu]=mmlu [gsm8k]=gsm8k [longbench]=longbench_single [longbench2]=longbench2_single )
declare -A LMEVAL_RUNS=()
LMEVAL_TOTAL=0
for ds in mmlu gsm8k longbench longbench2; do
    n="$(eval_runs "${ds}")"
    LMEVAL_RUNS[$ds]="${n}"
    LMEVAL_TOTAL=$((LMEVAL_TOTAL + n))
done

# RepoBench paths from env.yaml; relative values resolve against AUTO_ROOT.
DATASET_PATH="$(abspath "${DATASET_PATH:-$(yaml_get '.eval.datasets.repobench.dataset_path')}")"
REPOBENCH_DIR="$(abspath "${REPOBENCH_DIR:-$(yaml_get '.eval.datasets.repobench.repobench_dir')}")"

# RepoBench generation params (all overridable).
LANGUAGE="${LANGUAGE:-python}"
LEVELS="${LEVELS:-}"
NUM_PROMPTS="${NUM_PROMPTS:-${MAX_SAMPLES_PER_SETTING:-0}}"
EVAL_SETTING="${EVAL_SETTING:-in_file}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
CONCURRENCY="${CONCURRENCY:-4}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-3600}"
MAX_RETRIES="${MAX_RETRIES:-3}"
EXTRA_REQUEST_BODY="${EXTRA_REQUEST_BODY:-{}}"
COMPLETIONS_URL="${COMPLETIONS_URL:-}"
RESUME="${RESUME:-0}"

# lm_eval params.
LMEVAL_CONCURRENCY="${LMEVAL_CONCURRENCY:-8}"
LMEVAL_MAX_RETRIES="${LMEVAL_MAX_RETRIES:-3}"
LMEVAL_TIMEOUT="${LMEVAL_TIMEOUT:-3600}"

echo "=== auto_eval.sh started at $(date) ==="
echo "Backend=${BACKEND} port=${SERVER_PORT} model=${MODEL_PATH}"
echo "repobench runs=${N_REPOBENCH}  lm_eval runs: mmlu=${LMEVAL_RUNS[mmlu]} gsm8k=${LMEVAL_RUNS[gsm8k]} longbench=${LMEVAL_RUNS[longbench]} longbench2=${LMEVAL_RUNS[longbench2]}"

if [ "${N_REPOBENCH}" -gt 0 ]; then
    [ -d "${REPOBENCH_DIR}" ] || { echo "[ERROR] RepoBench dir not found: ${REPOBENCH_DIR}" >&2; exit 1; }
    [ -f "${DATASET_PATH}" ]  || { echo "[ERROR] dataset not found: ${DATASET_PATH}" >&2; exit 1; }
fi

if is_enabled "${AUTO_SERVE}" && [ "${BACKEND,,}" != "sglang" ] && [ ! -f "${PRESET_YAML}" ]; then
    echo "[ERROR] preset not found: ${PRESET_YAML}" >&2; exit 1
fi

export REPOBENCH_DIR MODEL_PATH DATASET_PATH LANGUAGE LEVELS NUM_PROMPTS EVAL_SETTING \
    MAX_NEW_TOKENS TEMPERATURE TOP_P CONCURRENCY REQUEST_TIMEOUT_SECS MAX_RETRIES \
    EXTRA_REQUEST_BODY COMPLETIONS_URL RESUME BASE_URL

if is_enabled "${AUTO_SERVE}"; then trap 'kill_server' EXIT; fi

# repobench_generate <out_dir> — generate predictions jsonl into out_dir.
repobench_generate() {
    OUTPUT_DIR="$1" python3 - <<'PY'
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests


def env_bool(name, default="0"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}


def env_int(name):
    return int(os.environ[name])


def env_float(name):
    return float(os.environ[name])


repobench_dir = Path(os.environ["REPOBENCH_DIR"]).resolve()
sys.path.insert(0, str(repobench_dir))

model_path = os.environ["MODEL_PATH"]
dataset_path = Path(os.environ["DATASET_PATH"])
language = os.environ["LANGUAGE"]
levels = set(os.environ["LEVELS"].replace(",", " ").split())
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

num_prompts = env_int("NUM_PROMPTS")
eval_setting = os.environ["EVAL_SETTING"]
max_new_tokens = env_int("MAX_NEW_TOKENS")
temperature = env_float("TEMPERATURE")
top_p = env_float("TOP_P")
concurrency = max(1, env_int("CONCURRENCY"))
timeout = env_int("REQUEST_TIMEOUT_SECS")
max_retries = max(1, env_int("MAX_RETRIES"))
resume = env_bool("RESUME")

try:
    extra_request_body = json.loads(os.environ.get("EXTRA_REQUEST_BODY", "{}") or "{}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"EXTRA_REQUEST_BODY must be valid JSON: {exc}") from exc


def auth_headers():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        return {"Authorization": api_key}
    return {}


def resolve_completions_url():
    explicit = os.environ.get("COMPLETIONS_URL", "").strip()
    if explicit:
        return explicit
    base = os.environ["BASE_URL"].rstrip("/")
    if base.endswith("/v1/completions") or base.endswith("/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/completions"
    return f"{base}/v1/completions"


headers = auth_headers()
completions_url = resolve_completions_url()


def get_first_line_not_comment(code, language="python"):
    assert language in ["python", "java"], "language must be one of [python, java]"
    code = code.lstrip("\n")
    lines = code.split("\n")
    in_multiline_comment = False
    if language == "python":
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not in_multiline_comment and (
                stripped.startswith('"""') or stripped.startswith("'''")
            ):
                in_multiline_comment = True
                continue
            if in_multiline_comment and (
                stripped.endswith('"""') or stripped.endswith("'''")
            ):
                in_multiline_comment = False
                continue
            if in_multiline_comment or stripped.startswith("#"):
                continue
            return line
    if language == "java":
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not in_multiline_comment and stripped.startswith("/*"):
                in_multiline_comment = True
                continue
            if in_multiline_comment and stripped.endswith("*/"):
                in_multiline_comment = False
                continue
            if in_multiline_comment or stripped.startswith("//"):
                continue
            return line
    return lines[0] if lines else ""


def load_local_dataset():
    with dataset_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)
    rows = []
    for source_idx, item in enumerate(raw_data):
        conversations = item.get("conversations") or item.get("conversation") or []
        if len(conversations) < 2:
            continue
        level = item.get("level", "")
        if levels and level not in levels:
            continue
        prompt = conversations[0].get("value", "")
        gt = conversations[1].get("value", "")
        if not prompt or not gt:
            continue
        rows.append({"idx": source_idx, "level": level, "prompt": prompt, "gt": gt})
    if num_prompts > 0:
        rows = rows[:num_prompts]
    return rows


def request_completion(prompt):
    payload = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stream": False,
        **extra_request_body,
    }
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                completions_url, json=payload, headers=headers, timeout=timeout
            )
            if response.status_code == 200:
                data = response.json()
                return True, data["choices"][0].get("text", ""), ""
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except Exception as exc:
            last_error = repr(exc)
        if attempt + 1 < max_retries:
            time.sleep(min(2 ** attempt, 30))
    return False, "", last_error


def seen_indices(path):
    if not resume or not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


print(f"Loading local RepoBench dataset from {dataset_path}")
data = load_local_dataset()
output_path = output_dir / f"{eval_setting}.jsonl"
if output_path.exists() and not resume:
    output_path.unlink()

skipped = seen_indices(output_path)
limit = len(data)
print(f"Generating {eval_setting}: {limit} examples -> {output_path}")
pending = {}
state = {"next_pos": 0, "submitted": 0}
done = 0


def submit_next(pool):
    while state["next_pos"] < limit and data[state["next_pos"]]["idx"] in skipped:
        state["next_pos"] += 1
    if state["next_pos"] >= limit:
        return False
    item = data[state["next_pos"]]
    state["next_pos"] += 1
    future = pool.submit(request_completion, item["prompt"])
    pending[future] = {"idx": item["idx"], "level": item["level"], "gt": item["gt"]}
    state["submitted"] += 1
    return True


with ThreadPoolExecutor(max_workers=concurrency) as pool:
    for _ in range(concurrency):
        if not submit_next(pool):
            break
    with output_path.open("a", encoding="utf-8") as file:
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                meta = pending.pop(future)
                ok, raw_pred, error = future.result()
                pred = (
                    get_first_line_not_comment(raw_pred, language=language) if ok else ""
                )
                row = {"idx": meta["idx"], "level": meta["level"], "pred": pred, "gt": meta["gt"]}
                if not ok:
                    row["success"] = False
                    row["error"] = error
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                done += 1
                submitted = state["submitted"]
                if done == submitted or done % max(1, min(50, submitted)) == 0:
                    print(f"  {eval_setting}: {done}/{submitted} written")
                submit_next(pool)

if skipped:
    print(f"  {eval_setting}: skipped {len(skipped)} existing rows")
print(f"RepoBench result files are in: {output_dir}")
PY
}

# run_repobench_once <out_dir> — generate then compute metrics (one wrapped run).
run_repobench_once() {
    local out="$1"; mkdir -p "${out}"
    echo "[repobench] generate -> ${out}"
    repobench_generate "${out}"
    echo "[repobench] metrics -> ${out}/eval.log"
    ( cd "${REPOBENCH_DIR}" && python3 eval.py --path "${out}" --language "${LANGUAGE}" ) \
        2>&1 | tee "${out}/eval.log"
}

# ---- RepoBench phase ----
if [ "${N_REPOBENCH}" -gt 0 ]; then
    # eval.py needs these; install only when repobench actually runs.
    if ! python3 -c 'import codebleu, tree_sitter, fuzzywuzzy, fire' 2>/dev/null; then
        echo "[repobench] installing eval deps..."
        pip install --no-cache-dir \
            codebleu==0.7.0 tree-sitter==0.22.3 tree-sitter-python==0.21.0 fuzzywuzzy fire >/dev/null \
            || { echo "[repobench][ERROR] pip install failed." >&2; exit 1; }
    fi
    if is_enabled "${AUTO_SERVE}"; then
        wait_for_gpu_free
        kill_server
        serve_backend "${RUN_DIR}/serve.log"
        wait_for_server || { echo "[ERROR] server failed for repobench." >&2; exit 1; }
    fi
    for i in $(seq 1 "${N_REPOBENCH}"); do
        echo "========== repobench run ${i}/${N_REPOBENCH} =========="
        run_repobench_once "${RUN_DIR}/repobench/run${i}"
    done
fi

# ---- lm_eval phase ----
if [ "${LMEVAL_TOTAL}" -gt 0 ]; then
    LMEVAL_NEEDS_LONGBENCH=0
    { [ "${LMEVAL_RUNS[longbench]}" -gt 0 ] || [ "${LMEVAL_RUNS[longbench2]}" -gt 0 ]; } && LMEVAL_NEEDS_LONGBENCH=1

    if is_enabled "${AUTO_SERVE}"; then
        SERVE_LOG_LMEVAL="${RUN_DIR}/serve_lmeval.log"
        kill_server
        if [ "${BACKEND,,}" = "sglang" ]; then
            serve_backend "${SERVE_LOG_LMEVAL}"
        else
            # PDSLoggingScheduler (scheduler-cls) is incompatible with lm_eval;
            # restart vLLM with a preset copy that strips it out.
            LMEVAL_PRESET="${RUN_DIR}/preset_no_scheduler_cls.yaml"
            yq 'del(.engine_args["scheduler-cls"])' "${PRESET_YAML}" > "${LMEVAL_PRESET}"
            echo "[serve] vLLM (no scheduler-cls) preset: ${LMEVAL_PRESET}"
            (bash "${SERVE_SH}" "${MODEL_PATH}" "${LMEVAL_PRESET}") >"${SERVE_LOG_LMEVAL}" 2>&1 &
            echo "[serve] PID: $!"
        fi
        wait_for_server || { echo "[ERROR] server failed for lm_eval." >&2; exit 1; }
    fi

    pip_targets=()
    command -v lm_eval >/dev/null 2>&1 || pip_targets+=('lm-eval[api]')
    [ "${LMEVAL_NEEDS_LONGBENCH}" -eq 1 ] && pip_targets+=(hf_transfer 'lm-eval[longbench]')
    if [ "${#pip_targets[@]}" -gt 0 ]; then
        echo "[lm_eval] installing: ${pip_targets[*]}"
        pip install "${pip_targets[@]}" >/dev/null || { echo "[lm_eval][ERROR] pip install failed." >&2; exit 1; }
    fi

    LMEVAL_URL="${COMPLETIONS_URL:-${BASE_URL%/}/v1/completions}"
    for ds in mmlu gsm8k longbench longbench2; do
        n="${LMEVAL_RUNS[$ds]}"
        [ "${n}" -gt 0 ] || continue
        task="${LMEVAL_TASK[$ds]}"
        # method: lm_eval (standard local-completions) or script
        # (datasets/<ds>/<ds>.py, e.g. gsm8k's chat-completions runner).
        method="$(yaml_get ".eval.datasets.${ds}.method" lm_eval)"
        for i in $(seq 1 "${n}"); do
            task_out="${RUN_DIR}/lm_eval/${task}/run${i}"
            mkdir -p "${task_out}"
            echo "========== ${ds} (${method}) run ${i}/${n} -> ${task_out} =========="
            if [ "${method}" = "script" ]; then
                ( cd "${AUTO_ROOT}/datasets/${ds}" && \
                  EVAL_MODEL="${MODEL_PATH}" \
                  EVAL_BASE_URL="${BASE_URL}" \
                  EVAL_TASKS="${ds}.yaml" \
                  EVAL_OUTPUT_PATH="${task_out}" \
                  EVAL_NUM_CONCURRENT="${LMEVAL_CONCURRENCY}" \
                  python3 "${ds}.py" ) 2>&1 | tee "${task_out}/eval.log"
            else
                lm_eval --model local-completions \
                    --tasks "${task}" \
                    --model_args "model=${MODEL_PATH},base_url=${LMEVAL_URL},num_concurrent=${LMEVAL_CONCURRENCY},max_retries=${LMEVAL_MAX_RETRIES},tokenized_requests=False,timeout=${LMEVAL_TIMEOUT}" \
                    --output_path "${task_out}" \
                    2>&1 | tee "${task_out}/eval.log"
            fi
            status=${PIPESTATUS[0]}
            [ "${status}" -ne 0 ] && echo "[eval][WARN] ${ds} run ${i} exited ${status}; continuing." >&2
        done
    done
fi

echo "=== auto_eval.sh done at $(date). Results under ${RUN_DIR} ==="
