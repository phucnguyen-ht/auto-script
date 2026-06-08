#!/usr/bin/env bash
set -euo pipefail

# Auto-serve vLLM then run RepoBench evaluation.
#
# Examples:
#   bash auto_eval.sh
#   NUM_PROMPTS=8 CONCURRENCY=8 bash auto_eval.sh
#   LEVELS=128k NUM_PROMPTS=8 bash auto_eval.sh
#   OUTPUT_DIR=/path/to/result GENERATE=0 bash auto_eval.sh
#   EXTRA_REQUEST_BODY='{"ignore_eos": true}' bash auto_eval.sh
#   AUTO_SERVE=0 bash auto_eval.sh            # bring your own server

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_YAML="${ROOT_DIR}/../env.yaml"
PRESETS_DIR="${ROOT_DIR}/../presets"
SERVE_SH="${ROOT_DIR}/../serve.sh"
BENCH_DIR="${ROOT_DIR}/../bench"

if ! command -v yq >/dev/null 2>&1; then
    echo "[ERROR] yq is required. Run serve.sh once to auto-install it."
    exit 1
fi

# Preset yaml to use when auto-starting the server.
# Override with: PRESET_YAML=/path/to/preset.yaml bash auto_eval.sh
PRESET_YAML="${PRESET_YAML:-${PRESETS_DIR}/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-profile.yaml}"

# Set AUTO_SERVE=0 to manage the server yourself.
AUTO_SERVE="${AUTO_SERVE:-1}"

AUTO_LOG_DIR="${ROOT_DIR}/logs/auto_eval"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${AUTO_LOG_DIR}/${ts}"
mkdir -p "${RUN_DIR}"

if command -v yq >/dev/null 2>&1 && [ -f "${ENV_YAML}" ]; then
    MODEL_PATH="${MODEL_PATH:-$(yq e '.model.path' "${ENV_YAML}")}"
else
    MODEL_PATH="${MODEL_PATH:-/remote/vast0/share-mv/zai-org/GLM-5-FP8}"
fi

REPOBENCH_DIR="${REPOBENCH_DIR:-/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-blocksize-64/customer-poc-delivery/repobench}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
COMPLETIONS_URL="${COMPLETIONS_URL:-}"

# Default dataset lives in the bench/ sibling directory.
DATASET_PATH="${DATASET_PATH:-${BENCH_DIR}/repobench_65k.json}"
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

DATASET_TAG="$(basename "${DATASET_PATH%.*}")"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/repobench_eval_results}"
EVAL_LOG="${EVAL_LOG:-${RUN_DIR}/eval.log}"

GENERATE="${GENERATE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ===========================================================
# Helper: wait until all GPUs report 0% utilization
# ===========================================================
wait_for_gpu_free() {
    local check_interval="${GPU_POLL_INTERVAL:-30}"
    local elapsed=0
    echo "[gpu-wait] Waiting for all GPUs to be free (poll every ${check_interval}s)..."
    while true; do
        local busy_count
        # A GPU holding a loaded model reports GPU use (%)=0 while idle, so we
        # gate on VRAM allocation instead: any GPU whose VRAM% exceeds the
        # threshold (default 10) counts as busy.
        busy_count=$(rocm-smi --showmemuse 2>/dev/null \
            | grep -E 'GPU Memory Allocated \(VRAM%\)' \
            | awk -F': ' '{print $NF+0}' \
            | awk -v thr="${GPU_VRAM_BUSY_THRESHOLD:-10}" '$1 > thr' \
            | wc -l)
        if [ "${busy_count}" -eq 0 ]; then
            echo "[gpu-wait] All GPUs are free after ${elapsed}s. Proceeding."
            return 0
        fi
        echo "[gpu-wait] ${busy_count} GPU(s) still busy (${elapsed}s elapsed). Retrying in ${check_interval}s..."
        sleep "${check_interval}"
        elapsed=$((elapsed + check_interval))
    done
}

# ===========================================================
# Helper: poll until server health endpoint responds
# ===========================================================
wait_for_server() {
    local max_wait="${SERVER_WAIT_TIMEOUT:-3600}"
    local elapsed=0
    echo "[wait] Polling ${BASE_URL}/health (timeout: ${max_wait}s)..."
    while ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "[ERROR] Server did not respond within ${max_wait}s"
            return 1
        fi
        echo "[wait] ${elapsed}s elapsed, still waiting..."
    done
    echo "[wait] Server ready after ${elapsed}s"
}

# ===========================================================
# Helper: kill all VLLM processes
# ===========================================================
kill_vllm() {
    echo "[kill] Sending SIGKILL to VLLM processes..."
    pkill -9 VLLM 2>/dev/null || true
    sleep 15
    echo "[kill] Done."
}

is_enabled() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

print_config() {
  cat <<EOF
RepoBench dir: $REPOBENCH_DIR
Model: $MODEL_PATH
Base URL: $BASE_URL
Completions URL: ${COMPLETIONS_URL:-<auto>}
Dataset path: $DATASET_PATH
Language: $LANGUAGE
Level filter: ${LEVELS:-<all>}
Num prompts: $NUM_PROMPTS
Eval setting file: $EVAL_SETTING.jsonl
Max new tokens: $MAX_NEW_TOKENS
Temperature: $TEMPERATURE
Top-p: $TOP_P
Concurrency: $CONCURRENCY
Output dir: $OUTPUT_DIR
Generate: $GENERATE
Run eval: $RUN_EVAL
Resume: $RESUME
EOF
}

if [[ ! -d "$REPOBENCH_DIR" ]]; then
  echo "RepoBench dir not found: $REPOBENCH_DIR" >&2
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH" >&2
  exit 1
fi

print_config

if is_enabled "$DRY_RUN"; then
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

export REPOBENCH_DIR
export MODEL_PATH
export BASE_URL
export COMPLETIONS_URL
export DATASET_PATH
export LANGUAGE
export LEVELS
export NUM_PROMPTS
export EVAL_SETTING
export MAX_NEW_TOKENS
export TEMPERATURE
export TOP_P
export CONCURRENCY
export REQUEST_TIMEOUT_SECS
export MAX_RETRIES
export EXTRA_REQUEST_BODY
export OUTPUT_DIR
export RESUME

if is_enabled "$AUTO_SERVE"; then
    if [ ! -f "${PRESET_YAML}" ]; then
        echo "[ERROR] Preset not found: ${PRESET_YAML}" >&2
        exit 1
    fi

    wait_for_gpu_free

    pkill -9 VLLM 2>/dev/null || true
    sleep 5

    SERVE_LOG="${RUN_DIR}/serve.log"
    echo "[serve] Starting server with preset: ${PRESET_YAML}"
    echo "[serve] Log: ${SERVE_LOG}"
    (bash "${SERVE_SH}" "${MODEL_PATH}" "${PRESET_YAML}") >"${SERVE_LOG}" 2>&1 &
    echo "[serve] PID: $!"

    trap 'kill_vllm' EXIT

    if ! wait_for_server; then
        echo "[ERROR] Server failed to start. Aborting."
        exit 1
    fi
fi

if is_enabled "$GENERATE"; then
  python3 - <<'PY'
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str) -> int:
    return int(os.environ[name])


def env_float(name: str) -> float:
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


def resolve_completions_url() -> str:
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


def get_first_line_not_comment(code: str, language: str = "python"):
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

        rows.append(
            {
                "idx": source_idx,
                "level": level,
                "prompt": prompt,
                "gt": gt,
            }
        )

    if num_prompts > 0:
        rows = rows[:num_prompts]

    return rows


def request_completion(prompt: str):
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
                completions_url,
                json=payload,
                headers=headers,
                timeout=timeout,
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


def seen_indices(path: Path):
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
    pending[future] = {
        "idx": item["idx"],
        "level": item["level"],
        "gt": item["gt"],
    }
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
                    get_first_line_not_comment(raw_pred, language=language)
                    if ok
                    else ""
                )
                row = {
                    "idx": meta["idx"],
                    "level": meta["level"],
                    "pred": pred,
                    "gt": meta["gt"],
                }
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
fi

if is_enabled "$RUN_EVAL"; then
  echo "Running RepoBench metrics..."
  (
    cd "$REPOBENCH_DIR"
    python3 eval.py --path "$OUTPUT_DIR" --language "$LANGUAGE"
  ) | tee "$EVAL_LOG"
fi

# ===========================================================
# Extra lm_eval datasets (default: mmlu, gsm8k) — chạy thẳng
# vào server đang sống (cùng base_url). Server được start ở đầu
# script khi AUTO_SERVE=1 và sẽ bị kill bởi trap EXIT, nên block
# này phải đứng TRƯỚC khi script thoát.
#
# PDSLoggingScheduler (scheduler-cls) is incompatible with lm_eval
# tasks (mmlu, gsm8k). When AUTO_SERVE=1, we restart the server with
# a temporary preset that has scheduler-cls stripped out.
#
# Tắt: RUN_LMEVAL=0 bash auto_eval.sh
# Đổi task: LMEVAL_TASKS="mmlu gsm8k longbench_single" bash auto_eval.sh
# ===========================================================
RUN_LMEVAL="${RUN_LMEVAL:-1}"
LMEVAL_TASKS="${LMEVAL_TASKS:-mmlu gsm8k}"
LMEVAL_CONCURRENCY="${LMEVAL_CONCURRENCY:-8}"
LMEVAL_MAX_RETRIES="${LMEVAL_MAX_RETRIES:-3}"
LMEVAL_TIMEOUT="${LMEVAL_TIMEOUT:-3600}"

if is_enabled "$RUN_LMEVAL" && is_enabled "$AUTO_SERVE"; then
    echo "[serve] Restarting server without scheduler-cls for lm_eval tasks..."
    kill_vllm

    LMEVAL_PRESET="${RUN_DIR}/preset_no_scheduler_cls.yaml"
    yq 'del(.engine_args["scheduler-cls"])' "${PRESET_YAML}" > "${LMEVAL_PRESET}"
    echo "[serve] lm_eval preset written to: ${LMEVAL_PRESET}"

    SERVE_LOG_LMEVAL="${RUN_DIR}/serve_lmeval.log"
    echo "[serve] Starting server for lm_eval: ${LMEVAL_PRESET}"
    echo "[serve] Log: ${SERVE_LOG_LMEVAL}"
    (bash "${SERVE_SH}" "${MODEL_PATH}" "${LMEVAL_PRESET}") >"${SERVE_LOG_LMEVAL}" 2>&1 &
    echo "[serve] PID: $!"

    if ! wait_for_server; then
        echo "[ERROR] Server failed to start for lm_eval. Aborting."
        exit 1
    fi
fi

if is_enabled "$RUN_LMEVAL"; then
    if ! command -v lm_eval >/dev/null 2>&1; then
        echo "[lm_eval] installing lm-eval[api]..."
        if ! pip install 'lm-eval[api]' >/dev/null; then
            echo "[lm_eval][ERROR] pip install failed; skipping lm_eval block." >&2
            RUN_LMEVAL=0
        fi
    fi
fi

if is_enabled "$RUN_LMEVAL"; then
    LMEVAL_URL="${COMPLETIONS_URL:-${BASE_URL%/}/v1/completions}"
    LMEVAL_DIR="${RUN_DIR}/lm_eval"
    mkdir -p "${LMEVAL_DIR}"
    echo "[lm_eval] base completions URL: ${LMEVAL_URL}"
    echo "[lm_eval] output root: ${LMEVAL_DIR}"

    for task in ${LMEVAL_TASKS}; do
        task_out="${LMEVAL_DIR}/${task}"
        task_log="${LMEVAL_DIR}/${task}.log"
        mkdir -p "${task_out}"
        echo ""
        echo "=========================================================="
        echo "[lm_eval] task=${task}"
        echo "  out: ${task_out}"
        echo "  log: ${task_log}"
        echo "=========================================================="
        # Use set +e for the task so a single failure doesn't abort the
        # whole script (e.g. one dataset missing should not skip the next).
        set +e
        lm_eval --model local-completions \
            --tasks "${task}" \
            --model_args "model=${MODEL_PATH},base_url=${LMEVAL_URL},num_concurrent=${LMEVAL_CONCURRENCY},max_retries=${LMEVAL_MAX_RETRIES},tokenized_requests=False,timeout=${LMEVAL_TIMEOUT}" \
            --output_path "${task_out}" \
            2>&1 | tee "${task_log}"
        status=${PIPESTATUS[0]}
        set -e
        if [ "${status}" -ne 0 ]; then
            echo "[lm_eval][WARN] task=${task} exited with ${status}; continuing." >&2
        fi
    done
    echo ""
    echo "[lm_eval] all tasks done. Results under ${LMEVAL_DIR}"
fi
