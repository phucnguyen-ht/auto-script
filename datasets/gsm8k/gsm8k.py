#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
import tempfile

# ----------------------------------------------------------------------------
# Config — env-overridable (auto_eval.sh sets EVAL_*); defaults for standalone.
# ----------------------------------------------------------------------------
MODEL = os.environ.get("EVAL_MODEL", "/remote/vast0/share-mv/moonshotai/Kimi-K2.5")
TASKS = os.environ.get("EVAL_TASKS", "gsm8k.yaml")
OUTPUT_PATH = os.environ.get("EVAL_OUTPUT_PATH", "zresult")


# gsm8k REQUIRES the chat-completions endpoint; only the host may vary, so
# normalize whatever EVAL_BASE_URL is given to end with /v1/chat/completions.
def _chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


BASE_URL = _chat_url(os.environ.get("EVAL_BASE_URL", "http://localhost:8008"))

NUM_CONCURRENT = int(os.environ.get("EVAL_NUM_CONCURRENT", "8"))
MAX_RETRIES = 1
REQUEST_TIMEOUT = 1800          # per-request HTTP timeout (model_args)
MAX_LENGTH = 262144
EOS_STRING = "</s>"

MAX_TOKENS = 16384
TEMPERATURE = 0
TOP_P = 1

DISABLE_THINKING = True         # see module docstring
PROCESS_TIMEOUT = 86400         # outer wall-clock cap for the whole run (s)

# ----------------------------------------------------------------------------
# sitecustomize patch (verbatim from GSM8KEvaluationClient)
# ----------------------------------------------------------------------------
_SITECUSTOMIZE = """\
import json
from lm_eval.models.openai_completions import LocalChatCompletion as _LCC

def _le_parse_generations(outputs, **kwargs):
    res = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in (outputs or []):
        try:
            choices = out.get("choices", [])
            tmp = ["" for _ in choices]
            for choice in choices:
                idx = choice.get("index", 0)
                msg = (choice.get("message") or {})
                content = msg.get("content")
                if content in (None, "", []):
                    content = msg.get("reasoning_content") or ""
                tmp[idx] = content
        except Exception:
            tmp = [""]
        res.extend(tmp)
    return res

_LCC.parse_generations = staticmethod(_le_parse_generations)

try:
    from lm_eval.models import api_models as _api_models
    _TemplateAPI = _api_models.TemplateAPI
    _JsonChatStr = _api_models.JsonChatStr
except Exception:
    _TemplateAPI = None
    _JsonChatStr = None

if _TemplateAPI is not None and _JsonChatStr is not None:
    def _patched_apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        elif self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            return _JsonChatStr(
                json.dumps([{**item} for item in chat_history], ensure_ascii=False)
            )
    _TemplateAPI.apply_chat_template = _patched_apply_chat_template
"""


def main() -> int:
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    model_args = (
        f"model={MODEL},"
        f"base_url={BASE_URL},"
        f"trust_remote_code=True,"
        f"eos_string={EOS_STRING},"
        f"max_retries={MAX_RETRIES},"
        f"num_concurrent={NUM_CONCURRENT},"
        f"timeout={REQUEST_TIMEOUT},"
        f"tokenized_requests=False,"
        f"max_length={MAX_LENGTH}"
    )

    # gen_kwargs as JSON (this lm_eval parses args with try_parse_json; a JSON
    # object is accepted directly and is required once chat_template_kwargs adds
    # a nested dict).
    gen_kwargs = {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P}
    if DISABLE_THINKING:
        gen_kwargs["chat_template_kwargs"] = {"thinking": False}

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--apply_chat_template",
        "--tasks", TASKS,
        "--output_path", OUTPUT_PATH,
        "--log_samples",
        "--model_args", model_args,
        "--gen_kwargs", json.dumps(gen_kwargs),
    ]

    patch_dir = tempfile.mkdtemp(prefix="gsm8k_patch_")
    try:
        with open(os.path.join(patch_dir, "sitecustomize.py"), "w") as f:
            f.write(_SITECUSTOMIZE)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{patch_dir}:{env.get('PYTHONPATH', '')}"

        print("Applying reasoning_content patch via:", patch_dir)
        print("thinking:", "OFF" if DISABLE_THINKING else "ON")
        print("Command:", " ".join(cmd))

        try:
            subprocess.run(cmd, env=env, check=True, timeout=PROCESS_TIMEOUT)
        except subprocess.CalledProcessError as e:
            print(f"lm_eval failed (exit {e.returncode})", file=sys.stderr)
            return e.returncode or 1
        except subprocess.TimeoutExpired:
            print(f"GSM8K eval timed out after {PROCESS_TIMEOUT}s", file=sys.stderr)
            return 124
    finally:
        shutil.rmtree(patch_dir, ignore_errors=True)

    print(f"GSM8K eval complete — results under {OUTPUT_PATH}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
