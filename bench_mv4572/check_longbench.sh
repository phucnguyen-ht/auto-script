#!/usr/bin/env bash
# LongBench-v2 100k spot-check vs a running vLLM (default :8000). Prints the reply
# (not scored — the custom jsonl has no answer key). Used standalone AND by the
# readable method (auto_readable_longbench2.sh passes EVAL_MODEL / EVAL_BASE_URL).
# Paths default to CONTAINER paths (podman mounts host .../share-mv -> /remote/vast0/
# share-mv), overridable via EVAL_MODEL / LONGBENCH_JSONL / EVAL_BASE_URL / LONGBENCH_MAX_TOKENS.
set -euo pipefail
python3 - <<'PY'
import json, os, requests
MODEL = os.environ.get("EVAL_MODEL", "/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/")
JSONL = os.environ.get("LONGBENCH_JSONL", "/remote/vast0/share-mv/longbenchv2-custom/longbenchv2-100k.jsonl")
URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000").rstrip("/") + "/v1/chat/completions"
# Kimi-K2.6 is a REASONING model: with too small a budget it is cut off mid-thinking
# and the parser returns neither content nor reasoning_content (-> prints None). Give room.
MAX_TOKENS = int(os.environ.get("LONGBENCH_MAX_TOKENS", "2048"))
TIMEOUT = int(os.environ.get("LONGBENCH_TIMEOUT", "3600"))
p = json.loads(open(JSONL).readline())     # _id 66fcffd9bb02136c067c94c5, 99999 input tokens
r = requests.post(URL, json={
    "model": MODEL,
    "messages": [{"role": "user", "content": p["prompt"]}],
    "max_tokens": MAX_TOKENS, "temperature": 0,
}, timeout=TIMEOUT)
r.raise_for_status()
ch = r.json()["choices"][0]
m = ch["message"]
out = m.get("reasoning_content") or m.get("content")
if out:
    print(out)
else:
    # empty: hit max_tokens mid-reasoning (finish_reason=length) so the reasoning parser
    # emitted neither field. Show WHY instead of a bare `None`.
    print(f"[longbench2][WARN] empty reply (finish_reason={ch.get('finish_reason')}); "
          f"likely hit max_tokens={MAX_TOKENS} while still reasoning -> raise LONGBENCH_MAX_TOKENS.")
    print("[longbench2] raw message:", json.dumps(m)[:800])
PY
