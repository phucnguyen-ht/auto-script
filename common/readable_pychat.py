#!/usr/bin/env python3
"""Readable smoke test: apply the chat template CLIENT-SIDE, then POST the raw
/v1/completions endpoint.

Why: applying the chat template here (instead of /v1/chat/completions) gives
exact control over prompt formatting / special tokens. If the server's built-in
chat template is wrong/mismatched, the chat endpoint yields degenerate output;
doing it client-side against /v1/completions sidesteps that.

Prompts come from --prompts-file (one per line; blank lines skipped).
"""

import argparse
import sys

import requests
from transformers import AutoTokenizer


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to --model).")
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--show-prompt", action="store_true")
    return p.parse_args()


def main():
    args = build_args()
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"

    with open(args.prompts_file) as fh:
        prompts = [ln.strip() for ln in fh if ln.strip()]

    tok_src = args.tokenizer or args.model
    print(f"Loading tokenizer from: {tok_src}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    session = requests.Session()
    for prompt in prompts:
        print(f"Making requests for prompt: {prompt}")
        print("------------------------------------------")

        messages = [{"role": "user", "content": prompt}]
        # add_generation_prompt=True appends the assistant turn marker.
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if args.show_prompt:
            print("--- templated prompt ---")
            print(templated)
            print("--- end prompt ---")

        payload = {
            "model": args.model,
            "prompt": templated,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            # The chat template already added special tokens as text; don't re-add.
            "add_special_tokens": False,
        }
        try:
            resp = session.post(endpoint, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[request error] {e}\n")
            continue
        except ValueError:
            print(f"[non-JSON response] {resp.text[:500]}\n")
            continue

        if "error" in data:
            print(f"[server error] {data['error']}")
        else:
            try:
                print(data["choices"][0]["text"].strip())
            except (KeyError, IndexError):
                print(f"[unexpected shape] {data}")
        print("\n")


if __name__ == "__main__":
    main()
