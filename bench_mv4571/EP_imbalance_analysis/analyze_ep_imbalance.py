#!/usr/bin/env python3
"""MV-4571: EP-imbalance analysis for GLM-5.2 (DP8/EP8), following the MV-3382
methodology.

Parses ``[TOPK] Expert selection counts: [...]`` lines (emitted per MoE-layer
forward when the server runs with VLLM_MOREH_EP_LOG=1) and computes, per step:

  * per-rank load  = sum of expert counts in contiguous blocks of
                     (num_experts / num_ranks) experts  -> EP rank load
  * rank imbalance = max_rank / mean_rank   (room-for-improvement metric used in
                     MV-3382's final analysis; also report max/min)
  * expert imbalance = max_expert / mean_expert

Steps are split into prefill/mixed vs decode by tokens-per-step (num_tokens =
sum(counts)/topk). Decode steps are the ones that matter for sustained TPOT.

Usage:
  analyze_ep_imbalance.py LOG [LOG ...] --experts 256 --ranks 8 --topk 8 \
      [--decode-max-tokens N] [--json out.json] [--label NAME]
"""
import argparse
import json
import statistics
import sys

TERM = "[TOPK] Expert selection counts: "


def parse_counts(paths):
    """Yield per-step expert-count lists from one or more log files."""
    for path in paths:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                idx = line.find(TERM)
                if idx == -1:
                    continue
                payload = line[idx + len(TERM):].strip()
                # tolerate trailing text after the list
                rb = payload.rfind("]")
                if rb != -1:
                    payload = payload[: rb + 1]
                payload = payload.strip().strip("[]")
                if not payload:
                    continue
                try:
                    yield [int(x) for x in payload.split(",")]
                except ValueError:
                    continue


def rank_loads(counts, num_ranks):
    """Sum per-expert counts into num_ranks contiguous expert blocks."""
    n = len(counts)
    per = n // num_ranks
    return [sum(counts[r * per:(r + 1) * per]) for r in range(num_ranks)]


def pct(x):
    return 100.0 * x


def summarize(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.fmean(s),
        "median": s[n // 2],
        "p90": s[min(n - 1, int(0.90 * n))],
        "p99": s[min(n - 1, int(0.99 * n))],
        "max": s[-1],
    }


def analyze(paths, num_experts, num_ranks, topk, decode_max_tokens):
    per = num_experts // num_ranks
    rows = []  # (num_tokens, rank_max_over_avg, rank_max_over_min, expert_max_over_avg)
    for counts in parse_counts(paths):
        if len(counts) != num_experts:
            # tolerate length mismatch only if divisible by num_ranks
            if len(counts) % num_ranks != 0:
                continue
        total = sum(counts)
        if total == 0:
            continue
        num_tokens = total / topk
        loads = rank_loads(counts, num_ranks)
        rmax, rmin = max(loads), max(min(loads), 1)
        ravg = total / num_ranks
        emax = max(counts)
        eavg = total / len(counts)
        rows.append((
            num_tokens,
            rmax / ravg,            # rank max/avg  (>=1; 1.0 == perfect)
            rmax / rmin,            # rank max/min
            emax / eavg if eavg else 0.0,  # expert max/avg
        ))

    if not rows:
        return None

    def split(pred):
        return [r for r in rows if pred(r[0])]

    decode = split(lambda t: t <= decode_max_tokens)
    prefill = split(lambda t: t > decode_max_tokens)

    def block(rs):
        if not rs:
            return None
        return {
            "rank_max_over_avg": summarize([r[1] for r in rs]),
            "rank_max_over_min": summarize([r[2] for r in rs]),
            "expert_max_over_avg": summarize([r[3] for r in rs]),
            # imbalance as a percentage: extra load on the busiest rank vs average
            "rank_imbalance_pct_mean": pct(statistics.fmean([r[1] - 1 for r in rs])),
            "rank_imbalance_pct_p99": pct(sorted([r[1] - 1 for r in rs])[
                min(len(rs) - 1, int(0.99 * len(rs)))]),
        }

    return {
        "config": {
            "num_experts": num_experts, "num_ranks": num_ranks, "topk": topk,
            "experts_per_rank": per, "decode_max_tokens": decode_max_tokens,
        },
        "steps": {"total": len(rows), "decode": len(decode), "prefill_mixed": len(prefill)},
        "all": block(rows),
        "decode": block(decode),
        "prefill_mixed": block(prefill),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--decode-max-tokens", type=float, default=256.0,
                    help="steps with <= this many tokens are decode")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    res = analyze(args.logs, args.experts, args.ranks, args.topk, args.decode_max_tokens)
    if res is None:
        print(f"[{args.label}] no [TOPK] lines parsed from {args.logs}", file=sys.stderr)
        sys.exit(1)
    if args.label:
        res["label"] = args.label
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)

    def fmt(b, name):
        if not b:
            return f"  {name:14s}: (none)"
        rm = b["rank_max_over_avg"]
        return (f"  {name:14s}: rank max/avg mean={rm['mean']:.3f} "
                f"p99={rm['p99']:.3f} max={rm['max']:.3f} | "
                f"imbalance mean={b['rank_imbalance_pct_mean']:.1f}% "
                f"p99={b['rank_imbalance_pct_p99']:.1f}%")

    print(f"=== {args.label or args.logs} ===")
    print(f"  steps total={res['steps']['total']} "
          f"decode={res['steps']['decode']} prefill/mixed={res['steps']['prefill_mixed']}")
    print(fmt(res["decode"], "decode"))
    print(fmt(res["prefill_mixed"], "prefill/mixed"))
    print(fmt(res["all"], "all"))


if __name__ == "__main__":
    main()
