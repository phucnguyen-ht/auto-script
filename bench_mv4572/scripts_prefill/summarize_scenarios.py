#!/usr/bin/env python3
"""Aggregate every prefill benchmark scenario into one summary CSV.

multi_process_test.py writes one result directory per scenario, named
    longbenchv2-<dataset>_p<parallel_threads>_<timestamp>/
and inside each, unit_test.py appends a single
    [Summary] {<json>}
line with the prefill metrics (parallel_threads, encoding_size, requests,
TTFT percentiles, prompt-token counts, per-request and system prefill
throughput). The dataset is NOT in that JSON, so we read it from the dir name.

This script collects all scenarios into a single CSV (one row per scenario),
sorted by input length then concurrency -- so TTFT / prefill throughput across
the whole sweep can be eyeballed in one place.

Usage
-----
    python3 summarize_scenarios.py                  # defaults below
    python3 summarize_scenarios.py <results_dir> <out.csv>
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

DEFAULT_RESULTS = "/workspace/vllm-moreh/benchmark_prefill/results"
DEFAULT_OUT = "/workspace/vllm-moreh/benchmark_prefill/scenario_summary.csv"

# Scenario result dirs, both layouts:
#   multi_process_test:        longbenchv2-<dataset>_p<threads>_<timestamp>
#   single-shot (with seed):   single-longbenchv2-<dataset>_p<threads>_s<seed>_<timestamp>
# (warmup-* dirs are excluded: the regex is anchored and won't match "warmup-".)
DIR_RE = re.compile(
    r"(?:single-)?longbenchv2-(?P<dataset>[^_]+)_p(?P<threads>\d+)"
    r"(?:_s(?P<seed>\d+))?_(?P<ts>[\d.]+)$")

# Metrics from the [Summary] JSON, as (output_column, summary_key, multiplier).
# TTFT is stored in seconds in the summary but emitted in ms here (x1000).
# For the prefill bench, TTFT == prefill latency and prefill_throughput is the
# system-level input-token throughput (total prompt tokens / window span).
METRIC_SPEC = [
    ("requests", "requests", 1),
    ("mean_ttft_ms", "mean_ttft_s", 1000),
    ("p90_ttft_ms", "p90_ttft_s", 1000),
    ("p99_ttft_ms", "p99_ttft_s", 1000),
    ("mean_prompt_tokens", "mean_prompt_tokens", 1),
    ("total_prompt_tokens", "total_prompt_tokens", 1),
    ("window_span_s", "window_span_s", 1),
    ("mean_prefill_tps", "mean_prefill_tps", 1),
    ("prefill_throughput", "prefill_throughput", 1),
]
METRIC_FIELDS = [out for out, _, _ in METRIC_SPEC]

# Server-side prefill throughput, joined from the matching prefill_running
# totals CSV (per-step prompt throughput summed across DP ranks):
#   server_mean_prefill_tps         -- mean of the per-step cluster throughput
#                                      over ALL steps (includes decode/idle)
#   server_active_mean_prefill_tps  -- mean over prefill-active steps only
#                                      (per-step cluster throughput > 0)
#   server_max_prefill_tps          -- peak per-step cluster throughput
SERVER_FIELDS = ["server_mean_prefill_tps", "server_active_mean_prefill_tps",
                 "server_max_prefill_tps"]
FIELDS = (["concurrency", "ISL", "seed"] + METRIC_FIELDS + SERVER_FIELDS)


def isl_sort_key(isl):
    """Numeric sort key from an ISL label like '8k', '10k', '100k', '1M'."""
    s = str(isl).strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def read_summary(scenario_dir):
    """Return the parsed [Summary] dict from the scenario dir, or None."""
    for path in glob.glob(os.path.join(scenario_dir, "*.log")):
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("[Summary]"):
                    try:
                        return json.loads(line[len("[Summary]"):].strip())
                    except json.JSONDecodeError:
                        return None
    return None


def totals_stats(totals_dir, threads, dataset, seed):
    """Server-side prefill throughput stats for one scenario.

    Reads the totals CSV produced by extract_prefill_running.py (column
    `prefill_throughput` = per-step cluster prompt throughput summed across DP
    ranks) and returns {server_mean_prefill_tps, server_active_mean_prefill_tps,
    server_max_prefill_tps}. Returns all-None if the file is absent (e.g. the
    extractor was not run) or has no data rows.
    """
    none = {"server_mean_prefill_tps": None,
            "server_active_mean_prefill_tps": None,
            "server_max_prefill_tps": None}
    # extract_prefill_running names the totals CSV after its --scenarios label.
    # Support the dataset-first and threads-first conventions:
    #   <dataset>_p<threads>   (multi_process_test, dataset-first)
    #   p<threads>_d<dataset>  (threads-first)
    candidates = [
        f"prefill_running_{dataset}_p{threads}.totals.csv",
        f"prefill_running_p{threads}_d{dataset}.totals.csv",
    ]
    if seed is not None:
        candidates.append(
            f"prefill_running_p{threads}_d{dataset}_s{seed}.totals.csv")
    path = next((os.path.join(totals_dir, c) for c in candidates
                 if os.path.exists(os.path.join(totals_dir, c))), None)
    if path is None:
        return none
    vals = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            try:
                vals.append(float(rec["prefill_throughput"]))
            except (KeyError, ValueError):
                continue
    if not vals:
        return none
    active = [v for v in vals if v > 0.0]
    return {
        "server_mean_prefill_tps": round(sum(vals) / len(vals), 4),
        "server_active_mean_prefill_tps": (round(sum(active) / len(active), 4)
                                           if active else None),
        "server_max_prefill_tps": round(max(vals), 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate prefill-bench scenario [Summary] lines into one CSV.")
    ap.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS,
                    help=f"dir holding scenario subdirs (default: {DEFAULT_RESULTS})")
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT,
                    help=f"output CSV path (default: {DEFAULT_OUT})")
    ap.add_argument("--totals-dir", default=None,
                    help="dir holding extract_prefill_running totals CSVs for "
                         "server-side prefill throughput (default: the output "
                         "CSV's dir)")
    ap.add_argument("--min-ts", type=float, default=None,
                    help="only include result dirs whose trailing timestamp is "
                         ">= this epoch value -- use to scope to one run and "
                         "exclude older runs' dirs sharing the results/ folder")
    args = ap.parse_args()

    totals_dir = args.totals_dir or os.path.dirname(args.out)

    rows, missing = [], []
    for entry in sorted(os.listdir(args.results_dir)):
        m = DIR_RE.match(entry)
        if not m:
            continue
        scenario_dir = os.path.join(args.results_dir, entry)
        if not os.path.isdir(scenario_dir):
            continue
        seed = int(m.group("seed")) if m.group("seed") is not None else None
        try:
            ts = float(m.group("ts"))
        except (TypeError, ValueError):
            ts = 0.0
        if args.min_ts is not None and ts < args.min_ts:
            continue
        summary = read_summary(scenario_dir)
        if summary is None:
            missing.append(entry)
            continue
        row = {
            "concurrency": int(m.group("threads")),
            "ISL": m.group("dataset"),
            "seed": seed,
            "_ts": ts,
        }
        for out, src, mul in METRIC_SPEC:
            v = summary.get(src)
            row[out] = (round(v * mul, 2) if mul != 1 and isinstance(v, (int, float))
                        else v)
        row.update(totals_stats(
            totals_dir, m.group("threads"), m.group("dataset"), seed))
        rows.append(row)

    if not rows:
        sys.exit(f"No scenario [Summary] lines found under {args.results_dir}")

    # Dedup by (ISL, concurrency, seed): keep the latest-timestamp dir so stale
    # dirs from aborted runs don't produce duplicate rows.
    best = {}
    for r in rows:
        key = (r["ISL"], r["concurrency"], r["seed"])
        if key not in best or r["_ts"] > best[key]["_ts"]:
            best[key] = r
    dropped = len(rows) - len(best)
    rows = list(best.values())

    # Sort by input length (ISL) first, then concurrency, then seed.
    rows.sort(key=lambda r: (isl_sort_key(r["ISL"]), r["concurrency"],
                             r["seed"] if r["seed"] is not None else -1))

    # Only include a seed column if this run actually has seeds.
    has_seed = any(r["seed"] is not None for r in rows)
    fields = (["concurrency", "ISL"] + (["seed"] if has_seed else [])
              + METRIC_FIELDS + SERVER_FIELDS)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} scenario(s) -> {args.out}"
          + (f" (dropped {dropped} stale dup dir(s))" if dropped else ""))
    if missing:
        print(f"[warn] {len(missing)} dir(s) had no [Summary] line: "
              + ", ".join(missing))


if __name__ == "__main__":
    main()
