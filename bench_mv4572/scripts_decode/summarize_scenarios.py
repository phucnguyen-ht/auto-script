#!/usr/bin/env python3
"""Aggregate every single-shot benchmark scenario into one summary CSV.

multi_process_single_shot.py writes one result directory per scenario, named
    single-longbenchv2-<dataset>_p<parallel_threads>_s<seed>_<timestamp>/
and inside each, single_shot_test.py appends a single
    [Summary] {<json>}
line (parallel_threads, encoding_size, requests, TTFT/TPOT/decode-tps stats).
The dataset and seed are NOT in that JSON, so we read them from the dir name.

This script collects all scenarios into a single CSV (one row per scenario),
sorted by parallel_threads, then input length, then seed -- so TTFT / TPOT /
decode throughput across the whole sweep can be eyeballed in one place.

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

DEFAULT_RESULTS = "/workspace/vllm-moreh/benchmark_zhipu/results"
DEFAULT_OUT = ("/workspace/vllm-moreh/benchmark_zhipu/logs/"
               "20260629_191445/summary.csv")

# Scenario result dirs, both layouts:
#   multi_process_test:        longbenchv2-<dataset>_p<threads>_<timestamp>
#   single-shot (with seed):   single-longbenchv2-<dataset>_p<threads>_s<seed>_<timestamp>
# (warmup-* dirs are excluded: the regex is anchored and won't match "warmup-".)
DIR_RE = re.compile(
    r"(?:single-)?longbenchv2-(?P<dataset>[^_]+)_p(?P<threads>\d+)"
    r"(?:_s(?P<seed>\d+))?_(?P<ts>[\d.]+)$")

# Metrics from the [Summary] JSON, as (output_column, summary_key, multiplier).
# TTFT is stored in seconds in the summary but emitted in ms here (x1000).
METRIC_SPEC = [
    ("requests", "requests", 1),
    ("mean_ttft_ms", "mean_ttft_s", 1000),
    ("p90_ttft_ms", "p90_ttft_s", 1000),
    ("mean_decode_tps", "mean_decode_tps", 1),
    ("mean_tpot_ms", "mean_tpot_ms", 1),
    ("p50_tpot_ms", "p50_tpot_ms", 1),
    ("p90_tpot_ms", "p90_tpot_ms", 1),
    ("p99_tpot_ms", "p99_tpot_ms", 1),
]
METRIC_FIELDS = [out for out, _, _ in METRIC_SPEC]

# System output throughput, derived from output.jsonl:
#   total_output_tokens  -- sum of completion tokens over all logged requests
#   benchmark_duration_s -- wall-clock span the system produced output tokens
#                           for this scenario = max(last_token_ts) -
#                           min(first_token_ts) across all logged requests. For
#                           the sustained-load (unit_test loop) runs this is ~the
#                           measurement window (time_limit - ignore_start -
#                           ignore_end, ~200s); for single-shot it's the wave's
#                           generation span.
#   output_tps           -- total_output_tokens / benchmark_duration_s
THROUGHPUT_FIELDS = ["total_output_tokens", "benchmark_duration_s", "output_tps"]
# Server-side decode batch size, joined from the matching decode_running totals
# CSV (per-step running reqs summed across DP ranks):
#   mean_decode_batch_size           -- mean over all steps
#   sustained_max_decode_batch_size  -- peak of the 10-step rolling mean, i.e.
#                                       the highest batch size actually held
#                                       across a window (ignores 1-step spikes)
#   steps_at_max_decode_batch_size   -- how many steps hit the raw instantaneous
#                                       peak (small => that peak was transient)
ROLLING_WINDOW = 10
CONCURRENCY_FIELDS = ["mean_decode_batch_size", "sustained_max_decode_batch_size",
                      "steps_at_max_decode_batch_size"]
FIELDS = (["concurrency", "ISL", "seed"]
          + METRIC_FIELDS + THROUGHPUT_FIELDS + CONCURRENCY_FIELDS)

# [Test] <threads> <enc> <out_tokens> <ttft_s> <decode_tps> <chunks> <tpot_ms>
TEST_RE = re.compile(
    r"^\[Test\] \d+ \d+ (?P<tokens>\d+) (?P<ttft>\S+) (?P<tps>\S+) "
    r"\d+ -?\d+(?:\.\d+)?\s*$")


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


def throughput_stats(scenario_dir):
    """System output throughput for one scenario, from output.jsonl.

    total_output_tokens = sum of completion tokens over all logged requests.
    benchmark_duration_s = the wall-clock span the system was producing output
    tokens = max(last_token_ts) - min(first_token_ts) across all logged
    requests (absolute time.time() stamps). For the sustained-load unit_test
    loop this is ~the measurement window (~200s); for single-shot it's the
    wave's generation span. output_tps = tokens / duration. Returns all-None if
    output.jsonl is missing/empty.
    """
    none = {"total_output_tokens": None, "benchmark_duration_s": None,
            "output_tps": None}
    oj = os.path.join(scenario_dir, "output.jsonl")
    if not os.path.exists(oj):
        return none
    firsts, lasts, total = [], [], 0
    with open(oj, "r", errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = o.get("token_timestamps") or []
            if ts:
                firsts.append(ts[0])
                lasts.append(ts[-1])
            usage = o.get("usage") or {}
            total += usage.get("completion_tokens") or o.get("usage_tokens") or 0
    if not firsts or total <= 0:
        return none
    dur = max(lasts) - min(firsts)
    return {
        "total_output_tokens": total,
        "benchmark_duration_s": round(dur, 4),
        "output_tps": round(total / dur, 4) if dur > 0 else None,
    }


def totals_stats(totals_dir, threads, dataset, seed):
    """Per-step decode batch size stats for one scenario.

    Reads the totals CSV produced by extract_decode_running.py and returns
    {mean_decode_batch_size, sustained_max_decode_batch_size,
    steps_at_max_decode_batch_size} -- sustained max being the peak of the
    10-step rolling mean, and steps_at_max how many steps hit the raw peak.
    Tries every known label convention (see candidates below). Returns all-None
    if the file is absent (e.g. extractor not run) or has no data rows.
    """
    none = {"mean_decode_batch_size": None,
            "sustained_max_decode_batch_size": None,
            "steps_at_max_decode_batch_size": None}
    # extract_decode_running names the totals CSV after its --scenarios label.
    # Support every convention we emit:
    #   <dataset>_p<threads>       (multi_process_test, dataset-first)
    #   p<threads>_d<dataset>      (multi_process_test, threads-first / extract_all.sh)
    #   p<threads>_d<dataset>_s<seed>  (single-shot, with seed)
    candidates = [
        f"decode_running_{dataset}_p{threads}.totals.csv",
        f"decode_running_p{threads}_d{dataset}.totals.csv",
    ]
    if seed is not None:
        candidates.append(
            f"decode_running_p{threads}_d{dataset}_s{seed}.totals.csv")
    path = next((os.path.join(totals_dir, c) for c in candidates
                 if os.path.exists(os.path.join(totals_dir, c))), None)
    if path is None:
        return none
    vals = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            try:
                vals.append(int(rec["decode_batch_size"]))
            except (KeyError, ValueError):
                continue
    if not vals:
        return none
    peak = max(vals)
    # Sustained max = peak of the rolling mean (window clamped to series length),
    # so a 1-step spike can't define the maximum.
    w = min(ROLLING_WINDOW, len(vals))
    rolling = [sum(vals[i:i + w]) / w for i in range(len(vals) - w + 1)]
    return {
        "mean_decode_batch_size": round(sum(vals) / len(vals), 4),
        "sustained_max_decode_batch_size": round(max(rolling), 4),
        "steps_at_max_decode_batch_size": sum(1 for v in vals if v == peak),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate single-shot scenario [Summary] lines into one CSV.")
    ap.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS,
                    help=f"dir holding scenario subdirs (default: {DEFAULT_RESULTS})")
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT,
                    help=f"output CSV path (default: {DEFAULT_OUT})")
    ap.add_argument("--totals-dir", default=None,
                    help="dir holding extract_decode_running totals CSVs for "
                         "mean_total_running (default: the output CSV's dir)")
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
        row.update(throughput_stats(scenario_dir))
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
              + METRIC_FIELDS + THROUGHPUT_FIELDS + CONCURRENCY_FIELDS)

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
