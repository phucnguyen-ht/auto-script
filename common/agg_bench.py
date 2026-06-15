#!/usr/bin/env python3
"""Build CSV tables from random-dataset `vllm bench serve --save-result` results.

Usage: agg_bench.py <run_dir>
Reads <run_dir>/run<i>/<scenario>.json and writes:
  <run_dir>/run<i>.csv   # one table per run  (rows = scenarios, cols = metrics)
  <run_dir>/mean.csv     # mean over the runs (rows = scenarios, cols = metrics)
  <run_dir>/std.csv      # sample std over the runs
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

run_dir = sys.argv[1]

# Metric columns (X axis), in display order; only those present are emitted.
COLS = [
    "duration", "completed", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p50_itl_ms", "p90_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p50_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
]


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# runs[i] = {scenario_label: {metric: value}}
runs = {}
for rd in glob.glob(os.path.join(run_dir, "run*/")):
    m = re.search(r"run(\d+)", os.path.basename(os.path.normpath(rd)))
    if not m:
        continue
    i = int(m.group(1))
    runs[i] = {}
    for jf in sorted(glob.glob(os.path.join(rd, "*.json"))):
        scen = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf) as fh:
                runs[i][scen] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[agg] skip {jf}: {exc}")

if not runs:
    print(f"[agg] no run*/ result json under {run_dir}")
    sys.exit(0)

present = set()
for i in runs:
    for d in runs[i].values():
        present |= {k for k, v in d.items() if is_num(v)}
cols = [c for c in COLS if c in present]

# scenarios in first-seen order across runs
scenarios = []
for i in sorted(runs):
    for scen in runs[i]:
        if scen not in scenarios:
            scenarios.append(scen)


def write_table(path, cell):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario"] + cols)
        for scen in scenarios:
            w.writerow([scen] + [cell(scen, c) for c in cols])


# one CSV per run
for i in sorted(runs):
    write_table(
        os.path.join(run_dir, f"run{i}.csv"),
        lambda scen, c, i=i: runs[i].get(scen, {}).get(c, ""),
    )


def across(scen, c, how):
    vals = [
        runs[i][scen][c]
        for i in runs
        if scen in runs[i] and is_num(runs[i][scen].get(c))
    ]
    if not vals:
        return ""
    if how == "mean":
        return round(statistics.fmean(vals), 4)
    return round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0


write_table(os.path.join(run_dir, "mean.csv"), lambda scen, c: across(scen, c, "mean"))
write_table(os.path.join(run_dir, "std.csv"), lambda scen, c: across(scen, c, "std"))

print(
    f"[agg] {len(runs)} run(s), {len(scenarios)} scenario(s) -> "
    f"run<i>.csv + mean.csv/std.csv in {run_dir}"
)
