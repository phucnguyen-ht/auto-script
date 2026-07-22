#!/usr/bin/env python3
"""Extract per-rank, per-step *running-request* counts from a vllm-moreh serve
log (PDSLoggingScheduler) and summarise the cluster-wide concurrency.

Requires the preset to log every step:
    VLLM_MOREH_SCHEDULER_LOGGING=1
    VLLM_MOREH_SCHEDULER_LOG_INTERVAL=0
(the GLM-5.2 MTP dp8-ep8 preset already sets these).

How it works
------------
* Only the log between `bench_start` and `bench_end` markers (the
  formal measurement window emitted by unit_test.py) is used. A single serve
  log usually holds *several* such windows -- one per benchmark scenario, in
  the order they were run. multi_process_test.py runs `for dataset: for
  parallel_threads:` (dataset-outer, concurrency-inner, no seeds), so window
  order is 8k/p16, 8k/p32, ... 8k/p256, 10k/p16, ... By default every window is
  extracted into its **own** result file; `--window N` restricts to one.
* Each `[Server-side (scheduler) stats]` line is exactly one engine step
  (LOG_INTERVAL=0 -> printed every step). With worker mode "both" each line
  carries a [Prefill] and a [Decode] section:
      prompt throughput     > 0  -> prefill ran this step
      generation throughput > 0  -> decode tokens generated this step
* running_reqs assigned per step:
      decode step (gen>0 and prompt==0) -> `Running: N reqs` from the log
      otherwise (prefill / idle step)   -> 0
  i.e. a prefill step is treated as a decode step with running_reqs = 0.

Cluster total per step
----------------------
DP ranks run asynchronously, so there is no shared wall clock. We align by
per-rank step ordinal: at ordinal i the cluster total = sum of running_reqs
over every rank that has an i-th step. We then report mean / max / median / p90
of those per-step totals.

Outputs (one set per scenario/window)
-------------------------------------
* <base>_<scenario>.csv             long CSV: rank, step, phase, running_reqs
* <base>_<scenario>.totals.csv      per-step totals: step, total_running,
                                     ranks_present
and prints the mean/max/median/p90 summary for each scenario to stdout.

`<scenario>` is taken from --scenarios (default: the dataset-outer
<dataset>_p<threads> sweep, in window order) falling back to `wN` when there
are more windows than labels.

Examples
--------
    # all windows -> decode_running_8k_p16.csv, _8k_p32.csv, ... _100k_p256.csv
    python3 extract_decode_running.py

    # custom log + output base
    python3 extract_decode_running.py server.log out

    # just the 2nd window
    python3 extract_decode_running.py server.log out --window 2
"""
import argparse
import csv
import os
import re
import sys

DEFAULT_LOG = "/workspace/vllm-moreh/benchmark_zhipu/server.log"
DEFAULT_OUT = "/workspace/vllm-moreh/benchmark_zhipu/decode_running"

# Sweep definition -- MUST stay in sync with multi_process_test.py. The serve
# log bench_start/bench_end markers do NOT carry scenario identity, so each
# window is labelled purely by its order in the log. multi_process_test.py runs
# `for dataset: for parallel_threads:` (dataset-outer, concurrency-inner, no
# seeds), so the label list below is built in that exact order and window i maps
# to SWEEP labels[i]. (Override with --scenarios for other sweeps, e.g. the
# single-shot p<threads>_d<dataset>_s<seed> layout.)
SWEEP_DATASETS = ["8k", "10k", "100k"]
SWEEP_PARALLEL_THREADS = [16, 32, 64, 96, 128, 192, 256]
DEFAULT_SCENARIOS = [
    f"{d}_p{p}"
    for d in SWEEP_DATASETS
    for p in SWEEP_PARALLEL_THREADS
]

ANSI = re.compile(r"\033\[[0-9;]*m")
RANK = re.compile(r"\[PDS - (DP\d+)\]")
RUNNING = re.compile(r"Running:\s*(\d+)\s*reqs")
PROMPT_TPUT = re.compile(r"Avg prompt throughput:\s*([\d.]+)\s*toks/sec")
GEN_TPUT = re.compile(r"Avg generation throughput:\s*([\d.]+)\s*toks/sec")
TIMESTAMP = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STATS_MARK = "[Server-side (scheduler) stats]"


def percentile(xs, q):
    """np.percentile-equivalent (linear interpolation). 0.0 on empty input."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def find_windows(lines):
    """Return [(start_idx, end_idx), ...] for every bench_start..end pair.

    A start with no following end (truncated log) extends to EOF.
    """
    windows, start = [], None
    for i, ln in enumerate(lines):
        if "bench_start" in ln:
            if start is not None:           # back-to-back start: close prior here
                windows.append((start, i))
            start = i
        elif start is not None and "bench_end" in ln:
            windows.append((start, i))
            start = None
    if start is not None:
        print("[warn] last bench_start has no matching end; using EOF.")
        windows.append((start, len(lines)))
    return windows


def parse_step(line):
    """-> (timestamp, rank, running, prompt_tput, gen_tput) or None."""
    s = ANSI.sub("", line)
    mr, mrun, mg, mt = (RANK.search(s), RUNNING.search(s),
                        GEN_TPUT.search(s), TIMESTAMP.search(s))
    if not (mr and mrun and mg and mt):
        return None
    mp = PROMPT_TPUT.search(s)
    return (mt.group(1), mr.group(1), int(mrun.group(1)),
            float(mp.group(1)) if mp else 0.0, float(mg.group(1)))


def extract_window(window_lines):
    """Parse the stats lines of one window.

    Returns (long_rows, per_rank) or (None, None) if no step lines found.
    """
    per_rank = {}     # rank -> [running_value, ...] in step order
    long_rows = []    # (rank, step, phase, running_value)
    for ln in window_lines:
        if STATS_MARK not in ln:
            continue
        parsed = parse_step(ln)
        if parsed is None:
            continue
        _ts, rank, running, prompt_tput, gen_tput = parsed
        is_decode = gen_tput > 0.0 and prompt_tput == 0.0
        value = running if is_decode else 0          # prefill/idle -> 0
        phase = "decode" if is_decode else ("prefill" if prompt_tput > 0 else "idle")
        seq = per_rank.setdefault(rank, [])
        long_rows.append((rank, len(seq), phase, value))
        seq.append(value)
    if not per_rank:
        return None, None
    return long_rows, per_rank


def write_scenario(base, scenario, long_rows, per_rank):
    """Write the detail + totals CSVs for one scenario and print its summary."""
    detail_csv = f"{base}_{scenario}.csv"
    totals_csv = f"{base}_{scenario}.totals.csv"

    with open(detail_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "step", "phase", "running_reqs"])
        w.writerows(long_rows)

    # Cluster total per step ordinal: sum over ranks present at that ordinal.
    max_len = max(len(v) for v in per_rank.values())
    totals, totals_rows = [], []
    for i in range(max_len):
        present = [v[i] for v in per_rank.values() if i < len(v)]
        totals.append(sum(present))
        totals_rows.append((i, sum(present), len(present)))

    with open(totals_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "decode_batch_size", "ranks_present"])
        w.writerows(totals_rows)

    print(f"\n=== scenario '{scenario}' ===")
    print(f"  Ranks: {len(per_rank)}  "
          f"steps/rank: {min(len(v) for v in per_rank.values())}..{max_len}")
    print(f"  Detail CSV : {detail_csv} ({len(long_rows)} rows)")
    print(f"  Totals CSV : {totals_csv} ({len(totals)} steps)")
    print("  Per-step decode batch size (running reqs summed across ranks):")
    print(f"    mean   : {sum(totals) / len(totals):.4f}")
    print(f"    max    : {max(totals)}")
    print(f"    median : {median(totals):.4f}")
    print(f"    p90    : {percentile(totals, 90):.4f}")


def main():
    ap = argparse.ArgumentParser(
        description="Per-rank per-step decode running requests + cluster totals "
                    "from a vllm-moreh PDS scheduler serve log. Splits each "
                    "benchmark scenario (bench window) into its own CSV.")
    ap.add_argument("log", nargs="?", default=DEFAULT_LOG,
                    help=f"serve log path (default: {DEFAULT_LOG})")
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT,
                    help="output base path; results are written as "
                         f"<out>_<scenario>.csv (default: {DEFAULT_OUT})")
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS,
                    help="scenario labels by window order (default: the "
                         f"dataset-outer <dataset>_p<threads> sweep, "
                         f"{len(DEFAULT_SCENARIOS)} labels)")
    ap.add_argument("--window", type=int, default=None,
                    help="extract only this 1-based window (default: all)")
    args = ap.parse_args()

    with open(args.log, "r", errors="replace") as fh:
        lines = fh.readlines()

    windows = find_windows(lines)
    if not windows:
        sys.exit("No bench_start..bench_end windows found in the log.")

    # Drop the .csv extension if the user passed one as the output base.
    base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
    out_dir = os.path.dirname(base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Parse every window and keep only the ones that actually contain scheduler
    # step lines. *Empty* windows (e.g. a stray bench_start/bench_end pair from
    # a manual marker test, or a scenario that produced no scheduler stats) must
    # NOT consume a scenario label -- otherwise every later window would be
    # mislabelled by one. So labels are assigned by order among the real
    # (non-empty) windows only.
    real = []   # (orig_window_no, start, end, long_rows, per_rank)
    empties = []
    for win_no, (start, end) in enumerate(windows, 1):
        long_rows, per_rank = extract_window(lines[start + 1:end])
        if per_rank is None:
            empties.append((win_no, start, end))
            continue
        real.append((win_no, start, end, long_rows, per_rank))

    if empties:
        print(f"[info] {len(empties)} empty window(s) skipped "
              f"(no scheduler step lines), not consuming a label: "
              + ", ".join(f"#{n} (lines {s + 1}..{e + 1})"
                          for n, s, e in empties))

    if not real:
        sys.exit("No scheduler step lines found in any window.")

    print(f"Found {len(windows)} window(s); {len(real)} non-empty "
          f"-> {len(real)} scenario(s).")

    # Select which real windows to process (--window is 1-based over the real
    # windows, so it matches the scenario numbering printed below).
    if args.window is not None:
        if not (1 <= args.window <= len(real)):
            sys.exit(f"--window {args.window} out of range "
                     f"(found {len(real)} non-empty window(s)).")
        selected = [real[args.window - 1]]
        label_offset = args.window - 1
    else:
        selected = real
        label_offset = 0

    written = 0
    for i, (win_no, start, end, long_rows, per_rank) in enumerate(selected):
        label_idx = label_offset + i
        scenario = (args.scenarios[label_idx]
                    if label_idx < len(args.scenarios) else f"w{label_idx + 1}")
        print(f"\nScenario #{label_idx + 1} '{scenario}' "
              f"(log window #{win_no}, lines {start + 1}..{end + 1}, "
              f"{end - start} lines)")
        write_scenario(base, scenario, long_rows, per_rank)
        written += 1

    if written == 0:
        sys.exit("No scheduler step lines found in any selected window.")
    print(f"\nDone: wrote results for {written} scenario(s).")


if __name__ == "__main__":
    main()
