#!/usr/bin/env python3
"""Gom kết quả sweep Kimi-K2.6-MXFP4 (EPLB) vào 1 file CSV.

Quét <sweep_root>/<run>/<preset>/conc<N>/<ISL>/ , mỗi thư mục lá gồm:
  - scenario_summary.csv : TTFT / TPOT / throughput / decode-batch-size (do summarize_scenarios.py sinh)
  - serve.log            : dùng để đếm 4 cột thêm quanh MARKER `bench_start`

4 cột thêm (mốc = dòng access-log chứa "bench_start" / "bench_end" trong serve.log):
  warmup_steps       : số engine-step (dòng "[Server-side (scheduler) stats]" của DP0) TRƯỚC bench_start
                       (tức từ sau application-start tới lúc bench bắt đầu -> chủ yếu là warmup)
  bench_steps        : số step (DP0) giữa bench_start..bench_end (cửa sổ đo)
  warmup_rearranges  : số lần "Rearranging experts" (BỎ dòng "(profile)") TRƯỚC bench_start
  bench_rearranges   : số lần "Rearranging experts" (BỎ "(profile)") giữa bench_start..bench_end

Ghi chú:
  - "step" đếm theo 1 rank (DP0) vì PDSLoggingScheduler in 1 dòng/step/rank (LOG_INTERVAL=0);
    các DP rank chạy async nên số step có thể lệch nhẹ giữa rank -> lấy DP0 làm chuẩn.
  - rearrange chỉ được log ở rank chính (ep_rank==0) 1 dòng/lần -> đếm không trùng.
  - Dòng rearrange lúc profile-run (startup, "(profile)") bị loại vì không do tải thật.
  - preset name được tách thành eplb/communicator/mode/step_interval/redundant để dễ gom cụm.

Cách dùng:
    python3 aggregate_sweep.py [sweep_root] [out.csv]
Mặc định: sweep_root = <thư mục script>/../logs/sweep_results
          out        = <sweep_root>/aggregated_sweep.csv
"""
import csv
import glob
import os
import re
import sys

ANSI = re.compile(r"\033\[[0-9;]*m")
STATS_MARK = "[Server-side (scheduler) stats]"
DP0_TAG = "[PDS - DP0]"
REARRANGE = "Rearranging experts"
PROFILE = "(profile)"

METRIC_COLS = [
    "requests", "mean_ttft_ms", "p90_ttft_ms", "mean_decode_tps", "mean_tpot_ms",
    "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "total_output_tokens",
    "benchmark_duration_s", "output_tps", "mean_decode_batch_size",
    "sustained_max_decode_batch_size", "steps_at_max_decode_batch_size",
]
EXTRA_COLS = ["warmup_steps", "bench_steps", "warmup_rearranges", "bench_rearranges"]
CONFIG_COLS = ["run", "preset", "eplb", "communicator", "mode", "step_interval",
               "redundant", "concurrency", "ISL"]
OUT_COLS = CONFIG_COLS + EXTRA_COLS + METRIC_COLS


def parse_preset(p):
    """base | base-eplb-<comm>-<mode>-<step>-<red>  ->  dict cột cấu hình."""
    if p == "base":
        return dict(eplb="off", communicator="", mode="", step_interval="", redundant="")
    if p.startswith("base-eplb-"):
        toks = p[len("base-eplb-"):].split("-")
        comm = toks[0] if len(toks) > 0 else ""
        mode = toks[1] if len(toks) > 1 else ""
        step = toks[2] if len(toks) > 2 else ""   # s250 / default
        red = toks[3] if len(toks) > 3 else ""     # r0 / r8 / r16
        step = step[1:] if step.startswith("s") and step[1:].isdigit() else step
        red = red[1:] if red.startswith("r") and red[1:].isdigit() else red
        return dict(eplb="on", communicator=comm, mode=mode, step_interval=step, redundant=red)
    return dict(eplb="?", communicator="", mode="", step_interval="", redundant="")


def scan_serve_log(path):
    """1 lần duyệt (streaming, tiết kiệm RAM) -> 4 cột thêm.

    Trả về "" hết nếu thiếu serve.log hoặc không thấy marker bench_start."""
    empty = dict.fromkeys(EXTRA_COLS, "")
    if not os.path.exists(path):
        return empty

    res = {k: 0 for k in EXTRA_COLS}
    seen_start = seen_end = has_start = False
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = ANSI.sub("", raw)
            # mốc cửa sổ (là dòng access-log của GET /v1/bench_start | bench_end)
            if not seen_start and "bench_start" in line:
                seen_start = has_start = True
                continue
            if seen_start and not seen_end and "bench_end" in line:
                seen_end = True
                continue
            is_step = STATS_MARK in line and DP0_TAG in line
            is_rearr = (REARRANGE in line) and (PROFILE not in line)
            if not (is_step or is_rearr):
                continue
            if not seen_start:                       # trước bench_start = warmup
                if is_step:
                    res["warmup_steps"] += 1
                if is_rearr:
                    res["warmup_rearranges"] += 1
            elif not seen_end:                       # trong cửa sổ bench
                if is_step:
                    res["bench_steps"] += 1
                if is_rearr:
                    res["bench_rearranges"] += 1
            # sau bench_end: bỏ qua
    if not has_start:
        return empty
    return res


def sort_key(r):
    # Thứ tự: 1) concurrency  2) eplb off -> on  3) default interval -> các interval số
    #         4) redundant tăng dần  (còn lại: communicator/mode/run làm tiebreak ổn định)
    conc_n = int(r["concurrency"]) if str(r["concurrency"]).isdigit() else -1
    eplb_rank = 0 if r["eplb"] == "off" else 1
    step = r["step_interval"]
    if r["eplb"] == "off":
        step_rank = -1                     # base (off): xếp trước, không xét interval
    elif step == "default":
        step_rank = 0                      # default TRƯỚC các interval số
    elif str(step).isdigit():
        step_rank = int(step)              # 100 < 250 < 500 < 1000
    else:
        step_rank = 10 ** 9
    red_n = int(r["redundant"]) if str(r["redundant"]).isdigit() else -1
    return (conc_n, eplb_rank, step_rank, red_n, r["communicator"], r["mode"], r["run"])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "logs", "sweep_results")
    root = os.path.abspath(root)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "aggregated_sweep.csv")

    rows = []
    pattern = os.path.join(root, "*", "*", "conc*", "*", "scenario_summary.csv")
    for csv_path in glob.glob(pattern):
        leaf = os.path.dirname(csv_path)                 # <root>/<run>/<preset>/concN/<ISL>
        rel = os.path.relpath(leaf, root).split(os.sep)  # [run, preset, concN, ISL]
        run, preset = rel[0], rel[1]
        cfg = parse_preset(preset)
        extra = scan_serve_log(os.path.join(leaf, "serve.log"))

        with open(csv_path, "r", errors="replace") as fh:
            data_rows = list(csv.DictReader(fh))
        if not data_rows:
            print(f"[warn] empty summary: {csv_path}", file=sys.stderr)
            continue
        for dr in data_rows:
            row = {"run": run, "preset": preset, **cfg}
            row["concurrency"] = dr.get("concurrency", "")
            row["ISL"] = dr.get("ISL", "")
            row.update(extra)
            for c in METRIC_COLS:
                row[c] = dr.get(c, "")
            rows.append(row)

    if not rows:
        sys.exit(f"Không tìm thấy scenario_summary.csv nào dưới {root}")

    rows.sort(key=sort_key)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] {len(rows)} dòng -> {out}")


if __name__ == "__main__":
    main()
