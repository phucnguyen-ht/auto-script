#!/usr/bin/env python3
# [MV-4572] Chart phân phối SCHEDULE (per-step decode batch size) để so >=2 run.
# Nguồn: <run>/decode_running/decode_running_<ds>_p<conc>.totals.csv (cột: step,decode_batch_size,ranks_present)
#   -> đã do bench harness ghi sẵn, KHÔNG cần chạy thêm GPU.
# Mục đích: kiểm tra A0 vs A8 có được scheduler xếp batch GIỐNG nhau không (điều kiện để so
#   mean_fwd_dev_ms cho công bằng). Nếu 2 phân phối trùng -> khác biệt device-time là do EPLB,
#   không phải do batch khác.
#
# Dùng:
#   python3 plot_schedule_dist.py OUT.png "A0:/.../A0/decode_running/decode_running_8k_p16.totals.csv" \
#                                         "A8:/.../A8/decode_running/decode_running_8k_p16.totals.csv"
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    df = pd.read_csv(path)
    return df["decode_batch_size"].to_numpy()

def main():
    out = sys.argv[1]
    series = []  # (label, data)
    for spec in sys.argv[2:]:
        label, path = spec.split(":", 1)
        series.append((label, load(path)))

    # ---- summary stats ----
    print(f"{'run':<8} {'steps':>7} {'mean':>8} {'median':>7} {'p90':>5} {'p99':>5} {'max':>4} {'min':>4}")
    for label, d in series:
        print(f"{label:<8} {len(d):>7} {d.mean():>8.3f} {np.median(d):>7.1f} "
              f"{np.percentile(d,90):>5.0f} {np.percentile(d,99):>5.0f} {d.max():>4} {d.min():>4}")

    # ---- KS distance giữa 2 phân phối đầu (nếu có >=2) ----
    if len(series) >= 2:
        try:
            from scipy.stats import ks_2samp
            ks = ks_2samp(series[0][1], series[1][1])
            print(f"\n[KS] {series[0][0]} vs {series[1][0]}: D={ks.statistic:.4f} p={ks.pvalue:.3g}"
                  "  (D nhỏ + p lớn => 2 phân phối schedule ~ trùng)")
        except Exception as e:
            print(f"[KS] skip ({e})")

    # ---- chart: histogram chuẩn hoá (PMF) theo batch-size nguyên ----
    alld = np.concatenate([d for _, d in series])
    lo, hi = int(alld.min()), int(alld.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    centers = np.arange(lo, hi + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    width = 0.8 / len(series)
    for i, (label, d) in enumerate(series):
        h, _ = np.histogram(d, bins=bins, density=False)
        pmf = h / h.sum()
        ax1.bar(centers + (i - (len(series)-1)/2) * width, pmf, width=width,
                label=f"{label} (n={len(d)}, mean={d.mean():.2f})", color=colors[i % len(colors)], alpha=0.85)
    ax1.set_xlabel("per-step decode batch size (running reqs, summed 8 ranks)")
    ax1.set_ylabel("fraction of steps (PMF)")
    ax1.set_title("Phân phối schedule per-step — so sánh run")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # CDF để thấy dịch phân phối
    for i, (label, d) in enumerate(series):
        xs = np.sort(d)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, label=label, color=colors[i % len(colors)], lw=2)
    ax2.set_xlabel("per-step decode batch size")
    ax2.set_ylabel("CDF")
    ax2.set_title("CDF (trùng nhau => schedule giống nhau)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\n[saved] {out}")

if __name__ == "__main__":
    main()
