#!/usr/bin/env python3
# [MV-4572] Vẽ schedule distribution: (1) TOTAL cluster (Σ ranks) + (2) TỪNG rank (DP0..DP7).
# So >=2 run (A0 vs A8). Bench-only (data lấy từ decode_running đã cắt theo bench_start..bench_end).
#   total  : totals.csv  step,decode_batch_size,ranks_present   (lọc ranks_present==8 = vùng sạch)
#   per-rank: detail csv  rank,step,phase,running_reqs           (phase==decode)
# Dùng: python3 plot_schedule_vis.py OUTDIR DS PCONC "A0:/.../A0" "A8:/.../A8"
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

def ksD(a, b):
    xs = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))

def pmf(d, centers):
    h, _ = np.histogram(d, bins=np.append(centers - 0.5, centers[-1] + 0.5))
    return h / h.sum()

def main():
    outdir, ds, pc = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    runs = []  # (label, totals_df, detail_df)
    for spec in sys.argv[4:]:
        label, d = spec.split(":", 1)
        tot = pd.read_csv(f"{d}/decode_running/decode_running_{ds}_p{pc}.totals.csv")
        det = pd.read_csv(f"{d}/decode_running/decode_running_{ds}_p{pc}.csv")
        runs.append((label, tot, det))

    # ================= (1) TOTAL cluster =================
    clusters = []
    for label, tot, _ in runs:
        c = tot.loc[tot["ranks_present"] == 8, "decode_batch_size"].to_numpy()
        clusters.append((label, c))
    allc = np.concatenate([c for _, c in clusters])
    centers = np.arange(int(allc.min()), int(allc.max()) + 1)
    w = 0.8 / len(clusters)

    fig, (axp, axc, axt) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, c) in enumerate(clusters):
        axp.bar(centers + (i - (len(clusters)-1)/2)*w, pmf(c, centers), width=w,
                color=COLORS[i % 4], alpha=0.85,
                label=f"{label} (n={len(c)}, mean={c.mean():.2f}, med={np.median(c):.0f}, max={c.max()})")
    axp.set_title(f"TOTAL cluster decode batch (Σ 8 rank, ranks_present==8)\n{ds} conc{pc} — bench-only")
    axp.set_xlabel("decode_batch_size (Σ 8 rank)"); axp.set_ylabel("fraction of steps")
    axp.legend(); axp.grid(axis="y", alpha=0.3)

    for i, (label, c) in enumerate(clusters):
        xs = np.sort(c); ys = np.arange(1, len(xs)+1)/len(xs)
        axc.plot(xs, ys, color=COLORS[i % 4], lw=2, label=label)
    ttl = "CDF cluster"
    if len(clusters) >= 2:
        ttl += f"  (KS D={ksD(clusters[0][1], clusters[1][1]):.3f})"
    axc.set_title(ttl); axc.set_xlabel("decode_batch_size (Σ 8 rank)"); axc.set_ylabel("CDF")
    axc.legend(); axc.grid(alpha=0.3)

    # timeseries total (order) — full totals, không lọc
    for i, (label, tot, _) in enumerate(runs):
        axt.plot(tot["step"], tot["decode_batch_size"], lw=0.3, color=COLORS[i % 4], alpha=0.8, label=label)
    axt.set_title("TOTAL time-series (giữ order)"); axt.set_xlabel("step index"); axt.set_ylabel("Σ 8 rank")
    axt.legend(); axt.grid(alpha=0.3)
    fig.tight_layout()
    p1 = f"{outdir}/total_{ds}_p{pc}.png"; fig.savefig(p1, dpi=120); plt.close(fig)
    print(f"[saved] {p1}")

    # ================= (2) TỪNG rank DP0..DP7 =================
    ranks = sorted(set().union(*[set(det["rank"].unique()) for _, _, det in runs]),
                   key=lambda r: int(r.replace("DP", "")))
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    axes = axes.ravel()
    # domain chung cho per-rank
    allpr = np.concatenate([det.loc[det["phase"] == "decode", "running_reqs"].to_numpy()
                            for _, _, det in runs])
    prc = np.arange(int(allpr.min()), int(allpr.max()) + 1)
    for ri, rk in enumerate(ranks):
        ax = axes[ri]
        lines = []
        for i, (label, _, det) in enumerate(runs):
            d = det.loc[(det["rank"] == rk) & (det["phase"] == "decode"), "running_reqs"].to_numpy()
            if len(d) == 0:
                continue
            ax.bar(prc + (i - (len(runs)-1)/2)*w, pmf(d, prc), width=w,
                   color=COLORS[i % 4], alpha=0.85, label=f"{label} (mean={d.mean():.2f})")
            lines.append((label, d))
        ttl = rk
        if len(lines) >= 2:
            ttl += f"  KS D={ksD(lines[0][1], lines[1][1]):.3f}"
        ax.set_title(ttl); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        if ri >= 4:
            ax.set_xlabel("running_reqs / decode-step")
        if ri % 4 == 0:
            ax.set_ylabel("fraction of rank-steps")
    fig.suptitle(f"PER-RANK decode running_reqs (width forward THẬT mỗi rank) — {ds} conc{pc}, bench-only", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = f"{outdir}/per_rank_{ds}_p{pc}.png"; fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"[saved] {p2}")

    # summary số liệu per-rank
    print("\nrank  " + "  ".join(f"{lb}_mean" for lb, _, _ in runs) + "   KS(1v2)")
    for rk in ranks:
        vals = []
        arrs = []
        for _, _, det in runs:
            d = det.loc[(det["rank"] == rk) & (det["phase"] == "decode"), "running_reqs"].to_numpy()
            vals.append(f"{d.mean():.3f}" if len(d) else "n/a")
            arrs.append(d)
        ks = f"{ksD(arrs[0], arrs[1]):.3f}" if len(arrs) >= 2 and len(arrs[0]) and len(arrs[1]) else ""
        print(f"{rk:<5} " + "     ".join(vals) + f"     {ks}")

if __name__ == "__main__":
    main()
