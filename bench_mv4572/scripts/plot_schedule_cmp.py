#!/usr/bin/env python3
# [MV-4572] So phân phối SCHEDULE giữa các run (vd A0 vs A8) — bench-only, đã lọc artifact.
# Input: mỗi run 1 thư mục có decode_running/decode_running_<ds>_p<conc>.{csv,totals.csv}.
#   - totals.csv : step, decode_batch_size (Σ 8 rank), ranks_present
#   - detail csv : rank, step, phase, running_reqs  (width forward THẬT mỗi rank)
# 2 view:
#   (A) CLUSTER decode batch (lọc ranks_present==8 -> bỏ tail ordinal-artifact)
#   (B) PER-RANK running_reqs (phase==decode) -> đúng cái mỗi rank forward xử lý
# Mục đích: xác nhận A0/A8 có được scheduler feed cùng phân phối không (điều kiện so device-time
#   công bằng). Trùng -> chênh device-time là do EPLB; lệch -> phải bucket theo batch.
#
# Dùng: python3 plot_schedule_cmp.py OUT.png DS PCONC "A0:/.../A0" "A8:/.../A8"
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ks(a, b):
    try:
        from scipy.stats import ks_2samp
        r = ks_2samp(a, b)
        return f"D={r.statistic:.4f} p={r.pvalue:.3g}"
    except Exception:
        # KS thủ công (không cần scipy)
        xs = np.sort(np.unique(np.concatenate([a, b])))
        ca = np.searchsorted(np.sort(a), xs, side="right") / len(a)
        cb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
        return f"D={np.max(np.abs(ca - cb)):.4f} (manual)"

def stats(d):
    return (f"n={len(d)} mean={d.mean():.3f} median={np.median(d):.1f} "
            f"p90={np.percentile(d,90):.0f} p99={np.percentile(d,99):.0f} max={d.max()} min={d.min()}")

def main():
    out, ds, pc = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = []  # (label, cluster_full8, perrank_decode)
    for spec in sys.argv[4:]:
        label, d = spec.split(":", 1)
        tot = pd.read_csv(f"{d}/decode_running/decode_running_{ds}_p{pc}.totals.csv")
        det = pd.read_csv(f"{d}/decode_running/decode_running_{ds}_p{pc}.csv")
        cluster = tot.loc[tot["ranks_present"] == 8, "decode_batch_size"].to_numpy()
        perrank = det.loc[det["phase"] == "decode", "running_reqs"].to_numpy()
        runs.append((label, cluster, perrank))
        print(f"[{label}] CLUSTER(ranks=8): {stats(cluster)}")
        print(f"[{label}] PER-RANK(decode): {stats(perrank)}")
    if len(runs) >= 2:
        print(f"\n[KS cluster] {runs[0][0]} vs {runs[1][0]}: {ks(runs[0][1], runs[1][1])}")
        print(f"[KS per-rank] {runs[0][0]} vs {runs[1][0]}: {ks(runs[0][2], runs[1][2])}")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # (A) cluster PMF
    ax = axes[0]
    alld = np.concatenate([c for _, c, _ in runs])
    bins = np.arange(alld.min() - 0.5, alld.max() + 1.5)
    centers = np.arange(int(alld.min()), int(alld.max()) + 1)
    w = 0.8 / len(runs)
    for i, (lb, c, _) in enumerate(runs):
        h, _ = np.histogram(c, bins=bins)
        ax.bar(centers + (i - (len(runs)-1)/2)*w, h/h.sum(), width=w,
               color=colors[i % 4], alpha=0.85, label=f"{lb} (mean={c.mean():.2f})")
    ax.set_title(f"(A) CLUSTER decode batch (ranks_present==8)\nΣ 8 rank / step — {ds} conc{pc}")
    ax.set_xlabel("decode_batch_size (Σ 8 rank)"); ax.set_ylabel("fraction of steps"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    # (B) per-rank PMF
    ax = axes[1]
    allp = np.concatenate([p for _, _, p in runs])
    bins2 = np.arange(allp.min() - 0.5, allp.max() + 1.5)
    centers2 = np.arange(int(allp.min()), int(allp.max()) + 1)
    for i, (lb, _, p) in enumerate(runs):
        h, _ = np.histogram(p, bins=bins2)
        ax.bar(centers2 + (i - (len(runs)-1)/2)*w, h/h.sum(), width=w,
               color=colors[i % 4], alpha=0.85, label=f"{lb} (mean={p.mean():.2f})")
    ax.set_title("(B) PER-RANK running_reqs (decode)\nwidth forward THẬT mỗi rank")
    ax.set_xlabel("running_reqs / rank / decode-step"); ax.set_ylabel("fraction of rank-steps"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    # (C) CDF cluster
    ax = axes[2]
    for i, (lb, c, _) in enumerate(runs):
        xs = np.sort(c); ys = np.arange(1, len(xs)+1)/len(xs)
        ax.plot(xs, ys, color=colors[i % 4], lw=2, label=lb)
    ax.set_title("(C) CDF cluster batch (trùng => schedule giống)")
    ax.set_xlabel("decode_batch_size (Σ 8 rank)"); ax.set_ylabel("CDF"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\n[saved] {out}")

if __name__ == "__main__":
    main()
