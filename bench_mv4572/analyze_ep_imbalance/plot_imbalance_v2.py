#!/usr/bin/env python3
# Plot EP time-imbalance (barrier-aligned) — EPLB ON (r0) vs OFF (base).
# Consumes the *_v2_arrays.npz written by analyze_time_v2.py.
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    z = np.load(p)
    return z["step_busy"], z["per_step_mm"], z["ranks"]


def windowed_curve(step_busy, Ws):
    R, S = step_busy.shape
    xs, ys = [], []
    for W in Ws:
        nW = S // W
        if nW == 0: continue
        block = step_busy[:, :nW*W].reshape(R, nW, W).sum(2)
        mm = block.max(0) / np.maximum(block.min(0), 1e-9)
        xs.append(W); ys.append(mm.mean())
    return xs, ys


def roll(a, w):
    if len(a) < w: return a
    return np.convolve(a, np.ones(w)/w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="base (EPLB off) *_v2_arrays.npz")
    ap.add_argument("--on", required=True, help="r0 (EPLB on) *_v2_arrays.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="EP MoE-busy imbalance — Kimi-2.6-MXFP4 100k/c16 (barrier-aligned)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sb_off, mm_off, ranks = load(args.off)
    sb_on, mm_on, _ = load(args.on)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    # A: per-step imbalance over time (rolling mean, W=50 steps)
    a = ax[0, 0]
    w = 50
    a.plot(roll(mm_off, w), color="#2166ac", lw=1.1, label=f"OFF (base) mean={mm_off.mean():.3f}")
    a.plot(roll(mm_on, w), color="#b2182b", lw=1.1, label=f"ON (r0)   mean={mm_on.mean():.3f}")
    a.axhline(mm_off.mean(), color="#2166ac", ls="--", lw=0.7, alpha=0.6)
    a.axhline(mm_on.mean(), color="#b2182b", ls="--", lw=0.7, alpha=0.6)
    a.set_title(f"A. Per-step imbalance max/min over time (rolling {w} steps)")
    a.set_xlabel("step"); a.set_ylabel("max_rank / min_rank MoE-busy"); a.legend(); a.grid(alpha=0.3)

    # B: histogram of per-step imbalance
    b = ax[0, 1]
    bins = np.linspace(1.0, np.percentile(np.concatenate([mm_off, mm_on]), 99.5), 60)
    b.hist(mm_off, bins=bins, color="#2166ac", alpha=0.55, label=f"OFF median={np.median(mm_off):.3f}")
    b.hist(mm_on, bins=bins, color="#b2182b", alpha=0.55, label=f"ON median={np.median(mm_on):.3f}")
    b.axvline(np.median(mm_off), color="#2166ac", ls="--", lw=1)
    b.axvline(np.median(mm_on), color="#b2182b", ls="--", lw=1)
    b.set_title("B. Distribution of per-step imbalance"); b.set_xlabel("max/min"); b.set_ylabel("#steps"); b.legend(); b.grid(alpha=0.3)

    # C: per-rank mean MoE-busy per step (who is hot)
    c = ax[1, 0]
    x = np.arange(len(ranks)); wbar = 0.4
    c.bar(x-wbar/2, sb_off.mean(1), wbar, color="#2166ac", label="OFF")
    c.bar(x+wbar/2, sb_on.mean(1), wbar, color="#b2182b", label="ON")
    c.set_xticks(x); c.set_xticklabels([f"r{r}" for r in ranks])
    c.set_title("C. Per-rank mean MoE-busy per step (µs)"); c.set_xlabel("rank"); c.set_ylabel("mean busy/step (µs)"); c.legend(); c.grid(alpha=0.3, axis="y")

    # D: windowed imbalance vs window size
    d = ax[1, 1]
    Ws = [1, 100, 250, 500, 1000, 2000, 3000]
    xo, yo = windowed_curve(sb_off, Ws); xn, yn = windowed_curve(sb_on, Ws)
    d.plot(xo, yo, "o-", color="#2166ac", label="OFF (base)")
    d.plot(xn, yn, "o-", color="#b2182b", label="ON (r0)")
    d.set_xscale("log"); d.set_title("D. Imbalance vs aggregation window W (steps)")
    d.set_xlabel("W (steps, log)"); d.set_ylabel("mean max/min"); d.legend(); d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outp = os.path.join(args.out, "imbalance_on_vs_off.png")
    fig.savefig(outp, dpi=130)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
