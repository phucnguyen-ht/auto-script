#!/usr/bin/env python3
# [MV-4572] Per-step scheduling timeseries across all 8 DP ranks, V1 vs V2 (EPLB-on norearr).
# Parses [MV-4572][V1|V2] serve-log probes (dp_rank/num_reqs/num_toks/padded/cg_mode),
# reconstructs the per-rank step trajectory, and renders charts to docs/regression/vis/.
# Goal (user): visualize the FULL #tokens/#requests-scheduled trajectory per step per rank,
# to test whether V2 dispatches differently (more/smaller forwards) than V1.
#   python3 sched_timeseries.py <V2_serve.log> <V1_serve.log> <out_dir>
import sys, re, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"\[MV-4572\]\[(V\d)\] dp_rank=(\d+) num_reqs=(\d+) num_toks=(\d+) padded=(\d+) cg_mode=(\w+)"
)
PREFILL_TOK = 32  # decode batch (conc16/dp8 ~ few reqs/rank) is small; >32 tok => (chunked) prefill

def parse(path):
    # per rank: list of (num_reqs, num_toks, padded, cg_mode) in step order
    ranks = collections.defaultdict(list)
    ver = None
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            m = LINE.search(ln)
            if not m:
                continue
            ver = m.group(1)
            r = int(m.group(2))
            ranks[r].append((int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)))
    return ver, ranks

def stats(ranks):
    out = {}
    for r, seq in ranks.items():
        reqs = np.array([s[0] for s in seq])
        toks = np.array([s[1] for s in seq])
        is_pf = toks > PREFILL_TOK
        dec = toks[~is_pf]
        out[r] = dict(
            n_steps=len(seq),
            n_prefill=int(is_pf.sum()),
            n_decode=int((~is_pf).sum()),
            tot_tok=int(toks.sum()),
            tot_dec_tok=int(dec.sum()),
            tot_pf_tok=int(toks[is_pf].sum()),
            mean_dec_batch=float(dec.mean()) if len(dec) else 0.0,
            reqs=reqs, toks=toks, is_pf=is_pf,
        )
    return out

def main():
    v2p, v1p, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    _, r2 = parse(v2p)
    _, r1 = parse(v1p)
    s2, s1 = stats(r2), stats(r1)

    # ---------- text summary + reconciliation ----------
    lines = []
    def p(x): lines.append(x); print(x)
    p("================ SCHEDULING RECONCILIATION (V2 vs V1, EPLB-on norearr) ================")
    p(f"{'rank':>4} | {'V2_steps':>9} {'V1_steps':>9} {'stepΔ%':>7} | "
      f"{'V2_decTok':>10} {'V1_decTok':>10} {'tokΔ%':>7} | "
      f"{'V2_batch':>8} {'V1_batch':>8} | {'V2_pf':>5} {'V1_pf':>5}")
    agg = collections.defaultdict(float)
    for r in range(8):
        if r not in s2 or r not in s1:
            continue
        a, b = s2[r], s1[r]
        stepd = 100 * (a["n_steps"] - b["n_steps"]) / b["n_steps"]
        tokd = 100 * (a["tot_dec_tok"] - b["tot_dec_tok"]) / b["tot_dec_tok"] if b["tot_dec_tok"] else 0
        p(f"{r:>4} | {a['n_steps']:>9} {b['n_steps']:>9} {stepd:>6.1f}% | "
          f"{a['tot_dec_tok']:>10} {b['tot_dec_tok']:>10} {tokd:>6.1f}% | "
          f"{a['mean_dec_batch']:>8.3f} {b['mean_dec_batch']:>8.3f} | "
          f"{a['n_prefill']:>5} {b['n_prefill']:>5}")
        agg["v2_steps"] += a["n_steps"]; agg["v1_steps"] += b["n_steps"]
        agg["v2_tok"] += a["tot_dec_tok"]; agg["v1_tok"] += b["tot_dec_tok"]
    p("-" * 96)
    spt2 = agg['v2_steps'] / agg['v2_tok']
    spt1 = agg['v1_steps'] / agg['v1_tok']
    p(f"TOTAL steps  V2={int(agg['v2_steps'])}  V1={int(agg['v1_steps'])}  "
      f"(V2/V1={agg['v2_steps']/agg['v1_steps']:.3f})")
    p(f"TOTAL decTok V2={int(agg['v2_tok'])}  V1={int(agg['v1_tok'])}  "
      f"(V2/V1={agg['v2_tok']/agg['v1_tok']:.3f})")
    p(f"** WORK-NORMALIZED: steps/decode-token  V2={spt2:.4f}  V1={spt1:.4f}  "
      f"(V2/V1={spt2/spt1:.3f})  <- confound-robust metric **")
    p("")
    p("READ (EPLB-on axis r0_v2 vs r0_noV2, per rule: base = reference only):")
    p(f"  - V2 = strict DP lockstep: all 8 ranks uniform forward count (16547). V1 independent:")
    p("    idle ranks skip forwards (uneven 9152..13205). => V2 issues far more total forwards.")
    p(f"  - Raw totals confounded (warmup-in-log + ~1% num_prompts from adaptive KV); use the")
    p(f"    RATIO. Work-normalized: V2 spends ~{100*(spt2/spt1-1):.0f}% more forward-steps per decode-token")
    p("    than V1 -- close to the measured +10.5% tpot regression.")
    p("  - CANDIDATE cause (lockstep x EPLB interaction, NOT refuted by base): V2's many lockstep")
    p("    forwards each pay EPLB per-forward cost / wait on the slowest rank's MoE. To CONFIRM:")
    p("    clean matched bench (equal KV, warmup excluded) + per-step wall-time. NOT yet causal.")
    with open(os.path.join(outdir, "SCHED_SUMMARY.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------- Chart 1: step-count per rank (headline: lockstep vs independent) ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(8)
    w = 0.38
    v2s = [s2[r]["n_steps"] if r in s2 else 0 for r in range(8)]
    v1s = [s1[r]["n_steps"] if r in s1 else 0 for r in range(8)]
    ax.bar(xs - w/2, v2s, w, label="V2 (MODEL_RUNNER_V2)", color="#d1495b")
    ax.bar(xs + w/2, v1s, w, label="V1", color="#3d7ea6")
    ax.set_xlabel("DP rank"); ax.set_ylabel("total forward steps (full bench)")
    ax.set_title("Forward-step count per rank — V2 lockstep (uniform) vs V1 independent (uneven)\n"
                 "EPLB-on norearr, longbenchv2-8k conc16")
    ax.set_xticks(xs); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i in range(8):
        ax.text(i - w/2, v2s[i], str(v2s[i]), ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, v1s[i], str(v1s[i]), ha="center", va="bottom", fontsize=7)
    # optional base_v2 reference (context: lockstep is a V2 trait, seen at base too)
    if len(sys.argv) > 4 and os.path.exists(sys.argv[4]):
        _, rb = parse(sys.argv[4]); sb = stats(rb)
        bsteps = [sb[r]["n_steps"] if r in sb else 0 for r in range(8)]
        ax.plot(xs, bsteps, "k--o", ms=4, lw=1, label="base_v2 (EPLB-off, reference)")
        ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sched_stepcount_per_rank.png"), dpi=130)
    plt.close(fig)

    # ---------- Chart 1b: WORK-NORMALIZED steps per decode-token per rank (money metric) ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    v2n = [s2[r]["n_steps"]/s2[r]["tot_dec_tok"] if r in s2 and s2[r]["tot_dec_tok"] else 0 for r in range(8)]
    v1n = [s1[r]["n_steps"]/s1[r]["tot_dec_tok"] if r in s1 and s1[r]["tot_dec_tok"] else 0 for r in range(8)]
    ax.bar(xs - w/2, v2n, w, label="V2", color="#d1495b")
    ax.bar(xs + w/2, v1n, w, label="V1", color="#3d7ea6")
    ax.set_xlabel("DP rank"); ax.set_ylabel("forward-steps / decode-token")
    ax.set_title("Work-normalized dispatch cost (EPLB-on) — V2 issues more forwards per token generated\n"
                 "confound-robust: removes num_prompts / warmup differences")
    ax.set_xticks(xs); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sched_steps_per_token.png"), dpi=130)
    plt.close(fig)

    # ---------- Chart 2 (V2) + Chart 3 (V1): per-rank num_reqs & num_toks trajectory ----------
    for ver, sdat, fname in [("V2", s2, "sched_timeseries_V2.png"),
                             ("V1", s1, "sched_timeseries_V1.png")]:
        fig, axes = plt.subplots(8, 1, figsize=(13, 15), sharex=False)
        for r in range(8):
            ax = axes[r]
            if r not in sdat:
                ax.set_visible(False); continue
            reqs = sdat[r]["reqs"]; toks = sdat[r]["toks"]
            x = np.arange(len(reqs))
            ax.plot(x, toks, lw=0.4, color="#d1495b", rasterized=True, label="num_toks")
            ax.plot(x, reqs, lw=0.4, color="#2e7d32", rasterized=True, alpha=0.8, label="num_reqs")
            ax.set_yscale("symlog")
            ax.set_ylabel(f"rank{r}", fontsize=9)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"{ver}: per-step scheduling trajectory (all 8 ranks) — "
                             f"num_toks (red) & num_reqs (green), symlog y")
                ax.legend(loc="upper right", fontsize=8)
            if r == 7:
                ax.set_xlabel("forward step index")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), dpi=110)
        plt.close(fig)

    # ---------- Chart 4: decode batch-size histogram (V1 vs V2) ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d2 = np.concatenate([s2[r]["toks"][~s2[r]["is_pf"]] for r in s2])
    d1 = np.concatenate([s1[r]["toks"][~s1[r]["is_pf"]] for r in s1])
    bins = np.arange(0.5, max(d2.max(), d1.max()) + 1.5, 1)
    ax.hist(d2, bins=bins, alpha=0.55, label=f"V2 (n={len(d2)}, mean={d2.mean():.2f})", color="#d1495b")
    ax.hist(d1, bins=bins, alpha=0.55, label=f"V1 (n={len(d1)}, mean={d1.mean():.2f})", color="#3d7ea6")
    ax.set_xlabel("decode num_toks per forward (all ranks pooled)")
    ax.set_ylabel("count of forward steps")
    ax.set_title("Decode batch-size distribution — does V2 fragment into smaller forwards?\n"
                 "EPLB-on norearr, longbenchv2-8k conc16")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sched_decode_batchsize_hist.png"), dpi=130)
    plt.close(fig)

    # ---------- Chart 5: cumulative decode tokens over steps (rank0) ----------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for ver, sdat, col in [("V2", s2, "#d1495b"), ("V1", s1, "#3d7ea6")]:
        r0 = sdat[0]
        toks = r0["toks"].copy(); toks[r0["is_pf"]] = 0  # decode only
        ax.plot(np.arange(len(toks)), np.cumsum(toks), color=col, lw=1.3,
                label=f"{ver} rank0 (final={int(np.cumsum(toks)[-1])} in {len(toks)} steps)")
    ax.set_xlabel("forward step index (rank0)")
    ax.set_ylabel("cumulative decode tokens")
    ax.set_title("Cumulative decode tokens vs step (rank0) — same total work, V2 needs more steps")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sched_cumulative_tokens_rank0.png"), dpi=130)
    plt.close(fig)

    print(f"\nWrote charts + SCHED_SUMMARY.txt to {outdir}")

if __name__ == "__main__":
    main()
