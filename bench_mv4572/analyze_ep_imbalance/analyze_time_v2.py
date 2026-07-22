#!/usr/bin/env python3
# =============================================================================
# MV-4572 — EP time-imbalance v2 (BARRIER-ALIGNED cross-rank).
#
# WHY v2: analyze_time_mxfp4.py aligns ranks by np.stack(reg[r][:K]) — INDEX from 0.
# But EP dispatch/combine is a *collective barrier*: all 8 ranks run the SAME number of
# collectives per step (DP runs dummy forwards to stay lockstep). So comm_events must be
# EQUAL across ranks in principle; the ~0.5% spread seen is trace-capture start/stop skew
# at the edges, NOT workload. Index-from-0 then compares cluster i of rank A (step s) with
# cluster i of rank B (step s+2) -> unphysical maxmin_max ~92x.
#
# FIX: align by the collective itself. gather_end (dispatch-comm END) is barrier-synchronized
# (~equal ts across ranks). We (1) trim to the common time window [max first, min last],
# (2) match each rank's collectives to a reference timeline by nearest gather_end within a
# gap-adaptive tolerance (never matches adjacent collectives), (3) keep only collectives
# matched on ALL ranks, (4) recompute imbalance on that. Also prints the OLD index-aligned
# numbers next to the NEW ones, plus edge-skew diagnostics that prove the model.
#
# Parses via analyze_time_mxfp4.parse_rank (same cluster logic), caches per-rank arrays to
# npz so re-runs (and plotting) are cheap. Saves imbalance_v2.json/.txt + per-step arrays.
# =============================================================================
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_time_mxfp4 import parse_rank, rank_of, list_files, windowed_imbalance, detect_layers_per_step, Clusters


def parse_or_cache(files, cache, tag, comm_substr, moe_substrs):
    """Parse each rank once; cache (gather_end, rs_start, moe_busy, cg_phase) to npz."""
    reg = {}; commcount = {}
    for f in files:
        r = rank_of(f)
        cf = os.path.join(cache, f"{tag}_rank{r}.npz")
        if os.path.exists(cf):
            z = np.load(cf)
            reg[r] = dict(gather_end=z["gather_end"], rs_start=z["rs_start"],
                          moe_busy=z["moe_busy"], cg_phase=z["cg_phase"])
            commcount[r] = int(z["comm_total"])
            print(f"  rank{r}: cached clusters={len(reg[r]['moe_busy']):6d} comm={commcount[r]}")
            continue
        arr, ct, hist = parse_rank(f, comm_substr, moe_substrs)
        reg[r] = dict(gather_end=arr["gather_end"], rs_start=arr["rs_start"],
                      moe_busy=arr["moe_busy"], cg_phase=arr["cg_phase"])
        commcount[r] = ct
        np.savez(cf, gather_end=arr["gather_end"], rs_start=arr["rs_start"],
                 moe_busy=arr["moe_busy"], cg_phase=arr["cg_phase"], comm_total=ct)
        print(f"  rank{r}: parsed clusters={len(arr):6d} comm={ct} -> cached")
    return reg, commcount


def barrier_align(reg, ranks):
    """Align clusters across ranks by gather_end (barrier ts). Returns:
       M_busy [R, J], M_ge [R, J] for fully-matched collectives, and a diagnostics dict."""
    ge = {r: reg[r]["gather_end"] for r in ranks}
    bz = {r: reg[r]["moe_busy"] for r in ranks}
    # common time window
    t_lo = max(ge[r][0] for r in ranks)
    t_hi = min(ge[r][-1] for r in ranks)
    # per-rank in-window slices
    win = {}
    for r in ranks:
        m = (ge[r] >= t_lo) & (ge[r] <= t_hi)
        win[r] = (ge[r][m], bz[r][m])
    in_counts = {r: len(win[r][0]) for r in ranks}
    # reference = rank with the MOST in-window collectives
    ref = max(ranks, key=lambda r: in_counts[r])
    ref_ge = win[ref][0]
    J = len(ref_ge)
    # gap-adaptive tolerance: 0.45 * distance to nearest neighbour on the ref timeline
    d_prev = np.empty(J); d_next = np.empty(J)
    d_prev[1:] = np.diff(ref_ge); d_prev[0] = np.inf
    d_next[:-1] = np.diff(ref_ge); d_next[-1] = np.inf
    tol = 0.45 * np.minimum(d_prev, d_next)
    M_busy = np.full((len(ranks), J), np.nan)
    M_ge = np.full((len(ranks), J), np.nan)
    for ri, r in enumerate(ranks):
        g, b = win[r]
        idx = np.searchsorted(g, ref_ge)
        for j in range(J):
            best = -1; bestd = np.inf
            for k in (idx[j]-1, idx[j]):
                if 0 <= k < len(g):
                    dd = abs(g[k] - ref_ge[j])
                    if dd < bestd: bestd = dd; best = k
            if best >= 0 and bestd <= tol[j]:
                M_busy[ri, j] = b[best]; M_ge[ri, j] = g[best]
    full = ~np.isnan(M_busy).any(0)
    diag = dict(t_lo=float(t_lo), t_hi=float(t_hi), window_s=float((t_hi-t_lo)/1e6),
                in_counts=in_counts, ref=int(ref), J_ref=int(J),
                J_full=int(full.sum()), match_rate=round(float(full.mean()), 4))
    # barrier spread on fully-matched columns (proves synchronization)
    ge_full = M_ge[:, full]
    spread = ge_full.max(0) - ge_full.min(0)
    diag["barrier_spread_med_us"] = round(float(np.median(spread)), 2)
    diag["barrier_spread_p99_us"] = round(float(np.percentile(spread, 99)), 2)
    return M_busy[:, full], M_ge[:, full], diag


def imbalance_table(metric, L, windows):
    """metric [R, K] -> per-cluster + per-step(W=1) + windowed rows."""
    R, K = metric.shape
    n_steps = K // L
    step_busy = metric[:, :n_steps*L].reshape(R, n_steps, L).sum(2) if n_steps else metric.sum(1, keepdims=True)
    def stat(mm, ma, n):
        return dict(n=int(n), maxmin_mean=round(float(mm.mean()),3), maxmin_median=round(float(np.median(mm)),3),
                    maxmin_p99=round(float(np.percentile(mm,99)),3), maxmin_max=round(float(mm.max()),3),
                    maxavg_mean=round(float(ma.mean()),3))
    rows = []
    pc_mm = metric.max(0)/np.maximum(metric.min(0),1e-9)
    pc_ma = metric.max(0)/np.maximum(metric.mean(0),1e-9)
    rows.append(("per-cluster", stat(pc_mm, pc_ma, K)))
    per_step_mm = None
    for W in [1]+[int(x) for x in windows.split(",")]:
        mm, ma, nW = windowed_imbalance(step_busy, W)
        if W == 1: per_step_mm = mm
        rows.append((f"W={W}step" if W>1 else "per-step(W=1)", stat(mm, ma, nW)))
    return rows, n_steps, step_busy, per_step_mm


def old_index_align(reg, ranks):
    """Reproduce analyze_time_mxfp4 [:K] index alignment for side-by-side comparison."""
    K = min(len(reg[r]["moe_busy"]) for r in ranks)
    return np.stack([reg[r]["moe_busy"][:K] for r in ranks])


def fmt_table(rows):
    out = [f"{'granularity':<16}{'n':>8}{'maxmin_mean':>13}{'maxmin_med':>12}{'maxmin_p99':>12}{'maxmin_max':>12}{'maxavg_mean':>13}"]
    for name, s in rows:
        out.append(f"{name:<16}{s['n']:>8}{s['maxmin_mean']:>13}{s['maxmin_median']:>12}{s['maxmin_p99']:>12}{s['maxmin_max']:>12}{s['maxavg_mean']:>13}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="EP time-imbalance v2 (barrier-aligned).")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="e.g. base_c16_100k or r0_c16_100k")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--comm-substr", default="ncclDevKernel")
    ap.add_argument("--moe-substr", default="mfma_moe,_gemm_afp4wfp4,_batched_gemm_a16wfp4")
    ap.add_argument("--windows", default="100,250,500,1000,2000,3000")
    ap.add_argument("--layers-per-step", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    moe_substrs = [s for s in args.moe_substr.split(",") if s]
    files = list_files(args.trace_dir)
    assert files, f"no traces in {args.trace_dir}"
    print(f"TAG={args.tag}  TRACE_DIR={args.trace_dir}\n{len(files)} files")

    reg, commcount = parse_or_cache(files, args.cache, args.tag, args.comm_substr, moe_substrs)
    ranks = sorted(reg)

    # edge-skew diagnostics
    lines = [f"TAG={args.tag}", f"TRACE_DIR={args.trace_dir}", ""]
    lines.append("=== per-rank comm/cluster counts + capture edges (gather_end us) ===")
    for r in ranks:
        g = reg[r]["gather_end"]
        lines.append(f"  rank{r}: clusters={len(g):7d} comm={commcount[r]:7d}  first={g[0]:.0f} last={g[-1]:.0f}")
    cc = [commcount[r] for r in ranks]
    lines.append(f"  comm spread: min={min(cc)} max={max(cc)} diff={max(cc)-min(cc)} ({100*(max(cc)-min(cc))/max(cc):.2f}%)")

    Malign, Mge, diag = barrier_align(reg, ranks)
    L = args.layers_per_step or detect_layers_per_step(
        Clusters(ranks=np.array(ranks), gather_end=np.stack([reg[r]["gather_end"][:min(len(reg[rr]["gather_end"]) for rr in ranks)] for r in ranks]),
                 rs_start=None, moe_busy=None, cg_phase=None)) or 120
    lines.append("")
    lines.append("=== BARRIER ALIGNMENT diagnostics ===")
    lines.append(f"  common window = {diag['window_s']:.2f}s  in-window counts={diag['in_counts']}")
    lines.append(f"  reference rank={diag['ref']}  ref collectives J={diag['J_ref']}  fully-matched J={diag['J_full']}  match_rate={diag['match_rate']}")
    lines.append(f"  barrier spread across ranks (should be << inter-collective gap): median={diag['barrier_spread_med_us']}us p99={diag['barrier_spread_p99_us']}us")
    lines.append(f"  L (layers/step) = {L}")

    # NEW (barrier-aligned) vs OLD (index [:K])
    new_rows, n_steps, step_busy, per_step_mm = imbalance_table(Malign, L, args.windows)
    old_M = old_index_align(reg, ranks)
    old_rows, _, _, _ = imbalance_table(old_M, L, args.windows)
    lines.append("")
    lines.append(f"=== EP MoE-busy imbalance — NEW barrier-aligned (K={Malign.shape[1]}, steps={n_steps}) ===")
    lines.append(fmt_table(new_rows))
    lines.append("")
    lines.append(f"=== OLD index-aligned [:K] (K={old_M.shape[1]}) — for comparison (artifact) ===")
    lines.append(fmt_table(old_rows))

    txt = "\n".join(lines)
    print("\n"+txt)
    with open(os.path.join(args.out, f"{args.tag}_v2.txt"), "w") as f: f.write(txt+"\n")
    summary = dict(tag=args.tag, trace_dir=args.trace_dir, ranks=ranks, L=int(L),
                   comm_counts=commcount, align_diag=diag,
                   new_barrier_aligned={n:s for n,s in new_rows},
                   old_index_aligned={n:s for n,s in old_rows})
    with open(os.path.join(args.out, f"{args.tag}_v2.json"), "w") as f: json.dump(summary, f, indent=2)
    # arrays for plotting
    np.savez(os.path.join(args.out, f"{args.tag}_v2_arrays.npz"),
             step_busy=step_busy, per_step_mm=per_step_mm, ranks=np.array(ranks))
    print(f"\nsaved -> {args.out}/{args.tag}_v2.{{txt,json}} + _arrays.npz")


if __name__ == "__main__":
    main()
