#!/usr/bin/env python3
# =============================================================================
# MV-4572 — Phân rã THỜI GIAN comm (gather/reduce-scatter) vs MoE-compute, + chart SYNC BARRIER.
#
# Trả lời 2 câu hỏi:
#   (1) time của communication (gather / reduce-scatter) vs MoE
#   (2) chart hình ảnh sync barrier (độ lệch điểm barrier giữa 8 rank)
#
# NỀN TẢNG (xem analyze_time_mxfp4.parse_rank): mỗi cluster = 1 MoE-layer trên 1 rank =
#   [gather=dispatch all-to-all] -> (MoE-compute kernels) -> [reduce-scatter=combine].
#   gather_start/end, rs_start/end là 2 comm-kernel ncclDevKernel bọc 2 đầu; moe_busy =
#   tổng dur các kernel non-comm ở giữa (gather_end, rs_start).
# Định nghĩa THỜI GIAN (µs, per-cluster):
#   gather_dur = gather_end - gather_start     (comm dispatch)
#   rs_dur     = rs_end     - rs_start         (comm combine)
#   moe_busy   = tổng dur kernel compute        (compute HỮU ÍCH)
#   moe_span   = rs_start   - gather_end        (cửa sổ MoE: compute + bubble chờ)
#   moe_bubble = moe_span   - moe_busy          (idle/chờ trong cửa sổ — dấu hiệu comm-bound)
#   cluster_wall = rs_end - gather_start = gather_dur + moe_span + rs_dur
#
# SYNC BARRIER: dispatch/combine là collective => gather_end (điểm KẾT dispatch) đồng bộ
#   ~bằng nhau qua 8 rank. Ta barrier-align theo gather_end (như analyze_time_v2), rồi đo:
#     barrier_spread[j] = max_r(gather_end) - min_r(gather_end)  của collective j  (µs)
#     arrival_offset[r] = mean_j( gather_end[r,j] - mean_r(gather_end[:,j]) )       (rank nào tới trễ/sớm)
#   Spread nhỏ (<< khoảng cách giữa 2 collective) => 8 rank chạy lockstep, imbalance KHÔNG
#   phải do trôi pha mà do tải compute per-collective.
#
# CPU-only, đọc trace đã có sẵn (KHÔNG cần GPU). Parse SONG SONG 8 rank + cache full arrays.
# Usage:
#   python3 analyze_comm_moe.py --trace-dir <profiling_result> --out <imbalance_results> \
#       --tag r0_023_c16_100k --cache <cache_dir> [--layers-per-step 120] [--procs 8]
# =============================================================================
import os, sys, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_time_mxfp4 import parse_rank, rank_of, list_files

COMM_SUBSTR = "ncclDevKernel"
MOE_SUBSTRS = ["mfma_moe", "_gemm_afp4wfp4", "_batched_gemm_a16wfp4"]
FIELDS = ["gather_start", "gather_end", "rs_start", "rs_end", "moe_busy", "cg_phase"]

C_GATHER = "#4a90d9"   # dispatch comm (blue)
C_MOE    = "#e07b39"   # moe compute   (orange)
C_BUBBLE = "#c9ccd1"   # bubble/idle   (grey)
C_RS     = "#7b52ab"   # combine comm  (purple)


def _worker(args):
    """Parse 1 rank -> (rank, dict-of-arrays, comm_total). Top-level cho Pool."""
    path, cache, tag = args
    r = rank_of(path)
    cf = os.path.join(cache, f"{tag}_full_rank{r}.npz")
    if os.path.exists(cf):
        z = np.load(cf)
        return r, {k: z[k] for k in FIELDS}, int(z["comm_total"])
    arr, ct, _ = parse_rank(path, COMM_SUBSTR, MOE_SUBSTRS)
    d = {k: arr[k] for k in FIELDS}
    np.savez(cf, comm_total=ct, **d)
    return r, d, ct


def parse_all(files, cache, tag, procs):
    print(f"parsing {len(files)} ranks với {procs} process ...", flush=True)
    with Pool(procs) as pool:
        res = pool.map(_worker, [(f, cache, tag) for f in files])
    reg = {}; comm_total = {}
    for r, d, ct in res:
        reg[r] = d; comm_total[r] = ct
        print(f"  rank{r}: clusters={len(d['gather_end']):7d} comm_total={ct}", flush=True)
    return reg, comm_total


def barrier_align(reg, ranks):
    """Align các cluster qua rank theo gather_end (barrier ts). Trả idx[R,J_full] = chỉ số
    (trong mảng gốc mỗi rank) của các collective khớp trên MỌI rank, + diag."""
    ge = {r: reg[r]["gather_end"] for r in ranks}
    t_lo = max(ge[r][0] for r in ranks)
    t_hi = min(ge[r][-1] for r in ranks)
    win_idx = {}
    for r in ranks:
        m = np.where((ge[r] >= t_lo) & (ge[r] <= t_hi))[0]
        win_idx[r] = m
    in_counts = {r: int(len(win_idx[r])) for r in ranks}
    ref = max(ranks, key=lambda r: in_counts[r])
    ref_ge = ge[ref][win_idx[ref]]
    J = len(ref_ge)
    d_prev = np.empty(J); d_next = np.empty(J)
    d_prev[1:] = np.diff(ref_ge); d_prev[0] = np.inf
    d_next[:-1] = np.diff(ref_ge); d_next[-1] = np.inf
    tol = 0.45 * np.minimum(d_prev, d_next)
    R = len(ranks)
    idx = np.full((R, J), -1, dtype=np.int64)
    for ri, r in enumerate(ranks):
        g = ge[r]
        ins = np.searchsorted(g, ref_ge)
        for j in range(J):
            best = -1; bestd = np.inf
            for k in (ins[j]-1, ins[j]):
                if 0 <= k < len(g):
                    dd = abs(g[k] - ref_ge[j])
                    if dd < bestd:
                        bestd = dd; best = k
            if best >= 0 and bestd <= tol[j]:
                idx[ri, j] = best
    full = (idx >= 0).all(0)
    idx = idx[:, full]
    ge_al = np.stack([ge[r][idx[ri]] for ri, r in enumerate(ranks)])
    spread = ge_al.max(0) - ge_al.min(0)
    diag = dict(t_lo=float(t_lo), t_hi=float(t_hi), window_s=float((t_hi-t_lo)/1e6),
                in_counts=in_counts, ref=int(ref), J_ref=int(J), J_full=int(idx.shape[1]),
                match_rate=round(float(full.mean()), 4),
                barrier_spread_med_us=round(float(np.median(spread)), 3),
                barrier_spread_p99_us=round(float(np.percentile(spread, 99)), 3),
                barrier_spread_max_us=round(float(spread.max()), 3))
    return idx, diag


def gather_aligned(reg, ranks, idx, field):
    return np.stack([reg[r][field][idx[ri]] for ri, r in enumerate(ranks)])


def stats(a):
    return dict(mean=round(float(np.mean(a)), 3), median=round(float(np.median(a)), 3),
                p99=round(float(np.percentile(a, 99)), 3), max=round(float(np.max(a)), 3))


# ------------------------------- charts -------------------------------
def plot_commmoe(tag, ranks, per_rank, mean_break, dist, comm_frac_series, L, outp):
    """4 panel: (A) stacked bar per-rank, (B) donut mean, (C) violin phân phối, (D) comm-frac theo thời gian."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Comm (gather/reduce-scatter) vs MoE-compute time — {tag}", fontsize=13, fontweight="bold")

    # A: stacked horizontal bar mỗi rank (µs trung bình / cluster)
    a = ax[0, 0]
    g = np.array([per_rank[r]["gather"] for r in ranks])
    mo = np.array([per_rank[r]["moe_busy"] for r in ranks])
    bb = np.array([per_rank[r]["bubble"] for r in ranks])
    rs = np.array([per_rank[r]["rs"] for r in ranks])
    y = np.arange(len(ranks))
    a.barh(y, g, color=C_GATHER, label="gather (dispatch)")
    a.barh(y, mo, left=g, color=C_MOE, label="MoE compute")
    a.barh(y, bb, left=g+mo, color=C_BUBBLE, label="bubble (span−busy)")
    a.barh(y, rs, left=g+mo+bb, color=C_RS, label="reduce-scatter (combine)")
    a.set_yticks(y); a.set_yticklabels([f"dp{r}" for r in ranks]); a.invert_yaxis()
    a.set_xlabel("µs trung bình / cluster (MoE-layer)")
    a.set_title("A. Thời gian/ cluster mỗi rank (comm vs compute)")
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3, axis="x")

    # B: donut trung bình toàn cục
    b = ax[0, 1]
    vals = [mean_break["gather"], mean_break["moe_busy"], mean_break["bubble"], mean_break["rs"]]
    labs = ["gather", "MoE compute", "bubble", "reduce-scatter"]
    cols = [C_GATHER, C_MOE, C_BUBBLE, C_RS]
    tot = sum(vals)
    w, _, _ = b.pie(vals, colors=cols, startangle=90, counterclock=False,
                    wedgeprops=dict(width=0.42, edgecolor="w"),
                    autopct=lambda p: f"{p:.1f}%", pctdistance=0.79, textprops=dict(fontsize=9))
    comm_pct = 100*(mean_break["gather"]+mean_break["rs"])/tot
    b.text(0, 0, f"comm\n{comm_pct:.0f}%", ha="center", va="center", fontsize=13, fontweight="bold")
    b.legend(labs, fontsize=8, loc="lower left", bbox_to_anchor=(-0.15, -0.05))
    b.set_title(f"B. Cơ cấu thời gian trung bình/cluster (wall={tot:.1f}µs)")

    # C: violin phân phối 4 đại lượng (aligned, gộp mọi rank)
    c = ax[1, 0]
    data = [dist["gather"], dist["moe_busy"], dist["bubble"], dist["rs"]]
    parts = c.violinplot(data, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [C_GATHER, C_MOE, C_BUBBLE, C_RS]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    c.set_xticks([1, 2, 3, 4]); c.set_xticklabels(["gather", "MoE\nbusy", "bubble", "reduce\nscatter"])
    c.set_ylabel("µs / cluster"); c.set_yscale("symlog")
    c.set_title("C. Phân phối thời gian/cluster (symlog)"); c.grid(alpha=0.3, axis="y")

    # D: comm fraction theo thời gian (per-step, rolling)
    d = ax[1, 1]
    cf = comm_frac_series
    w = max(1, len(cf)//200)
    cf_s = np.convolve(cf, np.ones(w)/w, mode="valid") if len(cf) >= w else cf
    d.plot(cf_s*100, color="#b2182b", lw=1.1)
    d.axhline(cf.mean()*100, color="#2166ac", ls="--", lw=0.9, label=f"mean={cf.mean()*100:.1f}%")
    d.set_xlabel(f"step (L={L} cluster/step, rolling {w})")
    d.set_ylabel("comm fraction = (gather+RS)/wall  [%]")
    d.set_title("D. Tỷ trọng comm theo thời gian (comm-bound?)")
    d.legend(); d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outp, dpi=130); plt.close(fig)
    print(f"saved -> {outp}", flush=True)


def plot_barrier(tag, ranks, spread, arrival_off, spread_series, rs_spread, diag, L, outp):
    """4 panel: (A) hist spread, (B) CDF spread, (C) spread theo thời gian, (D) arrival offset per-rank."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Sync barrier — độ lệch điểm barrier giữa {len(ranks)} rank — {tag}",
                 fontsize=13, fontweight="bold")

    med = float(np.median(spread)); p99 = float(np.percentile(spread, 99))

    # A: hist barrier spread (gather_end)
    a = ax[0, 0]
    hi = np.percentile(spread, 99.5)
    a.hist(spread, bins=np.linspace(0, hi, 60), color=C_GATHER, alpha=0.8)
    a.axvline(med, color="#b2182b", ls="--", lw=1.2, label=f"median={med:.2f}µs")
    a.axvline(p99, color="#333", ls=":", lw=1.2, label=f"p99={p99:.2f}µs")
    a.set_xlabel("barrier spread = max−min gather_end (µs)"); a.set_ylabel("#collective")
    a.set_title("A. Phân phối độ lệch barrier (dispatch)")
    a.legend(); a.grid(alpha=0.3)

    # B: CDF
    b = ax[0, 1]
    s = np.sort(spread); yv = np.arange(1, len(s)+1)/len(s)
    b.plot(s, yv*100, color=C_GATHER, lw=1.4)
    b.axvline(med, color="#b2182b", ls="--", lw=1, label=f"median={med:.2f}µs")
    b.axhline(99, color="#333", ls=":", lw=0.8)
    b.set_xlim(0, np.percentile(spread, 99.8))
    b.set_xlabel("barrier spread (µs)"); b.set_ylabel("CDF (%)")
    b.set_title("B. CDF độ lệch barrier"); b.legend(); b.grid(alpha=0.3)

    # C: spread theo thời gian (rolling)
    c = ax[1, 0]
    w = max(1, len(spread_series)//300)
    ss = np.convolve(spread_series, np.ones(w)/w, mode="valid") if len(spread_series) >= w else spread_series
    c.plot(ss, color=C_GATHER, lw=0.9, label=f"gather_end (rolling {w})")
    if rs_spread is not None:
        rr = np.convolve(rs_spread, np.ones(w)/w, mode="valid") if len(rs_spread) >= w else rs_spread
        c.plot(rr, color=C_RS, lw=0.9, alpha=0.8, label=f"rs_start (rolling {w})")
    c.set_xlabel("collective #"); c.set_ylabel("barrier spread (µs)")
    c.set_title("C. Độ lệch barrier theo thời gian"); c.legend(); c.grid(alpha=0.3)

    # D: arrival offset per-rank (rank nào tới barrier trễ/sớm)
    d = ax[1, 1]
    off = np.array([arrival_off[r] for r in ranks])
    cols = ["#b2182b" if v > 0 else "#2166ac" for v in off]
    x = np.arange(len(ranks))
    d.bar(x, off, color=cols)
    d.axhline(0, color="k", lw=0.8)
    d.set_xticks(x); d.set_xticklabels([f"dp{r}" for r in ranks])
    d.set_ylabel("offset trung bình vs mean-rank (µs)")
    d.set_title("D. Rank tới barrier trễ(+)/sớm(−)")
    d.grid(alpha=0.3, axis="y")
    for xi, v in zip(x, off):
        d.text(xi, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outp, dpi=130); plt.close(fig)
    print(f"saved -> {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--layers-per-step", type=int, default=120)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    files = list_files(args.trace_dir)
    assert files, f"no traces in {args.trace_dir}"
    print(f"TAG={args.tag}  TRACE_DIR={args.trace_dir}\n{len(files)} files", flush=True)

    reg, comm_total = parse_all(files, args.cache, args.tag, args.procs)
    ranks = sorted(reg)
    L = args.layers_per_step

    idx, diag = barrier_align(reg, ranks)
    print(f"barrier-align: window={diag['window_s']:.1f}s J_full={diag['J_full']} "
          f"spread med={diag['barrier_spread_med_us']}µs p99={diag['barrier_spread_p99_us']}µs", flush=True)

    # aligned fields [R, J]
    gs = gather_aligned(reg, ranks, idx, "gather_start")
    ge = gather_aligned(reg, ranks, idx, "gather_end")
    rss = gather_aligned(reg, ranks, idx, "rs_start")
    rse = gather_aligned(reg, ranks, idx, "rs_end")
    mb = gather_aligned(reg, ranks, idx, "moe_busy")
    gather_dur = ge - gs
    rs_dur = rse - rss
    span = rss - ge
    bubble = np.maximum(span - mb, 0.0)
    wall = rse - gs
    R, J = gather_dur.shape

    # per-rank trung bình/cluster (µs)
    per_rank = {}
    for ri, r in enumerate(ranks):
        per_rank[r] = dict(gather=float(gather_dur[ri].mean()), moe_busy=float(mb[ri].mean()),
                           bubble=float(bubble[ri].mean()), rs=float(rs_dur[ri].mean()),
                           span=float(span[ri].mean()), wall=float(wall[ri].mean()))
    mean_break = dict(gather=float(gather_dur.mean()), moe_busy=float(mb.mean()),
                      bubble=float(bubble.mean()), rs=float(rs_dur.mean()),
                      span=float(span.mean()), wall=float(wall.mean()))
    dist = dict(gather=gather_dur.ravel(), moe_busy=mb.ravel(), bubble=bubble.ravel(), rs=rs_dur.ravel())

    # comm fraction theo step (dùng rank-mean per collective, gộp L cluster/step)
    nsteps = J // L
    comm = (gather_dur + rs_dur).mean(0)   # [J]
    wall_c = wall.mean(0)
    if nsteps:
        comm_step = comm[:nsteps*L].reshape(nsteps, L).sum(1)
        wall_step = wall_c[:nsteps*L].reshape(nsteps, L).sum(1)
        comm_frac_series = comm_step / np.maximum(wall_step, 1e-9)
    else:
        comm_frac_series = (comm / np.maximum(wall_c, 1e-9))

    # sync barrier
    spread = ge.max(0) - ge.min(0)
    rs_spread = rss.max(0) - rss.min(0)
    arrival = ge - ge.mean(0, keepdims=True)
    arrival_off = {r: float(arrival[ri].mean()) for ri, r in enumerate(ranks)}

    # ---- text + json ----
    tot = mean_break["wall"]
    def pct(x): return 100*x/tot
    lines = [f"TAG={args.tag}", f"TRACE_DIR={args.trace_dir}",
             f"ranks={ranks}  L={L}  aligned collectives J={J} ({nsteps} steps)", ""]
    lines.append("=== (1) THỜI GIAN comm vs MoE — trung bình / cluster (µs) [barrier-aligned] ===")
    lines.append(f"{'':<14}{'gather':>10}{'moe_busy':>10}{'bubble':>10}{'rs':>10}{'span':>10}{'wall':>10}")
    for r in ranks:
        p = per_rank[r]
        lines.append(f"  dp{r:<11}{p['gather']:>10.2f}{p['moe_busy']:>10.2f}{p['bubble']:>10.2f}"
                     f"{p['rs']:>10.2f}{p['span']:>10.2f}{p['wall']:>10.2f}")
    m = mean_break
    lines.append(f"  {'MEAN':<11}{m['gather']:>10.2f}{m['moe_busy']:>10.2f}{m['bubble']:>10.2f}"
                 f"{m['rs']:>10.2f}{m['span']:>10.2f}{m['wall']:>10.2f}")
    lines.append(f"  {'%wall':<11}{pct(m['gather']):>9.1f}%{pct(m['moe_busy']):>9.1f}%{pct(m['bubble']):>9.1f}%"
                 f"{pct(m['rs']):>9.1f}%{pct(m['span']):>9.1f}%{100.0:>9.1f}%")
    comm_pct = pct(m['gather'] + m['rs'])
    lines.append(f"  => comm (gather+RS) = {m['gather']+m['rs']:.2f}µs ({comm_pct:.1f}% wall); "
                 f"MoE compute = {m['moe_busy']:.2f}µs ({pct(m['moe_busy']):.1f}%); "
                 f"bubble = {m['bubble']:.2f}µs ({pct(m['bubble']):.1f}%)")
    lines.append("")
    lines.append("  Phân phối (mọi rank×collective):")
    for k in ["gather", "moe_busy", "bubble", "rs"]:
        s = stats(dist[k])
        lines.append(f"    {k:<10} med={s['median']:>7.2f}  mean={s['mean']:>7.2f}  p99={s['p99']:>7.2f}  max={s['max']:>8.2f}")
    lines.append(f"  comm fraction /step: mean={comm_frac_series.mean()*100:.1f}%  "
                 f"min={comm_frac_series.min()*100:.1f}%  max={comm_frac_series.max()*100:.1f}%")
    lines.append("")
    lines.append("=== (2) SYNC BARRIER (align theo gather_end) ===")
    lines.append(f"  common window = {diag['window_s']:.2f}s  J_full={diag['J_full']}  match_rate={diag['match_rate']}")
    lines.append(f"  barrier spread (gather_end)  : median={diag['barrier_spread_med_us']}µs  "
                 f"p99={diag['barrier_spread_p99_us']}µs  max={diag['barrier_spread_max_us']}µs")
    lines.append(f"  barrier spread (rs_start)    : median={np.median(rs_spread):.3f}µs  "
                 f"p99={np.percentile(rs_spread,99):.3f}µs")
    med_gap = float(np.median(np.diff(ge.mean(0))))
    lines.append(f"  median inter-collective gap  = {med_gap:.2f}µs  => spread/gap = "
                 f"{diag['barrier_spread_med_us']/med_gap*100:.2f}%  (nhỏ => lockstep)")
    lines.append("  arrival offset per-rank (µs, +trễ / −sớm):")
    for r in ranks:
        lines.append(f"    dp{r}: {arrival_off[r]:+.3f}")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    with open(os.path.join(args.out, f"{args.tag}_commmoe.txt"), "w") as f:
        f.write(txt + "\n")
    summary = dict(tag=args.tag, trace_dir=args.trace_dir, ranks=ranks, L=L, J=int(J), steps=int(nsteps),
                   per_rank_us=per_rank, mean_break_us=mean_break,
                   pct_wall=dict(gather=pct(m['gather']), moe_busy=pct(m['moe_busy']),
                                 bubble=pct(m['bubble']), rs=pct(m['rs']), comm=comm_pct),
                   dist=dict((k, stats(dist[k])) for k in dist),
                   comm_frac_step=dict(mean=float(comm_frac_series.mean()),
                                       min=float(comm_frac_series.min()), max=float(comm_frac_series.max())),
                   barrier=dict(diag, rs_spread_med_us=round(float(np.median(rs_spread)), 3),
                                rs_spread_p99_us=round(float(np.percentile(rs_spread, 99)), 3),
                                median_inter_collective_gap_us=round(med_gap, 3),
                                arrival_offset_us=arrival_off))
    with open(os.path.join(args.out, f"{args.tag}_commmoe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out, f"{args.tag}_commmoe_arrays.npz"),
             gather_dur=gather_dur, rs_dur=rs_dur, moe_busy=mb, span=span, bubble=bubble,
             barrier_spread=spread, rs_spread=rs_spread, arrival=arrival, ranks=np.array(ranks))

    # ---- charts ----
    plot_commmoe(args.tag, ranks, per_rank, mean_break, dist, comm_frac_series, L,
                 os.path.join(args.out, f"commmoe_{args.tag}.png"))
    plot_barrier(args.tag, ranks, spread, arrival_off, spread, rs_spread, diag, L,
                 os.path.join(args.out, f"barrier_{args.tag}.png"))
    print(f"\nsaved -> {args.out}/{args.tag}_commmoe.{{txt,json}} + commmoe_/barrier_ PNG + _arrays.npz", flush=True)


if __name__ == "__main__":
    main()
