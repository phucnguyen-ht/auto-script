#!/usr/bin/env python3
# =============================================================================
# MV-4572 — EP time-imbalance for Kimi-K2.6-MXFP4 (dp8/ep8), from 8 torch-profiler
# traces. MXFP4-ONLY: this deliberately drops the GLM-5.2-FP8 / Kimi-2.6-FP8 kernel
# handling of the 4571 analyze_time.py. It does NOT hard-code MoE kernel names —
# per the data-flow, the MoE compute of one layer is simply *everything the GPU runs
# between that layer's EP all-gather (dispatch) and reduce-scatter (combine)*. So a
# cluster = [all-gather] -> (non-comm kernels = MoE) -> [reduce-scatter]; the region
# between a reduce-scatter and the next all-gather (attention/norm) is NOT a cluster.
#
# Adds WINDOWED imbalance: the per-step (per-cluster) max/min is very skewed; we also
# aggregate per-rank MoE time over windows of W steps (W in 100/250/500/1000/2000/3000,
# = the EPLB step_interval knobs) and report max/min & max/avg per window. This shows
# how the imbalance EPLB actually sees settles as the rebalance interval grows.
#
# Usage:
#   python3 analyze_time_mxfp4.py --trace-dir <dir with dp*_rank0.*.pt.trace.json.gz> \
#       --out <dir> [--layers-per-step N] [--comm-substr ncclDevKernel] \
#       [--gather-substr AllGather] [--scatter-substr ReduceScatter] [--metric busy|span]
#
# NOTE: the comm-kernel names for MXFP4 are confirmed against a real trace via
# --dump-kernels (prints the comm kernel names + a sample cluster). Defaults match
# rccl/nccl ('ncclDevKernel' + 'AllGather'/'ReduceScatter'); override if a trace shows
# otherwise (e.g. all-to-all dispatch).
# =============================================================================
import os, glob, gzip, re, json, argparse, bisect, collections
from dataclasses import dataclass
import numpy as np

try:
    import orjson as _json
    def _load_gz(p):
        with gzip.open(p, "rb") as fh: return _json.loads(fh.read())
except ImportError:
    def _load_gz(p):
        with gzip.open(p, "rb") as fh: return json.load(fh)


def iter_events(path):
    d = _load_gz(path)
    yield from (d["traceEvents"] if isinstance(d, dict) else d)


def rank_of(path):
    m = re.search(r"dp(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_files(trace_dir):
    fs = sorted(glob.glob(os.path.join(trace_dir, "dp*_rank0.*.pt.trace.json.gz")))
    if not fs:  # fall back to any per-rank torch traces
        fs = sorted(glob.glob(os.path.join(trace_dir, "*.pt.trace.json.gz")))
    return fs


# ----------------------------- cluster struct -----------------------------
# Each cluster = one MoE-layer invocation on one rank: gather (dispatch) -> MoE
# compute -> reduce-scatter (combine).
REGION_DT = np.dtype([
    ("gather_start", "f8"), ("gather_end", "f8"),
    ("rs_start", "f8"), ("rs_end", "f8"),
    ("moe_busy", "f8"),     # sum of non-comm kernel durations in (gather_end, rs_start)
    ("cg_phase", "i4"),     # 1 prefill / 2 decode / 0 unknown (cudagraph arrow heuristic)
])
CG_NOARROW_THR = 0.5


def parse_rank(path, comm_substr, moe_substrs):
    """Parse one rank's trace into MoE clusters. At v0.24 the EP dispatch (all-gather) and
    combine (reduce-scatter) are the SAME kernel — `ncclDevKernel_Generic_1` — so they are
    NAME-indistinguishable (confirmed via --dump-kernels). We therefore anchor on the MoE
    *compute* instead of on comm names: a pair of consecutive comm kernels that has >=1
    MoE-matmul kernel (name in *moe_substrs*) starting between them brackets exactly one MoE
    layer — the earlier comm = dispatch (gather), the later = combine (reduce-scatter).
    Consecutive comm pairs with NO MoE between them (that gap is the attention/norm between
    a combine and the next dispatch) are skipped, and stray ncclDevKernel during attention
    (DP/TP all-reduce) are naturally ignored. This also handles multi-call dispatch/combine:
    only the LAST comm before the MoE run and the FIRST comm after it form the pair.
    Everything non-comm inside is MoE compute (busy). cudagraph phase via 'ac2g' flow arrows
    (prefill=eager has arrows, decode=graph replay)."""
    comm_evs = []       # (ts, end)  EP comm kernels (ncclDevKernel — dispatch AND combine)
    kern_evs = []       # (ts, end)  non-comm GPU kernels (=> MoE compute when inside a cluster)
    moe_starts = []     # ts of MoE-matmul kernels — anchor to tell a MoE region from attention
    s_ids = set(); f_ts = []; f_id = []
    for e in iter_events(path):
        cat = e.get("cat"); ph = e.get("ph")
        if cat == "ac2g":
            if ph == "s": s_ids.add(e.get("id"))
            elif ph == "f":
                t = e.get("ts")
                if t is not None: f_ts.append(float(t)); f_id.append(e.get("id"))
            continue
        if ph != "X" or cat != "kernel": continue
        ts = e.get("ts"); dur = e.get("dur"); n = e.get("name", "")
        if ts is None or dur is None: continue
        ts = float(ts); end = ts + float(dur)
        if comm_substr in n:
            comm_evs.append((ts, end))
        else:
            kern_evs.append((ts, end))
            if any(sub in n for sub in moe_substrs):
                moe_starts.append(ts)
    comm_evs.sort(); kern_evs.sort(); moe_starts.sort()
    kern_starts = [k[0] for k in kern_evs]

    # cudagraph arrow lookup (vectorized): flow 'f' with matching 's' id == eager (prefill)
    if f_ts:
        fts = np.asarray(f_ts); fid = np.asarray(f_id)
        order = np.argsort(fts); fts = fts[order]
        s_arr = np.fromiter(s_ids, dtype=fid.dtype, count=len(s_ids)) if s_ids else np.empty(0, dtype=fid.dtype)
        has_s = np.isin(fid[order], s_arr)
    else:
        fts = np.empty(0); has_s = np.empty(0, bool)

    def cg_label(gs, rse):
        a = bisect.bisect_left(fts, gs); b = bisect.bisect_right(fts, rse)
        if b <= a: return 0
        return 2 if (1.0 - has_s[a:b].mean()) >= CG_NOARROW_THR else 1

    def busy_between(t0, t1):
        # sum of non-comm kernel durations whose start is in [t0, t1)
        lo = bisect.bisect_left(kern_starts, t0); hi = bisect.bisect_left(kern_starts, t1)
        return sum(min(kern_evs[i][1], t1) - kern_evs[i][0] for i in range(lo, hi))

    # MoE-anchored pairing: a consecutive comm pair with >=1 MoE-matmul start between them
    # brackets one MoE layer (dispatch -> compute -> combine); pairs with none between are
    # the attention/norm gap and are skipped. See parse_rank docstring for why this is robust
    # to same-name dispatch/combine and multi-call collectives at v0.24.
    regs = []
    for i in range(len(comm_evs) - 1):
        g_ts, g_end = comm_evs[i]
        s_ts, s_end = comm_evs[i + 1]
        lo = bisect.bisect_left(moe_starts, g_end)
        hi = bisect.bisect_left(moe_starts, s_ts)
        if hi > lo:  # >=1 MoE-matmul kernel starts in (dispatch_end, combine_start)
            regs.append((g_ts, g_end, s_ts, s_end, busy_between(g_end, s_ts), cg_label(g_ts, s_end)))
    hist = collections.Counter({"moe_pairs": len(regs), "comm_total": len(comm_evs), "moe_kernels": len(moe_starts)})
    return np.array(regs, dtype=REGION_DT), len(comm_evs), hist


@dataclass
class Clusters:
    ranks: np.ndarray
    gather_end: np.ndarray
    rs_start: np.ndarray
    moe_busy: np.ndarray
    cg_phase: np.ndarray
    @property
    def moe_span(self): return self.rs_start - self.gather_end   # per-rank MoE window
    @property
    def shape(self): return self.moe_busy.shape


def build(files, comm_substr, moe_substrs):
    reg = {}; commcount = {}; kinds_hist = collections.Counter()
    for f in files:
        r = rank_of(f)
        reg[r], commcount[r], kh = parse_rank(f, comm_substr, moe_substrs)
        kinds_hist += kh
        print(f"  rank{r}: MoE clusters={len(reg[r]):6d}  comm_events={commcount[r]}")
    print(f"histogram (moe_pairs=bracketed MoE layers, comm_total=all ncclDevKernel, "
          f"moe_kernels=MoE-matmul anchors): {dict(kinds_hist)}")
    if kinds_hist.get("moe_pairs", 0) == 0:
        print("!! WARNING: no MoE layer bracketed — check --comm-substr / --moe-substr against the "
              "real trace (run --dump-kernels). Expected comm=ncclDevKernel, moe=mfma_moe/_gemm_afp4wfp4.")
    ranks = sorted(reg)
    K = min(len(reg[r]) for r in ranks)
    def field(name): return np.stack([reg[r][name][:K] for r in ranks])
    C = Clusters(ranks=np.array(ranks), gather_end=field("gather_end"), rs_start=field("rs_start"),
                 moe_busy=field("moe_busy"), cg_phase=field("cg_phase"))
    print(f"ranks={list(C.ranks)}  R={C.shape[0]}  aligned clusters K={C.shape[1]}  "
          f"MoE-busy median={np.median(C.moe_busy):.1f}us")
    return C


def detect_layers_per_step(C):
    """Heuristic: clusters within one forward step run back-to-back; the gap before the
    FIRST cluster of the next step (extra sampling/scheduling) is larger. L = median run
    length between such large gaps. Falls back to 1 if undetectable."""
    ge = C.gather_end[0]; gs_next = C.gather_end[0]  # use rank0 timeline
    starts = C.gather_end[0]
    gaps = np.diff(starts)
    if len(gaps) < 3: return None
    thr = np.median(gaps) * 3.0
    bounds = np.where(gaps > thr)[0]
    if len(bounds) < 2: return None
    runs = np.diff(bounds)
    if len(runs) == 0: return None
    L = int(np.median(runs))
    return L if L >= 1 else None


def windowed_imbalance(step_busy, W):
    """step_busy: [R, n_steps] per-rank MoE-busy summed per step. Window into W-step blocks;
    per window imbalance = max_r(sum) / min_r(sum) and max_r/avg_r. Returns arrays over windows."""
    R, S = step_busy.shape
    nW = S // W
    if nW == 0:
        block = step_busy.sum(1, keepdims=True); nW = 1
    else:
        block = step_busy[:, :nW * W].reshape(R, nW, W).sum(2)   # [R, nW]
    mx = block.max(0); mn = np.maximum(block.min(0), 1e-9); av = np.maximum(block.mean(0), 1e-9)
    return mx / mn, mx / av, nW


def main():
    ap = argparse.ArgumentParser(description="MV-4572 Kimi-2.6-MXFP4 EP time-imbalance (per-step + windowed).")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers-per-step", type=int, default=0, help="clusters/step (0 = auto-detect).")
    ap.add_argument("--windows", default="100,250,500,1000,2000,3000")
    ap.add_argument("--metric", choices=["busy", "span"], default="busy",
                    help="busy = sum of MoE kernel durations (compute load); span = rs_start-gather_end.")
    ap.add_argument("--comm-substr", default="ncclDevKernel",
                    help="EP dispatch+combine kernel (v0.24: ncclDevKernel_Generic_1, same for both).")
    ap.add_argument("--moe-substr", default="mfma_moe,_gemm_afp4wfp4,_batched_gemm_a16wfp4",
                    help="comma-list of MXFP4 MoE-matmul kernel substrings — anchors a MoE region "
                         "between two comm kernels (confirmed from trace: mfma_moe1/mfma_moe2 experts, "
                         "_gemm_afp4wfp4, _batched_gemm_a16wfp4).")
    ap.add_argument("--phase", choices=["all", "decode", "prefill"], default="all")
    ap.add_argument("--dump-kernels", action="store_true", help="print comm kernel names then exit (trace calibration).")
    args = ap.parse_args()

    files = list_files(args.trace_dir)
    assert files, f"no dp*_rank0.*.pt.trace.json.gz in {args.trace_dir}"
    os.makedirs(args.out, exist_ok=True)
    print(f"TRACE_DIR={args.trace_dir}\n{len(files)} files")

    moe_substrs = [s for s in args.moe_substr.split(",") if s]
    if args.dump_kernels:
        comm_names = collections.Counter(); moe_names = collections.Counter()
        for e in iter_events(files[0]):
            if e.get("ph") == "X" and e.get("cat") == "kernel":
                n = e.get("name", "")
                if args.comm_substr in n or "ccl" in n.lower() or "gather" in n.lower() or "scatter" in n.lower() or "all2all" in n.lower() or "alltoall" in n.lower():
                    comm_names[n] += 1
                if any(sub in n for sub in moe_substrs):
                    moe_names[n] += 1
        print("=== comm-ish kernel names (rank0) ===")
        for n, c in comm_names.most_common(30): print(f"  x{c:6d}  {n}")
        print(f"\n=== MoE-matmul anchor kernels (match --moe-substr={args.moe_substr}) ===")
        for n, c in moe_names.most_common(30): print(f"  x{c:6d}  {n}")
        return

    C = build(files, args.comm_substr, moe_substrs)
    metric = C.moe_busy if args.metric == "busy" else C.moe_span

    # phase mask (cudagraph majority vote)
    if args.phase != "all":
        dec = (C.cg_phase == 2).sum(0) >= (C.cg_phase == 1).sum(0)
        keep = dec if args.phase == "decode" else ~dec
        metric = metric[:, keep]
        print(f"phase={args.phase}: kept {int(keep.sum())}/{C.shape[1]} clusters")

    R, K = metric.shape
    L = args.layers_per_step or detect_layers_per_step(C) or 1
    print(f"layers-per-step L = {L}  ({'given' if args.layers_per_step else 'auto/fallback'})")
    n_steps = K // L
    step_busy = metric[:, :n_steps * L].reshape(R, n_steps, L).sum(2) if n_steps else metric.sum(1, keepdims=True)
    print(f"steps = {n_steps}  (K={K} clusters / L={L})")

    # per-cluster (finest, = current metric) + per-step (W=1) + windows
    percluster_mm = metric.max(0) / np.maximum(metric.min(0), 1e-9)
    rows = []
    def stat(mm, ma, n):
        return dict(n=int(n), maxmin_mean=round(float(mm.mean()), 3), maxmin_median=round(float(np.median(mm)), 3),
                    maxmin_p99=round(float(np.percentile(mm, 99)), 3), maxmin_max=round(float(mm.max()), 3),
                    maxavg_mean=round(float(ma.mean()), 3), maxavg_p99=round(float(np.percentile(ma, 99)), 3))
    # per-cluster maxavg
    pc_ma = metric.max(0) / np.maximum(metric.mean(0), 1e-9)
    rows.append(("per-cluster", stat(percluster_mm, pc_ma, K)))
    for W in [1] + [int(x) for x in args.windows.split(",")]:
        mm, ma, nW = windowed_imbalance(step_busy, W)
        rows.append((f"W={W}step" if W > 1 else "per-step(W=1)", stat(mm, ma, nW)))

    # print table
    print(f"\n=== EP MoE-{args.metric} imbalance (phase={args.phase}) ===")
    print(f"{'granularity':<16}{'n':>7}{'maxmin_mean':>13}{'maxmin_med':>12}{'maxmin_p99':>12}{'maxmin_max':>12}{'maxavg_mean':>13}")
    for name, s in rows:
        print(f"{name:<16}{s['n']:>7}{s['maxmin_mean']:>13}{s['maxmin_median']:>12}{s['maxmin_p99']:>12}{s['maxmin_max']:>12}{s['maxavg_mean']:>13}")

    summary = {"trace_dir": args.trace_dir, "metric": args.metric, "phase": args.phase,
               "ranks": [int(x) for x in C.ranks], "clusters": int(K), "layers_per_step": int(L),
               "steps": int(n_steps), "granularity": {name: s for name, s in rows}}
    with open(os.path.join(args.out, "summary_time_mxfp4.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {os.path.join(args.out, 'summary_time_mxfp4.json')}")


if __name__ == "__main__":
    main()
