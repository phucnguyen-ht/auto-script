#!/usr/bin/env python3
"""MV-4572: GPU-bubble analyzer (host-bound hypothesis §16.3).

Reuses the orjson fast-load of analyze_time_mxfp4.py. For each rank trace:
  - extract GPU kernel (ts,dur) [cat=="kernel"], + gpu_memcpy (D2H etc.)
  - window to STEADY-STATE decode: middle [WIN_LO, WIN_HI] fraction of the kernel
    timeline (skips prefill/warmup at start + teardown at end)
  - union-merge kernel intervals -> GPU active wall-time in the window
  - active_frac = union_busy / window_span   (KEY: lower => more idle => host-bound)
  - kernel_time = sum kernel durations (compute proxy; ~equal across configs => bubble,
    not extra compute)
  - gaps between merged busy intervals; big = gap > GAP_MS (inter-forward bubble)

Compare labels (A20 recover / A23 regress / V1on / V2on):
  A23 (or V2on) active_frac < A20 (or V1on) with kernel_time ~equal => HOST-BOUND BUBBLE.

Usage: analyze_gpu_bubble.py LABEL:trace_dir [LABEL:trace_dir ...]
  trace_dir contains dp*_rank0.*.pt.trace.json.gz (8 ranks).
Env: WIN_LO=0.30 WIN_HI=0.80 GAP_MS=0.20
"""
import os, sys, glob, gzip, statistics as st
import numpy as np

WIN_LO = float(os.environ.get("WIN_LO", "0.30"))
WIN_HI = float(os.environ.get("WIN_HI", "0.80"))
GAP_MS = float(os.environ.get("GAP_MS", "0.20"))

try:
    import orjson as _j
    def _load(p):
        with gzip.open(p, "rb") as f: return _j.loads(f.read())
except ImportError:
    import json as _j
    def _load(p):
        with gzip.open(p, "rt") as f: return _j.load(f)

def list_files(trace_dir):
    fs = sorted(glob.glob(os.path.join(trace_dir, "dp*_rank0.*.pt.trace.json.gz")))
    return fs or sorted(glob.glob(os.path.join(trace_dir, "*.pt.trace.json*")))

def extract_kernels(path):
    """Return (ts[us], dur[us]) numpy arrays for GPU kernels; del big obj asap."""
    d = _load(path)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    ts = []; du = []
    for e in ev:
        if e.get("ph") != "X": continue
        c = e.get("cat")
        if c != "kernel" and c != "gpu_memcpy" and c != "gpu_memset": continue
        t = e.get("ts"); dd = e.get("dur")
        if t is None or dd is None or dd <= 0: continue
        ts.append(t); du.append(dd)
    del d, ev
    a = np.asarray(ts, dtype=np.float64); b = np.asarray(du, dtype=np.float64)
    o = np.argsort(a)
    return a[o], b[o]

def analyze(path):
    ts, du = extract_kernels(path)
    if ts.size == 0: return None
    t0, t1 = ts[0], ts[-1] + du[-1]
    T = t1 - t0
    lo = t0 + WIN_LO * T; hi = t0 + WIN_HI * T
    m = (ts >= lo) & (ts < hi)
    ts = ts[m]; du = du[m]
    if ts.size == 0: return None
    starts = ts; ends = ts + du
    # union-merge
    order = np.argsort(starts)
    s = starts[order]; e = ends[order]
    merged_s = [s[0]]; merged_e = [e[0]]
    for i in range(1, len(s)):
        if s[i] <= merged_e[-1]:
            if e[i] > merged_e[-1]: merged_e[-1] = e[i]
        else:
            merged_s.append(s[i]); merged_e.append(e[i])
    ms = np.array(merged_s); me = np.array(merged_e)
    span = me[-1] - ms[0]
    busy = float((me - ms).sum())
    gaps = ms[1:] - me[:-1]
    gaps = gaps[gaps > 0]
    big = gaps[gaps / 1000.0 > GAP_MS]
    return dict(
        active_frac=busy / span if span else 0.0,
        busy_ms=busy / 1000.0, span_ms=span / 1000.0, idle_ms=(span - busy) / 1000.0,
        ktime_ms=float(du.sum()) / 1000.0, kcount=int(ts.size),
        ngaps=int(gaps.size), nbig=int(big.size), big_idle_ms=float(big.sum()) / 1000.0,
        gap_p90_ms=float(np.percentile(gaps, 90) / 1000.0) if gaps.size else 0.0,
        gap_max_ms=float(gaps.max() / 1000.0) if gaps.size else 0.0,
    )

def main():
    print(f"window=[{WIN_LO},{WIN_HI}] of timeline | big-gap > {GAP_MS}ms\n")
    hdr = f"{'label':>7} {'rk':>3} | {'active%':>7} {'busy_ms':>8} {'idle_ms':>8} {'span_ms':>8} | {'ktime_ms':>8} {'kcount':>8} | {'nbig':>5} {'bigIdle':>8} {'gapP90':>7} {'gapMax':>7}"
    print(hdr); print("-"*len(hdr))
    agg = {}
    for arg in sys.argv[1:]:
        label, td = arg.split(":", 1)
        rows = []
        for i, fp in enumerate(list_files(td)):
            try:
                r = analyze(fp)
            except Exception as ex:
                print(f"{label:>7} {i:>3} | ERROR {ex}"); continue
            if r is None: continue
            rows.append(r)
            print(f"{label:>7} {i:>3} | {r['active_frac']*100:>6.1f}% {r['busy_ms']:>8.1f} {r['idle_ms']:>8.1f} {r['span_ms']:>8.1f} | "
                  f"{r['ktime_ms']:>8.1f} {r['kcount']:>8} | {r['nbig']:>5} {r['big_idle_ms']:>8.1f} {r['gap_p90_ms']:>7.3f} {r['gap_max_ms']:>7.3f}", flush=True)
        agg[label] = rows
    print("="*len(hdr))
    for label, rows in agg.items():
        if not rows: print(f"{label:>7}  (no traces)"); continue
        m = lambda k: st.mean(r[k] for r in rows)
        print(f"{label:>7} {'AVG':>3} | {m('active_frac')*100:>6.1f}% {m('busy_ms'):>8.1f} {m('idle_ms'):>8.1f} {m('span_ms'):>8.1f} | "
              f"{m('ktime_ms'):>8.1f} {m('kcount'):>8.0f} | {m('nbig'):>5.1f} {m('big_idle_ms'):>8.1f} {m('gap_p90_ms'):>7.3f} {m('gap_max_ms'):>7.3f}")
    print()
    labs = [l for l in agg if agg[l]]
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            a, b = labs[i], labs[j]
            fa = st.mean(r['active_frac'] for r in agg[a]); fb = st.mean(r['active_frac'] for r in agg[b])
            ka = st.mean(r['ktime_ms'] for r in agg[a]); kb = st.mean(r['ktime_ms'] for r in agg[b])
            print(f"[Δ {a} vs {b}] active_frac {fa*100:.1f}% vs {fb*100:.1f}% (Δ {(fb-fa)*100:+.1f}pp) | "
                  f"kernel_time {ka:.1f} vs {kb:.1f}ms (Δ {(kb-ka)/ka*100:+.1f}%)")

if __name__ == "__main__":
    main()
