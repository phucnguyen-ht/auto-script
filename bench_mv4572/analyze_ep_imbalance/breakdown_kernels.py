#!/usr/bin/env python3
# [MV-4572 Task2 B] Performance breakdown per-kernel/CPU-op từ torch trace, so V2 vs V1.
# Mục tiêu: op/section nào TĂNG time ở V2 (vs V1) -> trace back code. (kiểu perfetto breakdown.)
# Aggregate duration theo TÊN (normalize số/shape) cho GPU kernel + CPU op, per-rank.
#   python3 breakdown_kernels.py <V2_profiling_result_dir> <V1_profiling_result_dir> [--ranks 0,1,..]
import sys, gzip, json, re, glob, os, collections, argparse

def norm(n):
    n = re.sub(r'\b\d[\d,]*\b', 'N', n)            # numbers/shapes -> N
    n = re.sub(r'0x[0-9a-fA-F]+', 'H', n)
    return n[:90]

def rank_of(p):
    m = re.search(r'dp(\d+)_', os.path.basename(p)); return int(m.group(1)) if m else -1

def breakdown(path):
    with gzip.open(path, 'rt') as f: d = json.load(f)
    evs = d["traceEvents"] if isinstance(d, dict) else d
    gpu = collections.defaultdict(lambda:[0.0,0]); cpu = collections.defaultdict(lambda:[0.0,0])
    tg=tc=0.0
    for e in evs:
        if e.get("ph") != "X": continue
        dur = e.get("dur");  n = e.get("name","");  cat = e.get("cat")
        if dur is None: continue
        dur=float(dur)
        if cat == "kernel": gpu[norm(n)][0]+=dur; gpu[norm(n)][1]+=1; tg+=dur
        elif cat in ("cpu_op","user_annotation","python_function"): cpu[norm(n)][0]+=dur; cpu[norm(n)][1]+=1; tc+=dur
    return gpu, cpu, tg, tc

def agg_dir(d, ranks):
    fs = sorted(glob.glob(os.path.join(d, "dp*_rank0.*.pt.trace.json.gz")))
    if ranks is not None: fs = [f for f in fs if rank_of(f) in ranks]
    G=collections.defaultdict(lambda:[0.0,0]); C=collections.defaultdict(lambda:[0.0,0]); TG=TC=0.0; nr=0
    for f in fs:
        g,c,tg,tc = breakdown(f); nr+=1
        for k,v in g.items(): G[k][0]+=v[0]; G[k][1]+=v[1]
        for k,v in c.items(): C[k][0]+=v[0]; C[k][1]+=v[1]
        TG+=tg; TC+=tc
        print(f"    {os.path.basename(f)}: gpu_tot={tg/1e6:.2f}s cpu_tot={tc/1e6:.2f}s", flush=True)
    return G,C,TG,TC,nr

def show(name, A, B, totA, totB, nrank, topn=30):
    print(f"\n########## {name}: V2 tot={totA/1e6:.2f}s | V1 tot={totB/1e6:.2f}s | V2-V1={ (totA-totB)/1e6:+.2f}s ({nrank} rank) ##########")
    keys=set(A)|set(B)
    rows=[]
    for k in keys:
        a=A.get(k,[0,0]); b=B.get(k,[0,0]); rows.append((a[0]-b[0], k, a[0],a[1], b[0],b[1]))
    rows.sort(reverse=True, key=lambda r:abs(r[0]))
    print(f"{'Δ(V2-V1) s':>12} {'V2 s':>9} {'V2 cnt':>9} {'V1 s':>9} {'V1 cnt':>9}  name")
    for dl,k,as_,ac,bs,bc in rows[:topn]:
        print(f"{dl/1e6:>+12.3f} {as_/1e6:>9.3f} {ac:>9d} {bs/1e6:>9.3f} {bc:>9d}  {k}")

ap=argparse.ArgumentParser(); ap.add_argument("v2"); ap.add_argument("v1"); ap.add_argument("--ranks",default=None)
a=ap.parse_args()
ranks=set(int(x) for x in a.ranks.split(",")) if a.ranks else None
print("=== V2 dir:", a.v2); G2,C2,TG2,TC2,nr2=agg_dir(a.v2, ranks)
print("=== V1 dir:", a.v1); G1,C1,TG1,TC1,nr1=agg_dir(a.v1, ranks)
show("GPU KERNELS", G2,G1,TG2,TG1,nr2)
show("CPU OPS", C2,C1,TC2,TC1,nr2)
