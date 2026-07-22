#!/usr/bin/env python3
# [MV-4572 Task2 §4.9.3] GPU-busy / GPU-idle (bubble) analyzer từ torch trace.
# Phân biệt: P1 host/pipeline bubble (idle TĂNG) vs P2 GPU work (compute TĂNG) vs comm-wait (nccl TĂNG).
# GPU-busy = UNION các interval kernel (mọi stream) = lúc CÓ kernel chạy. idle = wall - busy.
#   python3 gpu_idle.py <profiling_result_dir> [--ranks 0] [--label NAME]
import sys, gzip, json, re, glob, os, argparse, collections

def rank_of(p):
    m = re.search(r'dp(\d+)_', os.path.basename(p)); return int(m.group(1)) if m else -1

def analyze(path):
    with gzip.open(path, 'rt') as f: d = json.load(f)
    evs = d["traceEvents"] if isinstance(d, dict) else d
    ivs=[]            # (start, end) mọi GPU kernel
    nccl=0.0; comp=0.0; ncnt=0; ccnt=0
    for e in evs:
        if e.get("ph")!="X" or e.get("cat")!="kernel": continue
        dur=e.get("dur");  ts=e.get("ts")
        if dur is None or ts is None: continue
        dur=float(dur); ts=float(ts)
        ivs.append((ts, ts+dur))
        nm=e.get("name","")
        if "nccl" in nm.lower() or "ncclDevKernel" in nm: nccl+=dur; ncnt+=1
        else: comp+=dur; ccnt+=1
    if not ivs: return None
    ivs.sort()
    # union of intervals -> busy
    busy=0.0; cs,ce=ivs[0]
    for s,e in ivs[1:]:
        if s>ce: busy+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    busy+=ce-cs
    wall=ivs[-1][1]-ivs[0][0]
    return dict(wall=wall/1e6, busy=busy/1e6, idle=(wall-busy)/1e6,
                nccl=nccl/1e6, comp=comp/1e6, nsum=(nccl+comp)/1e6, ncnt=ncnt, ccnt=ccnt)

ap=argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--ranks",default="0"); ap.add_argument("--label",default="")
a=ap.parse_args()
ranks=[int(x) for x in a.ranks.split(",")]
fs=sorted(glob.glob(os.path.join(a.dir,"dp*_rank0.*.pt.trace.json.gz")))
fs=[f for f in fs if rank_of(f) in ranks]
print(f"=== {a.label or a.dir} (ranks={ranks}) ===")
print(f"{'rank':>4} {'wall_s':>8} {'busy_s':>8} {'idle_s':>8} {'busy%':>6} {'nccl_s':>8} {'comp_s':>8}")
agg=collections.defaultdict(float); n=0
for f in fs:
    r=analyze(f)
    if not r: continue
    n+=1
    for k in ("wall","busy","idle","nccl","comp"): agg[k]+=r[k]
    print(f"{rank_of(f):>4} {r['wall']:>8.2f} {r['busy']:>8.2f} {r['idle']:>8.2f} {100*r['busy']/r['wall']:>5.1f}% {r['nccl']:>8.2f} {r['comp']:>8.2f}", flush=True)
if n:
    print(f"{'AVG':>4} {agg['wall']/n:>8.2f} {agg['busy']/n:>8.2f} {agg['idle']/n:>8.2f} {100*agg['busy']/agg['wall']:>5.1f}% {agg['nccl']/n:>8.2f} {agg['comp']/n:>8.2f}")
