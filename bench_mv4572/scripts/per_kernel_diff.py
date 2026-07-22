#!/usr/bin/env python3
"""Per-kernel-name GPU-time breakdown for 2 traces (find WHERE the +10% goes).
Decode-isolated: windows to [WIN_LO,WIN_HI] of the timeline (late = decode-heavy at conc16/8k),
histograms cat==kernel by name -> total dur + count, prints top-N side by side + delta.
If a kernel grows in the regressed trace -> extra compute; if totals ~equal but wall grows -> bubble.
Usage: per_kernel_diff.py LABEL_A:traceA.gz LABEL_B:traceB.gz  (one rank each)
Env: WIN_LO=0.55 WIN_HI=0.95 TOPN=30
"""
import os, sys, gzip, collections
WIN_LO=float(os.environ.get("WIN_LO","0.55")); WIN_HI=float(os.environ.get("WIN_HI","0.95")); TOPN=int(os.environ.get("TOPN","30"))
try:
    import orjson as J
    load=lambda p: J.loads(gzip.open(p,"rb").read())
except ImportError:
    import json as J
    load=lambda p: J.load(gzip.open(p,"rt"))

def hist(path):
    d=load(path); ev=d["traceEvents"] if isinstance(d,dict) else d
    ks=[(e["ts"],e.get("dur",0),e.get("name","")) for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel" and e.get("dur",0)>0]
    del d,ev
    ks.sort()
    t0=ks[0][0]; t1=ks[-1][0]+ks[-1][1]; T=t1-t0
    lo=t0+WIN_LO*T; hi=t0+WIN_HI*T
    ks=[(ts,du,n) for ts,du,n in ks if lo<=ts<hi]
    tot=collections.Counter(); cnt=collections.Counter()
    for ts,du,n in ks: tot[n]+=du; cnt[n]+=1
    total=sum(tot.values())
    return tot,cnt,total,len(ks),(hi-lo)/1000.0

def main():
    (la,pa),(lb,pb)=[a.split(":",1) for a in sys.argv[1:3]]
    ta,ca,tota,na,spa=hist(pa); tb,cb,totb,nb,spb=hist(pb)
    print(f"window=[{WIN_LO},{WIN_HI}] | {la}: total_kernel={tota/1000:.1f}ms over {spa:.1f}ms wall, {na} kernels | {lb}: total_kernel={totb/1000:.1f}ms over {spb:.1f}ms wall, {nb} kernels")
    print(f"{la} kernel-busy/wall = {tota/1000/spa*100:.1f}% | {lb} = {totb/1000/spb*100:.1f}%")
    # normalize per wall-ms so window-length confound removed:
    print(f"\n=== per-kernel time, NORMALIZED to us-per-wall-second (removes window-length confound) ===")
    print(f"{'kernel (trunc 46)':<48}{la+'_us/s':>12}{lb+'_us/s':>12}{'Δ_us/s':>11}{'Δ%':>8}")
    print("-"*99)
    allk=set(ta)|set(tb)
    rows=[]
    for k in allk:
        a=ta[k]/spa*1000/1000.0  # us per wall-second: (us)/(wall_ms)*1000ms/1000? keep as us per wall-sec
        pass
    # simpler: rate = total_us / wall_seconds
    wa=spa/1000.0; wb=spb/1000.0
    for k in allk:
        ra=ta[k]/wa; rb=tb[k]/wb  # us of this kernel per wall-second
        rows.append((rb-ra, k, ra, rb))
    rows.sort(key=lambda x: abs(x[0]), reverse=True)
    for dr,k,ra,rb in rows[:TOPN]:
        pct = (rb-ra)/ra*100 if ra>0 else float('inf')
        print(f"{k[:46]:<48}{ra:>12.0f}{rb:>12.0f}{dr:>11.0f}{pct:>7.0f}%")
    print("-"*99)
    print(f"{'TOTAL us/wall-sec':<48}{tota/wa:>12.0f}{totb/wb:>12.0f}{totb/wb-tota/wa:>11.0f}{(totb/wb-tota/wa)/(tota/wa)*100:>7.1f}%")
    print("(rate ~equal => same compute/sec, regress = fewer tokens/sec at same GPU work? ; a kernel's rate UP in regress => extra compute)")

if __name__=="__main__": main()
