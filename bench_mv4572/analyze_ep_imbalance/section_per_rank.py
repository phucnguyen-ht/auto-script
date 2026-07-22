#!/usr/bin/env python3
# [MV-4572] Per-RANK, per-SECTION kernel-time breakdown from torch traces.
# Focus: r0-norearr V2 vs V1 (base = reference only). Check if "Others/Comm inflated"
# holds across ALL 8 ranks or is rank0-only. Reports section TOTAL + COUNT + avg-per-call
# (avg/call removes the bench-window #calls confound).
#   python3 section_per_rank.py <V2_dir> <V1_dir>
import sys, gzip, json, re, glob, os, collections

def section_of(n):
    s = n.lower()
    if "nccl" in s: return "Communication"
    if "indexer" in s or "sparse_attn" in s: return "Indexer"
    if "mla" in s or "fused_qk_rmsnorm" in s or "concat_and_cache" in s or "flash" in s or "attention" in s:
        return "Attention"
    if ("moe" in s or "afp4wfp4" in s or "wvsplitk" in s or "eplb" in s or "grouped_topk" in s
        or "mxfp4_quant" in s or "batched_gemm_a16wfp4" in s or "silu" in s):
        return "MoE"
    if "cijk" in s or "hgemm" in s or "unquantized_gemm" in s: return "Linear"
    return "Others"

def rank_of(p):
    m = re.search(r'dp(\d+)_', os.path.basename(p)); return int(m.group(1)) if m else -1

_seccache={}
def parse(path):
    with gzip.open(path,'rt') as f: d=json.load(f)
    evs=d["traceEvents"] if isinstance(d,dict) else d
    sec=collections.defaultdict(lambda:[0.0,0])  # section -> [time_us, count]
    nfwd_norm=0  # concat_and_cache_mla count ~ forwards*attn_layers (per-forward normalizer)
    sc=_seccache
    for e in evs:
        if e.get("cat")!="kernel": continue
        dur=e.get("dur")
        if dur is None: continue
        nm=e["name"]
        s=sc.get(nm)
        if s is None:
            s=section_of(nm); sc[nm]=s
        r=sec[s]; r[0]+=dur; r[1]+=1
        if s=="Attention" and "concat_and_cache" in nm: nfwd_norm+=1
    return sec, nfwd_norm

def agg(dirp):
    fs=sorted(glob.glob(os.path.join(dirp,"dp*_rank0.*.pt.trace.json.gz")))
    out={}
    for f in fs:
        r=rank_of(f); sec,nf=parse(f); out[r]=(sec,nf)
        print(f"  parsed rank{r}: norm(concat_cache)={nf}", flush=True)
    return out

SECTIONS=["MoE","Communication","Attention","Indexer","Linear","Others"]
v2=agg(sys.argv[1]); print("--- V2 done ---",flush=True)
v1=agg(sys.argv[2]); print("--- V1 done ---",flush=True)

print("\n================ PER-RANK SECTION: total_ms (V2 | V1 | V2/V1) ================")
for sctn in SECTIONS:
    print(f"\n--- {sctn} ---")
    print(f"{'rank':>4} {'V2_ms':>9} {'V1_ms':>9} {'V2/V1':>6} {'V2/fwd_us':>10} {'V1/fwd_us':>10} {'perfwd_ratio':>12}")
    for r in range(8):
        if r not in v2 or r not in v1: continue
        s2,n2=v2[r]; s1,n1=v1[r]
        t2=s2[sctn][0]/1e3; t1=s1[sctn][0]/1e3
        # per-forward: section_us / normalizer_count  (removes window #calls confound)
        pf2=s2[sctn][0]/n2 if n2 else 0; pf1=s1[sctn][0]/n1 if n1 else 0
        print(f"{r:>4} {t2:>9.1f} {t1:>9.1f} {t2/t1 if t1 else 0:>6.3f} {pf2:>10.3f} {pf1:>10.3f} {pf2/pf1 if pf1 else 0:>12.3f}")
print("\n(perfwd = section_time / concat_and_cache_count = per-forward-per-attnlayer time, confound-free of #forwards)")
