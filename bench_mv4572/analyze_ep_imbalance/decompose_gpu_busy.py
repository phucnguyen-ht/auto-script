#!/usr/bin/env python3
# MV-4572 — Phân rã GPU-busy theo LOẠI kernel cho 1 preset (8 rank trace), để đo EPLB ăn bao
# nhiêu GPU. So r0(on) vs base(off): phần GPU-time TĂNG ở r0 nằm ở loại kernel nào?
#   - moe_matmul  : mfma_moe1/2, _gemm_afp4wfp4, _batched_gemm_a16wfp4 (compute expert - HỮU ÍCH)
#   - moe_route   : opus_moe_sort/MoeSorting/fused_mx_quant_moe_sort/grouped_topk (routing)
#   - comm_nccl   : ncclDevKernel (dispatch/combine EP + DP all_reduce)
#   - copy_xfer   : __amd_rocclr_copyBuffer / *Memcpy* / hipMemcpy (ỨNG VIÊN: NIXL rocm_ipc khuân weight EPLB)
#   - attn_mla    : mla / paged_attention / concat_and_cache / flash
#   - quant_act   : quant / silu / act / elementwise / norm
#   - other
# GPU-busy = union các khoảng kernel (KHÔNG cộng chồng — nhiều kernel song song/overlap). Ta báo:
#   (a) sum-dur mỗi loại (có thể > span do overlap giữa loại), (b) % so tong-sum, (c) per-step.
# Metric so sánh chính: sum-dur mỗi loại / #step (r0 vs base). Loại nào r0 >> base = chi phí EPLB.
import os, glob, gzip, json, argparse, collections
try:
    import orjson as _j
    def _load(p):
        with gzip.open(p,"rb") as f: return _j.loads(f.read())
except ImportError:
    def _load(p):
        with gzip.open(p,"rb") as f: return json.load(f)

CATS = [
    ("moe_matmul", ("mfma_moe","_gemm_afp4wfp4","_batched_gemm_a16wfp4")),
    ("moe_route",  ("opus_moe_sort","MoeSorting","fused_mx_quant_moe_sort","grouped_topk","moe_align")),
    ("comm_nccl",  ("ncclDevKernel","mscclKernel","rccl")),
    ("copy_xfer",  ("copyBuffer","Memcpy","memcpy","__amd_rocclr_copy")),
    ("attn_mla",   ("mla_","_mla","paged_attention","concat_and_cache","flash","attn")),
    ("quant_act",  ("quant","silu","_act","elementwise","rmsnorm","layernorm","_norm","rope","embed")),
]
def categorize(n):
    for cat, subs in CATS:
        for s in subs:
            if s in n: return cat
    return "other"

def rank_of(path):
    import re
    m = re.search(r"dp(\d+)_", os.path.basename(path)); return int(m.group(1)) if m else -1

def parse(path):
    dur = collections.Counter(); cnt = collections.Counter()
    tmin = None; tmax = None; total = 0.0
    d = _load(path); evs = d["traceEvents"] if isinstance(d,dict) else d
    for e in evs:
        if e.get("ph")!="X" or e.get("cat")!="kernel": continue
        ts=e.get("ts"); du=e.get("dur"); n=e.get("name","")
        if ts is None or du is None: continue
        ts=float(ts); du=float(du); end=ts+du
        c=categorize(n); dur[c]+=du; cnt[c]+=1; total+=du
        if tmin is None or ts<tmin: tmin=ts
        if tmax is None or end>tmax: tmax=end
    span=(tmax-tmin)/1e6 if tmin is not None else 0.0
    return dur, cnt, span, total

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace-dir",required=True); ap.add_argument("--tag",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=0,help="#step de chia per-step")
    a=ap.parse_args()
    fs=sorted(glob.glob(os.path.join(a.trace_dir,"dp*_rank0.*.pt.trace.json.gz"))) or \
       sorted(glob.glob(os.path.join(a.trace_dir,"*.pt.trace.json.gz")))
    assert fs, f"no traces in {a.trace_dir}"
    os.makedirs(a.out,exist_ok=True)
    agg=collections.Counter(); aggc=collections.Counter(); spans=[]; totals=[]
    print(f"TAG={a.tag}  {len(fs)} files")
    for f in fs:
        dur,cnt,span,total=parse(f)
        spans.append(span); totals.append(total)
        for c,_ in CATS+[("other",())]: agg[c]+=dur.get(c,0.0); aggc[c]+=cnt.get(c,0)
        print(f"  rank{rank_of(f)}: span={span:.1f}s sum_kernel={total/1e6:.1f}s "
              f"copy={dur.get('copy_xfer',0)/1e6:.2f}s moe={dur.get('moe_matmul',0)/1e6:.1f}s comm={dur.get('comm_nccl',0)/1e6:.1f}s")
    R=len(fs); span_mean=sum(spans)/R; total_mean=sum(totals)/R
    steps=a.steps or 1
    rows=[]
    print(f"\n=== GPU sum-dur theo loai (mean/rank, R={R}) — span_mean={span_mean:.1f}s steps={a.steps} ===")
    print(f"{'cat':<12}{'sum_s':>10}{'%oftot':>9}{'per_step_ms':>13}{'count':>12}")
    order=[c for c,_ in CATS]+["other"]
    for c in order:
        s=agg[c]/R/1e6; pct=100*agg[c]/max(sum(agg.values()),1); ps=agg[c]/R/1e3/steps if a.steps else 0
        rows.append(dict(cat=c,sum_s=round(s,2),pct=round(pct,1),per_step_ms=round(ps,4),count=int(aggc[c]/R)))
        print(f"{c:<12}{s:>10.2f}{pct:>8.1f}%{ps:>13.4f}{int(aggc[c]/R):>12}")
    out=dict(tag=a.tag,trace_dir=a.trace_dir,ranks=R,span_mean_s=round(span_mean,2),
             sumkernel_mean_s=round(total_mean/1e6,2),steps=a.steps,cats=rows)
    with open(os.path.join(a.out,f"{a.tag}_gpubusy.json"),"w") as fo: json.dump(out,fo,indent=2)
    print(f"saved -> {os.path.join(a.out,a.tag+'_gpubusy.json')}")

if __name__=="__main__": main()
