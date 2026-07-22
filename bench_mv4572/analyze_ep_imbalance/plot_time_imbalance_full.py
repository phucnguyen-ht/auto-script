#!/usr/bin/env python3
# MV-4572 — Phân tích time-imbalance ĐẦY ĐỦ PER-CASE (kiểu bench_mv4571/analyze_time.py), MXFP4.
# Đọc *_v2_arrays.npz (step_busy [R,S], per_step_mm, ranks) + *_gpubusy.json (nếu có).
# OUTPUT:
#   - PER-CASE (đặt trong imbalance_results/ CỦA CHÍNH config đó): hist / perrank_load / windowed /
#     gpu_breakdown / INFO_<tag>.md  — phân tích đầy đủ về time của case đó, co-located.
#   - CROSS-OVER (1 folder chung --cross-out): onoff_<ver>_<case> / crossver_<case> / SUMMARY.md
#     — chỉ chart liên quan tới 2 phiên bản (0.23 vs 0.24) + on-vs-off.
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P="/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572/auto-script/bench_mv4572/logs_v0.24.0_analysis/profile"
CASES=[
 ("0.24","off","100k","base_c16_100k",   f"{P}/nonmtp_base_c16_100k/base"),
 ("0.24","on", "100k","r0_c16_100k",     f"{P}/nonmtp_r0_c16_100k/base-eplb-nixl-async-default-r0"),
 ("0.24","off","8k",  "base_c64_8k",      f"{P}/nonmtp_base_c64_8k/base"),
 ("0.24","on", "8k",  "r0_c64_8k",        f"{P}/nonmtp_r0_c64_8k/base-eplb-nixl-async-default-r0"),
 ("0.23","off","100k","base_023_c16_100k",f"{P}/v023_base_c16_100k/base"),
 ("0.23","on", "100k","r0_023_c16_100k",  f"{P}/v023_r0_c16_100k/base-eplb-nixl-async-default-r0"),
 ("0.23","off","8k",  "base_023_c64_8k",  f"{P}/v023_base_c64_8k/base"),
 ("0.23","on", "8k",  "r0_023_c64_8k",    f"{P}/v023_r0_c64_8k/base-eplb-nixl-async-default-r0"),
]
def npz_of(d,tag):
    f=os.path.join(d,"imbalance_results",tag+"_v2_arrays.npz"); return np.load(f) if os.path.exists(f) else None
def gpubusy_of(d,tag):
    f=os.path.join(d,"imbalance_results",tag+"_gpubusy.json"); return json.load(open(f)) if os.path.exists(f) else None
def maxavg(sb): return sb.max(0)/np.maximum(sb.mean(0),1e-9)
def maxmin(sb): return sb.max(0)/np.maximum(sb.min(0),1e-9)
def win(v,w):
    n=len(v)//w; return v[:n*w].reshape(n,w).mean(1) if n else np.array([v.mean()])

def per_case(e,tag,out):
    """Phân tích ĐẦY ĐỦ time cho 1 case -> out (imbalance_results/ của config đó)."""
    sb=e["sb"]; R,S=sb.shape; mm=maxmin(sb); ma=maxavg(sb); ranks=list(range(R))
    title=f"{e['ver']} {e['eplb']} {e['case']} (tag {tag})"
    # (a) histogram max/min + max/avg
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    ax[0].hist(np.clip(mm,1,np.percentile(mm,99)),bins=50,color="#f0a0a0",edgecolor="k")
    ax[0].set_title(f"max/min per-step (median {np.median(mm):.3f}, max {mm.max():.1f})"); ax[0].set_xlabel("max/min")
    ax[1].hist(np.clip(ma,1,np.percentile(ma,99.5)),bins=50,color="#a0c0f0",edgecolor="k")
    ax[1].set_title(f"max/avg per-step (mean {ma.mean():.3f}) [gate throughput]"); ax[1].set_xlabel("max/avg")
    fig.suptitle(f"Time-imbalance histogram — {title}"); fig.tight_layout()
    fig.savefig(os.path.join(out,f"hist_{tag}.png"),dpi=120); plt.close(fig)
    # (b) per-RANK MoE-busy load (tong qua step, chuan hoa theo mean) -> rank nao straggler
    tot=sb.sum(1); rel=tot/tot.mean()
    fig,ax=plt.subplots(figsize=(8,4)); bars=ax.bar([f"dp{r}" for r in ranks],rel,color="#8fbf8f",edgecolor="k")
    ax.axhline(1.0,color="gray",ls="--",lw=1); ax.set_ylabel("MoE-busy / mean-rank");
    for i,v in enumerate(rel): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
    ax.set_title(f"Per-rank MoE-busy load (spread max/min {tot.max()/tot.min():.3f}) — {title}")
    ax.grid(axis="y",alpha=.4); fig.tight_layout(); fig.savefig(os.path.join(out,f"perrank_load_{tag}.png"),dpi=120); plt.close(fig)
    # (c) windowed imbalance curve (max/avg + max/min vs W)
    Ws=[1,50,100,250,500,1000,2000,3000]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(range(len(Ws)),[np.mean(win(ma,w)) for w in Ws],marker="o",label="max/avg mean",color="#a0c0f0")
    ax.plot(range(len(Ws)),[np.median(win(mm,w)) for w in Ws],marker="s",label="max/min median",color="#f0a0a0")
    ax.set_xticks(range(len(Ws))); ax.set_xticklabels(Ws); ax.set_xlabel("window (step) = EPLB rebalance interval")
    ax.set_ylabel("imbalance"); ax.set_title(f"Imbalance vs window — {title}"); ax.legend(); ax.grid(alpha=.4)
    fig.tight_layout(); fig.savefig(os.path.join(out,f"windowed_{tag}.png"),dpi=120); plt.close(fig)
    # (d) GPU-busy breakdown theo loai kernel (neu co gpubusy.json)
    gb=e["gb"]
    if gb:
        cats=[c["cat"] for c in gb["cats"]]; ss=[c["sum_s"] for c in gb["cats"]]
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(cats,ss,color="#c9b7e0",edgecolor="k")
        for i,v in enumerate(ss): ax.text(i,v,f"{v:.1f}",ha="center",va="bottom",fontsize=8)
        ax.set_ylabel("GPU sum-dur (s, mean/rank)"); ax.set_title(f"GPU-busy breakdown (span {gb['span_mean_s']}s) — {title}")
        plt.xticks(rotation=30,ha="right"); ax.grid(axis="y",alpha=.4)
        fig.tight_layout(); fig.savefig(os.path.join(out,f"gpu_breakdown_{tag}.png"),dpi=120); plt.close(fig)
    # (e) INFO md — so lieu + cach doc
    L=[f"# Time-imbalance (đầy đủ) — {title}","",
       f"- ranks×steps: {R}×{S}",
       f"- **per-step max/avg** (gate throughput): mean **{ma.mean():.3f}**, median {np.median(ma):.3f}, p99 {np.percentile(ma,99):.3f}",
       f"- per-step max/min: median {np.median(mm):.3f}, mean {mm.mean():.3f}, max {mm.max():.1f} (max/min max lớn = bệnh min→0 dummy-rank, KHÔNG gate throughput)",
       f"- per-rank MoE-busy load spread (tổng qua run) max/min = **{tot.max()/tot.min():.3f}**; rank bận nhất dp{int(tot.argmax())} ({rel.max():.3f}×), rảnh nhất dp{int(tot.argmin())} ({rel.min():.3f}×)"]
    if gb:
        L.append(f"- GPU-busy (mean/rank, span {gb['span_mean_s']}s): "+", ".join(f"{c['cat']} {c['sum_s']:.1f}s ({c['pct']}%)" for c in gb['cats']))
    L+= ["","**Charts:** `hist_{t}.png` · `perrank_load_{t}.png` · `windowed_{t}.png`"
         .format(t=tag)+(" · `gpu_breakdown_{t}.png`".format(t=tag) if gb else "")+
         " · info `{t}_v2.txt`/`{t}_v2.json` (v2 barrier-align).".format(t=tag),
         "So 2 phiên bản / on-vs-off: xem folder `../../../cross-over-0.23-0.24/`."]
    open(os.path.join(out,f"INFO_{tag}.md"),"w").write("\n".join(L))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cross-out",required=True,help="folder chung cho chart 2 phien ban (cross-over-0.23-0.24)")
    ap.add_argument("--only",default="",help="chi chay 1 tag (vd base_c16_100k) de test")
    a=ap.parse_args(); os.makedirs(a.cross_out,exist_ok=True)
    D={}
    for ver,eplb,case,tag,d in CASES:
        z=npz_of(d,tag)
        if z is None: print(f"skip {tag} (no npz)"); continue
        D[tag]=dict(sb=z["step_busy"],ver=ver,eplb=eplb,case=case,dir=d,gb=gpubusy_of(d,tag))
    # PER-CASE: dat trong imbalance_results/ cua tung config
    for tag,e in D.items():
        if a.only and tag!=a.only: continue
        out=os.path.join(e["dir"],"imbalance_results"); os.makedirs(out,exist_ok=True)
        per_case(e,tag,out); print(f"per-case {tag} -> {out}")
    # CROSS-OVER: on-vs-off + 0.23-vs-0.24
    Ws=[1,50,100,250,500,1000]
    for ver in ["0.24","0.23"]:
        for case in ["100k","8k"]:
            on=[e for t,e in D.items() if e["ver"]==ver and e["eplb"]=="on" and e["case"]==case]
            off=[e for t,e in D.items() if e["ver"]==ver and e["eplb"]=="off" and e["case"]==case]
            if not on or not off: continue
            on,off=on[0],off[0]; fig,ax=plt.subplots(1,2,figsize=(13,4.2))
            for lbl,e,c in [("off",off,"#3070c0"),("on(r0)",on,"#c03030")]:
                ma=maxavg(e["sb"]); mm=maxmin(e["sb"])
                ax[0].plot([np.mean(win(ma,w)) for w in Ws],marker="o",label=lbl,color=c)
                ax[1].plot([np.median(win(mm,w)) for w in Ws],marker="o",label=lbl,color=c)
            ax[0].set_xticks(range(len(Ws)));ax[0].set_xticklabels(Ws);ax[0].set_title("max/avg (gate throughput)");ax[0].legend();ax[0].grid(alpha=.4)
            ax[1].set_xticks(range(len(Ws)));ax[1].set_xticklabels(Ws);ax[1].set_title("max/min (median)");ax[1].legend();ax[1].grid(alpha=.4)
            fig.suptitle(f"EPLB on-vs-off — v{ver} {case}"); fig.tight_layout()
            fig.savefig(os.path.join(a.cross_out,f"onoff_{ver}_{case}.png"),dpi=120); plt.close(fig)
    for case in ["100k","8k"]:
        fig,ax=plt.subplots(figsize=(7,4.5)); labels=[];vals=[];cols=[]
        for ver in ["0.23","0.24"]:
            for eplb,c in [("off","#3070c0"),("on","#c03030")]:
                e=[x for t,x in D.items() if x["ver"]==ver and x["eplb"]==eplb and x["case"]==case]
                if not e: continue
                labels.append(f"v{ver}\n{eplb}"); vals.append(float(maxavg(e[0]["sb"]).mean())); cols.append(c)
        ax.bar(labels,vals,color=cols); ax.set_ylim(1.0,(max(vals)*1.05) if vals else 1.2)
        for i,v in enumerate(vals): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom")
        ax.set_ylabel("per-step max/avg (mean)"); ax.set_title(f"Imbalance max/avg — 0.23 vs 0.24 — {case}")
        ax.grid(axis="y",alpha=.4); fig.tight_layout(); fig.savefig(os.path.join(a.cross_out,f"crossver_{case}.png"),dpi=120); plt.close(fig)
    # SUMMARY (cross-over)
    L=["# Cross-over 0.23 vs 0.24 — time-imbalance (MoE-busy, max/avg gate throughput)","",
       "Per-case charts + info: trong `imbalance_results/` của TỪNG config profile. Đây là folder so sánh 2 phiên bản.","",
       "| ver | EPLB | case | per-step max/avg mean | max/min median |","|---|---|---|---|---|"]
    for ver,eplb,case,tag,d in CASES:
        if tag not in D: continue
        sb=D[tag]["sb"]; L.append(f"| {ver} | {eplb} | {case} | {maxavg(sb).mean():.3f} | {np.median(maxmin(sb)):.3f} |")
    L+=["","**Charts:** `onoff_<ver>_<case>.png` (on-vs-off mỗi version) · `crossver_<case>.png` (0.23 vs 0.24).",
        "**Kết luận:** on≈off theo max/avg ở CẢ 2 version → imbalance KHÔNG giải thích regression (chi tiết "
        "`docs/followup/time-imbalance.md` §B–E). Token-imbalance chưa làm (cần EP_COLLECT instrumentation)."]
    open(os.path.join(a.cross_out,"SUMMARY.md"),"w").write("\n".join(L))
    print("cross-over ->",a.cross_out); print("\n".join(os.listdir(a.cross_out)))

if __name__=="__main__": main()
