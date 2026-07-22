#!/usr/bin/env python3
# [MV-4572 Task2] Aggregate npz -> compact JSON để visualize toàn cảnh EPLB qua cả run.
#  - tokens collected (timeline + total, per-rank), dummy pattern
#  - mỗi rearrange: input load per logical (hotspot), churn per-rank (experts exchanged),
#    per-rank load BEFORE vs AFTER (rearrange có cân bằng lại không), maxavg before/after.
#   python3 agg_eplb_viz.py <run_dir> <out.json>   (run_dir = .../logs_analysis/<TS>)
import numpy as np, glob, json, os, sys

def agg(eplb_dir):
    fs = sorted(glob.glob(f"{eplb_dir}/eplb_tokens_rank*.npz"))
    if not fs: return None
    per_step = None; nmin = None; dummy_any = None; per_rank_tot = []
    for f in fs:
        z = np.load(f)
        load = z['load']                              # (n,1,L,P)
        isd = z['is_dummy']
        tks = load.reshape(load.shape[0], -1).sum(1).astype(np.int64)   # tokens/step this rank
        per_rank_tot.append(int(tks.sum()))
        n = len(tks)
        if per_step is None:
            nmin=n; per_step=tks.copy(); dummy_any=isd.copy()
        else:
            nmin=min(nmin,n); per_step=per_step[:nmin]+tks[:nmin]; dummy_any=dummy_any[:nmin]|isd[:nmin]
    per_step=per_step[:nmin]; dummy_any=dummy_any[:nmin]
    # downsample timeline -> ~600 pts (mean per bin)
    B=600; step=max(1,nmin//B)
    ts_tok=[int(per_step[i:i+step].mean()) for i in range(0,nmin,step)]
    ts_dummy=[float(dummy_any[i:i+step].mean()) for i in range(0,nmin,step)]  # frac steps with any dummy
    out=dict(n_ranks=len(fs), n_samples=int(nmin), total_tokens=int(per_step.sum()),
             per_rank_total_tokens=per_rank_tot,
             timeline=dict(tokens_per_step=ts_tok, dummy_frac=ts_dummy, bin=step),
             rearranges=[])
    # rearrange (rank0)
    z0=np.load(fs[0])
    if 'rearr_input' in z0.files:
        ri,om,nm=z0['rearr_input'],z0['rearr_old_map'],z0['rearr_new_map']  # (R,L,P)
        R,L,P=ri.shape; eps=P//len(fs)
        for r in range(R):
            inp=ri[r].sum(0).astype(float)            # (P/logical) total load per logical (sum layers)
            order=np.argsort(inp)[::-1]
            top=[[int(order[i]), float(inp[order[i]])] for i in range(min(20,P))]
            changed=(om[r]!=nm[r])                     # (L,P) slot đổi logical
            churn_rank=[int(changed[:, g*eps:(g+1)*eps].sum()) for g in range(len(fs))]
            def rank_load(mp):                         # per-rank load = sum logical-load của slot rank giữ (chia replica)
                pr=np.zeros(len(fs))
                for lyr in range(L):
                    m=mp[lyr]; rc=np.bincount(m, minlength=len(inp))
                    pl=ri[r][lyr][m]/np.maximum(rc[m],1)     # (P) per-physical load layer này
                    for g in range(len(fs)): pr[g]+=pl[g*eps:(g+1)*eps].sum()
                return pr
            prb=rank_load(om[r]); pra=rank_load(nm[r])
            mav=lambda x:float(np.max(x)/max(np.mean(x),1e-9))
            out['rearranges'].append(dict(idx=r, total_load=float(inp.sum()),
                maxavg_logical=round(mav(inp),3), hotspot_top20=top,
                churn_total=int(changed.sum()), churn_per_rank=churn_rank,
                per_rank_load_before=[round(x,1) for x in prb.tolist()],
                per_rank_load_after=[round(x,1) for x in pra.tolist()],
                maxavg_rank_before=round(mav(prb),3), maxavg_rank_after=round(mav(pra),3)))
    return out

run=sys.argv[1]; out=sys.argv[2]
res={}
for tag,sub in [("V2","v0.24.0_r0_v2_prof_c16_8k/r0_v2_prof"),("V1","v0.24.0_r0_noV2_prof_c16_8k/r0_noV2_prof")]:
    d=f"{run}/{sub}/eplb_tokens"
    print(f"agg {tag}: {d}", flush=True)
    res[tag]=agg(d)
    if res[tag]: print(f"  {tag}: total_tokens={res[tag]['total_tokens']} n_samples={res[tag]['n_samples']} rearr={len(res[tag]['rearranges'])}", flush=True)
json.dump(res, open(out,'w'))
print("WROTE", out)
