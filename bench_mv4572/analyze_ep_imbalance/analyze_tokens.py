#!/usr/bin/env python3
# =============================================================================
# MV-4572 — Phân tích TOKEN (#tokens/expert, token-imbalance) + REARRANGE (input/output EPLB)
# từ buffer do endpoint /collect_eplb dump ra: eplb_tokens_rank{0..7}.npz.
#
# Mỗi file rank chứa (xem gpu_worker.collect_eplb_buffer + eplb_state buffer):
#   - load[n_samples, n_models, L, P]  : per-step, THIS-rank LOCAL, per-PHYSICAL-expert
#       (= #token của rank này route tới mỗi GLOBAL physical expert; DENSE). Append MỖI step.
#   - rearrangement_step[n_samples]    : counter EPLB (reset về 0 ngay sau mỗi rearrange).
#   - is_dummy[n_samples]              : True = step dummy (load = 0) -> DROP.
#   rank0 THÊM (rearrange snapshot, async decision point):
#   - rearr_input[n_rearr, L, num_logical] : INPUT policy thật (global, all-reduced, per-LOGICAL).
#   - rearr_old_map / rearr_new_map[n_rearr, L, P] : placement TRƯỚC/SAU (physical_to_logical_map).
#   - rearr_model[n_rearr].
#
# GLOBAL per-expert load per step = SUM `load` qua 8 rank (index-aligned vì mọi rank append cùng
# lockstep step). Per-rank load (gate throughput) = gom cột physical theo block rank (p in
# [d*M,(d+1)*M) = rank d) rồi sum. Token-imbalance cross-rank = max_rank/avg_rank.
#
# #experts rearranged — 2 đại lượng:
#   (A) transfer THẬT qua mạng = recv_count, TÁI TẠO offline từ old/new map (replay
#       rebalance_execute.move_to_buffer) — loại move nội bộ + unchanged.
#   (B) placement churn = (old_map != new_map).sum() theo layer.
#
# Usage:
#   python3 analyze_tokens.py --dir <.../eplb_tokens> --out <outdir> [--tag NAME]
#   python3 analyze_tokens.py --dir <onA> --dir2 <onB> --label NAMEA --label2 NAMEB --out <o>  # so 0.23 vs 0.24
# =============================================================================
import os, re, glob, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dir(d):
    files = sorted(glob.glob(os.path.join(d, "eplb_tokens_rank*.npz")))
    assert files, f"no eplb_tokens_rank*.npz in {d}"
    ranks = {}
    rearr = None
    for f in files:
        r = int(re.search(r"rank(\d+)", os.path.basename(f)).group(1))
        z = np.load(f, allow_pickle=True)
        keys = set(z.files)
        # Eager-load `load` per rank (kept in RAM): needed BOTH for the global sum AND for the
        # PER-RANK check2 window recon (the EPLB ring advances only on that rank's non-dummy
        # steps). 8 x ~1.5GB ~= 12GB int32; box has ample RAM. If RAM ever tight, stream instead
        # and drop check2's per-rank recon.
        ranks[r] = dict(
            load=z["load"] if "load" in keys else None,
            step=z["rearrangement_step"] if "rearrangement_step" in keys else None,
            dummy=z["is_dummy"] if "is_dummy" in keys else None,
        )
        if "rearr_input" in keys:
            rearr = dict(input=z["rearr_input"], old=z["rearr_old_map"],
                         new=z["rearr_new_map"], model=z["rearr_model"])
    return ranks, rearr


def global_load(ranks):
    """SUM per-rank local `load` -> global per-physical-expert load per step.
    Returns (load_g[n_samples, n_models, L, P], step[n_samples], dummy[n_samples]).
    Verifies all ranks index-aligned (same n_samples)."""
    rs = sorted(ranks)
    have = [r for r in rs if ranks[r]["load"] is not None]
    assert have, "no token buffers (log_balancedness off?)"
    ns = [ranks[r]["load"].shape[0] for r in have]
    n = min(ns)
    if len(set(ns)) != 1:
        print(f"  WARN: rank sample counts differ {dict(zip(have,ns))} -> trim to {n}")
    load_g = None
    for r in have:                               # accumulate int64 in a loop (no 8x intermediates)
        a = ranks[r]["load"][:n].astype(np.int64)
        load_g = a if load_g is None else load_g + a
    ref = have[0]
    step = ranks[ref]["step"][:n]
    dummy = ranks[ref]["dummy"][:n]
    return load_g, step, dummy, len(have)


def per_rank_from_physical(load_g, n_ranks):
    """load_g[..., P] -> per-rank tokens [..., n_ranks] by summing each rank's expert block."""
    *lead, P = load_g.shape
    M = P // n_ranks
    return load_g.reshape(*lead, n_ranks, M).sum(-1)  # [..., n_ranks]


def recv_count_from_maps(old_map, new_map, n_ranks):
    """Replay rebalance_execute.move_to_buffer: #experts EACH rank pulls from REMOTE.
    old_map/new_map: (L, P). Returns (total, per_layer[L], per_rank[n_ranks])."""
    L, P = old_map.shape
    M = P // n_ranks
    per_layer = np.zeros(L, np.int64)
    per_rank = np.zeros(n_ranks, np.int64)
    for l in range(L):
        for r in range(n_ranks):
            b = r * M
            ol = old_map[l, b:b + M]
            nl = new_map[l, b:b + M]
            valid = nl != -1
            recvd_local = (ol == nl) | (valid & np.isin(nl, ol))
            need = (~recvd_local) & valid
            c = int(np.unique(nl[need]).size)
            per_layer[l] += c
            per_rank[r] += c
    return int(per_layer.sum()), per_layer, per_rank


def dummy_report(ranks, n, tag):
    """WHICH steps are dummy (per-rank + cross-rank). is_dummy is PER-RANK (một rank idle ->
    dummy forward, các rank khác có thể KHÔNG), nên báo cả: đếm/rank, có phải mọi rank dummy
    cùng lúc không, và tập step dummy có LIỀN từ đầu không (giả thuyết: dummy chỉ lúc init)."""
    rs = sorted(r for r in ranks if ranks[r]["dummy"] is not None)
    D = {r: ranks[r]["dummy"][:n].astype(bool) for r in rs}
    any_d = np.zeros(n, bool); ref = D[rs[0]]; all_same = True
    for r in rs:
        any_d |= D[r]
        if not np.array_equal(D[r], ref):
            all_same = False
    di = np.where(any_d)[0]
    out = dict(
        per_rank_n_dummy={int(r): int(D[r].sum()) for r in rs},
        all_ranks_same_dummy=bool(all_same),
        n_steps_any_dummy=int(di.size), n_total=int(n),
        contiguous_from_start=bool(di.size > 0 and di[0] == 0 and di[-1] == di.size - 1),
        first_idx=int(di[0]) if di.size else -1,
        last_idx=int(di[-1]) if di.size else -1,
        idx_head=di[:40].tolist(),
    )
    print(f"[{tag}] DUMMY: any-rank dummy tại {di.size}/{n} step; "
          f"contiguous_from_start(init-only?)={out['contiguous_from_start']}; "
          f"first={out['first_idx']} last={out['last_idx']}; all_ranks_same={all_same}; "
          f"per_rank={out['per_rank_n_dummy']}")
    return out


def validate(load_g, step, dummy, rearr, ranks, n_ranks, step_interval, window_size, tag):
    """Self-consistency checks confirming the logged data is correct.
    Check 1: per-step total assignments (sum over ALL physical experts); a zero/degenerate total
             flags a broken buffer (cv is naturally high: prefill vs decode steps mix).
    Check 2: RE-DERIVE the rearrange INPUT from the per-step buffer and compare to the
             independently-logged rearr_input. The EPLB load window is a RING that advances ONLY
             on NON-DUMMY recorded steps (eplb_state.py:609-621), and is_dummy is PER-RANK -> so
             reconstruct PER RANK: each rank's window before rearrange k = ITS last `window_size`
             non-dummy steps up to the rearrange; sum per-rank windows -> global window; map
             physical->logical via old_map. Must equal rearr_input (ratio~1, mean_rel~0).
             (should_record is always True here: log_balancedness -> steps_until_log<=window_size,
             so every non-dummy step is recorded; dummy steps skip the ring.)"""
    out = {}
    tot = load_g[:, 0].reshape(load_g.shape[0], -1).sum(1)
    nz = tot[tot > 0]   # non-all-dummy steps (is_dummy is PER-RANK; global total>0 = some real)
    out["check1_perstep_total"] = dict(
        mean=float(nz.mean()) if len(nz) else 0.0,
        min=int(nz.min()) if len(nz) else 0, max=int(nz.max()) if len(nz) else 0,
        cv=float(nz.std() / max(nz.mean(), 1e-9)) if len(nz) else 0.0,
    )
    if rearr is not None:
        old = rearr["old"]; inp = rearr["input"]; n_rearr = old.shape[0]
        n = load_g.shape[0]
        st = np.asarray(step[:n]).astype(np.int64)
        # rearrange k fires at end of cycle k (counter hits step_interval then resets to 0);
        # buffer index just after = a reset (step drops). seg_ends[k] = index where rearr k fired.
        seg_ends = list(np.where(np.diff(st) < 0)[0] + 1) + [n]
        rrs = sorted(r for r in ranks if ranks[r].get("load") is not None)
        recon_g = {}; win_len = {}
        for r in rrs:                                    # PER-RANK ring recon
            lr = ranks[r]["load"][:n, 0]                 # [n, L, P] this rank (physical)
            nd = np.where(~np.asarray(ranks[r]["dummy"][:n]).astype(bool))[0]  # non-dummy indices
            for k in range(min(n_rearr, len(seg_ends))):
                w = nd[nd < seg_ends[k]][-window_size:]  # last window_size non-dummy before rearr k
                if len(w) == 0:
                    continue
                s = lr[w].astype(np.int64).sum(0)        # [L, P] this rank's window
                recon_g[k] = s if k not in recon_g else recon_g[k] + s
                win_len[k] = min(win_len.get(k, 1 << 30), len(w))
            del lr
        checks = []
        for k in sorted(recon_g):
            phys = recon_g[k]; om = old[k]
            L, P = phys.shape
            num_logical = inp[k].shape[1]
            recon = np.zeros((L, num_logical), np.int64)
            for l in range(L):
                np.add.at(recon[l], om[l], phys[l])
            logged = inp[k].astype(np.int64)
            diff = np.abs(recon - logged)
            checks.append(dict(
                rearr=k, n_nondummy_win=int(win_len[k]),
                max_abs=int(diff.max()), mean_rel=round(float((diff / np.maximum(logged, 1)).mean()), 5),
                ratio=round(float(recon.sum() / max(logged.sum(), 1)), 4),
            ))
        out["check2_input_recon"] = checks
        oks = [c for c in checks if abs(c["ratio"] - 1.0) < 0.02 and c["mean_rel"] < 0.02]
        out["check2_pass"] = (len(oks) == len(checks) and len(checks) > 0)
        print(f"[{tag}] VALIDATE check1 total/step mean={out['check1_perstep_total']['mean']:.0f} "
              f"cv={out['check1_perstep_total']['cv']:.3f} | check2 input-recon pass={out.get('check2_pass')} "
              f"({[c['ratio'] for c in checks]})")
    return out


def analyze(d, tag, out, step_interval=3000, window_size=1000):
    os.makedirs(out, exist_ok=True)
    ranks, rearr = load_dir(d)
    load_g, step, dummy, nr = global_load(ranks)
    n_samples, n_models, L, P = load_g.shape
    # keep = steps with ANY real traffic (global load>0). is_dummy is PER-RANK, so a single
    # rank's dummy flag must NOT gate the global sum -- dummy ranks already contribute 0 here.
    tot_step = load_g[:, 0].reshape(n_samples, -1).sum(1)
    keep = tot_step > 0
    lg = load_g[keep]; st = step[keep]
    print(f"[{tag}] n_samples={n_samples} (non-dummy {keep.sum()}) n_models={n_models} L={L} P={P} ranks={nr}")

    res = dict(tag=tag, dir=d, n_samples=int(n_samples), n_nondummy=int(keep.sum()),
               n_models=n_models, L=L, P=P, n_ranks=nr)
    res["validation"] = validate(load_g, step, dummy, rearr, ranks, nr, step_interval, window_size, tag)
    res["dummy"] = dummy_report(ranks, n_samples, tag)

    # ---- token cross-rank imbalance over time (model 0 = main) ----
    prk = per_rank_from_physical(lg[:, 0], nr)            # [n, L, nr] tokens per rank per layer
    prk_step = prk.sum(1)                                  # [n, nr] tokens per rank per step (sum layers)
    mx = prk_step.max(1); av = prk_step.mean(1)
    maxavg = mx / np.maximum(av, 1e-9)                     # cross-rank imbalance per step
    res["token_maxavg_cross_rank"] = dict(mean=float(maxavg.mean()), median=float(np.median(maxavg)),
                                          p99=float(np.percentile(maxavg, 99)), max=float(maxavg.max()))
    # per-rank mean share
    res["per_rank_mean_share"] = (prk_step.mean(0) / prk_step.mean(0).mean()).round(4).tolist()

    # ---- per-expert hotspot (global, summed over steps + layers) ----
    exp_load = lg[:, 0].sum(0).sum(0)                      # [P] total tokens per physical expert
    order = np.argsort(exp_load)[::-1]
    res["hotspot_top10_physexpert"] = [(int(i), int(exp_load[i])) for i in order[:10]]
    res["expert_load_maxavg"] = float(exp_load.max() / max(exp_load.mean(), 1e-9))

    # ---- rearrange input/output (rank0) ----
    if rearr is not None:
        old = rearr["old"]; new = rearr["new"]; inp = rearr["input"]
        n_rearr = old.shape[0]
        churn = [(int((old[k] != new[k]).sum()), int(old[k].size)) for k in range(n_rearr)]  # (#slot changed, total)
        recv = [recv_count_from_maps(old[k], new[k], nr) for k in range(n_rearr)]
        # input per-logical imbalance (max/avg over logical experts, summed over layers)
        inp_imb = []
        for k in range(n_rearr):
            e = inp[k].sum(0).astype(np.float64)          # [num_logical] load per logical (sum layers)
            inp_imb.append(float(e.max() / max(e.mean(), 1e-9)))
        res["rearrange"] = dict(
            n_rearr=n_rearr,
            churn_slots=[c[0] for c in churn],
            churn_total_slots=churn[0][1] if churn else 0,
            experts_transferred_remote=[c[0] for c in recv],     # (A) global remote transfers/rearrange
            input_maxavg_logical=[round(x, 4) for x in inp_imb],
            models=[str(m) for m in rearr["model"].tolist()],
        )
        print(f"[{tag}] rearranges={n_rearr}  churn(slots)={[c[0] for c in churn]}  "
              f"remote_transfers={[c[0] for c in recv]}  input_maxavg={[round(x,3) for x in inp_imb]}")

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"EPLB token analysis — {tag}", fontsize=13, fontweight="bold")
    # A: token cross-rank imbalance over time (+ rearrange markers)
    a = ax[0, 0]
    w = max(1, len(maxavg) // 300)
    a.plot(np.convolve(maxavg, np.ones(w)/w, "valid"), color="#b2182b", lw=1.0, label=f"max/avg (roll {w})")
    # rearrange boundaries = where step resets to 0
    resets = np.where(np.diff(st.astype(np.int64)) < 0)[0]
    for x in resets:
        a.axvline(x, color="#2166ac", ls=":", lw=0.7, alpha=0.6)
    a.axhline(maxavg.mean(), color="k", ls="--", lw=0.7, label=f"mean={maxavg.mean():.3f}")
    a.set_title("A. Token cross-rank imbalance (max/avg) theo thời gian; xanh chấm = rearrange")
    a.set_xlabel("sample (step)"); a.set_ylabel("max_rank/avg_rank"); a.legend(); a.grid(alpha=0.3)
    # B: per-rank mean token share
    b = ax[0, 1]
    x = np.arange(nr)
    b.bar(x, prk_step.mean(0), color="#4a90d9")
    b.set_xticks(x); b.set_xticklabels([f"dp{r}" for r in range(nr)])
    b.set_title("B. Token trung bình/step mỗi rank"); b.set_xlabel("rank"); b.set_ylabel("tokens/step"); b.grid(alpha=0.3, axis="y")
    # C: per-expert load (sorted) = hotspot
    c = ax[1, 0]
    c.plot(np.sort(exp_load)[::-1], color="#e07b39")
    c.set_title(f"C. Load per physical-expert (sorted) — max/avg={res['expert_load_maxavg']:.2f}")
    c.set_xlabel("expert rank (sorted desc)"); c.set_ylabel("tổng tokens"); c.grid(alpha=0.3)
    # D: #experts rearranged per rearrange (churn vs remote-transfer)
    d = ax[1, 1]
    if rearr is not None and n_rearr:
        xr = np.arange(n_rearr); wb = 0.4
        d.bar(xr - wb/2, [c[0] for c in churn], wb, color="#7b52ab", label="churn (slots changed)")
        d.bar(xr + wb/2, [c[0] for c in recv], wb, color="#2166ac", label="remote transfers (A)")
        d.set_xticks(xr); d.set_xlabel("rearrange #"); d.set_ylabel("#experts")
        d.set_title("D. #experts rearranged mỗi lần: churn vs transfer thật"); d.legend(); d.grid(alpha=0.3, axis="y")
    else:
        d.text(0.5, 0.5, "no rearrange snapshot (rank0 file only)", ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out, f"tokens_{tag}.png"); fig.savefig(p, dpi=130); plt.close(fig)
    with open(os.path.join(out, f"tokens_{tag}.json"), "w") as fo:
        json.dump(res, fo, indent=2)
    print(f"[{tag}] saved -> {p} + tokens_{tag}.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eplb_tokens dir (config A)")
    ap.add_argument("--dir2", default=None, help="eplb_tokens dir (config B, để so sánh)")
    ap.add_argument("--label", default="A"); ap.add_argument("--label2", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-interval", type=int, default=3000)
    ap.add_argument("--window-size", type=int, default=1000)
    args = ap.parse_args()
    rA = analyze(args.dir, args.label, args.out, args.step_interval, args.window_size)
    if args.dir2:
        rB = analyze(args.dir2, args.label2, args.out, args.step_interval, args.window_size)
        cmp = {args.label: rA.get("token_maxavg_cross_rank"),
               args.label2: rB.get("token_maxavg_cross_rank"),
               "rearrange_"+args.label: rA.get("rearrange"),
               "rearrange_"+args.label2: rB.get("rearrange")}
        with open(os.path.join(args.out, "compare.json"), "w") as fo:
            json.dump(cmp, fo, indent=2)
        print(f"compare -> {os.path.join(args.out,'compare.json')}")


if __name__ == "__main__":
    main()
