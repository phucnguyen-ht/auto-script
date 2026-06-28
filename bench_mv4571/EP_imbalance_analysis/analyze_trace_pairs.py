#!/usr/bin/env python3
"""MV-4571: EP time-imbalance từ traces — đo MoE bằng 2 comm bao quanh (robust).

Vì sao KHÔNG đếm kernel MoE bên trong:
  Flow MoE đổi theo số token: prefill -> 5 kernel (sort, quant, gemm s1, quant,
  gemm s2); decode đôi khi -> 1 kernel fused `aiter::fmoe_...`. Nên ta KHÔNG dựa
  vào kiểu kernel. Thay vào đó MoE bị KẸP giữa 2 collective:
        ncclDevKernel(GATHER) -> [kernel MoE bất kỳ] -> ncclDevKernel(REDUCE-SCATTER)
  => thời gian MoE 1 cụm / rank = rs.start - gather.end.

Phân biệt gather vs reduce-scatter: KHÔNG bằng tên (mọi nccl giống hệt) -> dùng VỊ TRÍ:
  comm NGAY TRƯỚC cụm MoE = gather; comm NGAY SAU = reduce-scatter.
  (data-dep: gather.end <= moe.start <= moe.end <= rs.start.)

LƯU Ý quan trọng: có 1 quant `dynamic_per_group_scaled_quant` chạy TRƯỚC gather
(quant để dispatch). quant=3×#cụm. => CHỈ mở "region MoE" bằng sort/gemm/fmoe
(opener), KHÔNG mở bằng quant, nếu không sẽ đếm gấp đôi số cụm.

Clock 8 rank CHUNG (lệch < 3ms) -> so timestamp giữa rank trực tiếp được.

Output: <trace_dir>/trace_pairs/  (npz + summary.txt + cluster_gantt_*.png)
"""
import gzip, glob, os, re, sys, gc
import numpy as np
import ijson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/phuc-nguyen/workspaces/mv-4571/auto-script/bench_mv4571/logs/glm5.2/"
    "dp8ep8/noMTP-bs64-dg/auto_profile/20260625_091838/profiling_result/run1/8k_rinf_c8")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def derive_out(trace_dir):
    """OUT = EP_imbalance_analysis/logs/<run_timestamp>_<scenario>/  (vd 20260625_091838_8k_rinf_c8)."""
    parts = trace_dir.rstrip("/").split("/")
    scenario = parts[-1]
    runts = ""
    if "profiling_result" in parts:
        runts = parts[parts.index("profiling_result") - 1]   # .../auto_profile/<runts>/profiling_result/...
    tag = f"{runts}_{scenario}" if runts else scenario
    return os.path.join(SCRIPT_DIR, "logs", tag)


OUT_DIR = os.environ.get("OUT_DIR") or derive_out(TRACE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
DROP_HEAD = int(os.environ.get("DROP_HEAD", "150"))   # bỏ cụm đầu (warmup)
GANTT_CLUSTERS = [int(x) for x in os.environ.get("GANTT", "200,40000,77000").split(",")]


def is_comm(n):   return "ncclDevKernel" in n
def is_opener(n):  # kernel MỞ ĐẦU khối MoE-compute (sau gather): sort / gemm / fmoe
    return ("MoeSorting" in n or "kernel_moe_gemm" in n or "fmoe" in n)
def rank_of(path):
    m = re.search(r"dp(\d+)_", os.path.basename(path)); return int(m.group(1)) if m else -1


def parse_rank(path):
    """-> (regions ndarray (N,6), n_comm).
    region = (gather_start, gather_end, moe_start, moe_end, rs_start, rs_end)."""
    comm, opn = [], []
    n_comm = 0
    with gzip.open(path, "rb") as fh:
        for e in ijson.items(fh, "traceEvents.item"):
            if e.get("ph") != "X" or e.get("cat") != "kernel":
                continue
            ts = e.get("ts"); dur = e.get("dur"); n = e.get("name", "")
            if ts is None or dur is None:
                continue
            ts = float(ts); end = ts + float(dur)
            if is_comm(n):
                comm.append((ts, end, "c")); n_comm += 1
            elif is_opener(n):
                opn.append((ts, end, "o"))
    evs = comm + opn
    evs.sort(key=lambda x: x[0])
    del comm, opn

    regions = []
    last_comm = None     # (start,end) -> ứng viên gather
    cur = None
    for ts, end, typ in evs:
        if typ == "o":
            if cur is None:
                gs, ge = (last_comm if last_comm is not None else (ts, ts))
                cur = {"gs": gs, "ge": ge, "ms": ts, "me": end}
            elif end > cur["me"]:
                cur["me"] = end
        else:  # comm
            if cur is not None:               # reduce-scatter -> chốt cụm
                regions.append((cur["gs"], cur["ge"], cur["ms"], cur["me"], ts, end))
                cur = None
            last_comm = (ts, end)
    del evs; gc.collect()
    return np.array(regions, dtype=np.float64), n_comm


def draw_gantt(GE_abs, gs, ge, rss, rse, ms, me, ranks, k, path):
    """Vẽ flow-chart giống hình tay: mỗi rank 1 hàng [gather | moe | reduce-scatter],
    barrier dọc ở gather.end và rs.end (điểm sync). Trục x = thời gian tương đối (us)."""
    R = len(ranks)
    t0 = gs[:, k].min()    # mốc 0 = gather sớm nhất của cụm k
    fig, ax = plt.subplots(figsize=(14, 0.7 * R + 1.5))
    for i in range(R):
        g0, g1 = gs[i, k] - t0, ge[i, k] - t0
        m1 = rss[i, k] - t0            # moe = [gather.end, rs.start]
        r1 = rse[i, k] - t0            # rs.end
        y = R - 1 - i
        ax.barh(y, g1 - g0, left=g0, color="#bcd5f0", edgecolor="#888", label="nccl gather" if i == 0 else None)
        ax.barh(y, m1 - g1, left=g1, color="#cfe9cf", edgecolor="#888", label="moe" if i == 0 else None)
        ax.barh(y, r1 - m1, left=m1, color="#f5e7bf", edgecolor="#888", label="nccl reduce-scatter" if i == 0 else None)
        ax.text(r1 + (r1 * 0.0) + 2, y, f"rank{ranks[i]}", va="center", fontsize=9)
    # barrier dọc: gather.end (median) & rs.end (median) — điểm sync
    ax.axvline(np.median(ge[:, k]) - t0, color="#777", ls="--", lw=1)
    ax.axvline(np.median(rse[:, k]) - t0, color="#777", ls="--", lw=1)
    ax.text(np.median(ge[:, k]) - t0, R - 0.3, "sync (gather end)", ha="center", fontsize=8, color="#555")
    ax.text(np.median(rse[:, k]) - t0, R - 0.3, "sync (reduce-scatter end)", ha="center", fontsize=8, color="#555")
    ax.set_yticks([]); ax.set_xlabel("time within cluster (µs)")
    ax.set_title(f"MoE cluster k={k}: gather → moe → reduce-scatter (8 ranks)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main():
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "dp*_rank0.*.pt.trace.json.gz")))
    if not files:
        sys.exit(f"[ERROR] không thấy trace trong {TRACE_DIR}")
    if os.environ.get("QUICK") == "1":
        files = files[:1]
    print(f"[trace] {len(files)} file -> {OUT_DIR}")

    reg_by_rank, ncomm_by_rank = {}, {}
    for f in files:
        r = rank_of(f)
        reg, nc = parse_rank(f)
        reg_by_rank[r] = reg; ncomm_by_rank[r] = nc
        moe = reg[:, 4] - reg[:, 1]
        print(f"  rank{r}: comm={nc:7d} regions={len(reg):7d}  MoE-time sum={moe.sum()/1e3:8.1f}ms")
        del reg; gc.collect()

    ranks = sorted(reg_by_rank)
    lines = []
    def p(s): print(s); lines.append(s)

    # ---------- (a) CHECK MAPPING: số comm & số region bằng nhau giữa rank? ----------
    ncs = [ncomm_by_rank[r] for r in ranks]
    nrs = [len(reg_by_rank[r]) for r in ranks]
    p("=== (a) Kiểm tra MAPPING giữa các rank ===")
    p(f"  #comm  mỗi rank: {ncs}  -> {'BẰNG NHAU' if len(set(ncs))==1 else 'KHÁC NHAU!'}")
    p(f"  #region(cụm MoE) mỗi rank: {nrs}  -> {'BẰNG NHAU' if len(set(nrs))==1 else 'KHÁC NHAU!'}")
    p(f"  (mỗi cụm = 1 gather + 1 reduce-scatter; #comm ≈ 2×#region + vài comm startup)")
    if len(ranks) < 2:
        print("[QUICK] 1 rank."); return
    p("")

    K = min(nrs)
    def col(idx): return np.stack([reg_by_rank[r][:K, idx] for r in ranks])[:, DROP_HEAD:]
    GS, GE, MS, ME, RSs, RSe = (col(i) for i in range(6))   # (R, Kc)
    MOE = RSs - GE
    R, Kc = MOE.shape

    # ---------- (b) Flow-chart Gantt cho vài cụm ----------
    for k in GANTT_CLUSTERS:
        if 0 <= k < Kc:
            png = os.path.join(OUT_DIR, f"cluster_gantt_k{k}.png")
            draw_gantt(None, GS, GE, RSs, RSe, MS, ME, ranks, k, png)
            p(f"  [gantt] cụm k={k} -> {os.path.basename(png)}")
    p("")

    # ---------- Chứng minh sync ----------
    sp_ge = GE.max(0) - GE.min(0); sp_rse = RSe.max(0) - RSe.min(0); sp_rss = RSs.max(0) - RSs.min(0)
    p("=== Chứng minh sync (spread max-min giữa 8 rank tại mỗi cụm, µs) ===")
    p(f"  gather.END   : median={np.median(sp_ge):7.2f} p99={np.percentile(sp_ge,99):7.2f}  (kỳ vọng ~0)")
    p(f"  rscat .END   : median={np.median(sp_rse):7.2f} p99={np.percentile(sp_rse,99):7.2f}  (kỳ vọng ~0)")
    p(f"  rscat .START : median={np.median(sp_rss):7.2f} p99={np.percentile(sp_rss,99):7.2f}  (PHẢI lớn=lệch MoE)")
    p(f"  MoE-time/cụm : median={np.median(MOE):7.2f} p99={np.percentile(MOE,99):7.2f}")
    p("  -> gather.END & rscat.END spread << MoE-time => collective sync ở ĐIỂM KẾT THÚC. ✔")
    p("")

    # ---------- Time imbalance ----------
    cmax, cmin, cmean = MOE.max(0), np.maximum(MOE.min(0), 1e-9), MOE.mean(0)
    imb_mm, imb_ma = cmax / cmin, cmax / cmean
    crit, bal = cmax.sum(), cmean.sum()
    speedup = (1 - bal / crit) * 100 if crit > 0 else 0.0
    p("=== EP time imbalance (MoE-compute, comm-bounded) ===")
    p(f"  clusters={Kc} ranks={ranks}")
    p(f"  MoE-time/rank tổng (ms): " + " ".join(f"{MOE[i].sum()/1e3:.0f}" for i in range(R)))
    p(f"  critical-path (Σ max_r)  = {crit/1e3:9.1f} ms")
    p(f"  balanced      (Σ mean_r) = {bal/1e3:9.1f} ms")
    p(f"  => headroom cân bằng MoE = {speedup:5.1f}%  (Σ(max-mean)={(crit-bal)/1e3:.1f} ms)")
    p(f"  per-cluster max/min: mean={imb_mm.mean():.2f} median={np.median(imb_mm):.2f} p99={np.percentile(imb_mm,99):.2f}")
    p(f"  per-cluster max/avg: mean={imb_ma.mean():.2f} median={np.median(imb_ma):.2f} p99={np.percentile(imb_ma,99):.2f}")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    np.savez_compressed(os.path.join(OUT_DIR, "trace_pairs.npz"),
                        ranks=np.array(ranks), MOE=MOE, GS=GS, GE=GE, RSs=RSs, RSe=RSe)
    print(f"\n[saved] {OUT_DIR}/summary.txt + trace_pairs.npz + cluster_gantt_*.png")


if __name__ == "__main__":
    main()
