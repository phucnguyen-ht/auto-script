#!/usr/bin/env python3
# =============================================================================
# MV-4571 — EP imbalance theo TIME (headless), từ 8 torch-profiler traces.
# Port process_trace_time_glm5.ipynb thành script 1 phát: parse comm-bounded
# (MoE-time = reduce_scatter.start − gather.end), verify, rồi LƯU mọi
# visualization ra <out>/ (backend Agg, không show).
#
# Dùng:
#   python3 analyze_time.py --trace-dir <dir 8 file dp*_rank0.*.pt.trace.json.gz> \
#       --out <dir> [--drop-head 0] [--rel-thr 0.2] [--thr-us 50] \
#       [--gantt-max-figs 200] [--no-gantt]
#
# Tương ứng notebook:
#   Bước 0 (CHECK 1..4) -> verify_schema()      (header/kernel/clock/mapping #comm)
#   Bước 1  (parse)     -> parse_all()          (REGION_DT, Clusters, phase)
#   1b                  -> verify_same_kernel()
#   1c                  -> verify_phase()
#   "Bảng theo RANK"    -> per_rank_table()     (MoE_busy/MoE_span/gather/rscat + breakdown.png)
#   STRUCT 3 Metrics    -> Metrics
#   Section 3           -> plot_time_imbalance_hist()  (3 hist max/min: all/prefill/decode)
#   Section 5 Gantt     -> plot_gantt()         (~200 ảnh, mỗi ảnh 3x3 cụm)
#   5b                  -> verify_barrier_rel() (REL_THR, default 0.2)
#   5c                  -> verify_barrier_abs() (THR_US µs, default 50)
#   Export              -> export()
# =============================================================================
import os, glob, gzip, re, json, collections, argparse, shutil, bisect
from dataclasses import dataclass
from enum import Enum
import numpy as np
import ijson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ----------------------------- I/O nền tảng -----------------------------
try:
    import orjson as _json
    def _load_gz(path):
        with gzip.open(path, "rb") as fh:
            return _json.loads(fh.read())
except ImportError:
    def _load_gz(path):
        with gzip.open(path, "rb") as fh:
            return json.load(fh)


def iter_events(path):
    """Duyệt traceEvents[] (orjson nhanh nhất; host đủ RAM, nạp 1 file/lần)."""
    d = _load_gz(path)
    yield from (d["traceEvents"] if isinstance(d, dict) else d)


def rank_of(path):
    return int(re.search(r"dp(\d+)_", os.path.basename(path)).group(1))


def list_files(trace_dir):
    return sorted(glob.glob(os.path.join(trace_dir, "dp*_rank0.*.pt.trace.json.gz")))


# ----------------------------- STRUCT 1: Region -----------------------------
REGION_DT = np.dtype([
    ("gather_start", "f8"), ("gather_end", "f8"),  # gather: bắt đầu / KẾT THÚC (-> MoE bắt đầu)
    ("moe_start",    "f8"), ("moe_end",    "f8"),  # kernel MoE đầu tiên / cuối cùng
    ("rs_start",     "f8"), ("rs_end",     "f8"),  # reduce-scatter: BẮT ĐẦU (-> MoE kết thúc) / kết thúc
    ("nstage",       "i4"), ("sig",        "i8"),  # #kernel matmul (1=fused/2=two-stage); id chữ ký kernel
    ("moe_busy",     "f8"),                        # tổng DUR kernel MoE (= perfetto "MoE")
    ("cg_phase",     "i4"),                         # phase theo cudagraph: 1 prefill / 2 decode / 0 unknown
])

# Phân chia prefill/decode theo dấu CUDA-GRAPH (tổng quát cho cả MTP lẫn non-MTP):
#   cudagraph_mode bật capture cho DECODE (replay) nhưng KHÔNG cho prefill.
#   -> kernel GPU của 1 cụm prefill được launch EAGER: mỗi kernel có 1 flow "ac2g"
#      (mũi tên nâu) nối từ op CPU (hipLaunchKernel) lên -> flow 'f' của nó CÓ 's' khớp id.
#   -> kernel GPU của 1 cụm decode được REPLAY từ graph: KHÔNG có mũi tên CPU riêng
#      -> flow 'f' KHÔNG có 's' khớp id.
#   Nhãn cụm = decode nếu phần lớn arrow trong cửa sổ [gather_start, rs_end] thiếu 's'.
#   (Khác với nstage 1/2-stage: nstage sai cho MTP vì MTP luôn dùng kernel fused -> decode≈0.)
CG_NOARROW_THR = 0.5


class Phase(str, Enum):
    PREFILL = "prefill"   # 1 kernel matmul (fused fmoe) — chunk lớn
    DECODE  = "decode"    # 2 kernel matmul (gemm1 + gemm2) — ít token


def is_comm(n):
    return "ncclDevKernel" in n


def moe_kind(n):
    # --- glm5.2 (AITER MoE) kernels ---
    if "MoeSortingClearWorkspace" in n: return "sortClear"
    if "MoeSortingMultiPhase"     in n: return "sortMP"
    if "MoeSorting"               in n: return "sort"
    if "kernel_moe_gemm" in n: return "gemm1" if "(ck::InMemoryDataOperationEnum)0" in n else "gemm2"
    if "fmoe" in n: return "fmoe"
    # --- kimi2.6 (VLLM_ROCM_USE_AITER_MOE=0: native/triton compressed-tensors MoE) ---
    if "moe_align_block_size" in n: return "sort"          # token->expert sorting (opener)
    if "count_and_sort_expert_tokens" in n: return "sort"
    if "fused_moe_kernel" in n: return "fmoe"              # expert matmul (up/down proj)
    return None


MATMUL = {"gemm1", "gemm2", "fmoe"}
_SIG = {}


def _sig_id(kinds):
    t = tuple(kinds)
    if t not in _SIG:
        _SIG[t] = len(_SIG)
    return _SIG[t]


def parse_rank_pairs(path):
    """1 rank -> np.ndarray[Region] + #comm. opener đầu MỞ cụm (gather=comm liền trước);
    comm kế tiếp ĐÓNG cụm (=reduce-scatter). Đồng thời thu flow "ac2g" để gán cg_phase
    (prefill=eager có arrow / decode=graph replay không arrow)."""
    comm, opn, ncomm = [], [], 0
    s_ids = set(); f_ts = []; f_id = []          # arrow: 's' (CPU launch) / 'f' (GPU kernel)
    for e in iter_events(path):
        cat = e.get("cat"); ph = e.get("ph")
        if cat == "ac2g":                         # flow event (mũi tên nâu CPU->GPU)
            if ph == "s":
                s_ids.add(e.get("id"))
            elif ph == "f":
                t = e.get("ts")
                if t is not None:
                    f_ts.append(float(t)); f_id.append(e.get("id"))
            continue
        if ph != "X" or cat != "kernel":
            continue
        ts = e.get("ts"); dur = e.get("dur"); n = e.get("name", "")
        if ts is None or dur is None:
            continue
        ts = float(ts); end = ts + float(dur)
        if is_comm(n):
            comm.append((ts, end, "c", None)); ncomm += 1
        else:
            k = moe_kind(n)
            if k is not None:
                opn.append((ts, end, "o", k))
    evs = sorted(comm + opn, key=lambda x: x[0])

    # arrow lookup: f_ts sorted + has_s (flow 'f' có 's' khớp id hay không), vectorized
    if f_ts:
        fts = np.asarray(f_ts); fid = np.asarray(f_id)
        order = np.argsort(fts); fts = fts[order]
        s_arr = np.fromiter(s_ids, dtype=fid.dtype, count=len(s_ids)) if s_ids \
            else np.empty(0, dtype=fid.dtype)
        has_s = np.isin(fid[order], s_arr)
    else:
        fts = np.empty(0); has_s = np.empty(0, bool)

    def cg_label(gs, rse):
        a = bisect.bisect_left(fts, gs); b = bisect.bisect_right(fts, rse)
        if b <= a:
            return 0                              # không arrow trong cửa sổ -> unknown
        noarrow = 1.0 - has_s[a:b].mean()
        return 2 if noarrow >= CG_NOARROW_THR else 1

    regs, last, cur = [], None, None
    for ts, end, typ, k in evs:
        if typ == "o":
            if cur is None:
                if last is None:
                    raise Exception("No previous communication event found for the first opener")
                gs, ge = last
                cur = {"gather_start": gs, "gather_end": ge, "moe_start": ts, "moe_end": end,
                       "kinds": [k], "busy": end - ts}
            else:
                if end > cur["moe_end"]:
                    cur["moe_end"] = end
                cur["kinds"].append(k); cur["busy"] += end - ts
        else:
            if cur is not None:
                regs.append((cur["gather_start"], cur["gather_end"], cur["moe_start"], cur["moe_end"],
                             ts, end, int(sum(x in MATMUL for x in cur["kinds"])),
                             _sig_id(cur["kinds"]), cur["busy"],
                             cg_label(cur["gather_start"], end)))
                cur = None
            last = (ts, end)
    return np.array(regs, dtype=REGION_DT), ncomm


# ----------------------------- STRUCT 2: Clusters -----------------------------
@dataclass
class Clusters:
    ranks:        np.ndarray
    gather_start: np.ndarray
    gather_end:   np.ndarray
    rs_start:     np.ndarray
    rs_end:       np.ndarray
    nstage:       np.ndarray
    sig:          np.ndarray
    moe_busy:     np.ndarray
    cg_phase:     np.ndarray = None
    @property
    def moe(self):        return self.rs_start - self.gather_end
    @property
    def gather_dur(self): return self.gather_end - self.gather_start
    @property
    def rs_dur(self):     return self.rs_end - self.rs_start
    @property
    def shape(self):      return self.moe.shape


@dataclass
class Metrics:
    cmax:       np.ndarray
    cmin:       np.ndarray
    cmean:      np.ndarray
    imb_maxmin: np.ndarray
    imb_maxavg: np.ndarray
    comm:       np.ndarray


def parse_all(files, drop_head, out):
    """Parse 8 rank -> Clusters C, phase masks. Lưu trace_pairs.npz. Trả ncomm/rank để verify."""
    reg, ncomm = {}, {}
    for f in files:
        r = rank_of(f)
        reg[r], ncomm[r] = parse_rank_pairs(f)
        print(f"  rank{r}: regions={len(reg[r]):7d}  comm={ncomm[r]}")
    ranks = sorted(reg)
    K = min(len(reg[r]) for r in ranks)

    def field(name):
        return np.stack([reg[r][name][:K] for r in ranks])[:, drop_head:]

    C = Clusters(ranks=np.array(ranks),
                 gather_start=field("gather_start"), gather_end=field("gather_end"),
                 rs_start=field("rs_start"), rs_end=field("rs_end"),
                 nstage=field("nstage"), sig=field("sig"), moe_busy=field("moe_busy"),
                 cg_phase=field("cg_phase"))
    np.savez_compressed(os.path.join(out, "trace_pairs.npz"), ranks=C.ranks, MOE=C.moe,
                        GS=C.gather_start, GE=C.gather_end, RSs=C.rs_start, RSe=C.rs_end,
                        NSTAGE=C.nstage, SIG=C.sig, MOE_BUSY=C.moe_busy, CG_PHASE=C.cg_phase)
    print(f"ranks={list(C.ranks)}  R={C.shape[0]}  clusters K={C.shape[1]}  "
          f"MoE-time median={np.median(C.moe):.1f}us")
    return C, ncomm


def phase_masks(C, method):
    """Trả (phase[str], pre, dec) theo method: 'kernel' (nstage 1/2) hoặc 'cudagraph'
    (majority arrow). 'auto' -> cudagraph nếu kernel suy biến (decode hoặc prefill = 0)."""
    nst0 = C.nstage[0]
    pre_k = nst0 == 1
    # cudagraph: majority vote 8 rank (decode nếu >=1/2 rank gán decode); bỏ unknown(0)
    cg = C.cg_phase
    dec_votes = (cg == 2).sum(0); pre_votes = (cg == 1).sum(0)
    dec_cg = dec_votes >= pre_votes                     # hòa -> decode (an toàn cho MTP)
    if method == "auto":
        method = "kernel" if (pre_k.any() and (~pre_k).any()) else "cudagraph"
    if method == "kernel":
        pre = pre_k
    else:
        pre = ~dec_cg
    dec = ~pre
    phase = np.where(pre, Phase.PREFILL.value, Phase.DECODE.value)
    print(f"phase method = {method}: "
          f"prefill={int(pre.sum())} decode={int(dec.sum())}")
    return method, phase, pre, dec


# ======================= Bước 0 — VERIFY schema =======================
def verify_schema(files, ncomm):
    """CHECK 1..4 gộp: header, phân loại kernel rank0, clock chung, mapping #comm.
    (#comm lấy từ parse_all để khỏi đọc lại 8 file lần nữa.)"""
    print("\n===== Bước 0 — VERIFY schema trace =====")
    # CHECK-1: header
    with gzip.open(files[0], "rb") as fh:
        print("[CHECK-1] header (900B):", fh.read(900).decode("utf-8", "ignore")[:300], "...")

    # CHECK-2: phân loại kernel rank0
    def cls(n):
        if "ncclDevKernel" in n: return "comm"
        if "MoeSorting" in n: return "sort"
        if "dynamic_per_group_scaled_quant" in n: return "quant"
        if "kernel_moe_gemm" in n: return "gemm1" if "(ck::InMemoryDataOperationEnum)0" in n else "gemm2"
        if "fmoe" in n: return "fmoe"
        return None
    cat = collections.Counter(); ph = collections.Counter(); kc = collections.Counter()
    gemm_dur = None
    for e in iter_events(files[0]):
        cat[e.get("cat")] += 1; ph[e.get("ph")] += 1
        if e.get("cat") == "kernel":
            c = cls(e.get("name", ""))
            if c:
                kc[c] += 1
            if c == "gemm1" and gemm_dur is None:
                gemm_dur = float(e["dur"])
    print(f"[CHECK-2] kernel MoE classes: {dict(kc)}")
    print(f"          gemm1+gemm2+2*fmoe = {kc.get('gemm1',0)+kc.get('gemm2',0)+2*kc.get('fmoe',0)} "
          f"vs comm = {kc.get('comm')}  (mỗi cụm = 1 gather + 1 reduce-scatter)")
    print(f"          gemm stage1 dur = {gemm_dur} us -> xác nhận ts/dur là µs")

    # CHECK-3: clock chung rank0 vs rank1
    def ts_range(path):
        lo, hi = float("inf"), -1.0
        for e in iter_events(path):
            if e.get("cat") != "kernel":
                continue
            t = float(e["ts"]); lo = min(lo, t); hi = max(hi, t)
        return lo, hi
    if len(files) >= 2:
        a = ts_range(files[0]); b = ts_range(files[1])
        print(f"[CHECK-3] rank0 ts={a}  rank1 ts={b}  offset(min) ms={(b[0]-a[0])/1e3:.3f}  "
              f"overlap={not (a[1] < b[0] or b[1] < a[0])}  -> clock CHUNG")

    # CHECK-4: mapping #comm bằng nhau (dùng ncomm từ parse)
    vals = set(ncomm.values())
    for r in sorted(ncomm):
        print(f"[CHECK-4] rank{r}: #comm={ncomm[r]}")
    print("[CHECK-4] =>", "BẰNG NHAU (map 1-1 giữa rank)" if len(vals) == 1 else f"KHÁC NHAU! {vals}")


# ======================= 1b — cùng kernel MoE =======================
def verify_same_kernel(C):
    K = C.shape[1]
    sig_same = (C.sig == C.sig[0]).all(axis=0)
    nst_same = (C.nstage == C.nstage[0]).all(axis=0)
    print("\n===== 1b — mỗi cụm 8 rank gọi CÙNG kernel MoE =====")
    print(f"clusters K = {K}")
    print(f"  sig    giống cả 8 rank: {sig_same.sum()}/{K}  ({100*sig_same.mean():.2f}%)")
    print(f"  nstage giống cả 8 rank: {nst_same.sum()}/{K}  ({100*nst_same.mean():.2f}%)")
    id2sig = {v: k for k, v in _SIG.items()} if _SIG else {}
    print("Top chữ ký (sig id -> chuỗi kind kernel : số cụm):")
    for sid, cnt in collections.Counter(C.sig[0].tolist()).most_common(6):
        print(f"  id={sid}  x{cnt:6d}  {id2sig.get(sid, '?')}")
    print("OK: mọi cụm 8 rank cùng kernel." if sig_same.all()
          else "Có cụm khác chữ ký: " + str(np.where(~sig_same)[0][:5].tolist()))
    return bool(sig_same.all())


# ======================= 1c — kiểm tra phase (theo method ĐÃ CHỌN) =======================
def verify_phase(C, pre, dec, method="cudagraph"):
    """Sanity-check trên mask phase ĐÃ CHỌN (pre/dec từ phase_masks, mặc định cudagraph).
    KHÔNG phải phân loại theo kernel — nhãn cũ '1-kernel/2-kernel' đã bỏ vì nstage là
    lựa chọn KERNEL của AITER (fused vs 2-stage gemm), không phải tín hiệu prefill/decode."""
    mt = C.moe.mean(0)
    med_pre = float(np.median(mt[pre])) if pre.any() else float("nan")
    med_dec = float(np.median(mt[dec])) if dec.any() else float("nan")
    c3 = med_pre > med_dec
    print(f"\n===== 1c — kiểm tra phase (method = {method}) =====")
    print("phase counts:", {"prefill": int(pre.sum()), "decode": int(dec.sum())})
    print(f"  prefill: MoE-time median = {med_pre:8.1f} us")
    print(f"  decode : MoE-time median = {med_dec:8.1f} us")
    print(f"  sanity time(prefill) > time(decode): {'OK' if c3 else 'FAIL/n.a.'}  "
          f"({med_pre:.0f} vs {med_dec:.0f} us)")
    return c3


# ======================= 1d — verify cudagraph vs kernel =======================
def cg_masks(C):
    """Majority-vote phase theo cudagraph arrow: (cg_dec bool, valid bool)."""
    cg = C.cg_phase
    dec_votes = (cg == 2).sum(0); pre_votes = (cg == 1).sum(0)
    cg_dec = dec_votes >= pre_votes            # hòa/không-arrow -> decode (an toàn MTP)
    valid = (cg != 0).any(0)                   # >=1 rank có thông tin arrow
    return cg_dec, valid


def verify_phase_method(C):
    """So khớp 2 cách chia prefill/decode: cudagraph(arrow) vs kernel(nstage 1/2).
    Báo match% + confusion. kernel sai cho MTP (decode≈0); cudagraph tổng quát."""
    nst0 = C.nstage[0]; kpre = nst0 == 1; kdec = ~kpre
    cg_dec, valid = cg_masks(C); cg_pre = ~cg_dec
    agree = (cg_dec == kdec) & valid
    print("\n===== 1d — VERIFY phase: cudagraph(arrow) vs kernel(nstage) =====")
    print(f"  kernel   : prefill={int(kpre.sum())} decode={int(kdec.sum())}")
    print(f"  cudagraph: prefill={int(cg_pre.sum())} decode={int(cg_dec.sum())} "
          f"(unknown cụm bỏ qua: {int((~valid).sum())})")
    print(f"  MATCH = {int(agree.sum())}/{int(valid.sum())} = "
          f"{100*agree.sum()/max(1,int(valid.sum())):.3f}%")
    print(f"   cg=prefill & kernel=prefill: {int((cg_pre & kpre & valid).sum())}")
    print(f"   cg=prefill & kernel=decode : {int((cg_pre & kdec & valid).sum())}")
    print(f"   cg=decode  & kernel=prefill: {int((cg_dec & kpre & valid).sum())}")
    print(f"   cg=decode  & kernel=decode : {int((cg_dec & kdec & valid).sum())}")
    if int(kdec.sum()) == 0:
        print("  [NOTE] kernel cho decode=0 (đặc trưng MTP: luôn dùng kernel fused) "
              "-> cudagraph mới đúng.")
    return round(100 * float(agree.sum()) / max(1, int(valid.sum())), 3)


# ======================= Bảng theo RANK (ms) =======================
def per_rank_table(C, out):
    R = C.shape[0]
    print("\n===== Bảng theo RANK (ms) =====")
    per_rank = []
    for i in range(R):
        g = float(C.gather_dur[i].sum()) / 1e3
        rs = float(C.rs_dur[i].sum()) / 1e3
        mb = float(C.moe_busy[i].sum()) / 1e3
        msp = float(C.moe[i].sum()) / 1e3
        per_rank.append((int(C.ranks[i]), mb, msp, g, rs, g + rs, mb + g + rs))
    print(f"{'rank':>4} {'MoE_busy':>9} {'MoE_span':>9} {'gather':>9} {'rscat':>9} {'comm':>9} {'total':>10}   (ms)")
    for r, mb, msp, g, rs, ct, tot in per_rank:
        print(f"{r:>4} {mb:9.1f} {msp:9.1f} {g:9.1f} {rs:9.1f} {ct:9.1f} {tot:10.1f}")
    A = np.array([[mb, msp, g, rs, ct, tot] for _, mb, msp, g, rs, ct, tot in per_rank])
    print(f"{'mean':>4} " + " ".join(f"{v:9.1f}" for v in A.mean(0)))
    print(f"{'max':>4} " + " ".join(f"{v:9.1f}" for v in A.max(0)))

    labels = [str(r) for r, *_ in per_rank]
    mm, gg, ss = A[:, 0], A[:, 2], A[:, 3]
    plt.figure(figsize=(12, 4))
    plt.bar(labels, gg, label="comm gather", color="#bcd5f0")
    plt.bar(labels, ss, bottom=gg, label="comm reduce-scatter", color="#f5e7bf")
    plt.bar(labels, mm, bottom=gg + ss, label="MoE compute (kernel-sum)", color="#cfe9cf")
    plt.ylabel("ms"); plt.xlabel("rank"); plt.legend()
    plt.title("Per-rank time breakdown: comm (gather+reduce-scatter) + MoE kernel-sum")
    plt.tight_layout(); plt.savefig(os.path.join(out, "per_rank_time_breakdown.png"), dpi=130); plt.close()
    print("  saved per_rank_time_breakdown.png")


# ======================= STRUCT 3 — Metrics =======================
def compute_metrics(C):
    moe = C.moe
    MET = Metrics(
        cmax=moe.max(0), cmin=np.maximum(moe.min(0), 1e-9), cmean=moe.mean(0),
        imb_maxmin=moe.max(0) / np.maximum(moe.min(0), 1e-9),
        imb_maxavg=moe.max(0) / moe.mean(0),
        comm=C.gather_dur.mean(0) + C.rs_dur.mean(0),
    )
    print(f"\nMoE-time/cụm (us): median={np.median(moe):.1f} p99={np.percentile(moe,99):.1f}")
    print(f"max/min: mean={MET.imb_maxmin.mean():.2f} median={np.median(MET.imb_maxmin):.2f} "
          f"p99={np.percentile(MET.imb_maxmin,99):.2f}")
    return MET


# ======================= Section 3 — hist max/min (TIME) =======================
def plot_time_imbalance_hist(MET, K, pre, dec, out):
    for name, mask, color in [("all", np.ones(K, bool), "skyblue"),
                              (Phase.PREFILL.value, pre, "#1f77b4"),
                              (Phase.DECODE.value, dec, "#ff7f0e")]:
        vals = MET.imb_maxmin[mask]
        if len(vals) == 0:
            print(f"{name}: (none)"); continue
        xmax = max(2.0, float(np.percentile(vals, 99)))
        n_over = int((vals > xmax).sum())          # số mẫu ĐUÔI (>xmax) BỊ ẨN (không dồn bin cuối)
        plt.figure(figsize=(15, 4))
        plt.hist(vals, bins=60, range=(1, xmax), color=color, edgecolor="black")
        plt.xlim(1, xmax)
        plt.title(f"{name}: MoE-compute TIME imbalance (max/min over ranks) — n={len(vals)}, "
                  f"mean={vals.mean():.2f}, median={np.median(vals):.2f}, p99={np.percentile(vals,99):.2f}  "
                  f"[{n_over} mẫu > {xmax:.2f} bị ẩn]")
        plt.xlabel("Max Gap (max GPU time / min GPU time)"); plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=.5)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"time_imbalance_hist_{name}.png"), dpi=130); plt.close()
    print("  saved time_imbalance_hist_{all,prefill,decode}.png")


# ======================= Section 5 — Gantt (~200 ảnh, 3x3 cụm) =======================
def plot_gantt(C, phase, out, cols=3, rows=3, max_figs=200):
    """Mỗi figure = cols x rows cụm; mỗi cụm = 8 rank [gather|moe|reduce-scatter] + 2 đường barrier.
    Lấy mẫu cụm ĐỀU trên toàn run để ra ~max_figs ảnh. Lưu vào out/gantt/ (xoá folder cũ)."""
    R, K = C.shape
    per_fig = cols * rows
    n_clusters = min(K, per_fig * max_figs)
    idxs = list(range(K)) if K <= n_clusters else \
        sorted(set(np.linspace(0, K - 1, n_clusters).astype(int).tolist()))
    n_fig = (len(idxs) + per_fig - 1) // per_fig

    gantt_dir = os.path.join(out, "gantt")
    if os.path.isdir(gantt_dir):
        shutil.rmtree(gantt_dir)
    os.makedirs(gantt_dir, exist_ok=True)
    print(f"\n===== Gantt: {len(idxs)} cụm (mẫu đều) -> {n_fig} ảnh ({cols}x{rows}) -> {gantt_dir} =====")

    leg = [Patch(facecolor="#bcd5f0", edgecolor="#888", label="nccl gather"),
           Patch(facecolor="#cfe9cf", edgecolor="#888", label="moe"),
           Patch(facecolor="#f5e7bf", edgecolor="#888", label="nccl reduce-scatter")]

    def draw(ax, k):
        t0 = C.gather_start[:, k].min()
        for i in range(R):
            g0, g1 = C.gather_start[i, k] - t0, C.gather_end[i, k] - t0
            m1, r1 = C.rs_start[i, k] - t0, C.rs_end[i, k] - t0
            y = R - 1 - i
            ax.barh(y, g1 - g0, left=g0, color="#bcd5f0", edgecolor="#888")
            ax.barh(y, m1 - g1, left=g1, color="#cfe9cf", edgecolor="#888")
            ax.barh(y, r1 - m1, left=m1, color="#f5e7bf", edgecolor="#888")
            ax.text(r1, y, f" r{C.ranks[i]}", va="center", fontsize=7)
        ax.axvline(np.median(C.gather_end[:, k]) - t0, color="#777", ls="--", lw=.8)
        ax.axvline(np.median(C.rs_end[:, k]) - t0, color="#777", ls="--", lw=.8)
        ax.set_yticks([]); ax.set_xlabel("time within cluster (us)", fontsize=8)
        ax.set_title(f"cluster k={k} ({phase[k]})", fontsize=10)

    for fi in range(n_fig):
        batch = idxs[fi * per_fig:(fi + 1) * per_fig]
        fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, (0.45 * R + 1.3) * rows), squeeze=False)
        axes = axes.ravel()
        for j, k in enumerate(batch):
            draw(axes[j], k)
        for j in range(len(batch), len(axes)):
            axes[j].axis("off")
        fig.legend(handles=leg, loc="upper center", ncol=3, fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(gantt_dir, f"gantt_{batch[0]:06d}_{batch[-1]:06d}.png"), dpi=120)
        plt.close(fig)
    print(f"  saved {n_fig} gantt ảnh.")


def _to_ranges(a):
    a = sorted(int(x) for x in a); o = []; s = pp = a[0]
    for x in a[1:]:
        if x == pp + 1:
            pp = x
        else:
            o.append((s, pp)); s = pp = x
    o.append((s, pp)); return o


# ======================= 5b — barrier TƯƠNG ĐỐI (rel) =======================
def verify_barrier_rel(C, out, rel_thr=0.2):
    K = C.shape[1]
    ge_dev = np.abs(C.gather_end - C.gather_end.mean(0)).max(0)
    rs_dev = np.abs(C.rs_end - C.rs_end.mean(0)).max(0)
    ge_rel = ge_dev / np.maximum(C.gather_dur.min(0), 1e-9)
    rs_rel = rs_dev / np.maximum(C.rs_dur.min(0), 1e-9)
    rel = np.maximum(ge_rel, rs_rel)
    print(f"\n===== 5b — Barrier sync TƯƠNG ĐỐI (REL_THR={rel_thr:.0%}) =====")
    print(f"ge_rel: median={np.median(ge_rel):.3f} p99={np.percentile(ge_rel,99):.2f} max={ge_rel.max():.1f}")
    print(f"rs_rel: median={np.median(rs_rel):.3f} p99={np.percentile(rs_rel,99):.2f} max={rs_rel.max():.1f}")
    print("sweep ngưỡng -> %cụm lỏng:",
          {f"{t:.0%}": f"{100*(rel>t).mean():.1f}%" for t in (0.05, 0.10, 0.20, 0.50, 1.00)})
    loose = np.where(rel > rel_thr)[0]
    print(f"#cụm barrier LỎNG (rel>{rel_thr:.0%}): {len(loose)}/{K}  ({100*len(loose)/K:.2f}%)  "
          f"| index>100: {(loose>100).sum()}")
    if len(loose):
        worst = loose[np.argsort(-rel[loose])][:10]
        print("  10 cụm rel cao nhất (idx | rel | ge_dev_µs | min_ge_dur_µs):")
        for k in worst:
            print(f"    k={int(k):6d}  rel={rel[k]:7.2f}  ge_dev={ge_dev[k]:8.0f}  "
                  f"min_ge_dur={C.gather_dur[:, k].min():8.0f}")
        print(f"  danh sách {len(loose)} cluster index lỏng: {loose.tolist()}")
        print("  -> dải liên tục (>=3 cụm): range | #cụm | ~step (75 layer/step):")
        for s, e in _to_ranges(loose):
            if e - s >= 2:
                print(f"     {s:6d}-{e:6d} ({e - s + 1:4d})  ~step {s // 75}-{e // 75}")
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(rel, lw=.4, color="#9467bd")
    ax.axhline(rel_thr, color="r", ls="--", lw=1, label=f"threshold {rel_thr:.0%}")
    ax.set_yscale("log"); ax.set_xlabel("cluster index"); ax.set_ylabel("barrier dev / min(dur) (log)")
    ax.set_title("Barrier looseness TƯƠNG ĐỐI theo cluster (max của ge_rel, rs_rel)")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "barrier_dev.png"), dpi=130); plt.close()
    return len(loose)


# ======================= 5c — barrier TUYỆT ĐỐI (µs) =======================
def verify_barrier_abs(C, out, thr_us=50.0):
    K = C.shape[1]
    ge_dev = np.abs(C.gather_end - C.gather_end.mean(0)).max(0)
    rs_dev = np.abs(C.rs_end - C.rs_end.mean(0)).max(0)
    barr_dev = np.maximum(ge_dev, rs_dev)
    moe = C.moe.mean(0)
    print(f"\n===== 5c — (cách CŨ) Barrier TUYỆT ĐỐI (THR={thr_us:.0f}µs) =====")
    print(f"barrier lệch-max vs mean (µs): median={np.median(barr_dev):.1f} "
          f"p99={np.percentile(barr_dev,99):.1f} max={barr_dev.max():.0f}")
    loose_abs = np.where(barr_dev > thr_us)[0]
    print(f"#cụm barrier LỎNG (lệch>{thr_us:.0f}µs): {len(loose_abs)}/{K}  ({100*len(loose_abs)/K:.2f}%)  "
          f"| index>100: {(loose_abs>100).sum()}")
    if len(loose_abs):
        worst = loose_abs[np.argsort(-barr_dev[loose_abs])][:10]
        print("  10 cụm lệch nhất (idx | dev_µs | moe_µs):")
        for k in worst:
            print(f"    k={int(k):6d}  dev={barr_dev[k]:8.0f}  moe={moe[k]:8.0f}")
        print(f"  danh sách {len(loose_abs)} cluster index lỏng: {loose_abs.tolist()}")
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(barr_dev, lw=.4, color="#8c564b")
    ax.axhline(thr_us, color="r", ls="--", lw=1, label=f"threshold {thr_us:.0f}µs")
    ax.set_yscale("log"); ax.set_xlabel("cluster index"); ax.set_ylabel("barrier dev (µs, log)")
    ax.set_title("(cách cũ) Barrier looseness TUYỆT ĐỐI theo cluster")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "barrier_dev_abs.png"), dpi=130); plt.close()
    return len(loose_abs)


# ======================= Export =======================
def export(C, MET, pre, dec, out, phase_method="cudagraph", phase_match_pct=None):
    K = C.shape[1]
    nst0 = C.nstage[0]; cg_dec, _ = cg_masks(C)

    def block(mask):
        v = MET.imb_maxmin[mask]
        return {"n": int(mask.sum()),
                "moe_time_median_us": round(float(np.median(C.moe[:, mask])), 1) if mask.any() else None,
                "maxmin_mean": round(float(v.mean()), 3) if len(v) else None,
                "maxmin_p99": round(float(np.percentile(v, 99)), 3) if len(v) else None}
    summary = {
        "out": out, "ranks": [int(x) for x in C.ranks], "clusters": int(K),
        "phase_method": phase_method,
        "phase_match_cudagraph_vs_kernel_pct": phase_match_pct,
        "phase": {"prefill": int(pre.sum()), "decode": int(dec.sum())},
        "phase_by_kernel": {"prefill(1-kernel/fmoe)": int((nst0 == 1).sum()),
                            "decode(2-kernel/gemm)": int((nst0 != 1).sum())},
        "phase_by_cudagraph": {"prefill": int((~cg_dec).sum()), "decode": int(cg_dec.sum())},
        "verify_same_kernel_pct": round(float((C.sig == C.sig[0]).all(0).mean()) * 100, 2),
        "moe_time_per_rank_ms": [round(float(x), 1) for x in (C.moe.sum(1) / 1e3)],
        "gather_time_per_rank_ms": [round(float(x), 1) for x in (C.gather_dur.sum(1) / 1e3)],
        "rscat_time_per_rank_ms": [round(float(x), 1) for x in (C.rs_dur.sum(1) / 1e3)],
        "critical_path_ms": round(float(MET.cmax.sum() / 1e3), 1),
        "balanced_ms": round(float(MET.cmean.sum() / 1e3), 1),
        "headroom_pct": round(float((1 - MET.cmean.sum() / MET.cmax.sum()) * 100), 2),
        "comm_ms": round(float(MET.comm.sum() / 1e3), 1),
        "all": block(np.ones(K, bool)), "prefill": block(pre), "decode": block(dec),
    }
    with open(os.path.join(out, "summary_time.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print("saved ->", out)
    return summary


def main():
    ap = argparse.ArgumentParser(description="MV-4571 EP imbalance theo TIME (headless).")
    ap.add_argument("--trace-dir", required=True, help="thư mục chứa 8 file dp*_rank0.*.pt.trace.json.gz")
    ap.add_argument("--out", required=True, help="thư mục lưu kết quả.")
    ap.add_argument("--drop-head", type=int, default=0, help="bỏ N cụm đầu (warmup).")
    ap.add_argument("--rel-thr", type=float, default=0.2, help="ngưỡng barrier TƯƠNG ĐỐI (5b).")
    ap.add_argument("--thr-us", type=float, default=50.0, help="ngưỡng barrier TUYỆT ĐỐI µs (5c).")
    ap.add_argument("--gantt-max-figs", type=int, default=200, help="số ảnh gantt tối đa (~200).")
    ap.add_argument("--no-gantt", action="store_true", help="bỏ qua vẽ gantt (nhanh).")
    ap.add_argument("--no-verify-schema", action="store_true", help="bỏ Bước-0 verify schema (nhanh).")
    ap.add_argument("--phase-method", choices=["cudagraph", "kernel", "auto"], default="cudagraph",
                    help="chia prefill/decode: cudagraph(arrow, tổng quát MTP+nonMTP, mặc định) | "
                         "kernel(nstage 1/2, sai cho MTP) | auto(cudagraph nếu kernel suy biến).")
    args = ap.parse_args()

    files = list_files(args.trace_dir)
    assert files, f"không thấy file trace dp*_rank0.*.pt.trace.json.gz trong {args.trace_dir}"
    os.makedirs(args.out, exist_ok=True)
    print(f"TRACE_DIR = {args.trace_dir}\nOUT       = {args.out}\n{len(files)} files")

    C, ncomm = parse_all(files, args.drop_head, args.out)                    # Bước 1
    if not args.no_verify_schema:
        verify_schema(files, ncomm)                                          # Bước 0
    verify_same_kernel(C)                                                    # 1b
    match_pct = verify_phase_method(C)                                       # 1d (cudagraph vs kernel)
    method, phase, pre, dec = phase_masks(C, args.phase_method)              # chọn phase
    verify_phase(C, pre, dec, method)                                        # 1c (kiểm tra masks đã chọn)
    per_rank_table(C, args.out)                                              # Bảng theo RANK
    MET = compute_metrics(C)                                                 # STRUCT 3
    plot_time_imbalance_hist(MET, C.shape[1], pre, dec, args.out)            # Section 3
    if not args.no_gantt:
        plot_gantt(C, phase, args.out, max_figs=args.gantt_max_figs)         # Section 5
    verify_barrier_rel(C, args.out, rel_thr=args.rel_thr)                    # 5b
    verify_barrier_abs(C, args.out, thr_us=args.thr_us)                      # 5c
    export(C, MET, pre, dec, args.out, phase_method=method, phase_match_pct=match_pct)  # Export


if __name__ == "__main__":
    main()
