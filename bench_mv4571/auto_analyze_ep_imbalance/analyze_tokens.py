#!/usr/bin/env python3
# =============================================================================
# MV-4571 — EP imbalance theo #TOKEN (headless).
# Port các cell CELL 3/4/5/12/13/14 của process_ep_logs_glm5_log8k.ipynb thành
# script chạy 1 phát: đọc serve.log (có dòng [EP_COLLECT] do VLLM_MOREH_EP_LOG=1),
# cộng histogram qua 8 DP-rank, tính imbalance per (layer,step), rồi LƯU mọi
# visualization ra <out>/ (không show — backend Agg).
#
# Dùng:
#   python3 analyze_tokens.py --log <serve.log|glob> --out <dir> \
#       --concurrency 8 [--decode-max-tokens 4xconc|auto|<số>] [--layers all|3,8,40]
#   (mặc định ngưỡng chia phase = 4*concurrency)
#
# Tương ứng notebook:
#   CELL 3  -> parse_and_aggregate()           (parse + aggregate qua rank)
#   CELL 4  -> build_steps_df()                (histogram global -> tải/rank + imbalance)
#   CELL 5  -> plot_imbalance_hist()           (3 hist max/min: all/prefill/decode)
#   CELL 12 -> plot_load_distribution()        (per-expert/per-rank + heatmap layer x {expert,rank})
#   CELL 13 -> plot_per_layer_load()           (per-rank + per-expert bar cho MỖI layer)
#   CELL 14 -> plot_per_layer_imbalance()      (bar mean max/min per layer + decode heatmap)
#   CELL 16 -> export()                        (parquet/npy/json)
# =============================================================================
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # headless: chỉ lưu file, không show
import matplotlib.pyplot as plt

# ----------------------------- CONFIG mặc định -----------------------------
STARTUP_MARKER = "Application startup complete."   # chỉ collect dòng SAU marker này
USE_LAST_MARKER = True                              # lấy lần xuất hiện CUỐI (mọi engine đã ready)

# --- Regex bóc trường 1 dòng [EP_COLLECT] (xem CELL 3) ---
HEADER_RE = re.compile(
    r"Worker_DP(\d+)_EP(\d+).*?\[EP_COLLECT\] layer=(\S+) it=(\d+) ntok=(\d+) E=(\d+) counts=\["
)
LAYER_IDX_RE = re.compile(r"layers\.(\d+)")        # "model.layers.5.mlp" -> 5


# ======================= CELL 3 — PARSE + AGGREGATE =======================
def find_cut_line(path, marker, use_last=True):
    """Số thứ tự dòng của marker startup; chỉ collect các dòng SAU dòng này."""
    cut = -1
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if marker in line:
                cut = i
                if not use_last:
                    break
    return cut


def parse_and_aggregate(path, marker=STARTUP_MARKER, use_last=USE_LAST_MARKER):
    """Đọc serve.log 2 lần (streaming, ít RAM):
      PASS A: min(it) mỗi (rank,layer) -> chuẩn hoá step = it - min_it.
      PASS B: cộng counts mọi rank cho mỗi (layer,step) -> histogram global g[E]."""
    cut = find_cut_line(path, marker, use_last)
    print(f"startup '{marker}' at line {cut} (collect strictly after this)")

    # ---- PASS A: min(it) cho mỗi (rank, layer) ----
    min_it = {}
    ranks_seen, layers_seen, E_seen = set(), set(), set()
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i <= cut:
                continue
            m = HEADER_RE.search(line)
            if not m:
                continue
            ep = int(m.group(2))                   # EP rank (TP=1 => DP==EP)
            lm = LAYER_IDX_RE.search(m.group(3))
            if not lm:
                continue
            it = int(m.group(4)); E_seen.add(int(m.group(6)))
            lidx = int(lm.group(1))
            ranks_seen.add(ep); layers_seen.add(lidx)
            k = (ep, lidx)
            if k not in min_it or it < min_it[k]:
                min_it[k] = it

    R = len(ranks_seen)
    E = max(E_seen) if E_seen else 0
    assert R > 0 and E > 0, "No [EP_COLLECT] lines found after startup marker"
    print(f"ranks(R)={R} ranks={sorted(ranks_seen)}  experts(E)={E}  moe_layers={len(layers_seen)}")

    # ---- PASS B: cộng counts mọi rank cho mỗi (layer, step) ----
    agg, agg_ranks, agg_ntok, topk_samples = {}, {}, {}, []
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i <= cut:
                continue
            m = HEADER_RE.search(line)
            if not m:
                continue
            ep = int(m.group(2)); it = int(m.group(4)); ntok = int(m.group(5))
            lm = LAYER_IDX_RE.search(m.group(3))
            if not lm:
                continue
            lidx = int(lm.group(1))
            start = m.end(); rb = line.find("]", start)
            if rb == -1:
                continue
            counts = np.array(line[start:rb].split(","), dtype=np.int64)
            if counts.shape[0] != E:
                continue
            step = it - min_it[(ep, lidx)]
            key = (lidx, step)
            if key not in agg:
                agg[key] = counts.copy(); agg_ranks[key] = {ep}; agg_ntok[key] = ntok
            else:
                agg[key] += counts; agg_ranks[key].add(ep); agg_ntok[key] += ntok
            if ntok > 0:
                topk_samples.append(counts.sum() / ntok)

    topk = int(round(np.median(topk_samples))) if topk_samples else 0
    return agg, agg_ranks, agg_ntok, R, E, topk, sorted(layers_seen)


# ======================= CELL 4 — TẢI/RANK + IMBALANCE =======================
def rank_loads(counts, R):
    """Chia histogram global g[E] thành R block expert liên tục, sum mỗi block.
    -> vector R = #token-routing mỗi EP-rank xử lý (placement 'linear', EPLB tắt)."""
    per = counts.shape[0] // R
    assert counts.shape[0] % R == 0
    return counts[: per * R].reshape(R, per).sum(axis=1)


def auto_decode_threshold(num_tokens):
    """Tự tìm ranh giới prefill/decode KHÔNG cần knob: num_tokens lưỡng cực (decode ≈ #seq
    đang chạy ~ conc, rất nhỏ; prefill ≈ ISL/chunk, hàng nghìn). Tách ở KHE TRỐNG lớn nhất
    trên thang log -> ngưỡng = trung điểm hình học khe đó. (Phân tích TIME chia phase bằng
    kernel; log token không có kernel nên dùng cách tương đương này thay DECODE_MAX_TOKENS.)"""
    u = np.unique(np.asarray(num_tokens, dtype=np.int64))
    u = u[u > 0]
    if len(u) < 2:
        return (float(u.max()) + 1) if len(u) else 1.0      # 1 loại step -> coi hết là decode
    j = int(np.argmax(np.diff(np.log10(u.astype(float)))))  # khe log lớn nhất giữa 2 giá trị kề
    return float(np.sqrt(u[j] * u[j + 1]))                  # trung điểm hình học của khe


def build_steps_df(agg, agg_ranks, R, E, topk, decode_max_tokens=None):
    """Mỗi (layer,step) đủ R rank -> 1 dòng steps_df. expert_dist_global = Σ toàn run.
    decode_max_tokens=None -> tự dò ngưỡng (auto_decode_threshold)."""
    rows, n_dropped = [], 0
    expert_dist_global = np.zeros(E, dtype=np.int64)
    for (lidx, step), counts in agg.items():
        if len(agg_ranks[(lidx, step)]) != R:      # bỏ group thiếu rank (biên startup)
            n_dropped += 1
            continue
        total = int(counts.sum())
        if total == 0:
            continue
        loads = rank_loads(counts, R)
        lmax = int(loads.max()); lmin = max(int(loads.min()), 1); lavg = total / R
        rows.append({
            "layer": lidx, "step": step,
            "num_tokens": total / topk, "total_routings": total,
            "rank_max_over_min": lmax / lmin,                  # METRIC CHÍNH (ticket)
            "rank_max_over_avg": lmax / lavg,                  # room-for-improvement
            "expert_max_over_avg": counts.max() / (total / E),
            "load_min": lmin, "load_max": lmax,
        })
        expert_dist_global += counts
    steps_df = pd.DataFrame(rows)
    thr = auto_decode_threshold(steps_df["num_tokens"].values) if decode_max_tokens is None \
        else float(decode_max_tokens)
    steps_df["phase"] = np.where(steps_df["num_tokens"] <= thr, "decode", "prefill_mixed")
    print(f"decode/prefill threshold (num_tokens) = {thr:.1f} "
          f"({'auto' if decode_max_tokens is None else 'manual'})")
    print(f"kept groups (full {R} ranks) = {len(steps_df)}   dropped (partial) = {n_dropped}")
    print(steps_df["phase"].value_counts())
    return steps_df, expert_dist_global


# ======================= CELL 5 — HIST imbalance max/min =======================
def plot_imbalance_hist(steps_df, out, x_clip_pct=99.0, n_bins=60):
    """3 figure riêng (all/prefill_mixed/decode): hist của max_gpu_load/min_gpu_load.
    Mỗi mẫu = 1 (layer,step). Giá trị > xmax CLIP vào bin cuối (không mất mẫu)."""
    subsets = [
        ("all",           steps_df["rank_max_over_min"].values,                                    "skyblue"),
        ("prefill_mixed", steps_df[steps_df.phase == "prefill_mixed"]["rank_max_over_min"].values, "#1f77b4"),
        ("decode",        steps_df[steps_df.phase == "decode"]["rank_max_over_min"].values,        "#ff7f0e"),
    ]
    for name, vals, color in subsets:
        plt.figure(figsize=(16, 4.5))
        if len(vals) == 0:
            plt.title(f"{name} (none)")
            plt.savefig(os.path.join(out, f"token_imbalance_hist_{name}.png"), dpi=130); plt.close(); continue
        xmax = max(2.0, float(np.percentile(vals, x_clip_pct)))
        n_over = int((vals > xmax).sum())
        vclip = np.clip(vals, 1.0, xmax)
        plt.hist(vclip, bins=np.linspace(1.0, xmax, n_bins + 1), color=color, edgecolor="black")
        plt.xlim(1.0, xmax)
        title = (f"{name}: GPU load imbalance (max/min)  — "
                 f"mean={vals.mean():.2f}, median={np.median(vals):.2f}, p99={np.percentile(vals, 99):.2f}")
        if n_over:
            title += f"   [{n_over} samples > {xmax:.1f} clipped into last bin]"
        plt.title(title)
        plt.xlabel("Max Gap (max GPU load / min GPU load)"); plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"token_imbalance_hist_{name}.png"), dpi=130); plt.close()
    print("  saved token_imbalance_hist_{all,prefill_mixed,decode}.png")


# ======================= CELL 12 — phân phối tải (#token-routing) =======================
def build_expert_dist_by_layer(agg, agg_ranks, R, E):
    d = {}
    for (lidx, step), counts in agg.items():
        if len(agg_ranks[(lidx, step)]) != R:
            continue
        if lidx not in d:
            d[lidx] = np.zeros(E, dtype=np.int64)
        d[lidx] += counts
    return d


def plot_load_distribution(expert_dist_global, expert_dist_by_layer, R, E, out):
    experts_per_rank = E // R
    layers_sorted = sorted(expert_dist_by_layer)
    M_expert = np.stack([expert_dist_by_layer[l] for l in layers_sorted])    # (L, E)
    M_rank = M_expert.reshape(len(layers_sorted), R, E // R).sum(axis=2)      # (L, R)

    # ---- PHẦN A: end-to-end ----
    rank_dist = rank_loads(expert_dist_global, R)
    plt.figure(figsize=(20, 4.5))
    plt.bar(range(E), expert_dist_global, width=1.0)
    for r in range(1, R):
        plt.axvline(r * experts_per_rank - 0.5, color="r", lw=0.6, alpha=.6)
    plt.title(f"[End-to-end] Per-expert total routings (E={E}); red lines = rank boundaries")
    plt.xlabel("global expert id"); plt.ylabel("total routings")
    plt.tight_layout(); plt.savefig(os.path.join(out, "load_e2e_per_expert.png"), dpi=130); plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.bar(range(R), rank_dist, color="#2ca02c")
    plt.axhline(rank_dist.mean(), color="k", ls="--", label=f"mean={rank_dist.mean():.0f}")
    plt.title(f"[End-to-end] Per-rank total load  (max/min = {rank_dist.max()/max(rank_dist.min(),1):.2f})")
    plt.xlabel("EP rank"); plt.ylabel("total routings"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "load_e2e_per_rank.png"), dpi=130); plt.close()

    # ---- PHẦN B: heatmap theo layer ----
    ystep = max(1, len(layers_sorted) // 25)
    yt = list(range(0, len(layers_sorted), ystep)); ytl = [layers_sorted[i] for i in yt]

    plt.figure(figsize=(20, 9))
    plt.imshow(M_expert, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    for r in range(1, R):
        plt.axvline(r * experts_per_rank - 0.5, color="r", lw=0.6, alpha=.6)
    plt.colorbar(label="total routings (summed over steps)")
    plt.yticks(yt, ytl)
    plt.title("[Per-layer] total routings per expert  (rows=layer, cols=global expert id; red=rank boundaries)")
    plt.xlabel("global expert id"); plt.ylabel("layer index")
    plt.tight_layout(); plt.savefig(os.path.join(out, "load_heatmap_layer_x_expert.png"), dpi=130); plt.close()

    plt.figure(figsize=(11, 9))
    plt.imshow(M_rank, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="total routings (summed over steps)")
    plt.yticks(yt, ytl); plt.xticks(range(R), range(R))
    plt.title("[Per-layer] total load per EP rank  (rows=layer, cols=EP rank)")
    plt.xlabel("EP rank"); plt.ylabel("layer index")
    if len(layers_sorted) <= 80:
        for i in range(len(layers_sorted)):
            for j in range(R):
                v = M_rank[i, j]
                plt.text(j, i, f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k",
                         ha="center", va="center", color="w", fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out, "load_heatmap_layer_x_rank.png"), dpi=130); plt.close()
    print("  saved load_e2e_*.png + load_heatmap_*.png")


# ======================= CELL 13 — per-rank load TỪNG layer =======================
def plot_per_layer_load(expert_dist_by_layer, R, E, out, layers):
    """layers = list layer cần vẽ (mặc định TẤT CẢ). Mỗi layer: per-rank bar + per-expert bar.
    Lưu vào out/per_layer_load/ (nhiều ảnh) để không rối thư mục chính."""
    experts_per_rank = E // R
    sub = os.path.join(out, "per_layer_load"); os.makedirs(sub, exist_ok=True)
    done = 0
    for L in layers:
        if L not in expert_dist_by_layer:
            continue
        vec = expert_dist_by_layer[L]
        loads = rank_loads(vec, R)
        mm = loads.max() / max(loads.min(), 1); ma = loads.max() / loads.mean()

        plt.figure(figsize=(10, 4))
        plt.bar(range(R), loads, color="#2ca02c")
        plt.axhline(loads.mean(), color="k", ls="--")
        for j, v in enumerate(loads):
            plt.text(j, v, f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k", ha="center", va="bottom", fontsize=8)
        plt.title(f"[Layer {L}] per-rank total load (sum over steps)  —  max/min={mm:.2f}, max/avg={ma:.2f}")
        plt.xlabel("EP rank"); plt.ylabel("total routings"); plt.xticks(range(R))
        plt.tight_layout(); plt.savefig(os.path.join(sub, f"layer{L:02d}_per_rank.png"), dpi=120); plt.close()

        plt.figure(figsize=(20, 3.2))
        plt.bar(range(E), vec, width=1.0)
        for r in range(1, R):
            plt.axvline(r * experts_per_rank - 0.5, color="r", lw=0.6, alpha=.6)
        plt.title(f"[Layer {L}] per-expert total routings (red = rank boundaries)")
        plt.xlabel("global expert id"); plt.ylabel("total routings")
        plt.tight_layout(); plt.savefig(os.path.join(sub, f"layer{L:02d}_per_expert.png"), dpi=120); plt.close()
        done += 1
    print(f"  saved per_layer_load/ ({done} layers x 2 figs)")


# ======================= CELL 14 — imbalance per layer =======================
def plot_per_layer_imbalance(steps_df, out):
    per_layer = steps_df.groupby(["layer", "phase"])["rank_max_over_min"].mean().unstack("phase")
    per_layer = per_layer.reindex(sorted(per_layer.index))
    ax = per_layer.plot(kind="bar", figsize=(16, 4), width=0.85)
    ax.set_title("Mean EP imbalance (max/min) per MoE layer")
    ax.set_xlabel("layer index"); ax.set_ylabel("mean max/min"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "imbalance_per_layer.png"), dpi=130); plt.close()

    dec_df = steps_df[steps_df.phase == "decode"]
    if len(dec_df):
        piv = dec_df.pivot_table(index="layer", columns="step", values="rank_max_over_min")
        piv = piv.reindex(sorted(piv.index))
        plt.figure(figsize=(16, 6))
        plt.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower",
                   extent=[piv.columns.min(), piv.columns.max(), 0, len(piv.index)])
        plt.colorbar(label="max/min")
        plt.title("Decode: EP imbalance heatmap (layer x step)")
        plt.xlabel("step"); plt.ylabel("layer (row order)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "imbalance_decode_heatmap.png"), dpi=130); plt.close()
    print("  saved imbalance_per_layer.png + imbalance_decode_heatmap.png")


# ======================= CELL 16 — EXPORT =======================
def summarize(arr):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3)}


def export(steps_df, expert_dist_global, R, E, topk, out):
    steps_df.to_parquet(os.path.join(out, "steps_imbalance.parquet"))
    np.save(os.path.join(out, "expert_dist_global.npy"), expert_dist_global)
    rank_dist = rank_loads(expert_dist_global, R)
    summary = {
        "ranks": R, "experts": E, "topk": topk, "experts_per_rank": E // R,
        "n_steps": int(len(steps_df)),
        "e2e_per_rank_load": [int(x) for x in rank_dist],
        "e2e_rank_max_over_min": round(float(rank_dist.max() / max(rank_dist.min(), 1)), 3),
        "all": summarize(steps_df["rank_max_over_min"]),
        "prefill_mixed": summarize(steps_df[steps_df.phase == "prefill_mixed"]["rank_max_over_min"]),
        "decode": summarize(steps_df[steps_df.phase == "decode"]["rank_max_over_min"]),
    }
    with open(os.path.join(out, "summary_tokens.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved ->", out)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="MV-4571 EP imbalance theo #token (headless).")
    ap.add_argument("--log", required=True, help="serve.log path hoặc glob (lấy file mới nhất nếu glob).")
    ap.add_argument("--out", required=True, help="thư mục lưu kết quả.")
    ap.add_argument("--decode-max-tokens", default="4xconc",
                    help="ngưỡng chia phase: step có num_tokens(global) <= ngưỡng => decode, else "
                         "prefill_mixed. '4xconc' (mặc định) = 4*concurrency; 'auto' = tự dò khe "
                         "lưỡng cực; hoặc 1 số cụ thể.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="số request đồng thời (driving '4xconc': ngưỡng decode = 4*concurrency).")
    ap.add_argument("--layers", default="all",
                    help="'all' (mặc định) hoặc list layer cho per-layer-load, vd '3,8,40,41'.")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.log)) if any(c in args.log for c in "*?[") else [args.log]
    assert matches, f"no serve.log matched: {args.log}"
    log_path = matches[-1]
    os.makedirs(args.out, exist_ok=True)
    print(f"LOG_PATH = {log_path}\nsize     = {os.path.getsize(log_path)/1e6:.1f} MB\nOUT      = {args.out}")

    agg, agg_ranks, agg_ntok, R, E, topk, moe_layers = parse_and_aggregate(log_path)
    print(f"TOPK(inferred)={topk}  EXPERTS_PER_RANK={E // R}  groups={len(agg)}")

    # Ngưỡng chia decode/prefill: mặc định 4*concurrency (theo yêu cầu ticket); 'auto' = tự dò;
    # hoặc 1 số cụ thể. (decode num_tokens ~ #seq đang chạy ~ conc, nên 4*conc tách an toàn.)
    _dmt = str(args.decode_max_tokens).strip().lower()
    if _dmt == "auto":
        dmt = None
    elif _dmt in ("4xconc", "4*conc", "conc4", ""):
        dmt = 4 * args.concurrency
    else:
        dmt = float(args.decode_max_tokens)
    print(f"decode-threshold source = {args.decode_max_tokens!r} "
          f"(concurrency={args.concurrency}) -> "
          f"{'auto-detect' if dmt is None else f'{dmt:g} tokens'}")
    steps_df, expert_dist_global = build_steps_df(agg, agg_ranks, R, E, topk, dmt)
    expert_dist_by_layer = build_expert_dist_by_layer(agg, agg_ranks, R, E)

    layers = sorted(expert_dist_by_layer) if args.layers.strip().lower() == "all" \
        else [int(x) for x in args.layers.split(",") if x.strip()]

    plot_imbalance_hist(steps_df, args.out)                                   # CELL 5
    plot_load_distribution(expert_dist_global, expert_dist_by_layer, R, E, args.out)  # CELL 12
    plot_per_layer_load(expert_dist_by_layer, R, E, args.out, layers)        # CELL 13 (all layers)
    plot_per_layer_imbalance(steps_df, args.out)                             # CELL 14
    export(steps_df, expert_dist_global, R, E, topk, args.out)               # CELL 16


if __name__ == "__main__":
    main()
