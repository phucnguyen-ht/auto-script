#!/usr/bin/env python3
# [MV-4572] Parse [MV4572 FWDDEV] lines từ serve.log -> so N config device-time/forward theo
# (cg_mode, padded) bucket. Xuất CSV + in bảng.
#
# CƠ CHẾ (cách phân tích):
#   1. FWDDEV log CUMULATIVE, in mỗi FWDDEV_EVERY forward, mỗi DÒNG = 1 (rank, cg_mode, padded) bucket:
#        [MV4572 FWDDEV] tag=fwd cg_mode=FULL padded=4 n_forwards=17400 samples=77088
#          mean_fwd_dev_ms=18.06 avg_real_toks=2 avg_real_reqs=1.7 avg_dp=[...] last_dp=[...]
#   2. Vì cumulative -> lấy DÒNG CUỐI (max samples) per (rank, cg, padded) = mean cả run của rank đó.
#   3. Aggregate 8 rank cùng (cg, padded): sample-weighted mean_ms = Σ(mean_r*samples_r)/Σsamples_r.
#   4. Sort theo tổng samples desc (bucket decode dominant lên đầu). So per-bucket giữa các config.
#      Bucket dominant = FULL padded nhỏ (decode). A_x/A0 > 1 cùng bucket => +Δ NẰM TRONG forward-GPU.
#
# CHẠY:
#   python3 parse_fwddev.py OUT.csv A0:/…/A0/serve.log A8:/…/A8/serve.log [A9:… A10:…]
import csv
import re
import sys
from collections import defaultdict

PAT = re.compile(
    r"Worker_DP(\d+)_EP\d+.*\[MV4572 FWDDEV\] tag=\w+ cg_mode=(\w+) padded=(\d+) "
    r"n_forwards=(\d+) samples=(\d+) mean_fwd_dev_ms=([\d.]+) "
    r"avg_real_toks=([\d.]+) avg_real_reqs=([\d.]+)"
)

def parse(path):
    best = {}  # (dp,cg,padded) -> (samples, mean, rtoks, rreqs)  [last/max-samples occurrence]
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = PAT.search(ln)
            if not m:
                continue
            dp, cg, padded = int(m.group(1)), m.group(2), int(m.group(3))
            samples, mean = int(m.group(5)), float(m.group(6))
            rt, rr = float(m.group(7)), float(m.group(8))
            k = (dp, cg, padded)
            if k not in best or samples >= best[k][0]:
                best[k] = (samples, mean, rt, rr)
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for (dp, cg, padded), (s, mean, rt, rr) in best.items():
        a = agg[(cg, padded)]
        a[0] += s; a[1] += mean * s; a[2] += rt * s; a[3] += rr * s
    return {k: (s, ms / s, rt / s, rr / s) for k, (s, ms, rt, rr) in agg.items()}

def main():
    out_csv = sys.argv[1]
    labels, data = [], {}
    for spec in sys.argv[2:]:
        lb, path = spec.split(":", 1)
        labels.append(lb); data[lb] = parse(path)
    keys = sorted(set().union(*[set(d) for d in data.values()]),
                  key=lambda k: -sum(data[lb].get(k, (0,))[0] for lb in labels))
    base = labels[0]
    # write CSV
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["cg_mode", "padded", "avg_reqs", "avg_toks"]
        for lb in labels:
            hdr += [f"{lb}_samples", f"{lb}_ms"]
        for lb in labels[1:]:
            hdr += [f"{lb}/{base}", f"{lb}-{base}_ms"]
        w.writerow(hdr)
        for k in keys:
            cg, padded = k
            rr = next((data[lb][k][3] for lb in labels if k in data[lb]), 0)
            rt = next((data[lb][k][2] for lb in labels if k in data[lb]), 0)
            row = [cg, padded, f"{rr:.1f}", f"{rt:.0f}"]
            mb = data[base].get(k, (0, 0, 0, 0))[1]
            for lb in labels:
                s, ms, _, _ = data[lb].get(k, (0, 0, 0, 0))
                row += [s, f"{ms:.3f}"]
            for lb in labels[1:]:
                ms = data[lb].get(k, (0, 0, 0, 0))[1]
                row += [f"{ms/mb:.3f}" if (mb and ms) else "", f"{ms-mb:+.3f}" if (mb and ms) else ""]
            w.writerow(row)
    # print table
    print(f"{'cg':<10}{'padded':>7}{'reqs':>6}{'toks':>6} | " +
          " | ".join(f"{lb+'_smpl':>9}{lb+'_ms':>9}" for lb in labels))
    print("-" * (30 + 20 * len(labels)))
    for k in keys[:24]:
        cg, padded = k
        rr = next((data[lb][k][3] for lb in labels if k in data[lb]), 0)
        cells = []
        for lb in labels:
            s, ms, _, _ = data[lb].get(k, (0, 0, 0, 0))
            cells.append(f"{s:>9}{ms:>9.3f}")
        print(f"{cg:<10}{padded:>7}{rr:>6.1f}{'':>6} | " + " | ".join(cells))
    print(f"\n[saved] {out_csv}")

if __name__ == "__main__":
    main()
