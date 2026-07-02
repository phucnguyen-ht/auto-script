# MTP5 (GLM5.2 speculative) — tổng hợp toàn bộ kết quả EPLB đã chạy

Model: GLM5.2, preset `MTP5-bs64-dg` (MTP/eagle speculative, num_spec=5), DP8/EP8/TP1, MI300/ROCm.
Metric imbalance = max/min per-cluster MoE-compute span qua 8 rank (profile → analyze_time).
Metric bench = throughput/latency (vllm bench serve, không profiler), mean±std qua 3 run.

---

## 0. Các cấu hình EPLB ĐÃ CHẠY trên MTP5

| # | config | eplb_config | phase đã chạy → **thư mục runs** |
|---|---|---|---|
| 1 | **baseline (EPLB off)**<br>`MTP5-bs64-dg.yaml` | — | imbalance matrix → [run_glm5/…/MTP5-bs64-dg](bench_mv4571/auto_analyze_ep_imbalance/logs/run_glm5/glm5.2/MTP5-bs64-dg) · imbalance 10k_c36 → [run_201534/…/MTP5-bs64-dg](bench_mv4571/auto_analyze_ep_imbalance/logs/run_20260630_201534/glm5.2/MTP5-bs64-dg) · verify bench → [eplb_verify_bench/…/MTP5-bs64-dg](bench_mv4571/auto_analyze_ep_imbalance/logs/eplb_verify_bench/glm5.2/dp8ep8/MTP5-bs64-dg) · cand bench 8k → [eplb_candidate_bench/…/MTP5-bs64-dg-base-8k](bench_mv4571/auto_analyze_ep_imbalance/logs/eplb_candidate_bench/glm5.2/dp8ep8/MTP5-bs64-dg-base-8k) |
| 2 | **eplb-freq**<br>`MTP5-bs64-dg-eplb-freq.yaml` | sync, torch_nccl, win100/step100 | imbalance @10k_c36 → [run_201534/…/MTP5-bs64-dg-eplb-freq](bench_mv4571/auto_analyze_ep_imbalance/logs/run_20260630_201534/glm5.2/MTP5-bs64-dg-eplb-freq) |
| 3 | **eplb-default**<br>`MTP5-bs64-dg-eplb-default.yaml` | sync, torch_nccl, win1000/step3000 | imbalance @10k_c36 → [run_201534/…/MTP5-bs64-dg-eplb-default](bench_mv4571/auto_analyze_ep_imbalance/logs/run_20260630_201534/glm5.2/MTP5-bs64-dg-eplb-default) |
| 4 | **eplb-freq-pynccl**<br>`MTP5-bs64-dg-eplb-freq-pynccl.yaml` | sync, pynccl, win100/step100 | verify bench 3× @10k_c36 → [eplb_verify_bench/…/MTP5-bs64-dg-eplb-freq-pynccl](bench_mv4571/auto_analyze_ep_imbalance/logs/eplb_verify_bench/glm5.2/dp8ep8/MTP5-bs64-dg-eplb-freq-pynccl) |
| 5 | **async-NIXL**<br>`MTP5-bs64-dg-async-8k.yaml` | **async, nixl**, win100/step100 | candidate bench 8k (c8-c52) → [eplb_candidate_bench/…/MTP5-bs64-dg-async-8k](bench_mv4571/auto_analyze_ep_imbalance/logs/eplb_candidate_bench/glm5.2/dp8ep8/MTP5-bs64-dg-async-8k) |

Preset ĐÃ TẠO nhưng CHƯA chạy trên MTP5: `-eplb-freq-red16` (bỏ vì red16+torch_nccl hang trên noMTP),
`-eplb-step500-pynccl` (chỉ chạy noMTP). MTP5 không dính crash nào (khác noMTP: sync step100 crash).

### Số liệu chính mỗi config (imbalance @10k_c36 + rearrange + throughput)

| config | time imbalance (max/min) | **rearrange thật** | headroom % | crit (ms) | bal (ms) | comm (ms) | bench out_tput (10k_c36, no profiler) |
|---|---|---|---|---|---|---|---|
| 1 baseline (off)        | **1.322** | — (off) | 14.42 | 7833 | 6703 | 7816 | **611.6 ± 95.7** (72/72/72) |
| 2 eplb-freq (nccl)      | **1.284** (−2.9%) | **4** | 13.80 | 7343 | 6329 | 6352 | — (chỉ đo imbalance) |
| 3 eplb-default (nccl)   | **1.318** (−0.3%) | **0** (chỉ profile-warmup) | 13.92 | 7561 | 6509 | 6451 | — |
| 4 eplb-freq-pynccl      | ≈1.28 ¹ | ≈4 ¹ | — | — | — | — | **584.1 ± 76.4** (72/72/72) |
| 5 async-NIXL            | n/a ² | n/a ² | — | — | — | — | 8k c8–c52 → section D |

- Định nghĩa: **imbalance** = max/min của per-cluster MoE-compute span qua 8 rank; **rearrange thật** =
  số lần EPLB thực sự tái sắp expert trong run (đếm dòng `Rearranged experts … in Xs`, loại
  `(profile)` warmup); **headroom%** = 1 − Σmean/Σmax (mức tiết kiệm compute nếu cân bằng hoàn hảo);
  **crit** = Σ per-cluster max (đường tới hạn thật), **bal** = Σ per-cluster mean (nếu cân bằng),
  **comm** = Σ (gather+reduce-scatter).
- ¹ imbalance **độc lập communicator** (nixl/pynccl/nccl cho cùng placement) → pynccl ≈ eplb-freq(nccl)
  = 1.284/4 rearrange; pynccl chỉ chạy phase bench nên không có summary imbalance riêng.
- ² MTP5 async-NIXL chỉ chạy **bench 8k** (chưa profile @10k_c36 → chưa có imbalance). Tham chiếu:
  noMTP async-NIXL imbalance đo được = 1.497 (run_20260701_050802).
- Nhận xét: EPLB **freq** hạ imbalance 1.322→1.284 với **4 rearrange**; **default** **0 rearrange**
  (step_interval=3000 > số step của run ngắn) → ≈ baseline. Nhưng −2.9% imbalance KHÔNG ra lợi ích
  throughput (611.6→584.1 nằm trong std ~±90 → ≈ noise). Xem chi tiết section B/C/D.

---

## A. Imbalance — baseline MTP5 (EPLB off), toàn bộ matrix MV-4571 (run_glm5)

| case | imbalance | headroom% | crit(ms) | comm(ms) | compute vs comm |
|---|---|---|---|---|---|
| 100k_c8  | 1.464 | 19.43 | 12678 | 27170 | comm-bound |
| 100k_c36 | 1.449 | 18.10 | 47989 | 62687 | comm-bound |
| 10k_c8   | 1.439 | 16.89 |  3405 |  4261 | comm-bound |
| 100k_c22 | 1.428 | 17.73 | 30275 | 27977 | ~ |
| 100k_c31 | 1.417 | 17.74 | 41607 | 50301 | comm-bound |
| 8k_c8    | 1.407 | 16.05 |  5492 |  5514 | ~ |
| 8k_c22   | 1.373 | 13.96 |  8373 |  6994 | compute-bound |
| 10k_c22  | 1.340 | 14.30 |  5498 |  5637 | ~ |
| 8k_c31   | 1.318 | 13.12 |  8704 |  7006 | compute-bound |
| 8k_c52   | 1.300 | 13.50 | 11325 |  8159 | compute-bound |
| 8k_c36   | 1.290 | 12.61 |  9391 |  8011 | compute-bound |

→ **MTP5 imbalance THẤP toàn dải (1.29–1.46)** — spec-decode gộp ~(1+spec) token/step → averaging routing tốt
→ ít lệch → ít headroom cho EPLB. (So sánh: noMTP cao hơn hẳn, tới 1.64.)

## B. Imbalance — thí nghiệm EPLB @ 10k_c36 (run_20260630_201534, torch_nccl, sync)

| config | imbalance | Δ vs baseline | headroom% | crit(ms) | bal(ms) | rearrange thật |
|---|---|---|---|---|---|---|
| baseline (off)      | 1.322 | — | 14.42 | 7833 | 6703 | — |
| **eplb-freq** (step100) | **1.284** | **−2.9%** | 13.80 | 7343 | 6329 | 4 |
| eplb-default (step3000) | 1.318 | −0.3% (≈0) | 13.92 | 7561 | 6509 | 0 (chỉ profile warmup) |

→ EPLB **freq** giảm imbalance nhẹ (1.322→1.284) với 4 lần rearrange; **default** không rearrange lần nào
(step3000 > số step của run ngắn) → ≈ baseline.

## C. Verify bench 3× @ 10k_c36 (eplb_verify_bench, sync pynccl, no profiler)

| config | completed (3 run) | out_tput (mean±std) | mean_itl (ms) | mean_ttft (ms) | ổn định |
|---|---|---|---|---|---|
| baseline (off)        | 72/72/72 | **611.6 ± 95.7** | 218.0 | 6175 | ✓ |
| eplb-freq-pynccl (step100) | 72/72/72 | **584.1 ± 76.4** | 224.8 | 6111 | ✓ |

→ Δ throughput = −4.5% **mean** NHƯNG std ~13–16% (baseline trải 518.8/606.0/709.9) → **nằm trong 1σ,
không phân biệt được với 0**. MTP5 EPLB sync-pynccl **ổn định** và chi phí throughput **≈ noise**.

## D. Candidate bench 8k — async-NIXL vs baseline (eplb_candidate_bench, no profiler, 3 run)

| conc | baseline out_tput (mean±std) | async-NIXL out_tput (mean±std) | Δ | baseline itl | async itl |
|---|---|---|---|---|---|
| c8  |  559 ± 118 |  576 ± 64  | +3% | 60.6 | 60.2 |
| c22 | 1060 ± 218 |  928 ± 100 | −12% | 90.1 | 94.5 |
| c31 | 1232 ± 243 | 1336 ± 229 | +8% | 105.9 | 99.5 |
| c36 | 1378 ± 190 | 1282 ± 257 | −7% | 110.5 | 120.1 |
| c52 | 1477 ± 176 | 1459 ± 183 | −1% | 148.7 | 150.3 |

(completed = num_prompts = 2×conc = 16/44/62/72/104 — tất cả hoàn tất đủ, không crash.)
→ Dấu Δ **đổi chiều lung tung** (+3/−12/+8/−7/−1%) và **mọi hiệu số nằm trong std** (CoV ~15–20%) →
async-NIXL trên MTP5 8k là **throughput-neutral**: không thắng cũng không thua baseline một cách có ý nghĩa.

---

## Kết luận cho MTP5
1. **EPLB có giảm imbalance nhẹ** (1.322→1.284 @10k_c36, −2.9%) khi step_interval đủ nhỏ để rearrange.
2. **Nhưng KHÔNG chuyển thành lợi ích throughput**: cả sync-pynccl (verify) lẫn async-NIXL (candidate) đều
   cho Δ throughput **nằm trong noise** vs baseline. Lý do: MTP5 baseline vốn đã cân bằng (imbalance thấp,
   1.29–1.46) → ít headroom → lợi ích cân bằng ≈ overhead rearrange ≈ nhiễu.
3. **MTP5 ổn định với EPLB** (không crash ở bất kỳ config nào đã chạy — khác noMTP nơi sync-step100 sập).
4. ⇒ Với MTP5, **EPLB gần như không đáng bật** (spec-decode đã tự làm phẳng routing). EPLB có giá trị hơn ở
   noMTP (imbalance cao hơn) — đang bench riêng để xác nhận.

## Nơi lưu dữ liệu
- Imbalance baseline matrix: `logs/run_glm5/glm5.2/MTP5-bs64-dg/<case>/time/analysis/summary_time.json`
- Imbalance EPLB @10k_c36: `logs/run_20260630_201534/glm5.2/MTP5-bs64-dg{,-eplb-freq,-eplb-default}/...`
- Verify bench 3×: `logs/eplb_verify_bench/glm5.2/dp8ep8/MTP5-bs64-dg{,-eplb-freq-pynccl}/auto_bench/<ts>/{run1,2,3,mean,std}.csv`
- Candidate bench 8k: `logs/eplb_candidate_bench/glm5.2/dp8ep8/MTP5-bs64-dg-{base,async}-8k/auto_bench/<ts>/...`
