# MV-4571 — EP imbalance theo TIME (từ torch profiler traces, 8 rank)

Phân tích thời gian thực của MoE từ trace 8 rank (case noMTP / 8k / conc8 / inf), bổ sung
cho phân tích token (`process_ep_logs_glm5.ipynb`). Script: `analyze_trace_pairs.py`.
Schema trace: xem `events.MD`. Luồng code MoE: xem `MOE_DATA_FLOW.md`.

## Phương pháp: đo MoE bằng 2 comm bao quanh (KHÔNG đếm kernel bên trong)

Flow MoE đổi theo số token (prefill = 5 kernel `sort→quant→gemm1→quant→gemm2`; decode đôi
khi = 1 kernel fused `aiter::fmoe_...`). Nên đo MoE-time mỗi cụm = **`reduce_scatter.start −
gather.end`** (khoảng kẹp giữa 2 collective), độc lập kiểu kernel.

- Phân biệt gather vs reduce-scatter: mọi `ncclDevKernel` giống hệt tên/args → dùng **vị trí**
  (comm trước cụm MoE = gather, comm sau = reduce-scatter). Hợp lệ nhờ data-dep.
- Mở "cụm MoE" bằng **sort/gemm/fmoe** (KHÔNG bằng quant — có 1 quant chạy trước gather để
  dispatch, `quant = 3×#cụm`; nếu mở bằng quant sẽ đếm gấp đôi).
- Clock 8 rank CHUNG (lệch < 3ms) → so timestamp giữa rank trực tiếp.

## (a) MAPPING giữa các rank — XÁC NHẬN

| | rank0..7 |
|---|---|
| #comm | **154,650** (bằng nhau cả 8 rank) |
| #cụm MoE (region) | **77,325** (bằng nhau cả 8 rank) |

`154,650 = 2 × 77,325` → mỗi cụm đúng 1 gather + 1 reduce-scatter; số cụm khớp tuyệt đối giữa
8 rank ⇒ **cụm thứ k map 1-1 giữa các rank** (đúng giả định "vị trí 1 ↔ vị trí 1").
(77,325 = 75 MoE-layer × ~1031 step; gồm 77,100 cụm 2-stage-gemm + 225 cụm fused fmoe.)

## (b) Flow-chart (Gantt thực từ trace)

`trace_pairs/cluster_gantt_k{200,40000,77000}.png` — mỗi rank 1 hàng `[gather | moe |
reduce-scatter]`, 2 đường đứt = barrier sync (gather.end & reduce-scatter.end). Khớp đúng
hình mô tả: gather kết thúc thẳng hàng, moe dài-ngắn khác nhau, reduce-scatter kết thúc thẳng
hàng (rank xong moe sớm phải chờ).

## Chứng minh SYNC (spread max−min giữa 8 rank tại mỗi cụm, µs)

| Mốc | median | p99 | kỳ vọng |
|---|---|---|---|
| **gather.END** | 1.74 | 10.9 | ~0 (sync) ✔ |
| **reduce-scatter.END** | 1.15 | 10.5 | ~0 (sync) ✔ |
| reduce-scatter.START | 47.9 | 104.5 | lớn (lệch do MoE) ✔ |
| MoE-time / cụm | 94.7 | 138.6 | — |

→ gather.END & reduce-scatter.END **spread ≈ 1µs << MoE-time ~95µs** ⇒ collective **sync ở
ĐIỂM KẾT THÚC** (barrier ở cuối). Chỉ `reduce-scatter.START` lệch (rank nào MoE lâu hơn thì
bắt đầu reduce-scatter muộn hơn) — đúng giả thuyết.

## Kết quả: time imbalance MoE-compute

| Metric | Giá trị |
|---|---|
| MoE-time / rank tổng (ms) | 7414 – 7577 (max/min ≈ **1.02** — cân bằng tích luỹ) |
| critical-path `Σ max_r` | **9561 ms** |
| balanced `Σ mean_r` | **7485 ms** |
| **headroom nếu cân bằng MoE-compute** | **21.7%** (≈ 2076 ms) |
| per-cluster max/min | mean **1.73**, median 1.68, p99 2.47 |
| per-cluster max/avg | mean 1.27, p99 1.58 |

## Diễn giải (cho ticket)

1. **Cân bằng tích luỹ, lệch theo từng step.** Tổng cả run 8 GPU gần bằng nhau (~1.02), nhưng
   mỗi step rank bận nhất chậm hơn rank rảnh ~1.73×. Vì reduce-scatter là barrier ⇒ critical
   path = Σ max ⇒ **~21.7% thời gian MoE-compute mất vì imbalance**.
2. **Time imbalance (1.73) thấp hơn token imbalance (~3.8 ở decode).** Xác nhận token-ratio
   phóng đại so với time (do tiling/padding + chi phí cố định) → **đo bằng time mới đúng**.
3. **MoE-compute (~7.5s) nhỏ so với communication.** Backend AgRs all-gather + reduce-scatter
   toàn bộ token rất nặng; comm chiếm phần lớn wall-time của MoE-layer. Cân bằng expert giúp
   ~21.7% của phần compute, nhưng muốn cải thiện end-to-end lớn phải động tới communication
   (vd all2all thật DeepEP/mori thay vì all-gather/reduce-scatter).

## Artifacts — `EP_imbalance_analysis/logs/<run_ts>_<scenario>/` (vd `20260625_091838_8k_rinf_c8/`)
Tạo bởi `analyze_trace_pairs.py` (heavy parse) + `process_trace_time_glm5.ipynb` (plot):
- `summary.txt`, `summary_time.json`, `trace_pairs.npz` (MOE, GS/GE/RSs/RSe).
- `cluster_gantt_k*.png` / `gantt_k*.png` — flow-chart 8 rank (gather→moe→reduce-scatter + barrier sync).
- `time_imbalance_hist_{all,prefill_mixed,decode}.png` — phân phối max/min (time).
- `time_over_clusters.png`, `per_rank_and_budget.png` (comm vs MoE budget), `phase_threshold_hist.png`.
- (`token_vs_time.png` nếu đã chạy `process_ep_logs_glm5*.ipynb` để có `steps_imbalance.parquet`.)

Script: `analyze_trace_pairs.py` = bản chính (comm-bounded, OUT tự suy ra `logs/<run_ts>_<scenario>/`).
Logic parse cũng đã nhúng trong `process_trace_time_glm5.ipynb` (Bước 1) — chạy notebook là đủ.
(Phương pháp cũ "gemm2-boundary" bỏ sót 225 cụm fused `fmoe` nên đã loại; vẫn còn dưới dạng cell
tùy chọn `RUN_GEMM2_METHOD` trong notebook để đối chiếu.)
