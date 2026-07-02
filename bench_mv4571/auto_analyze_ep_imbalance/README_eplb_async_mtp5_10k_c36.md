# README — async EPLB (pynccl & torch_nccl) trên MTP5 @ 10k_c36

## Mục tiêu
Kiểm chứng giả thuyết: `use_async: true` (rearrange chạy background thread, overlap compute) có giúp
**pynccl / torch_nccl** tránh được **worker-stall / RPC-timeout** đã làm sập EPLB sync dưới serving kéo dài
không? (Trước đó chỉ async-NIXL được xác nhận ổn định.) Đo cả **time-imbalance** (profile) và **bench 3×**
(throughput, fair — không profiler, không EP_LOG, cudagraph on).

## Cấu hình
- Model: GLM5.2, preset MTP5 (`MTP5-bs64-dg`, speculative num_spec=5), DP8/EP8/TP1, MI300/ROCm.
- Scenario: ISL 10k, conc 36, rate inf, OSL 500.
- 2 preset async (base MTP5 fair + thêm eplb_config):
  - `MTP5-bs64-dg-eplb-async-pynccl.yaml`: `eplb_config '{"use_async": true, "communicator": "pynccl",  "window_size": 100, "step_interval": 100}'`
  - `MTP5-bs64-dg-eplb-async-nccl.yaml`:   `eplb_config '{"use_async": true, "communicator": "torch_nccl","window_size": 100, "step_interval": 100}'`
  - (KHÔNG cần UCX_TLS/kv_cache_memory_bytes — đó là workaround riêng của NIXL.)
- Driver: `run_mtp5_async.sh` → (1) time-imbalance `PHASES=time`, (2) bench 3× `auto_bench.sh` (env
  `env_eplb_bench.yaml`, runs=3).

## Kết quả

| config | time-imbalance (profile) | bench 3× (fair, no profiler) | ổn định |
|---|---|---|---|
| **async-pynccl** | ✗ **CRASH** ở profile (RPC timeout) → không đo được | ✗ **CRASH 36/0/0** (out_tp 3.6, invalid) | ✗ |
| **async-nccl**   | ✓ imbalance **1.341** (9 traces) | ✗ **CRASH 68/0/0** (out_tp 21.6, invalid) | ✗ |
| *(so sánh)* baseline MTP5 | 1.322 | 611.6 ± 95.7 (72/72/72) | ✓ |
| *(so sánh)* sync-pynccl MTP5 | ≈1.284 | 584.1 ± 76.4 (72/72/72) | ✓ |
| *(so sánh)* async-NIXL (noMTP) | 1.497 | 72/72/72 (đã proven ổn) | ✓ |

## Kết luận (ngược giả thuyết)
**async KHÔNG cứu được pynccl/torch_nccl.** Cả 2 vẫn crash cùng lớp lỗi worker-stall:
```
EngineCore encountered a fatal error → multiproc_executor.py get_response →
shm_broadcast.py dequeue/acquire_read → TimeoutError → "RPC call to sample_tokens timed out"
```
- **async-pynccl**: crash NGAY ở profile (nặng hơn) → không lấy được imbalance; bench cũng crash (36/0/0).
- **async-nccl**: profile qua được (imbalance 1.341) nhưng bench kéo dài (np=72) crash sau run1 (68/0/0).
- Chỉ **async-NIXL** ổn định (NIXL dùng READ-transfer riêng qua UCX/RIXL, không nghẽn đường shm_broadcast/RPC
  giữa các DP worker như async-worker + torch-communicator).

⇒ Trên build này (vLLM 0.23.1.dev0, ROCm/MI300), để EPLB **active-rebalance** dưới serving kéo dài:
- **MTP5**: dùng **sync-pynccl** (ổn định, nhưng lợi ích throughput ≈ noise vì MTP5 vốn cân bằng) — hoặc
  đơn giản **tắt EPLB**.
- Nếu cần backend async ổn định: chỉ **NIXL** (async-NIXL), đổi lại overhead rearrange ~37s/lần (ẩn nhờ async).

## Nơi lưu dữ liệu
- Time-imbalance (profile): `logs/run_20260702_011841/glm5.2/MTP5-bs64-dg-eplb-async-{pynccl,nccl}/10k_rinf_c36/time/`
  (async-pynccl: serve.log có RPC-timeout, traces=0, no summary; async-nccl: summary_time.json + 9 traces)
- Bench 3×: `logs/eplb_mtp5_async_bench/glm5.2/dp8ep8/MTP5-bs64-dg-eplb-async-{pynccl,nccl}/auto_bench/<ts>/`
  (run1/2/3.csv: crash 36/0/0 & 68/0/0)
- Driver log: `claude-logs/artifacts/mtp5_async_<ts>.log`
- Tổng hợp toàn bộ case: `EPLB_BENCH_DOC.md`
