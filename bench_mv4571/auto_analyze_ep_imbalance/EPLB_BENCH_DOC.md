# EPLB — kết quả GLM5.2 @ 10k_c36, tách 4 nhóm: {mtp,non-mtp} × {default,freq}

Model GLM5.2, DP8/EP8/TP1, MI300/ROCm. Scenario ISL 10k, conc 36, rate inf, OSL 500.
Mỗi cell chỉ 2 folder dir: **time-imbalance** (profile) và **bench**. Chưa có → `none`.

## Cách đọc số bench (theo yêu cầu)
- Cột bench = **mean_tpot** (ms/token) = **mean ± std qua 3 run** + **completed n/n/n** (kỳ vọng 72 = 2×conc36).
- **mean_tpot = thời gian sinh trung bình mỗi output token — THẤP hơn = nhanh hơn** (thay cho throughput cũ 611.6 tok/s khó đọc).
- **CRASH n/n/n** = có run bị 0 completed (worker-stall/RPC-timeout) → số tpot vô nghĩa, chỉ ghi pattern completed.
- **⏳** = đang chạy (điền nốt khi xong).

## Quy tắc FAIR (bench)
Bench **KHÔNG fair** (→ `none`) nếu preset serve có **`enforce_eager: true`** HOẶC **`VLLM_MOREH_EP_LOG:'1'`**
(eager tắt cudagraph + logging → chậm/không công bằng). Time-imbalance luôn fair (harness tự set eager=false, EP_LOG=0).
- **MTP5**: base preset fair (cudagraph, no EP_LOG) → mọi bench MTP5 fair.
- **noMTP**: base MV-4571 mang `enforce_eager:true`+`EP_LOG:'1'` → chỉ preset `*-fair` (đặt eager=false, EP_LOG=0) mới tính.

Đường dẫn tương đối từ chính file doc này. `imbalance` = `all.maxmin_mean` (per-step max/min MoE-time qua 8 rank, mean).
Folder default nằm ở `logs_result_default/`, folder freq (+còn lại) ở `logs_results/`.

---

## 1. MTP · DEFAULT  ⭐ (ưu tiên) — `logs_result_default/`
eplb_config = sync (`use_async:false`), **KHÔNG set window/step** → mặc định window1000/step3000 (rất ít/không rearrange).

| # | config | imbalance | rearrange | time-imbalance | bench (mean_tpot, completed) |
|---|---|---|---|---|---|
| 1 | **baseline** (EPLB off) | 1.322 | — | [time](logs_result_default/mtp5-base-none-asyncNo-off-10k36/time_imbalance) | [bench](logs_result_default/mtp5-base-none-asyncNo-off-10k36/bench) → **43.7 ± 7.3 ms**, 72/72/72 ✓ |
| 2 | **default-nccl** | 1.318 | 0 / **1** ¹ | [time](logs_result_default/mtp5-eplb-nccl-asyncNo-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nccl-asyncNo-default-10k36/bench) → **50.4 ± 5.5 ms**, 72/72/71 ✓ |
| 3 | **default-pynccl** | 1.383 | 0 / **2** ¹ | [time](logs_result_default/mtp5-eplb-pynccl-asyncNo-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-pynccl-asyncNo-default-10k36/bench) → **47.2 ± 3.9 ms**, 72/72/72 ✓ |
| 4 | **default-nixl** (+UCX+kv) | 1.333 | 0 / **2** ¹ | [time](logs_result_default/mtp5-eplb-nixl-asyncNo-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nixl-asyncNo-default-10k36/bench) → **55.9 ± 21.2 ms**, 72/72/72 ✓ |

¹ rearrange THẬT (loại warmup `(profile)`), dạng **profile / bench**. Bộ đếm trigger nạp sẵn 2250 (=¾·3000), trigger
khi chạm 3000 → **lần đầu ở forward 750**, rồi mỗi 3000. PROFILE ngắn (~270–500 forward < 750) → **0 rearrange** →
imbalance = baseline + noise (1.32–1.38), EPLB-default KHÔNG cân bằng thêm. BENCH dài hơn → vượt 750 (rồi mỗi 3000)
**1–2 lần**: nccl 1 (3.08s), pynccl 2 (0.43+1.10s), nixl 2 (0.51s + **38.12s** — 1 lần rearrange NIXL sync rất chậm,
nguồn variance ±21.2 của nixl).
**mtp-default ĐỦ 4/4.** So baseline (43.7ms): nccl **50.4** (+15%), pynccl **47.2** (+8%), nixl **55.9±21.2** (+28%)
— trong/gần khoảng noise. **Cả 3 SỐNG dưới bench** (nccl 72/72/71, pynccl & nixl 72/72/72). ĐÍNH CHÍNH: default
KHÔNG phải 0 rearrange — profile ngắn thì 0, nhưng **bench có 1–2 rearrange thật**. Khác biệt default vs freq là
**TẦN SUẤT** (1–2 vs 9–18), không phải zero:
- **default-nccl SỐNG** (72/72/71, chỉ **1** rearrange) trong khi **freq-nccl CRASH** (72/68/0, ~9 rearrange lặp
  lại → worker-stall).
- **default-nixl 55.9ms < freq sync-nixl 123.8ms**: default chỉ dính **1×38s** rearrange (→ variance ±21.2), freq
  dính **18×37s**.
Toàn bộ 4 config đã nằm ở `logs_result_default/`.

### 1b. MTP · DEFAULT · **async:true** (cùng window/step mặc định 1000/3000)
Rearrange chạy background thread (overlap compute) → kỳ vọng **ẩn** overhead 1–2 rearrange của default.

| # | config | imbalance | time-imbalance | bench (mean_tpot, completed) |
|---|---|---|---|---|
| 1 | **async-nixl** (+UCX+kv) | 1.341 | [time](logs_result_default/mtp5-eplb-nixl-asyncYes-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nixl-asyncYes-default-10k36/bench) → **45.2 ± 4.2 ms**, 72/72/72 ✓ |
| 2 | **async-nccl** | 1.322 | [time](logs_result_default/mtp5-eplb-nccl-asyncYes-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nccl-asyncYes-default-10k36/bench) → **48.4 ± 1.5 ms**, 72/72/72 ✓ |
| 3 | **async-pynccl** | 1.340 | [time](logs_result_default/mtp5-eplb-pynccl-asyncYes-default-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-pynccl-asyncYes-default-10k36/bench) → **48.3 ± 2.9 ms**, 72/72/72 ✓ |

→ **async ẩn được overhead rearrange của default** (đúng kỳ vọng), thắng cả sync-default. **Cả 3 communicator SỐNG**
(nixl 45.2 / nccl 48.4 / pynccl 48.3, đều 72/72/72, gần baseline 43.7):
- **nixl**: sync **55.9±21.2** → async **45.2±4.2** — 1 lần rearrange 38s giờ chạy nền, không block serving → nhanh
  hơn + hết variance, ≈ baseline.
- **nccl & pynccl**: async-default **SỐNG cả 2** (48.4±1.5, 48.3±2.9) trong khi **freq async-nccl/pynccl đều CRASH**
  (68/0/0, 36/0/0). Default thưa rearrange (1–2 lần) → async xử lý gọn, không worker-stall như freq (rearrange dày).
- ⇒ Nếu bắt buộc bật EPLB trên MTP5, **async + default** là cấu hình an toàn & rẻ nhất (≈ baseline, không crash, không
  variance). Vẫn không giảm imbalance (0 rearrange trong profile) — nhưng MTP5 vốn cân bằng nên không mất gì.
- "Pure EPLB" (chỉ `enable_eplb:true`) chính là async-default-nixl này (bản có UCX fix); placeholder crash no-fix
  giữ ở `logs_result_default/mtp5-eplb-nixl-asyncYes-default-10k36-nofix-crash`.

### 1c. Time-imbalance profile np=72 (scale=2) — folder `time_imbalance_2/`
Profile mặc định dùng **np=36** (½ bench) → tpot thấp giả tạo (run nhẹ tải, drain sớm). Chạy lại profile với
`PROFILE_NP_SCALE=2` (**np=72, khớp bench**) → lưu ở `time_imbalance_2/` cạnh `time_imbalance/` (np=36).

| config | imbalance np36 / np72 | profile-tpot np36 | profile-tpot **np72** | bench (np72, 3×) |
|---|---|---|---|---|
| baseline (off) | 1.322 / 1.359 | 25.3 ms | **56.8 ms** | 43.7 ± 7.3 ms |
| async-nixl | 1.341 / 1.340 | 22.3 ms | **47.0 ms** | 45.2 ms |
| async-nccl | 1.322 / 1.330 | 24.8 ms | **48.9 ms** | 48.4 ms |
| async-pynccl | 1.340 / 1.356 | 24.5 ms | **47.2 ms** | 48.3 ms |

→ 2 kết luận: (1) **imbalance độc lập np** (np36 ≈ np72, chênh ≤0.02 noise) → đo imbalance ở np nào cũng được.
(2) **np là biến chính của profile-tpot**: np36 cho ~½ bench (sai rõ rệt), np72 vào **đúng cỡ** bench. NHƯNG profile
là **1 run đơn + có torch-profiler** → còn sai số ~±10ms (baseline np72 **56.8** vs bench 43.7 = +13; async 47–49 vs
45–48 = +1–4). ⇒ np72 chỉ để **xác nhận np là thủ phạm** khiến np36 thấp giả tạo; **KHÔNG thay thế** bench 3× fair
(muốn số tpot chuẩn vẫn phải bench). Folder: `mtp5-base-…-off-10k36/time_imbalance_2` + 3 async `…/time_imbalance_2`.

### 1d. MTP · DEFAULT · async · interval NHỎ (window/step = 250/250) — `logs_result_default/…-s250-…`
Default 1000/3000 **quá to** cho run ngắn 10k_c36 (trigger lần 1 tận forward 750, profile 0 rearrange). Hạ xuống
**250/250** để rearrange **thực sự nổ** trong run ngắn → kiểm chứng: EPLB per-layer có giảm imbalance + có lợi tpot không.

| # | config (async, 250/250) | imbalance | rearrange (profile) | time-imbalance | bench (mean_tpot, completed) |
|---|---|---|---|---|---|
| 1 | **s250-nixl** | **1.306** | **1** (nổ ngay trong profile ngắn) | [time](logs_result_default/mtp5-eplb-nixl-asyncYes-s250-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nixl-asyncYes-s250-10k36/bench) → **47.2 ± 4.7 ms**, 72/72/72 ✓ |
| 2 | **s250-nccl** | 1.321 | 1 | [time](logs_result_default/mtp5-eplb-nccl-asyncYes-s250-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-nccl-asyncYes-s250-10k36/bench) → ✗ **CRASH 72/0/0** (worker-stall) |
| 3 | **s250-pynccl** | CRASH (profile) | — | [time](logs_result_default/mtp5-eplb-pynccl-asyncYes-s250-10k36/time_imbalance) | [bench](logs_result_default/mtp5-eplb-pynccl-asyncYes-s250-10k36/bench) → ✗ **CRASH 36/0/0** |

→ **2 kết luận:**
- **nixl** (so cùng communicator): imbalance **default 1.341 → s250 1.306 → freq 1.315**; tpot **default 45.2 → s250
  47.2 → freq 47.9**. Interval nhỏ (250) **có** làm EPLB rearrange (khác default=0) → **imbalance nhích xuống** (−2.6%),
  NHƯNG **tpot KHÔNG cải thiện** (47.2 > 45.2) — overhead rearrange > lợi ích cân bằng nhỏ (headroom 14% + comm-bound).
- **nccl & pynccl s250 vẫn CRASH** (72/0/0, 36/0/0, worker-stall) — **250 vẫn đủ dày** để sập như freq(100). Chỉ
  **default** (rearrange thưa ~1 lần) mới cho nccl/pynccl sống; **nixl-async** chịu được mọi interval.
⇒ Nghịch lý cốt lõi: **muốn EPLB thật sự hoạt động phải hạ interval → nhưng interval nhỏ thì nccl/pynccl tự sát,
còn nixl thì rearrange 37s (async ẩn được) — và dù chạy được (nixl) tpot vẫn không lời trên MTP5.**

---

## 2. NON-MTP · DEFAULT — `logs_result_default/`

| # | config | imbalance | rearrange | time-imbalance | bench (fair) |
|---|---|---|---|---|---|
| 1 | **baseline** (EPLB off) | 1.529 | — | [time](logs_result_default/nomtp-base-none-asyncNo-off-10k36/time_imbalance) | [bench-fair](logs_result_default/nomtp-base-none-asyncNo-off-10k36-fair/bench) → **51.4 ± 2.1 ms**, 72/72/72 ✓ |
| 2 | **default-nccl** | 1.502 | 0 | [time](logs_result_default/nomtp-eplb-nccl-asyncNo-default-10k36/time_imbalance) | none |
| 3 | **default-nixl** (pure→async) | CRASH (NIXL init) | — | [time](logs_result_default/nomtp-eplb-nixl-asyncYes-default-10k36/time_imbalance) | none |

→ default nccl noMTP: imbalance 1.529→1.502 (≈0, 0 rearrange). Chưa chạy bench default noMTP (ngoài baseline-fair).

---

## 3. MTP · FREQ — `logs_results/`
eplb_config = window100/step100 (rearrange dày). sync trừ khi ghi async.

| # | config | imbalance | rearrange ¹ | time-imbalance | bench (mean_tpot, completed) |
|---|---|---|---|---|---|
| 1 | **baseline** (off) | 1.322 | — | [time](logs_results/mtp5-base-none-asyncNo-off-10k36/time_imbalance) | [bench](logs_results/mtp5-base-none-asyncNo-off-10k36/bench) → **43.7 ± 7.3 ms**, 72/72/72 ✓ |
| 2 | **freq-nccl** (sync) | 1.284 | 4 | [time](logs_results/mtp5-eplb-nccl-asyncNo-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-nccl-asyncNo-freq-10k36/bench) → ✗ **CRASH 72/68/0** |
| 3 | **freq-pynccl** (sync) | 1.300 | 4 | [time](logs_results/mtp5-eplb-pynccl-asyncNo-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-pynccl-asyncNo-freq-10k36/bench) → **45.4 ± 7.4 ms**, 72/72/72 ✓ |
| 4 | **sync-nixl** (+UCX) | 1.305 | 4 | [time](logs_results/mtp5-eplb-nixl-asyncNo-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-nixl-asyncNo-freq-10k36/bench) → **123.8 ± 6.3 ms**, 72/72/72 ✓ (**CHẬM**: rearrange NIXL sync ~37s×18) |
| 5 | **async-nixl** (+UCX) | 1.315 | 1 | [time](logs_results/mtp5-eplb-nixl-asyncYes-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-nixl-asyncYes-freq-10k36/bench) → **47.9 ± 7.4 ms**, 72/72/72 ✓ |
| 6 | **async-nccl** | 1.341 | 1 | [time](logs_results/mtp5-eplb-nccl-asyncYes-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-nccl-asyncYes-freq-10k36/bench) → ✗ **CRASH 68/0/0** |
| 7 | **async-pynccl** | CRASH (profile) | 1 | [time](logs_results/mtp5-eplb-pynccl-asyncYes-freq-10k36/time_imbalance) | [bench](logs_results/mtp5-eplb-pynccl-asyncYes-freq-10k36/bench) → ✗ **CRASH 36/0/0** |

¹ rearrange THẬT trong profile (loại warmup); bench dài hơn → nhiều hơn (serve.log: freq-nccl 9, sync-nixl 18, async-nixl 3).

---

## 4. NON-MTP · FREQ — `logs_results/`

| # | config | imbalance | time-imbalance | bench (fair) |
|---|---|---|---|---|
| 1 | **baseline** (off) | 1.529 | [time](logs_results/nomtp-base-none-asyncNo-off-10k36/time_imbalance) | [bench-fair](logs_results/nomtp-base-none-asyncNo-off-10k36-fair/bench) → **51.4 ± 2.1 ms**, 72/72/72 ✓ |
| 2 | **freq-nccl** (sync) | 1.394 | [time](logs_results/nomtp-eplb-nccl-asyncNo-freq-10k36/time_imbalance) | none |
| 3 | **freq-pynccl** (sync) | 1.414 | [time](logs_results/nomtp-eplb-pynccl-asyncNo-freq-10k36/time_imbalance) | [bench-fair](logs_results/nomtp-eplb-pynccl-asyncNo-freq-10k36-fair/bench) → ✗ **CRASH 72/0/0** |
| 4 | **freq-gloo** (sync) | 1.424 | [time](logs_results/nomtp-eplb-gloo-asyncNo-freq-10k36/time_imbalance) | none |
| 5 | **sync-nixl-fix** (+UCX) | 1.418 | [time](logs_results/nomtp-eplb-nixl-asyncNo-freq-10k36-fix/time_imbalance) | none |
| 6 | **sync-nixl** (chưa fix) | CRASH (NIXL init) | [time](logs_results/nomtp-eplb-nixl-asyncNo-freq-10k36-crash/time_imbalance) | none |
| 7 | **async-nixl** (+UCX) | 1.497 | [time](logs_results/nomtp-eplb-nixl-asyncYes-freq-10k36/time_imbalance) | none (unfair có: 153.8ms, 72/72/72) |
| 8 | **step500-pynccl** (sync) | 1.462 | [time](logs_results/nomtp-eplb-pynccl-asyncNo-step500-10k36/time_imbalance) | none (unfair có: 134.3ms, 72/72/72) |
| 9 | **freq-red16** (+16 redundant) | CRASH (NCCL hang) | [time](logs_results/nomtp-eplb-nccl-asyncNo-freq-10k36-red16/time_imbalance) | none |

→ noMTP fair bench: **baseline chạy được (51.4)** nhưng **freq-pynccl CRASH ngay cả khi fair** (72/0/0) — sync-pynccl
kém ổn định trên noMTP (khác MTP5 nơi sync-pynccl sống). Các bench noMTP khác đều eager+EP_LOG → unfair → none.

Ghi chú sweep (khác 10k_c36): 8k/100k multi-conc ở `logs_results/*-8ksweep`, `*-100ksweep` — không đưa vào 4 bảng trên.

---

## Kết luận sơ bộ
- **default (nccl/pynccl/nixl)** [ĐO XONG]: profile 0 rearrange → imbalance = baseline (1.32–1.38, EPLB-default
  KHÔNG cân bằng thêm); bench **1–2 rearrange thật** → tpot hơi cao hơn baseline: nccl 50.4 / pynccl 47.2 / nixl
  55.9 vs 43.7 (gần noise). Cả 3 SỐNG. ⇒ default trả giá 1–2 rearrange mà gần như không được lợi ích cân bằng.
- **freq (step100)**: rearrange dày → imbalance giảm nhẹ (MTP5 1.322→1.28–1.31) nhưng **overhead rearrange lớn**;
  chỉ **pynccl** và **async-nixl** sống (45.4 / 47.9 ms), **sync-nixl chậm thảm** (123.8ms do 37s×18 rearrange),
  **nccl-sync + async-nccl + async-pynccl đều CRASH**.
- ⇒ **Giả thuyết của bạn ĐÚNG** (default rẻ & ổn định hơn freq) nhưng do **tần suất rearrange thấp** (1–2 vs
  9–18), KHÔNG phải do 0 rearrange: (a) **nccl** default SỐNG vs freq CRASH; (b) **nixl** default 55.9 vs
  freq-sync 123.8. Mặt trái: rearrange quá ít → **không giảm imbalance** (≈ baseline). Trên MTP5 imbalance vốn
  thấp nên EPLB (default lẫn freq) gần như vô ích.
- [ĐANG CHẠY] **default + async:true** (nixl→nccl→pynccl) — kiểm tra async có ẩn được overhead 1–2 rearrange của
  default không (async overlap rearrange vào background thread).
