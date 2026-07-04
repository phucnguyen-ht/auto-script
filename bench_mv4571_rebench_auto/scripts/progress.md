# PROGRESS — EPLB bench trên MI300 (GLM-5.2 1P1D, image 260626-rc1)

Mục tiêu: chạy các cấu hình EPLB cho base preset `presets/glm5.2.rebench/MTP5-bs64-dg.yaml`
trên **MI300 (192 GiB/GPU)**. Trước đó fix đã làm trên **MI325 (256 GiB/GPU)** — xem
`CONTEXT_HANDOFF.md` + `DEBUG_ASYNC_HANG.md`. Câu hỏi: fix đó có còn khả thi trên MI300
(ít VRAM hơn) không? Workload: 100K ISL / concurrency 64.

Cập nhật liên tục theo dạng: **Đã làm gì → gặp bug gì → hướng fix → log ở đâu → kết quả**.

---

## TÓM TẮT KẾT LUẬN (điền dần)
- Fix nccl/pynccl→SYNC: áp dụng được cho MI300 (đang verify throughput).
- **nixl: BẤT KHẢ KHÁNG trên MI300** — UCX init fix (+tcp) OK, nhưng async transfer đầu tiên
  làm EngineCore sập (node không có RDMA; data-plane GPU-mem qua tcp crash). → LOẠI nixl khỏi sweep MI300.
- **gloo: async transfer chạy được** trên MI300 (r16 probe: 21 transfers, không sập).
- **max_model_len phải hạ 1M→512K** cho mọi EPLB (EPLB ăn ~10 GiB KV → 1M không fit).

---

## 0. Bối cảnh & phân tích ban đầu (chưa chạy gì)

### Môi trường thực tế (đã kiểm tra)
- Node: MI300, `gfx942`, **8× GPU, mỗi GPU VRAM total = 206,141,652,992 B ≈ 192 GiB**
  (`rocm-smi --showmeminfo vram`). Lúc bắt đầu tất cả GPU rảnh (~297 MB/ GPU).
- Container: `phuc-nguyen-mv-4571` (docker, KHÔNG phải podman như handoff), image
  `moreh-vllm:0.23.0-260626-rc1` (GIỐNG image MI325 → nên bug deadlock async là giống nhau).
- Có container khác `anhcao-bench` cùng image đang chạy → cần theo dõi tranh chấp GPU.
- Repo mount tại cùng path trong container: `/home/phuc-nguyen/workspaces/mv-4571-rebench`.
- Bench: `bench_mv4571_rebench_auto/scripts/run_and_bench.sh` (serve→bench→extract→stop),
  sweep: `sweep_presets.sh`.

### Fix trong CONTEXT_HANDOFF (làm trên MI325) tóm tắt
Vấn đề gốc trên MI325 = **deadlock NCCL khi EPLB async với communicator nccl/pynccl**
(background thread issue NCCL collective out-of-order → `sample_tokens` RPC timeout 300s →
engine chết). Đây là **giới hạn framework vLLM**, không sửa được bằng config.
Giải pháp: matrix VIABLE =
- `nixl`  → ASYNC (UCX RDMA, ngoài NCCL group) + cần `kv_cache_memory_bytes≥54.62GiB` (đặt 60GiB) + UCX env
- `torch_gloo` → ASYNC (CPU-staged, cpu_group)
- `torch_nccl`, `pynccl` → chỉ SYNC (async của chúng deadlock — bất khả kháng)
× {default, s250} × {r0, r8, r16} = 24 preset. Đã generate đủ 24 file.

### Phân tích khả thi trên MI300 (suy luận trước khi chạy)
1. **Bug deadlock async là do FRAMEWORK + IMAGE, không phải VRAM.** Image trên MI300 =
   image trên MI325 (260626-rc1) → cơ chế deadlock nccl/pynccl-async y hệt → **fix chọn
   nixl/gloo async + nccl/pynccl sync vẫn đúng về mặt logic trên MI300.**
2. **Rủi ro MỚI của MI300 = OOM do VRAM ít hơn (192 vs 256 GiB).** Ước lượng/GPU với
   `gpu_memory_utilization=0.9` → ~172.8 GiB dùng được:
   - Model ~107.6 GiB (số từ handoff) + KV.
   - Baseline & gloo/nccl/pynccl: KV auto-size theo util → luôn "vừa" (vLLM tự co KV).
     → rủi ro thấp, TRỪ KHI r8/r16 (redundant experts) làm model phình to.
   - **nixl: KV CỐ ĐỊNH 60 GiB** (bắt buộc, để register UCX) → 107.6 + 60 = 167.6 GiB,
     chỉ còn ~5 GiB cho activation/cudagraph/UCX buffer → **nguy cơ OOM cao nhất trên MI300**.
   - **r16 (16 redundant experts)** cộng thêm bộ nhớ expert mỗi GPU → ép thêm → nguy cơ OOM.
3. **Giả thuyết cần chứng minh bằng số liệu:**
   - (a) baseline + gloo/nccl/pynccl async|sync r0 CHẠY ĐƯỢC trên MI300 (KV auto co lại).
   - (b) nixl 60GiB KV có thể OOM trên MI300 → nếu OOM thì hạ `kv_cache_memory_bytes`
     (kéo theo phải hạ `max_model_len` < 1M vì KV phải đủ cho ≥1 req ở max_model_len).
   - (c) r16 có thể OOM → nếu OOM thì đó là case bất khả kháng cho redundant cao, ghi rõ số.

### Kế hoạch test (mỗi run ~12–14 phút, 8 GPU dùng chung nên chạy tuần tự)
Thứ tự chọn để dò trần VRAM sớm:
- T1: baseline `MTP5-bs64-dg.yaml` (no EPLB) — xác nhận base vừa MI300 + số tham chiếu.
- T2: `nixl-async-default-r0` — case KV cố định 60GiB, nguy cơ OOM cao nhất.
- T3: `gloo-async-default-r0` — async không dùng KV cố định, đối chứng.
- T4: `pynccl-sync-default-r0` — xác nhận sync path.
- T5: `gloo-async-default-r16` — dò trần bộ nhớ redundant experts.

Log trung gian: `bench_mv4571_rebench_auto/logs/mi300_probe/<preset>/serve.log|scenario_summary.csv`.

---

## 1. Khai thác log sweep2 có sẵn (chạy TRÊN CHÍNH MI300 này lúc 17:27 hôm nay)
Không cần chạy lại, rút ra số liệu MI300 thực tế từ `logs/sweep2/20260702_172743`:

| case | kết quả | bằng chứng |
|---|---|---|
| baseline (no EPLB) | ✅ 510 req, p50_tpot **21.86**, decode_tps 45.1 | `MTP5-bs64-dg/scenario_summary.csv` |
| nccl-async-default-r0 | ❌ worker chết | `serve.log`: `Worker proc VllmWorker-0 died unexpectedly` (deadlock async, giống MI325) |
| pynccl-async-default-r0 | ❌ worker chết | như trên |
| nixl-async-default-r0 (preset CŨ) | ❌ UCX backend fail | `serve.log`: `no active messages transport ... NIXL_ERR_BACKEND` |

**Số liệu VRAM MI300 (từ baseline serve.log):**
- Model load: **107.63 GiB/GPU**. `gpu_memory_utilization=0.9` → ~172.8 GiB dùng được.
- **Available KV cache memory: 59.1 GiB/GPU** (auto). GPU KV size 1,134,528 tokens.
- **Max concurrency cho 1M tokens/req = 1.08x** → base chỉ vừa đủ 1 request 1M ctx.
  → Overhead ngoài model+KV ≈ 172.8 − 107.63 − 59.1 = **6.07 GiB**.

### Chẩn đoán các lỗi MI300 (khác MI325 ở đâu)
1. **nccl/pynccl async chết** = ĐÚNG như MI325 (cùng image 260626, deadlock NCCL framework)
   → fix "nccl/pynccl → SYNC" **áp dụng nguyên vẹn cho MI300**. (sẽ verify 1 case)
2. **nixl fail = LỖI MỚI CỦA MI300, không phải KV/OOM như MI325.** Root cause:
   - `ibv_devices` **rỗng** → node MI300 (tw031) KHÔNG có NIC RDMA/InfiniBand.
   - `ucx_info -d`: transports khả dụng = `self, tcp(eno0/lo), sysv, posix` (+rocm_copy/ipc). KHÔNG có `rc_x`.
   - UCX_TLS trong gen hiện tại = `self,sm,rc_x,rocm_copy,rocm_ipc` → chứa `rc_x` (không tồn tại)
     và **thiếu `tcp`** → không transport nào cung cấp `am bcopy` cho intra-agent setup → `NIXL_ERR_BACKEND`.
   - **Hướng fix nixl trên MI300**: thêm `tcp` vào UCX_TLS (bỏ/giữ rc_x đều được, UCX tự skip cái thiếu).
   - **KV nixl trên MI300**: KV cố định 60 GiB > available 59.1 GiB → sẽ bị từ chối lúc init.
     Phải hạ `kv_cache_memory_bytes` xuống trong khoảng [54.62 (đủ 1M ctx), 59.1 (available)].
     Chọn **57 GiB** (margin 2 GiB dưới trần, vẫn đủ 1M ctx ~1.04x). Sẽ verify.
3. **gloo async CHƯA từng test trên MI300** (sweep2 dùng matrix cũ không có gloo). Phải verify.
4. **r8/r16 (redundant experts)** tăng model memory → với KV auto vLLM sẽ tự co KV; rủi ro chỉ khi
   model+overhead vượt trần. Phải đo r16 thực tế.

---

## 3. KẾT QUẢ P1 (gloo-async-r0) — LỖI KV, phát hiện case bất khả kháng cho max_model_len=1M

**Đã làm:** chạy `gloo-async-default-r0` (util 0.9, max_model_len mặc định = 1M).
Log: `logs/mi300_probe/gloo-async-default-r0/serve.log`.

**Bug:** engine init FAIL ở KV feasibility check (KHÔNG phải deadlock, KHÔNG phải UCX):
```
ValueError: To serve at least one request with the model's max seq len (1048576),
54.62 GiB KV cache is needed, which is larger than the available KV cache memory
(48.82–49.66 GiB per rank). Estimated maximum model length ≈ 937088.
```
- Model load gloo = **108.76 GiB** (baseline 107.63; EPLB thêm ~1.1 GiB weights).
- **Available KV khi BẬT EPLB (r0) = ~49 GiB** vs baseline **59.1 GiB** → EPLB machinery
  (rearrange working buffers) ăn thêm **~10 GiB/GPU**. 49 < 54.62 cần cho 1M → init fail.

**Kết luận (số liệu rõ ràng):** trên MI300 (192 GiB), **không thể giữ max_model_len=1M khi bật EPLB**.
Baseline (no EPLB) giữ 1M được vì có đủ 59.1 GiB KV; bật EPLB tụt còn ~49 GiB < 54.62 GiB.
Đây đúng là case user dự đoán. Hướng xử lý (chọn có căn cứ):
- **KHÔNG** tăng `gpu_memory_utilization` lên 0.95: MI300 overhead ngoài model+KV chỉ ~6 GiB;
  đẩy util cao dễ OOM lúc prefill 100K×conc64 (activation spike) → rủi ro sập giữa bench.
- **CHỌN: hạ `max_model_len`** cho các preset EPLB, giữ `util=0.9` (parity với baseline).
  Vì workload thực = 100K ISL ≪ max_model_len → với 100K request, hành vi paging KV y hệt
  dù max_model_len là 1M hay 512K → **kết quả EPLB vẫn so sánh được với baseline** (chỉ khác
  đúng biến EPLB on/off). max_model_len chỉ ảnh hưởng startup feasibility + trần 1 request.
- Phải chọn 1 giá trị max_model_len fit cả case NẶNG NHẤT (r16, ít KV nhất). Đang đo r16.

## 4. Đo trần VRAM r16 (case nặng nhất) — probe-gloo-r16-mml131072

**Đã làm:** chạy 1-off preset `_probe-gloo-r16-mml131072.yaml` (gloo r16, max_model_len=131072
để CHẮC CHẮN fit → đọc được available KV). Log: `logs/mi300_probe/probe-gloo-r16-mml131072/`.

**Kết quả:** model load **114.17 GiB** (r16 = 16 redundant experts = 2/rank thêm ~5.4 GiB so r0),
GPU KV size **826,944 tokens ≈ 43.1 GiB available**, max concurrency cho 131072 = 6.31x → serve OK, đang bench.

**Bảng trần KV theo cấu hình EPLB (MI300, util 0.9):**
| config | model (GiB) | avail KV (GiB) | max single-req len |
|---|---|---|---|
| baseline (no EPLB) | 107.63 | 59.1 | 1,048,576 (1.08x) |
| EPLB r0 | 108.76 | ~49 | ~940K |
| EPLB r16 | 114.17 | 43.1 | ~827K |

### CHỐT giá trị cho preset MI300 (căn cứ số liệu trên)
- **`max_model_len = 524288` (512K)** cho MỌI preset EPLB (giữ baseline 1M). 512K cần 27.3 GiB KV
  < 43.1 (r16) → fit tất cả, margin lớn. Workload 100K ≪ 512K → không ảnh hưởng kết quả bench.
- **nixl `kv_cache_memory_bytes = 40 GiB`** (không thể 57/60 GiB): KV nixl là CỐ ĐỊNH, phải fit
  case nặng nhất r16 (avail 43.1). 40 GiB < 43.1 → fit r0/r8/r16; ≥ 27.3 cần cho 512K.
  (Lưu ý: 40 GiB = 768K tokens; peak workload conc8×~100.5K ≈ 804K/rank → nixl có thể preempt
   nhẹ ~4% so với gloo auto 49 GiB — đánh đổi bắt buộc do KV cố định trên VRAM hẹp; sẽ đo thực tế.)
- **nixl UCX_TLS thêm `tcp`** (node không có RDMA).

## 6. KẾT QUẢ nixl-r0 @ 512K (KV 40GiB, UCX+tcp) — UCX OK nhưng CRASH khi transfer

**Đã làm:** chạy `nixl-async-default-r0` với preset đã fix (UCX_TLS +tcp, KV=40GiB, 512K).
Log: `logs/mi300_probe/nixl-async-default-r0/`.

**Diễn biến (theo timestamp serve.log):**
1. `Initialized NIXL agent: eplb-0..7` trên cả 8 rank — **UCX backend init THÀNH CÔNG** (fix +tcp OK,
   không còn NIXL_ERR_BACKEND như sweep2). KV 40GiB fit (không OOM init).
2. Serve OK, bench chạy, decode batch tăng bình thường 5→13→20 (`decode_running` totals).
3. **21:43:25** cả 8 rank: `async_worker.py:87 async worker woke up for EPLB transfer` (rearrange đầu tiên).
4. **21:43:38** (13s sau): `EngineDeadError: EngineCore encountered an issue` → toàn bộ ApiServer chết.
   KHÔNG có Python traceback từ Worker → crash cứng (segfault/HIP fault) ở data-plane transfer.

**Kết quả bench:** 0 request hoàn thành (engine chết giữa chừng).

**Chẩn đoán:** nixl = **receiver-initiated RDMA READ** bộ nhớ GPU remote. Node MI300 (tw031)
**không có NIC RDMA** (`ibv_devices` rỗng). UCX **control-plane** (active message) chạy được qua tcp
→ init OK. Nhưng **data-plane** (đọc khối lớn GPU/rocm memory của rank khác) qua tcp/rocm_ipc **crash**
ngay lần transfer đầu. Đối chứng: **gloo (CPU-staged) fire 21 transfers KHÔNG sập** ở r16 probe.

**KẾT LUẬN (bất khả kháng, có dẫn chứng):** nixl async KHÔNG khả thi trên node MI300 này vì thiếu RDMA.
Không phải lỗi config KV/UCX (đã fix init thành công) mà là giới hạn phần cứng (không RDMA) + đường
GPU-mem-over-tcp của UCX. → **LOẠI nixl khỏi sweep MI300**; dùng **gloo** làm async backend, cùng
nccl/pynccl SYNC. (Trên MI325 có RDMA nên nixl chạy — khác biệt phần cứng node, không sửa bằng preset được.)

## 7. CHẶN: node dùng chung bị user khác chiếm GPU (contention)
**Đã làm:** chạy `gloo-async-default-r0` @ 512K. **Bug:** engine kẹt ở CUDA graph capture
**~102 giây/graph** (25/32 sau 43 phút; bình thường ~4-5s/graph) → không kịp health → timeout.
Log: `logs/mi300_probe/gloo-async-default-r0-512k/serve.log`.
**Nguyên nhân (KHÔNG phải lỗi gloo):** lúc 21:47 container **`thanh-nguyenxuan-glm5-dev`** (91 proc
`VLLM::Worker_DP`) khởi động job 8-GPU, chiếm **92% cả 8 GPU** + CPU ~180%/proc → starve job của tôi.
`rocm-smi` = `92 92 93 93 92 92 92 92`. KV của tôi vẫn OK (49.66 GiB, 512K→1.79x) — chỉ là bị tranh chấp.
**Xử lý:** node dùng chung → phải CHỜ GPU rảnh (<~10% VRAM) rồi mới test lại gloo + launch sweep.
`sweep_presets.sh` đã có `wait_gpu_free` nên sweep sẽ tự chờ. (nixl crash lúc 21:43 xảy ra TRƯỚC 21:47
nên kết luận "nixl bất khả kháng" không bị ảnh hưởng bởi contention.)

## 8. nixl auto-check (theo yêu cầu user) + HANDOFF sang session mới

**Hiểu đúng:** nixl khả thi hay không **tùy PHẦN CỨNG node** (cần RDMA NIC). Máy khác của user có
RDMA → nixl chạy được. Node MI300 này không có RDMA → nixl crash. Nên KHÔNG hardcode bỏ nixl mà
**tự động check**.

**Đã thêm:** `scripts/check_nixl.sh` → exit 0 nếu node có RDMA (`ibv_devices` hoặc UCX rc/dc/ud), exit 1 nếu không.
`sweep_presets.sh` gọi nó và **tự thêm 6 preset nixl CHỈ KHI node có RDMA**; node này skip. Ép: `NIXL_FORCE=1/0`.
→ Cùng 1 script `sweep_presets.sh` chạy được cả máy có RDMA (có nixl) lẫn máy không (bỏ nixl), không cần sửa tay.
Test trên node này: `ibv_devices=0 ucx_rdma_transports=0 → NOT viable → skip` (đúng).

**Trạng thái bàn giao (đọc CONTEXT_HANDOFF.md §"MI300 HANDOFF"):**
- ✅ Đã xong: chẩn đoán, bộ preset (gen_eplb_presets.sh: 512K + nixl KV 40GiB + UCX tcp), sweep_presets.sh
  (19 preset → `logs/sweep_results/`, tự thêm nixl nếu RDMA), check_nixl.sh, docs.
- ⏳ CHẶN: GPU đang bị đồng nghiệp chiếm 92% → CHƯA có run throughput SẠCH.
- TODO khi GPU rảnh: (1) confirm 1 run gloo-r0 + 1 run pynccl-sync-r0 sạch, (2) launch full sweep detached
  ra `logs/sweep_results/`. Lệnh cụ thể trong CONTEXT_HANDOFF.md §"STATE / WHAT'S LEFT".

## 9. CHỐT CUỐI + LAUNCH SWEEP (session 2)
- **nixl xác nhận KHÔNG dùng được** trên node này (và cả máy cũ của user). Lưu ý: `nixl_probe.py`
  register 1 buffer 768MiB qua rocm_ipc thì OK (REGISTER_OK), NHƯNG đường EPLB transfer thật vẫn
  crash → **probe là false-positive, không dùng làm gate**. Giữ `check_nixl.sh` bản RDMA (node này
  trả not-viable → khối nixl có điều kiện tự bị bỏ).
- **Thêm biến thể s500** (window/step=500) vào `gen_eplb_presets.sh`, đã regenerate (12 file s500).
- `sweep_presets.sh`: list do user curate (baseline + gloo/nccl-sync/pynccl-sync × {default,s250,s500}
  + vài dòng nixl user để lại — sẽ fail & move on). Đã validate MỌI file preset tồn tại.
- **Đã launch sweep** detached: `logs/sweep_results/<TS>/`, log `<TS>.sweep.log`. Watcher chờ `[sweep] DONE`.
- Lưu ý vận hành: container `phuc-nguyen-mv-4571` từng tự Exit(137) 2 lần (không OOM) — nếu sweep
  gián đoạn, `docker start` lại rồi relaunch. Tránh `pkill -f VLLM` rộng (khớp cả shell exec → exit137).

## 5. Kế hoạch verify còn lại (sau khi có preset chốt)
Chạy tuần tự vào `logs/mi300_probe/`:
- [đang chạy] gloo-r16 (131072) — xác nhận EPLB r16 bench được trên MI300.
- P3: `nixl-async-default-r0` (UCX+tcp, KV=40GiB, 512K) — **gate: nixl có init UCX + bench được không.**
- P1b: `gloo-async-default-r0` (512K) — xác nhận gloo bench được (lần trước fail do 1M).
- P2: `pynccl-sync-default-r0` (512K) — xác nhận fix SYNC.
- Nếu nixl-r0 OK → thử `nixl-async-default-r16` (KV cố định 40GiB + r16, chặt nhất) để chắc.

---
