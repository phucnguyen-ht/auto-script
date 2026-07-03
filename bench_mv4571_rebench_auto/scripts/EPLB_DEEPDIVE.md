# EPLB DEEP-DIVE — Communicator deadlock, flow & shapes (GLM-5.2, mv-4571)

> Tài liệu giải thích **line-by-line** cơ chế EPLB của vLLM (image `moreh-vllm:0.23.0-260626-rc1`,
> vllm `0.23.1.dev0+g0fc695fc6`), tập trung vào **THE BIG PROBLEM** (async communicator deadlock)
> và **flow chạy EPLB** kèm shape. Đọc cùng: [`CONTEXT_HANDOFF.md`](CONTEXT_HANDOFF.md),
> [`DEBUG_ASYNC_HANG.md`](DEBUG_ASYNC_HANG.md), [`progress.md`](progress.md).

## 0. Source ở đâu & cách sửa live

Source EPLB **đã copy từ trong container ra host** (bằng `kubectl cp` → file thuộc `phuc-nguyen`, sửa được).
Đặt **NGOÀI** `mv-4571` tại `/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571-vllm-src/`, và có **symlink**
`auto-script/bench_mv4571_rebench_auto/vllm_src → mv-4571-vllm-src` để thấy/click trong IDE (mọi link dưới đây
dùng `../vllm_src/...`).

```
vllm_src/vllm/
  distributed/eplb/{eplb_communicator,async_worker,eplb_state,rebalance_execute,eplb_utils}.py   # LÕI EPLB
  distributed/eplb/policy/{__init__,abstract,default}.py                                          # thuật toán
  distributed/parallel_state.py         # get_eplb_group / get_ep_group / _EPLB group riêng  ★
  distributed/nixl_utils.py             # NixlWrapper (RDMA)
  distributed/stateless_coordinator.py  # StatelessGroupCoordinator (elastic EP)
  distributed/utils.py                  # is_weak_contiguous, StatelessProcessGroup
  distributed/device_communicators/pynccl.py, pynccl_wrapper.py   # PyNCCL send/recv
  distributed/elastic_ep/elastic_execute.py
  config/parallel.py                    # ParallelConfig + EPLBConfig (use_async, communicator, step_interval…)
  model_executor/models/interfaces.py   # MixtureOfExperts (expert_weights, num_moe_layers…)
  model_executor/layers/fused_moe/layer.py
  v1/executor/multiproc_executor.py     # RPC timeout (EngineDead)
  v1/worker/gpu_model_runner.py, gpu/model_runner.py, gpu/eplb_utils.py   # nơi gọi eplb_step()
  envs.py                               # VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
```
> Đây là **dependency-closure trực tiếp** của module EPLB (mọi `from vllm… import` trong `distributed/eplb/*.py`,
> trừ hạ tầng chung `logger`/`platforms`). Muốn thêm file nào vào diện copy: `kubectl cp` như các file trên.

**5 file EPLB được overlay-mount live vào container** (sửa host ⇒ container thấy ở lần `vllm-moreh serve` kế tiếp),
xem `mv-valhalla-cluster/dev/setup_dev_pod.sh` (mảng `EDITABLE_FILES`): `eplb_communicator.py`,
`async_worker.py`, `eplb_state.py`, `rebalance_execute.py`, `eplb_utils.py`. Đã verify: sửa host → Python trong
container import bản mới (không dính `.pyc` cũ). Recreate pod: `bash mv-valhalla-cluster/dev/k.sh down && bash …/k.sh up`.

---

## A. NỀN TẢNG — cho người "mù tịt" về EPLB & GPU-communication (đọc trước §1)

> Mục tiêu: đọc xong hiểu §1 mà không cần biết trước gì. Dùng nhiều **ví dụ + loại suy**.

### A.1. MoE và Expert Parallelism (EP) — bài toán gốc
- **MoE (Mixture-of-Experts):** mỗi MoE layer có **nhiều "expert"** (mỗi expert = 1 FFN nhỏ), nhưng **mỗi token
  chỉ đi qua vài expert**. GLM-5.2: 256 expert/layer, router chọn **top-8** expert hợp nhất cho mỗi token.
  ```
  token "cat" --router--> [E3, E17, E42, … 8 cái] --> chạy 8 expert --> cộng lại
  ```
- **Vì sao chia expert ra 8 GPU (Expert Parallelism):** 256 expert × 75 layer × ~36 MB = quá lớn cho 1 GPU →
  **rải ra 8 GPU**, mỗi GPU giữ 256/8 = **32 expert**. Token cần expert ở GPU khác → phải **trao đổi qua GPU** (all-to-all).
- **Lệch tải (load imbalance):** router chọn theo nội dung → có expert "hot" (nhiều token) và "cold". Nếu nhiều
  expert hot dồn 1 GPU → **GPU đó nghẽn, GPU khác rảnh** → cả cụm chờ theo GPU chậm nhất.
  ```
  GPU0: [E3🔥 E17🔥 E42🔥 …]  ← 80% token dồn vào → nghẽn
  GPU1: [E1  E5   E9  …]       ← rảnh
  ```

### A.2. EPLB (Expert-Parallel Load Balancing) — nó làm gì
Vòng lặp 3 bước, chạy định kỳ:
```
(1) ĐO   tải: đếm mỗi expert nhận bao nhiêu token (trong 1 cửa sổ vài trăm bước)
(2) TÍNH lại: thuật toán xếp lại chỗ — expert hot thì NHÂN BẢN / trải đều ra các GPU
(3) KHUÂN:    di chuyển WEIGHT expert giữa các GPU cho khớp sơ đồ mới  ← tốn kém + gốc của BIG PROBLEM
```
**Loại suy:** như siêu thị điều phối **quầy thu ngân**. Quầy đông (expert hot) → mở thêm quầy giống hệt / dời khách
sang quầy vắng. Bước (3) = **bê cái máy tính tiền (~36 MB/expert) từ quầy này sang quầy kia** qua "đường truyền".

**Ví dụ số nhỏ** (6 physical / 4 logical expert / 3 GPU — theo docstring `eplb_state.py`):
```
physical_to_logical_map = "slot vật lý nào đang giữ expert-logic nào"
       GPU:   G0       G1       G2
       slot: [0,1] |  [2,3] |  [4,5]
TRƯỚC: giữ:  [0,1]    [2,3]    [0,1]     ← expert 0,1 đang có 2 bản (đang hot)
(đo thấy expert-3 nóng lên → policy tính map mới, thêm bản expert-3)
SAU:   giữ:  [0,3]    [2,3]    [0,1]     ← G0 slot1 đổi 1→3 ⇒ phải NHẬN weight expert-3 từ G1 qua mạng
```

**Code cấp cao (ai gọi ai):**
```
mỗi forward:  gpu_model_runner.eplb_step()               # gpu_model_runner.py:4565
                └─ eplb_state.step()                     # eplb_state.py:474   (đếm bước)
                     └─ đủ interval → rearrange()          # eplb_state.py:604   bước (2)+(3)
                          ├─ all_reduce tải toàn cục       # :718  (gộp tải mọi GPU)
                          ├─ policy.rebalance_experts()    # :756  tính map mới
                          └─ sync:  rearrange_expert_weights_inplace()  # :766  KHUÂN ngay
                             async: rearrange_event.record()           # :808  báo THREAD PHỤ khuân
```
(Chi tiết line-by-line + shape ở §2.)

### A.3. "KHUÂN weight" = truyền dữ liệu qua liên kết giữa GPU
Weight 1 expert ~36 MB (FP8). 1 lần rearrange có thể khuân hàng chục–trăm expert × 75 layer ⇒ **hàng GB** đi qua
liên kết GPU↔GPU. Có **nhiều "con đường" (transport)** để truyền — chính chỗ này đẻ ra 4 communicator và các từ
NIXL / RDMA / UCX / gloo dưới đây.

### A.4. Từ điển thuật ngữ (kèm loại suy)
Mỗi **rank** = 1 tiến trình giữ 1 GPU. Chúng trao đổi qua **NVLink/xGMI** (trong 1 máy, rất nhanh) hoặc **NIC mạng**
(giữa máy). Cần 1 thư viện truyền thông lo việc "GPU0 gửi tensor cho GPU1".

| Term | Là gì / để làm gì | Loại suy |
|---|---|---|
| **rank / process group (communicator)** | 1 rank = 1 process + 1 GPU. Process group = **nhóm rank đã bắt tay** để cùng truyền. vLLM tạo nhiều nhóm: `EP` (forward), **`EPLB` riêng** (khuân expert). | Nhóm chat riêng — chỉ người trong nhóm nhận tin nhóm đó. |
| **NCCL** (AMD = RCCL) | Thư viện cho GPU truyền **thẳng trên GPU** (nhanh nhất). 2 kiểu: **collective** (mọi rank cùng 1 phép, vd `all_reduce`=cộng gộp) và **P2P** (send/recv 1-1). **Bắt buộc: mọi rank gọi các phép theo CÙNG THỨ TỰ**, sai thứ tự → treo. | Đội kéo co: cả đội phải "1-2-3 kéo" đúng nhịp; 1 người lệch nhịp → cả đội khựng. |
| **collective vs P2P** | collective = tất cả tham gia (all-reduce/all-gather); P2P = 1 gửi–1 nhận. Khuân expert = P2P; đo tải toàn cục = collective. | Họp toàn cty (collective) vs nhắn tin 1-1 (P2P). |
| **SM (Streaming Multiprocessor)** | "Nhân" tính toán của GPU. Kernel NCCL khi chạy **chiếm SM** và **đứng chờ** đủ rank. | Làn cao tốc; 1 đoàn xe (kernel) chiếm làn rồi dừng chờ đoàn kia. |
| **RDMA** (Remote Direct Memory Access) | NIC đặc biệt (InfiniBand/RoCE, vd Mellanox `mlx5`) cho **máy A đọc/ghi thẳng RAM/VRAM máy B** mà **không phiền CPU** máy B → nhanh, trễ thấp. **Phải có NIC RDMA** mới dùng được. | Có **chìa khoá kho** hàng xóm → tự vào lấy đồ, khỏi gọi chủ nhà. Không chìa → phải nhờ người (chậm/hỏng). |
| **UCX** | Thư viện **tự chọn đường truyền**. Các "transport": `rc_x`(=RDMA), `tcp`(mạng thường), `sm/posix`(chung máy). `UCX_TLS`=danh sách đường được phép. | App bản đồ: tự chọn máy bay (RDMA) / xe khách (tcp) tuỳ cái nào có. |
| **NIXL** | Lib của vLLM **bọc UCX**, truyền khối lớn (KV, weight expert) qua **RDMA READ**: bên **NHẬN** chủ động "đọc" thẳng VRAM bên GỬI. **Không qua NCCL, không chiếm SM.** Cần RDMA. | Dùng "chìa khoá kho" (RDMA) tự sang bê máy tính tiền quầy kia — khỏi nhờ nhân viên (CPU/NCCL). |
| **gloo / CPU-staging** | `gloo` = truyền **trên CPU + TCP** (không dùng GPU). CPU-staging = **copy VRAM→RAM (D2H) → gửi TCP → copy RAM→VRAM (H2D)**. Chậm hơn nhưng **không đụng NCCL/SM** ⇒ an toàn khi chạy song song. | Chuyển đồ qua **kho trung gian ngoài đường** (RAM/TCP) thay vì đường hầm nội bộ (NVLink) đang kẹt. |
| **KV cache** | Bộ đệm attention (mỗi token lưu key/value); ăn phần lớn VRAM còn lại sau khi nạp model. EPLB ăn bớt VRAM ⇒ KV còn ít (xem §3). | Sổ ghi hội thoại: càng dài càng tốn giấy (VRAM). |
| **sync vs async EPLB** | **sync**: main-thread **dừng** decode để khuân (mọi rank cùng lúc). **async**: **thread phụ** khuân **song song** khi main-thread vẫn decode → không khựng. | sync = đóng quầy 1 phút để dời máy; async = dời máy trong lúc vẫn bán (dễ va chạm). |
| **blocking rendezvous / deadlock** | Nhiều phép GPU-comm là "**hẹn gặp**": kernel đứng chờ tới khi **đủ mọi rank** cùng vào. **Deadlock** = A chờ B ở điểm X, B chờ A ở điểm Y → treo vĩnh viễn. | 2 người hẹn nhau: A đứng cổng trước, B đứng cổng sau, ai cũng chờ người kia → chờ mãi. |

> **3 ý cần nhớ để đọc §1:** (a) khuân expert = truyền dữ liệu giữa GPU; (b) **NCCL** nhanh nhưng **bắt mọi rank
> cùng thứ tự**, còn **gloo/RDMA thì không cần** → hợp chạy nền; (c) **async = thread phụ chạy song song** → nếu
> thread phụ dùng **NCCL** thì đâm vào NCCL của main-thread → **deadlock** (đó chính là BIG PROBLEM ở §1).

### A.5. Trường hợp THỰC TẾ của deployment này: **1 node, 1 CPU, 8 GPU, 8 tiến trình**

> Ở trên tôi hay nói "giữa 2 máy" cho dễ hình dung — nhưng mv-4571 chạy **1 máy (mi300-7), 8 GPU, 8 tiến trình
> (dp8/ep8)**, mọi thứ **trong CÙNG 1 node**. "rank↔rank" ở đây = **giữa 2 GPU / 2 tiến trình trên cùng máy**,
> KHÔNG phải 2 máy. Các "con đường truyền" đổi tương ứng (số liệu đo thực trên mi300-7):
>
> - Topology: 8 GPU nối **full-mesh bằng xGMI (Infinity Fabric)** — `rocm-smi --showtopotype` = `XGMI` mọi cặp.
> - UCX transports khả dụng (`ucx_info -d`): `self, tcp(eth0/lo), sysv/posix(shared-mem), cma, rocm_copy,
>   rocm_ipc`, **và** `rc_verbs/rc_mlx5/ud/dc_mlx5` (RDMA-verbs qua NIC `mlx5_0`).

| communicator | Đường đi INTRA-NODE (1 máy, 8 GPU) | NIC mlx5 (RDMA-verbs) có dùng? |
|---|---|---|
| torch_nccl / pynccl | NCCL/RCCL kernel chạy **thẳng trên GPU**, truyền qua **xGMI** giữa 8 GPU (peer-to-peer trên chip) | Không — intra-node NCCL đi xGMI |
| **nixl** | UCX chọn **`rocm_ipc`**: tiến trình rank r lấy **IPC handle** tới VRAM rank khác rồi **đọc thẳng qua xGMI**. Đây mới là "RDMA READ" thật ở đây. | **Không** — NIC chỉ dùng khi EP trải **nhiều node** |
| torch_gloo | copy VRAM→RAM (D2H) → gloo gửi qua **TCP loopback (`lo`) giữa 8 tiến trình** → RAM→VRAM (H2D) | Không |

**Đính chính hiểu lầm phổ biến:**
1. **"RDMA" ≠ bắt buộc phải có card mạng.** RDMA nghĩa rộng = "đọc/ghi thẳng bộ nhớ bên kia, khỏi làm phiền CPU
   bên đó". **Intra-node**, vai trò đó do **`rocm_ipc` + xGMI** đảm nhiệm (1 GPU đọc VRAM GPU khác qua Infinity
   Fabric). NIC InfiniBand chỉ vào cuộc khi **liên node**.
2. Do đó trên deployment 1-node này, **nixl khả thi hay không phụ thuộc `rocm_ipc`/xGMI-P2P**, KHÔNG phải card
   `mlx5`. mi300-7 có đủ cả (`rocm_ipc` + full xGMI) → nixl có đường intra-node hợp lệ (run #2 đang verify).
   ⇒ tinh chỉnh lại kết luận cũ: "MI300 có RDMA nên nixl chạy" — đúng hơn là **"nixl chạy nhờ rocm_ipc/xGMI"**.
3. **Deadlock nccl/pynccl async cũng là chuyện intra-node**: 8 tiến trình trên 8 GPU cùng máy, mỗi tiến trình có
   2 luồng (main + async) cùng bơm NCCL kernel lên GPU của nó → thứ tự rendezvous giữa 8 GPU lệch nhau → treo.
   **Không cần "2 máy" mới deadlock.**
4. **Khi nào NIC mlx5/RDMA-verbs mới thật sự quan trọng?** Khi **EP trải trên >1 node** (multi-node serving):
   nixl/UCX dùng `rc_mlx5` (RDMA qua InfiniBand) để đọc VRAM GPU **máy khác**; gloo dùng TCP qua mạng thật thay
   vì loopback; NCCL dùng NIC thay vì xGMI cho phần liên-node.

**Ví dụ CỤ THỂ — deadlock intra-node (8 GPU / 1 node, async + `nccl`):** đúng lúc EPLB rearrange:
- **MAIN** mọi rank đang decode: cuối mỗi MoE layer chạy `all_reduce` (NCCL, trên EP group).
- **ASYNC** mọi rank đang khuân expert: chạy P2P `isend/irecv` (NCCL, trên EPLB group).

Thu nhỏ còn 2 GPU cho dễ nhìn (thực tế 8):
```
thời điểm T:
 GPU0 (rank0):  MAIN  launch all_reduce#5     → GPU0 chạy kernel, ĐỨNG CHỜ rank1 cùng vào all_reduce#5
 GPU1 (rank1):  ASYNC launch P2P recv←rank0   → GPU1 chạy kernel, ĐỨNG CHỜ rank0 gửi P2P
   (rank1.MAIN chưa tới all_reduce#5 vì luồng ASYNC của nó được OS xếp chạy TRƯỚC)
   (rank0.ASYNC chưa gửi P2P vì luồng MAIN của nó đang kẹt trong all_reduce#5)
kết quả:  rank0 chờ rank1 (ở all_reduce) ┐
          rank1 chờ rank0 (ở P2P)        ┘ → VÒNG CHỜ CHÉO → cả 2 GPU treo vĩnh viễn
```
Nguyên nhân: **thứ tự launch NCCL KHÁC nhau giữa các rank** (rank0: all_reduce trước; rank1: P2P trước) — do 2 luồng
độc lập, OS/GIL xếp lịch phi tất định. NCCL kernel **chiếm SM và block tới khi đủ peer**, nên vòng chờ không bao giờ
gỡ. MAIN kẹt ⇒ `execute_model`/`sample_tokens` RPC không trả ⇒ 300s timeout ⇒ engine chết. **Tất cả trên 1 máy** —
không liên quan card mạng. (`gloo`/`nixl` không post NCCL kernel nên ASYNC không tạo được vòng chờ này → an toàn.)

**Ví dụ khi NIC `mlx5` MỚI vào cuộc (multi-node):** EP trải 2 node — rank0-3 ở node A, rank4-7 ở node B. Khi rank2
(node A, dùng nixl) cần đọc weight expert của rank5 (node B): dữ liệu **vượt mạng** → UCX chọn `rc_mlx5` (RDMA qua
InfiniBand `mlx5_0`) đọc thẳng VRAM GPU node B. Các hop **trong cùng node** vẫn đi xGMI/`rocm_ipc`. gloo lúc đó gửi
TCP qua mạng thật; NCCL dùng NIC IB cho phần liên-node. ⇒ NIC chỉ "nóng" khi có hop **cross-node**.

**Loại suy cập nhật:** thay vì "kho nhà hàng xóm" (2 máy), intra-node giống **8 quầy trong CÙNG 1 siêu thị** nối
bằng băng chuyền nội bộ (xGMI): **nixl** = quầy này thò tay qua băng chuyền lấy đồ quầy kia (`rocm_ipc`);
**gloo** = mang đồ ra kho tạm ngoài hành lang (RAM) rồi đẩy xe sang (TCP loopback); **NCCL** = cả 8 quầy đồng
diễn theo nhịp chung trên băng chuyền (sai nhịp → kẹt).

---

## 1. THE BIG PROBLEM (goal) + resolution

### 1.1. EPLB async định làm gì
MoE routing lệch tải: vài expert "nóng" bị nhiều token dồn vào → GPU chứa expert đó nghẽn, GPU khác rảnh →
throughput giảm. **EPLB (Expert-Parallel Load Balancing)** định kỳ đo tải, tính lại cách sắp expert lên GPU, rồi
**di chuyển weight của expert giữa các GPU** cho cân. Việc "di chuyển weight" = P2P **send/recv tensor giữa các rank**.

- **Sync EPLB**: chính main-thread (thread đang chạy forward) dừng lại để chuyển weight → mọi rank làm cùng lúc,
  lockstep. An toàn nhưng "khựng" 1 nhịp (stall) khi rearrange.
- **Async EPLB**: đẻ 1 **background thread** chuyển weight **song song** với main-thread đang decode → không stall.
  Đây là nguồn của BIG PROBLEM.

### 1.2. 4 communicator (cách "khuân" weight qua lại)
Factory: [`eplb_communicator.py:618` `create_eplb_communicator()`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py).
Chọn theo preset `eplb_config.communicator`:

| backend | class | transport | group dùng | async an toàn? |
|---|---|---|---|---|
| `torch_nccl` | `TorchDistNcclEplbCommunicator` (L98) | NCCL `batch_isend_irecv` trên **GPU** | `device_group` (NCCL) | ❌ **deadlock** |
| `pynccl` | `PyNcclEplbCommunicator` (L574) | NCCL `ncclSend/Recv` trên **GPU** | pynccl comm (NCCL) | ❌ **deadlock** |
| `torch_gloo` | `TorchDistGlooStagedEplbCommunicator` (L155) | copy D2H → **gloo TCP** → H2D | **`cpu_group`** (gloo) | ✅ |
| `nixl` | `NixlEplbCommunicator` (L241) | **RDMA READ** (UCX) | data-plane = NIC | ✅ (cần RDMA HW) |

Điểm chung: cả 4 implement `add_send()`, `add_recv()`, `execute()`. Sự khác biệt sống-còn nằm ở **transport của
`execute()`**: 2 thằng đầu chạy **NCCL trên GPU**, 2 thằng sau **né NCCL/GPU-SM**.

### 1.3. Root cause của deadlock (torch_nccl / pynccl async) — line by line

**Bước 1 — EPLB đã có process-group NCCL RIÊNG.**
[`parallel_state.py:1878`](../vllm_src/vllm/distributed/parallel_state.py):
```python
# Create EPLB group with the same ranks as EP if EPLB is enabled.
# This is a separate process group to isolate EPLB communications
# from MoE forward pass collectives and prevent deadlocks when
# using torch.distributed in execution with torch.distributed in EPLB.
_EPLB = init_model_parallel_group(group_ranks, ..., group_name="eplb")
```
→ `_EPLB` là **NCCL communicator riêng**, cùng tập rank với `_EP`. vLLM tách ra để tránh lỗi "op out-of-order
trên **cùng 1** communicator". **Việc tách này chỉ đủ cứu SYNC**, KHÔNG cứu được ASYNC (xem bước 3).

**Bước 2 — background thread post NCCL từ thread phụ.**
[`async_worker.py:79` `transfer_run_periodically()`](../vllm_src/vllm/distributed/eplb/async_worker.py):
```python
while True:
    state.rearrange_event.wait(stream=cuda_stream)   # L86  chờ main-thread báo "có rearrange"
    ...
    transfer_metadata = transfer_layer(..., communicator=model_state.communicator, ...)  # L114
```
`transfer_layer` → `move_to_buffer` → cuối cùng gọi **`communicator.execute()`**
([`rebalance_execute.py:339`](../vllm_src/vllm/distributed/eplb/rebalance_execute.py)).
Với `torch_nccl`:
```python
# eplb_communicator.py:143  (chạy trong BACKGROUND thread)
def execute(self):
    with torch.cuda.stream(self._cuda_stream):
        reqs = batch_isend_irecv(self._p2p_ops)   # ← post NCCL P2P kernel lên GPU
        for req in reqs: req.wait()                # ← chờ tại chỗ
```
Với `pynccl`: `group_start()/send()/recv()/group_end()` (L587–615) — cũng là **NCCL kernel trên GPU**.

**Bước 3 — vì sao vẫn deadlock dù communicator đã tách.**
NCCL kernel là **blocking collective/P2P**: nó chiếm **SM** của GPU và **chờ đủ rank cùng vào** mới xong
(rendezvous). Trên **mỗi GPU** lúc này có **2 thread cùng bơm NCCL kernel**:
- **main thread**: all-reduce / all-to-all của MoE forward trên `_EP`/`_TP`.
- **async thread**: P2P expert-transfer trên `_EPLB`.

Hai thread độc lập ⇒ **thứ tự launch kernel giữa các rank không đồng nhất** (do OS/GIL scheduling, phi tất định).
Kịch bản chết:
```
rank0 (GPU0):  launch  FORWARD all-reduce   → chiếm SM, chờ rank1 vào all-reduce
rank1 (GPU1):  launch  EPLB   P2P recv      → chiếm SM, chờ rank0 vào P2P
        ⇒ rank0 chờ rank1 (ở all-reduce), rank1 chờ rank0 (ở P2P)  → VÒNG CHỜ → treo vĩnh viễn
```
Đây **không** phải "sai config" mà là **giới hạn: 2 luồng NCCL đồng thời trên cùng GPU không thể đảm bảo thứ tự
rendezvous nhất quán giữa các rank**. Chính vLLM tự thừa nhận cơ chế này trong
[`eplb_utils.py:85-118` `override_envs_for_eplb()`](../vllm_src/vllm/distributed/eplb/eplb_utils.py):
> *"If rank A enters DeepEP LL in main thread while rank B is still executing NCCL in async thread, rank A can block
> waiting for SMs, while rank B can block inside NCCL waiting for rank A to participate in the collective. This
> circular wait causes a deadlock."*

Và hint ở [`eplb_state.py:240-243`](../vllm_src/vllm/distributed/eplb/eplb_state.py):
> *"all EP ranks need to have the same `expert_rearrangement_step` … Otherwise, the rearrangement will hang at
> collective communication calls."*

**Bước 4 — hệ quả: RPC timeout → EngineCore chết.**
Main thread treo trong NCCL ⇒ lần `execute_model`/`sample_tokens` kế không trả kết quả. Executor chờ có deadline:
```python
# multiproc_executor.py:319/327  sample_tokens(..., timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
# multiproc_executor.py:387-389
status, result = mq.dequeue(timeout=dequeue_timeout)
except TimeoutError as e:
    raise TimeoutError(f"RPC call to {method} timed out.") from e
```
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS = 300` ([`envs.py:210`](../vllm_src/vllm/envs.py)) → sau **300 s**
báo `RPC call to sample_tokens timed out` → `Worker proc … died` → **EngineDeadError**.
(Đúng log sweep `logs/sweep/20260702_165527` và `logs/sweep2/20260702_172743`.)

**Pseudo-code minh hoạ toàn cảnh (rút gọn 8 rank → 2 rank):**
```text
# torch_nccl / pynccl ASYNC  (❌)
main_thread(rank r):                     async_thread(rank r):
  for step in decode_loop:                 wait(rearrange_event)
    y = allreduce(x, group=EP)  # NCCL      for layer in 75:
    if step % N == 0:                          batch_isend_irecv(P2P, group=EPLB)  # NCCL
       rearrange_event.record()                req.wait()
# Thứ tự "allreduce(EP)" vs "isend_irecv(EPLB)" trên các GPU KHÁC NHAU
# ⇒ có cấu hình rank chờ chéo nhau ⇒ NCCL không bao giờ khớp ⇒ treo 300s ⇒ chết.
```

### 1.4. Vì sao `torch_gloo` async AN TOÀN
[`eplb_communicator.py:186` `TorchDistGlooStagedEplbCommunicator.execute()`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py):
```python
cpu_tensor = tensor.to("cpu", non_blocking=True)          # L196  D2H (copy weight xuống RAM)
p2p_ops.append(P2POp(isend, cpu_tensor, peer, self._cpu_group))  # L197  GỬI QUA GLOO (TCP), KHÔNG NCCL
...
self._cuda_stream.synchronize()                           # L226  đợi D2H xong
reqs = batch_isend_irecv(p2p_ops); [r.wait() for r in reqs]      # L230  gloo trao đổi trên cpu_group
dst_tensor.copy_(cpu_tensor, non_blocking=True)           # L238  H2D (copy ngược lên VRAM)
```
Transport là **`cpu_group` (gloo/TCP)** trên host network — **không đụng NCCL, không chiếm SM GPU**. Background
thread có bơm gì thì cũng không xen vào rendezvous NCCL của main thread ⇒ **không vòng chờ**. Đánh đổi: chậm hơn
(qua RAM + TCP) nhưng đúng. Verify: gloo r16 probe bắn 21 transfer không sập (progress.md §4).

### 1.5. Vì sao `nixl` async AN TOÀN — và cái "gate" phần cứng
[`NixlEplbCommunicator`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py) dùng **RDMA READ**:
- `add_recv()` (L357) **phát luôn** lệnh đọc RDMA: `self._nixl_wrapper.transfer(xfer_h)` (L399) — receiver chủ động
  **đọc thẳng VRAM của rank khác** qua NIC.
- `execute()` (L526): `_wait_for_all_transfers()` + `monitored_barrier(cpu_group)` (L537).

Data-plane = **NIC RDMA**, hoàn toàn **ngoài NCCL và ngoài SM GPU** ⇒ song song với forward vô tư. **NHƯNG** cần
**phần cứng RDMA** (InfiniBand/RoCE NIC). Đây là "gate": không có NIC ⇒ nixl init có thể qua tcp nhưng transfer
data-plane (đọc khối VRAM lớn qua tcp) **crash** (progress.md §6). Xem §3 để biết node nào có RDMA.

### 1.6. RESOLUTION (ma trận viable)
| communicator | mode chốt | lý do |
|---|---|---|
| `torch_nccl` | **SYNC** | async deadlock (NCCL 2-thread) — bất khả kháng |
| `pynccl` | **SYNC** | như trên |
| `torch_gloo` | **ASYNC** | cpu_group, né NCCL/SM |
| `nixl` | **ASYNC** *(chỉ khi node có RDMA)* | RDMA data-plane, né NCCL/SM |

`gen_eplb_presets.sh` sinh đúng ma trận này; `check_nixl.sh` tự bật nixl **chỉ khi** `ibv_devices`/UCX có RDMA.

---

## 2. FLOW chạy EPLB — line by line + shape

### 2.0. Demo trực quan (đọc trước — theo style §0 của `EPLB_MECHANISM.vi.md`)
Config mặc định (verify khớp image này: [`config/parallel.py:59-96`](../vllm_src/vllm/config/parallel.py)):
`window_size=1000`, `step_interval=3000`, `use_async=true`, `num_redundant_experts=0`.

**(a) Mỗi step làm gì (RẺ — chỉ ĐẾM):** 1 forward = 1 step. EPLB cộng dồn "expert nào nhận bao nhiêu token" vào
**cửa sổ trượt 1000 step gần nhất**. KHÔNG di chuyển gì.
```
run-step :   0    1   ...  750(TRIGGER#1)  ...  3750(TRIGGER#2)
counter  : 2250 2251  ...  3000→reset 0    ...  3000→reset 0
           └ init = ¾·3000 = 2250 (eplb_state.py:433) → chạm 3000 chỉ sau 750 step ┘
```
Trigger khi counter ≥ `step_interval`: **lần đầu ở step 750** (do nạp sẵn ¾), sau đó **mỗi 3000 step**.

**(b) Step TRIGGER làm gì (ĐẮT — có thể KHUÂN weight):** (1) lấy tải TB mỗi expert qua 1000 step; (2)
`policy.rebalance_experts()` tính chỗ đặt mới; (3) nếu khác → **khuân weight expert giữa các rank**.
```
Tải đo:  E0=1000🔥  E1=100  E2=120  E3=90       (thu nhỏ 2 rank / 4 expert)
TRƯỚC:                                  SAU (redundant=0, chỉ hoán vị):
 rank0=[E0,E1] tải 1100 ⬅ nghẽn          rank0=[E0]        tải 1000 (vẫn hot)
 rank1=[E2,E3] tải  210                  rank1=[E1,E2,E3]  tải  310
 max/min = 5.2x                          max/min = 3.2x  (phẳng hơn)
```
`redundant=0` chỉ hoán vị → 1 expert siêu-hot đơn lẻ (E0) vẫn là trần. Muốn phẳng hơn → `num_redundant_experts>0`
để **nhân bản E0 ra 2 rank** (chia đôi tải). (Đây là lý do preset default-r0 giảm imbalance rất ít.)

**(c) SYNC vs ASYNC** — khác nhau CHÍNH ở bước (3), chi tiết ở **§2.7** (sơ đồ 2 luồng + code).

> Phân tích **field-by-field** config + máy trạng thái đầy đủ có ở doc chị em (cây `3rdparty/vllm`):
> `bench_mv4571/auto_analyze_ep_imbalance/EPLB_MECHANISM.vi.md` (§0–§2). Đã verify config/logic **khớp** image
> này (moreh 0.23.1). Bên dưới (§2.1–§2.7) là bản line-by-line trỏ vào source đã copy `../vllm_src/`.

### 2.0b. Config EPLB — các núm vặn ([`config/parallel.py`](../vllm_src/vllm/config/parallel.py))
| trường | mặc định | ý nghĩa | code |
|---|---|---|---|
| `window_size` | 1000 | số step gộp tải trước khi rearrange | `:59` |
| `step_interval` | 3000 | bao lâu (step) rearrange 1 lần | `:61` |
| `num_redundant_experts` | 0 | số slot expert nhân bản thêm (≠0 khi EPLB off → lỗi) | `:69`, `:474` |
| `use_async` | true | true=thread nền; false=chặn forward (async chỉ với policy "default") | `:81`, `:101` |
| `policy` | "default" | thuật toán xếp chỗ (chỉ "default") | `:86` |
| `communicator` | None (auto) | transport khuân weight | `:89` |

**Auto-select** khi `communicator=None` ([`config/parallel.py:907-924`](../vllm_src/vllm/config/parallel.py)):
elastic-EP→`pynccl`; else `nixl` nếu có, else `torch_gloo`. **`torch_nccl` bị cố ý TRÁNH** — comment `:914`:
*"NCCL is fundamentally incompatible with async EPLB due to multi-stream conflicts, and batched isend/irecv hangs
under high load"* → chính là §1.

### 2.0c. `log_balancedness` — bật để ĐO độ cân bằng (câu hỏi Q4, lưu tại đây theo RULE)
**Quan trọng:** `log_balancedness_interval` **một mình KHÔNG có tác dụng** — nó chỉ có nghĩa khi
`log_balancedness: true`. Cặp đôi ([`config/parallel.py`](../vllm_src/vllm/config/parallel.py)):
- `log_balancedness: bool = false` (`:72`) — bật/tắt tính+log metric cân bằng. **Mặc định OFF** vì tốn **1 all_reduce
  mỗi lần log** (overhead hot path).
- `log_balancedness_interval: int = 1` (`:77`) — cứ mỗi bao nhiêu step log 1 lần. `=1` → **log MỖI step** (rất nhiều
  + all_reduce mỗi step → nặng). Nên đặt lớn (vd `50`/`100`).

**Nó log CÁI GÌ** — [`eplb_state.py:509-556`](../vllm_src/vllm/distributed/eplb/eplb_state.py): khi
`log_balancedness=true` **và** `expert_rearrangement_step % interval == 0`, **rank 0** in đúng 1 dòng:
```
EPLB step: <S> for model <name>: avg_tokens=<A>, max_tokens=<M>, balancedness=<B>, steps until the next rearrangement: <R>
```
| trường | nghĩa | code |
|---|---|---|
| `EPLB step: S` | bộ đếm trigger `expert_rearrangement_step` hiện tại | `:549` |
| `avg_tokens A` | tải TRUNG BÌNH (token) — tính từ `num_tokens_per_rank` shape `(layer, rank)=(75,8)` | `:522-534` |
| `max_tokens M` | tải LỚN NHẤT (chỗ bận nhất) | `:535` |
| `balancedness B` | **= avg/max ∈ (0,1]** — **1.0 = cân bằng hoàn hảo**; càng nhỏ càng lệch (0.5 = chỗ bận gấp 2× TB) | `:542` |
| `steps until next rearrangement R` | `= step_interval − expert_rearrangement_step` (còn mấy step tới rearrange) | `:554` |

Trước khi tính, [`_sync_load_pass()`](../vllm_src/vllm/distributed/eplb/eplb_state.py) all_reduce 1 bản clone của
`expert_load_pass (75,256)` qua EP group → **đó là chi phí** (lý do default OFF). `balancedness` chính là con số cho
biết **EPLB có đang cân tải hiệu quả không**: theo dõi nó **tăng dần về 1** qua các lần rearrange = EPLB đang giúp;
nằm im ≈ baseline = EPLB chưa tác dụng (vd với `r0` do không nhân bản expert siêu-hot — xem §2.0).

**Có nên thêm vào preset?** CÓ (để đo tác dụng EPLB), nhưng phải thêm **CẢ HAI**:
`"log_balancedness": true, "log_balancedness_interval": 50`. Đánh đổi: +1 all_reduce mỗi 50 step → **hơi lệch số
throughput** (và baseline no-EPLB không log được metric này nên không so trực tiếp được). ⇒ **khuyến nghị: giữ run
throughput hiện tại SẠCH, chạy 1 follow-up run riêng** với `LOG_BALANCEDNESS=1`. `gen_eplb_presets.sh` đã có opt-in:
```bash
LOG_BALANCEDNESS=1 LOG_BALANCEDNESS_INTERVAL=50 bash gen_eplb_presets.sh   # regenerate preset kèm log
```

### 2.1. Số liệu model GLM-5.2-FP8 (dp8 / **ep8**)  — dùng xuyên suốt
`config.json`: `num_hidden_layers=78`, `first_k_dense_replace=3` ⇒ **`num_moe_layers L = 75`**;
`hidden_size H = 6144`; `moe_intermediate_size I = 2048`; `n_routed_experts = 256` (**logical**); top-8; FP8 block-quant.

| ký hiệu | ý nghĩa | r0 | r8 | r16 |
|---|---|---|---|---|
| `num_logical_experts` | expert "gốc" | 256 | 256 | 256 |
| `num_redundant_experts` | bản sao thêm | 0 | 8 | 16 |
| `P = num_physical_experts` | slot vật lý = logical+redundant | 256 | 264 | 272 |
| `ep_size` | số rank EP | 8 | 8 | 8 |
| `E = num_local_physical_experts = P/ep_size` | expert/GPU | 32 | 33 | 34 |

Weight 1 expert (FP8, 1 byte/phần tử), điển hình gồm 2 tensor: `w13 (gate+up) ≈ (2·I, H)=(4096,6144)`,
`w2 (down) ≈ (H, I)=(6144,2048)` → **~36 MB/expert** (chưa kể block-scale). `expert_weights[layer]` là list các
tensor, tensor đầu shape `(E, 4096, 6144)`, tensor sau `(E, 6144, 2048)`.

### 2.2. Khởi tạo — `EplbState.add_model()` [`eplb_state.py:342`](../vllm_src/vllm/distributed/eplb/eplb_state.py)
```python
self.is_async = parallel_config.eplb_config.use_async                       # L351
physical_to_logical_map = build_initial_global_physical_to_logical_map(...) # L353 → shape (P,), vd r0: [0..255]
physical_to_logical_map = ...expand(L, -1)                                  # L389 → (L, P) = (75, 256)
logical_to_physical_map  → (L, num_logical, 1024)                           # L397  (sparse, -1 = trống)
logical_replica_count    → (L, num_logical) = (75, 256)                     # L406
expert_load_pass         → (L, P)   int32  = (75, 256)                      # L415  tải forward hiện tại
expert_load_window       → (window_size, L, P) int32                        # L421  sliding window
self.expert_rearrangement_step = interval - interval//4                     # L433  khởi động ở 3/4 (rearrange sớm)
expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]      # L448  ← BUFFER = 1 LAYER experts
communicator = create_eplb_communicator(get_eplb_group(), backend, ...)     # L450
```
- `expert_buffer` = **đúng 1 layer** expert weights: r0 ≈ `32 × 36MB ≈ 1.15 GiB`/GPU (thường trực). Đây là 1 phần
  của "EPLB ăn thêm ~10 GiB" (progress.md §3). Phần còn lại: (a) NCCL comm-buffer reserve lúc profile (§2.5), và
  (b) weight của redundant experts (r8/r16 model to thêm ~5.4 GiB ở r16).

### 2.3. Trigger mỗi forward — `eplb_step()` → `step()`
[`gpu_model_runner.py:4564`](../vllm_src/vllm/v1/worker/gpu_model_runner.py): **cuối mỗi forward** gọi:
```python
with record_function_or_nullcontext("gpu_model_runner: eplb"):
    self.eplb_step()                       # L4565 → L3300 → eplb_state.step(is_dummy, is_profile, log_stats)
```
[`eplb_state.py:474` `step()`](../vllm_src/vllm/distributed/eplb/eplb_state.py):
```python
self.expert_rearrangement_step += 1                                        # L577  đếm bước
if self.is_async:                                                          # L579  (async) thử "thu hoạch" kết quả
    for ms in self.model_states.values():
        if ms.rebalanced and self._all_ranks_result_ready(ms):             # L585  MỌI rank có layer sẵn?
            _move_to_workspace(ms, ep_group.rank())                        # L588  copy buffer→weight (main thread)
if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:  # L593  tới hạn?
    if self.is_async and any(ms.rebalanced ...):                           # L594  đang rearrange dở → khoan reset
        self._update_layer_should_record(...); return                     # L601-602
    self.expert_rearrangement_step = 0                                     # L603
    self.rearrange()                                                       # L604  ← KÍCH HOẠT rearrange
```
- Nhịp: cứ đủ `step_interval` forward (preset `default` ~ mặc định, `s250` = 250) thì `rearrange()` 1 lần.
- `_all_ranks_result_ready` (L824): **all_reduce cờ `has_result`** trên cpu/device group để **mọi rank đồng ý**
  layer đã sẵn rồi mới commit → giữ lockstep phần "thu hoạch".

### 2.4. `rearrange()` — tính cách sắp expert mới [`eplb_state.py:658`](../vllm_src/vllm/distributed/eplb/eplb_state.py)
```python
# (a) physical load → logical load, cộng dồn qua window
logical_expert_load_window: (window_size, L, num_logical)                  # L697
logical_expert_load_window.scatter_add_(dim=-1, index=phys_to_logical, src=expert_load_window)  # L704
global_expert_load_window = logical_expert_load_window.sum(dim=0)          # L715 → (L, num_logical)=(75,256)
# (b) all-reduce tải qua toàn bộ EP ranks (CỘNG tải mọi GPU)  ← NCCL collective (main thread, lockstep)
global_expert_load_windows = self._allreduce_list([...])                   # L718
num_replicas = model.num_physical_experts   # =P                           # L723
num_groups   = model.num_expert_groups                                     # L724
num_nodes, num_gpus = get_node_count(), ep_group.size()                    # L739-740
# (c) chạy thuật toán → map mới
new_physical_to_logical_map = self.policy.rebalance_experts(               # L756  (SYNC path)
      global_expert_load_window.cpu(),   # (75, 256)  tải logical
      num_replicas,      # P=256
      num_groups, num_nodes, num_gpus,   # 1, 1, 8
      physical_to_logical_map.cpu())     # (75, 256) map cũ
#   → new_physical_to_logical_map: (L, P) = (75, 256)
```
`policy.rebalance_experts` ([`policy/default.py:274`](../vllm_src/vllm/distributed/eplb/policy/default.py)):
`balanced_packing` (pack group→node) → `replicate_experts` (nhân bản expert nóng, L76) →
`balanced_packing` (pack physical→GPU) → `preserve_intragpu_slots` (giữ nguyên slot cho expert **không đổi GPU** để
đỡ copy, L192). **In: `weight (75,256)` + `old (75,256)` → Out: `new (75,256)`.**

Sau đó chia 2 nhánh:
- **SYNC** (`not is_async`): gọi luôn `rearrange_expert_weights_inplace(...)` (L766) + `_commit_eplb_maps` (L778).
- **ASYNC** (`is_async`): **không** transfer ở đây; chỉ snapshot stats + bật cờ rồi báo thread phụ:
```python
eplb_model_state.eplb_stats = EplbStats(global_expert_load_window.clone(), ...)  # L795
eplb_model_state.rebalanced = True                                               # L805
self.rearrange_event.record()                                                    # L808  ← đánh thức async thread
```

### 2.5. Nhánh SYNC — `rearrange_expert_weights_inplace()` [`rebalance_execute.py:511`](../vllm_src/vllm/distributed/eplb/rebalance_execute.py)
```python
if is_profile:                                                             # L573  (chạy lúc profile_run)
    if communicator.needs_profile_buffer_reservation:                      # nccl/pynccl/gloo = True; nixl = False
        profile_buffer = [torch.empty_like(w) for w in first_layer_weights]
        all_gather(dummy_recv_buffer, weight, group=ep_group)              # L584  ← RESERVE NCCL comm buffer
    return
for layer_idx in range(num_moe_layers):        # 0..74                     # L596
    transfer_metadata = move_to_buffer(..., old_indices=old[layer], new_indices=new[layer], ...)  # L597
    move_from_buffer(..., transfer_metadata, new[layer], ep_rank)          # L609
```
- `is_profile=True` chạy 1 lần trong `profile_run` để **đặt trước** NCCL communication buffer (all_gather dummy) →
  vLLM tính headroom này vào ⇒ **KV khả dụng tụt** (chính là ~10 GiB, xem §3). `nixl` đặt `needs_profile_buffer_reservation=False`
  (L311) vì nó tự register buffer riêng.

**`move_to_buffer()` [`rebalance_execute.py:172`](../vllm_src/vllm/distributed/eplb/rebalance_execute.py)** (cho **1 layer**):
```python
old_indices, new_indices: (P,) = (256,)                                    # map cũ/mới của layer
# tính mask trên E=32 slot local:
is_unchanged            : (E,) bool   # slot giữ nguyên expert            # L217
is_received_locally     : (E,) bool   # lấy được từ chính GPU này         # L224
send_expert_ids/src_rows: (E,)        # expert cần GỬI + hàng nguồn        # L238-239
recv_expert_ids/dst_rows: (E,)        # expert cần NHẬN + hàng đích        # L250-251
# 1) copy nội bộ (đổi slot trong cùng GPU): b[dst].copy_(w[src_local])     # L266-268
communicator.set_transfer_context(old_indices, layer_idx)                  # L270 (nixl: precompute src rows)
# 2) post SEND: với mỗi expert cần gửi, add_send(expert_tensors, dst_rank) # L303
# 3) post RECV: add_recv([buffer rows], src_rank, expert_id)               # L332  ← ghi vào expert_buffer
communicator.execute()                                                     # L339  ← TRANSPORT thực thi ở đây
return TransferMetadata(...)                                               # L340
```
- Input tensor mỗi expert: `w[src]` shape `(4096,6144)` & `(6144,2048)` (FP8). `add_recv` nhận vào **`expert_buffer`**
  (không ghi đè weight đang dùng) → tránh hỏng weight khi đang forward.

**`move_from_buffer()` [`rebalance_execute.py:350`](../vllm_src/vllm/distributed/eplb/rebalance_execute.py)**:
copy từ `expert_buffer` **trở lại** `expert_weights[layer]` cho các slot đã nhận (L384-386), rồi nhân bản
row cho các slot "duplicate" nếu có (L422-424). Sau bước này layer đã dùng map mới.

### 2.6. Nhánh ASYNC — event → background thread → main thread thu hoạch
```
main thread                              async thread (async_worker.transfer_run_periodically, L79)
────────────                             ───────────────────────────────────────────────────
rearrange(): set rebalanced=True         wait(rearrange_event)                      # L86  (ngủ tới khi có việc)
             rearrange_event.record() ─▶ new_map = run_rebalance_experts(...)       # L101 (tính map, giống §2.4)
                                         for layer in 0..74 while rebalanced:        # L113
step() mỗi forward:                          transfer_layer(old[layer], new[layer], # L114 → move_to_buffer → execute()
  if rebalanced & all_ranks_ready:               expert_buffer, communicator, ...)         (ghi vào expert_buffer)
     _move_to_workspace(ms)  ◀───────────    cuda_stream.synchronize()              # L128
       move_from_buffer(...)                    pending_result = AsyncEplbLayerResult(layer,new_map[layer],meta,evt)  # L135
       _commit_eplb_maps_for_layer(...)         consumed_event.wait(stream)         # L145  ← CHỜ main thu hoạch xong
       result.consumed_event.record() ────▶  (thức dậy) layer_idx += 1             # L148
```
Đồng bộ 2 thread bằng:
- **`CpuGpuEvent`** [`eplb_utils.py:16`](../vllm_src/vllm/distributed/eplb/eplb_utils.py): ghép
  `torch.cuda.Event` + `threading.Event` để đảm bảo `record()` (main) xảy ra trước `wait()` (async) — 1 producer,
  1 consumer.
- **`rebalanced` / `pending_result`**: dựa vào **GIL** để đồng bộ (comment L184, L205).
- **`_all_ranks_result_ready`** (L824): all_reduce cờ để **mọi rank cùng** `_move_to_workspace` 1 layer/nhịp →
  phần "commit" vẫn lockstep, chỉ có phần "khuân bytes" là async.
> Với `torch_nccl`/`pynccl`, chính `communicator.execute()` ở **async thread** (L114→L339→NCCL) là chỗ đâm vào
> NCCL của main thread → deadlock (§1.3). Với `gloo`/`nixl`, `execute()` né NCCL nên async chạy êm.

### 2.7. SYNC vs ASYNC — sơ đồ 2 LUỒNG (per-step, per-layer) + trỏ code
Phần quan trọng nhất để hiểu §1. **sync = 1 luồng (1 row); async = 2 luồng (2 row).** Bối cảnh: GLM-5.2 = **75
MoE layer**; default `step_interval=3000`, init ¾ ⇒ **trigger lần đầu ở step 750**.

#### (A) SYNC (`use_async=false`) — CHỈ 1 luồng (main); rearrange CHẶN cả forward loop
```
LUỒNG MAIN (forward/decode)                                          │ code
────────────────────────────────────────────────────────────────── │ ───────────────────────────
step 749  forward; step(): counter 2999 (chưa tới)                   │ eplb_state.py:577 (++counter)
step 750  forward; step(): counter→3000 ≥ 3000 → rearrange()         │ eplb_state.py:593,604
  └ rearrange():                                                     │
      gộp tải logical + all_reduce toàn cục                          │ eplb_state.py:704-718
      policy.rebalance_experts() → new_map (75,256)                  │ eplb_state.py:756
      rearrange_expert_weights_inplace():   ◄══ CHẶN Ở ĐÂY           │ eplb_state.py:766
         for layer in 0..74:                                         │ rebalance_execute.py:596
            move_to_buffer(layer): mask + add_send/add_recv          │ rebalance_execute.py:597,172
               communicator.execute()  (khuân bytes layer này)       │ rebalance_execute.py:339
            move_from_buffer(layer): buffer → weight[layer]          │ rebalance_execute.py:609
      _commit_eplb_maps() (map mới có hiệu lực)                      │ eplb_state.py:778
  ══ engine ĐỨNG IM 3–38s: mọi request treo (spike latency) ══       │ (log "Rearranged experts in Xs" :789)
step 751  forward (bình thường trở lại)
```
1 luồng ⇒ **không xung đột NCCL** ⇒ `torch_nccl`/`pynccl` **dùng được** ở SYNC. Giá phải trả: **stall 3–38s**.

#### (B) ASYNC (`use_async=true`) — 2 LUỒNG song song (producer/consumer, ĐÚNG 1 layer / forward step)
Row trái = **MAIN** (vẫn decode), Row phải = **ASYNC worker** ([`async_worker.py`](../vllm_src/vllm/distributed/eplb/async_worker.py)).
```
TIME│ LUỒNG MAIN (forward + step())                   │ LUỒNG ASYNC (transfer_run_periodically)        │ code
────┼─────────────────────────────────────────────── ┼──────────────────────────────────────────────┼─────────────
 t0 │ step750: counter→3000 → rearrange():            │ (ngủ) rearrange_event.wait()                   │ async_worker.py:86
    │   gộp tải + all_reduce                          │                                                │ eplb_state.py:718
    │   (async) KHÔNG khuân; rebalanced=True          │                                                │ eplb_state.py:805
    │   rearrange_event.record() ─────────────────────┼──► THỨC DẬY                                    │ eplb_state.py:808
 t1 │ step751: forward; step():                       │ run_rebalance_experts() → new_map (75,256)     │ async_worker.py:101
    │   rebalanced? yes; all_ranks_ready? CHƯA        │ layer0: transfer_layer() → ghi expert_buffer   │ async_worker.py:114
    │   → chưa move                                    │ cuda_stream.synchronize()                      │ async_worker.py:128
    │                                                  │ publish pending_result(layer0)                 │ async_worker.py:135
    │                                                  │ consumed_event.wait() ◄── chờ MAIN tiêu thụ    │ async_worker.py:145
 t2 │ step752: step():                                │ (đang chờ)                                     │
    │   rebalanced & _all_ranks_result_ready()? YES ──┼─(all_reduce cờ: MỌI rank có layer0 sẵn?)       │ eplb_state.py:585,824
    │   _move_to_workspace(layer0):                   │                                                │ eplb_state.py:588
    │     move_from_buffer → weight[0]                 │                                                │ eplb_state.py:1160
    │     _commit_eplb_maps_for_layer(0)              │                                                │ eplb_state.py:1168
    │     consumed_event.record() ────────────────────┼──► THỨC: layer1: transfer_layer() → buffer     │ :1179 → async_worker.py:148,114
 t3 │ step753: _move_to_workspace(layer1)             │ layer2: transfer_layer() → buffer              │ (lặp)
 .. │  … mỗi forward step tiêu thụ ĐÚNG 1 layer …     │  … mỗi vòng chuẩn bị 1 layer rồi chờ MAIN …    │
 tN │ step750+75: _move_to_workspace(layer74 cuối):   │ layer74 xong → thoát vòng while                │ async_worker.py:113
    │   rebalanced=False (xong 1 rearrange)           │ → quay lại rearrange_event.wait()              │ eplb_state.py:1174
```
- **Khuân bytes = LUỒNG ASYNC** (`transfer_layer→communicator.execute`), **song song** MAIN đang decode ⇒ không
  stall. `expert_buffer` = vùng đệm để không đè weight đang dùng.
- **Commit (đổi map) = LUỒNG MAIN** trong `step()`, **1 layer/forward**, và **chỉ khi `_all_ranks_result_ready`**
  (all_reduce cờ) xác nhận **mọi rank** đã chuẩn bị xong layer đó ⇒ 8 rank vẫn **lockstep** ở bước commit.
- Đồng bộ 2 luồng: `rearrange_event` (đánh thức), `consumed_event`/[`CpuGpuEvent`](../vllm_src/vllm/distributed/eplb/eplb_utils.py) (bắt tay từng layer), cờ `rebalanced`/`pending_result` (dựa GIL).

#### (C) Vì sao ASYNC + nccl/pynccl → DEADLOCK (ghép 2 row lại)
Ở ASYNC, **LUỒNG ASYNC** gọi `communicator.execute()`; với `torch_nccl`/`pynccl` nó **post NCCL kernel lên GPU**
([`eplb_communicator.py:143`/`:612`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py)) **cùng lúc** LUỒNG
MAIN post NCCL collective của forward. 2 luồng ⇒ **thứ tự launch NCCL giữa 8 GPU lệch nhau** ⇒ vòng chờ chéo (§1.3):
```
GPU rank r  MAIN :  forward all_reduce(EP)   ─┐ chờ mọi rank vào all_reduce
            ASYNC:  execute() P2P(EPLB, NCCL) ┘ chờ mọi rank vào P2P
   → các rank vào 2 phép theo thứ tự khác nhau → KẸT → treo 300s → RPC timed out (multiproc_executor.py:389)
```
`gloo` (execute qua `cpu_group`, [`:186`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py)) và `nixl`
(execute qua `rocm_ipc`/RDMA, [`:526`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py)) **KHÔNG post NCCL
kernel** ⇒ LUỒNG ASYNC không đụng NCCL của MAIN ⇒ **không deadlock**. Đó là lý do async chỉ đi với gloo/nixl.

### 2.8. Data-path của 3 transport — XÁC NHẬN (câu hỏi Q5)
"Ai đọc/ghi gì" ở mức đường dữ liệu, đối chiếu code ([`eplb_communicator.py`](../vllm_src/vllm/distributed/eplb/eplb_communicator.py)):

| backend | mô hình data-path | đối chiếu hình dung | code |
|---|---|---|---|
| **nixl** | **1 phía — RECEIVER PULL**: GPU B **đọc thẳng VRAM GPU A**; GPU A thụ động (weight đã register sẵn). Intra-node = `rocm_ipc` qua xGMI. | ✓ đúng: "GPU B đọc VRAM trực tiếp từ GPU A" | add_send=no-op `:337`; add_recv phát READ `:399`; `make_prepped_xfer("READ")` `:517` |
| **nccl / pynccl** | **2 phía — PUSH + match**: A `isend`, B `irecv`, NCCL khớp cặp, chuyển **thẳng GPU→GPU qua xGMI (không qua CPU)**. | ✓ gần đúng ("peer-to-peer A→B") — thêm: **2 phía**, không phải chỉ A đẩy | isend/irecv `:120/:136`; `batch_isend_irecv`+wait `:148-150` |
| **gloo** | **CPU-staging, 2 copy**: GPU A `.to(cpu)` (D2H→host-RAM **tiến trình A**) → gloo `isend` qua **cpu_group (TCP loopback)** → tiến trình B `irecv` vào **host-RAM tiến trình B** → `.copy_` H2D lên VRAM GPU B. | ⚠️ ý tưởng đúng, **2 chỗ cần sửa** (dưới) | D2H `:196`; isend cpu_group `:197`; irecv `:207`; sync `:226`; H2D `:238` |

**2 chỗ cần sửa ở gloo (hiểu lầm hay gặp):**
1. **KHÔNG có "1 buffer CPU chung".** 8 rank = **8 tiến trình riêng**, mỗi tiến trình **host RAM riêng**. gloo
   **copy bytes giữa 2 vùng nhớ host khác nhau** (A→B) qua cpu_group; trên 1 node đường đó là **TCP loopback (`lo`)**.
   Cùng CPU/RAM vật lý (cùng máy) **nhưng khác vùng nhớ** — không phải cả hai chạm chung 1 buffer.
2. **KHÔNG phải pinned memory.** Code dùng `tensor.to(device="cpu")` (`:196`) + `torch.empty_like(tensor, device="cpu")`
   (`:206`) → **host RAM thường (pageable)**, không `pin_memory=True`. (`non_blocking=True` chỉ thật sự async nếu buffer
   pinned — ở đây không pinned ⇒ thực chất đồng bộ.)

→ Mô tả đúng gloo (4 bước): **GPU A —D2H→ host-RAM(A) —gloo/TCP-loopback→ host-RAM(B) —H2D→ GPU B**.
Điểm chung nixl+gloo (khác nccl): **execute() KHÔNG post NCCL kernel** ⇒ async an toàn (§2.7-C).

---

## 3. Vì sao MI325 chạy được mà MI300 hay lỗi — theo TỪNG phần

> **Cảnh báo quan trọng:** progress.md/CONTEXT_HANDOFF mô tả 1 node MI300 tên **`tw031`** (docker, **KHÔNG có RDMA**).
> Node mục tiêu hiện tại là **`mi300-7` trong cụm valhalla — node này CÓ RDMA** (`ibv_devices` = `mlx5_0`, `mlx5_1`).
> ⇒ Một số kết luận "MI300 fail" là do **đặc thù node tw031**, KHÔNG phải do kiến trúc MI300. Xem cột "mi300-7".

| Vấn đề | MI325 (256 GiB, có RDMA) | MI300 `tw031` (192 GiB, KHÔNG RDMA) | **MI300 `mi300-7`** (192 GiB, **CÓ RDMA**) | Bản chất |
|---|---|---|---|---|
| **nccl/pynccl async deadlock** | ❌ deadlock | ❌ deadlock | ❌ deadlock (chưa test nhưng chắc chắn) | **Framework+NCCL** (image 260626) — **giống hệt**, không liên quan HW. Fix SYNC dùng chung. |
| **VRAM / KV feasibility** | ✅ dư (util .9 → ~230 GiB) | ⚠️ chật | ⚠️ chật (192 GiB) | **Kiến trúc**: model ~107.6 + EPLB ~10 GiB. MI300 chỉ còn KV ~49 (r0)/43.1 (r16) < 54.62 cần cho 1M → **phải hạ max_model_len 1M→512K**. MI325 256 GiB nên 1M vẫn fit. |
| **nixl async** | ✅ chạy (RDMA thật) | ❌ init tcp OK nhưng transfer crash | **✅ NÊN chạy** (có mlx5 RDMA giống MI325) | **Phần cứng NODE** (cần RDMA NIC), KHÔNG phải GPU-arch. `check_nixl.sh` tự bật khi có RDMA. |
| **gloo async** | ✅ | ✅ (21 transfer, ko sập) | ✅ | cpu_group/TCP — **độc lập HW**, chạy mọi nơi. |
| **redundant r8/r16** | ✅ dư chỗ | ⚠️ r16 model 114.17 GiB, KV 43.1 | ⚠️ như tw031 (cùng 192 GiB) | Kiến trúc VRAM: KV auto co lại; chọn 512K để r16 vẫn fit. |

**Diễn giải gọn:**
1. **Deadlock nccl/pynccl async** = lỗi framework (2-thread NCCL) → **giống nhau trên mọi GPU**; MI325 cũng chết. Không phải "MI325 miễn nhiễm".
2. **KV/VRAM** = khác biệt **thật** do MI300 ít VRAM hơn (192 vs 256). Đây là lý do chính "MI300 khó" → cap `max_model_len=512K` cho preset EPLB (workload 100K ISL không bị ảnh hưởng).
3. **nixl** = phụ thuộc **NIC RDMA của node**, KHÔNG phải MI300-vs-MI325. tw031 thiếu RDMA nên crash; **mi300-7 có RDMA nên nixl nhiều khả năng chạy như MI325** — cần verify lại trên mi300-7 (đừng bê nguyên kết luận "loại nixl" của tw031).
4. **gloo** = an toàn ở mọi node.

### 3.1. Kết quả nixl THỰC ĐO trên mi300-7 (2026-07-03, `logs_debugs/2-nixl-async-default-r0`)
- **Với `rc_x` trong UCX_TLS (preset cũ) → init FAIL:** `ibv_reg_mr(rocm VRAM, len=805306368) on md[4]=mlx5_0 → Bad address` → `NIXL_ERR_BACKEND` tại `register_memory` ([eplb_communicator.py:424](../vllm_src/vllm/distributed/eplb/eplb_communicator.py)). Nguyên nhân: **GPUDirect-RDMA (amdgpu↔mlx5 peermem/dmabuf) không hoạt động** trên node → NIXL không đăng ký được VRAM với NIC IB.
- **Fix = bỏ `rc_x`** → `UCX_TLS=tcp,self,sm,rocm_copy,rocm_ipc`: probe [`nixl_probe.py`](nixl_probe.py) → `REGISTER_OK`; full serve → **init OK + async transfer FIRE trên cả 8 rank + engine KHÔNG sập** (`console.log`: HEALTH OK → bench DONE → server stopped). ⇒ **nixl hoạt động chức năng trên mi300-7** (khác node tw031 vốn crash lúc transfer). Intra-node dùng `rocm_ipc`/xGMI, không cần IB.
- **Vì sao throughput = 0/null (ĐÃ xác nhận từ log, KHÔNG phải crash):** formal test 240s **có chạy** nhưng ghi
  **0 request**. Điều kiện đếm 1 request ([`unit_test.py:435`](../vllm_src/../scripts/unit_test.py) — cùng cây scripts):
  `test_start_time >= official_start_time AND test_end_time <= official_end_time` → chỉ tính request **vừa bắt đầu
  vừa kết thúc trong cửa sổ đo `[start+20s, start+220s]`**. Bằng chứng: nixl formal log = **0 dòng `[Test]`**
  (baseline = **416** dòng, TTFT ~2.8s). nixl có **116 response `200`** trong pha formal nhưng chúng **quá chậm**
  (100k prefill × conc64 dưới **KV cố định 40 GiB** + EPLB overhead + **contention đồng nghiệp 15:11–15:15**) → bắt
  đầu/kết thúc **vượt ngoài** cửa sổ → không cái nào được tính. `EngineDeadError` trong serve.log chỉ là artifact
  của §6 force-kill. → cần **re-run lúc node SẠCH** (và cân nhắc **nâng KV nixl > 40 GiB** trên mi300-7) để có
  throughput thật. **"nixl không chạy được" là SAI** — nó init+transfer OK; chỉ là workload 100k quá nặng cho cửa
  sổ 240s khi KV nhỏ + contention (baseline no-EPLB KV 59 GiB thì kịp).

---

## 3.2. Prefix-cache fairness — `data[:10]` phải ADAPT theo KV cache (câu hỏi Q8)
Bench giữ `len(data)` prompt **phân biệt** ([`unit_test.py:512-513`](../vllm_src/../scripts/unit_test.py):
`if encoding_size == 100000: data = data[:10]`); 64 thread quay vòng trên chúng. Để **TPOT không bị TTFT
dominate**, cần **prefix cache ~100%** ⇒ tất cả prefix (100k token/prompt) phải nằm lại trong KV cùng lúc với
working-set decode. Điều kiện: `X·ISL + decode_ws(≈50K) ≤ KV_tokens` ⇒ **`X = ⌊KV_tokens/ISL⌋ − 1`** (−1 chừa
headroom cho decode 64-luồng). Baseline: `⌊1,134,528/100,000⌋−1 = 10` (nên là 10, không phải 11 — 11 prefix=1.1M
ăn trọn KV, decode evict prefix → hit<100%).

**EPLB co KV ⇒ `data[:10]` tràn ⇒ prefix hit tụt ⇒ TTFT dominate (đo được):**
| config | KV cache (tokens) | `data[:10]`=1.0M vừa? | prefix hit | X đúng (`⌊KV/ISL⌋−1`) |
|---|---|---|---|---|
| baseline | 1,134,528 | ✅ (dư 134K) | **99.56%** | 10 |
| gloo r0 | 937,536 | ❌ | **94.88%** | **8** |
| nixl r0 (KV 40GiB) | 767,872 | ❌ | **68.85%** | **6** |

⇒ Đây là **thủ phạm chính** của nixl 0-req (68.85% hit → re-prefill 100k liên tục → TTFT khổng lồ → không kịp cửa
sổ 240s) và gloo TTFT 29.5s. **Fix:** `run_and_bench.sh` grep `GPU KV cache size: N tokens` (serve.log, sau HEALTH)
→ truyền `--kv_cache_tokens N` xuống `unit_test.py` → tính `X=⌊N/ISL⌋−1` → `data[:X]`. Khôi phục prefix ~100% cho
mọi config ⇒ so sánh TPOT công bằng (chỉ khác đúng biến EPLB communicator, không lẫn hiệu ứng prefix-thrash).

## 4. Reproduce / patch nhanh
```bash
# vào pod (đã ở mi300-7)
cd /remote/vast0/phuc-nguyen/workspace/tickets/mv-4571/mv-valhalla-cluster/dev
bash k.sh shell
# sửa 1 file EPLB trên host (VD thêm log) rồi recreate để ăn:
#   $EDIT  mv-4571-vllm-src/vllm/distributed/eplb/eplb_communicator.py
bash k.sh down && bash k.sh up
# chạy 1 preset:
bash k.sh x -- bash -lc 'RUN=<abs> PRESET=<abs.yaml> bash <scripts>/run_and_bench.sh'
```

## 5. Chỉ mục code (file:line)
- 4 communicator + factory: `eplb_communicator.py` — nccl `:98/:143`, gloo `:155/:186`, nixl `:241/:357/:526`, pynccl `:574/:612`, factory `:618` (chọn group `:649-653`, dispatch `:706-734`).
- Async worker (bg thread): `async_worker.py:79` (event `:86`, transfer_layer `:114`, consumed_event `:145`).
- State machine: `eplb_state.py` — `add_model:342` (buffer `:448`), `step:474` (đếm `:577`, async harvest `:585`, trigger `:604`), `rearrange:658` (allreduce `:718`, policy `:756`, async signal `:805-808`), hint deadlock `:240-243`, `_move_to_workspace:1154`.
- Transfer thực thi: `rebalance_execute.py` — `rearrange_expert_weights_inplace:511` (profile reserve `:573-589`), `move_to_buffer:172` (`execute()` `:339`), `move_from_buffer:350`, `transfer_layer:427`.
- Policy: `policy/default.py:274` (`rebalance_experts_hierarchical:104`, `replicate_experts:76`, `preserve_intragpu_slots:192`).
- RPC timeout: `multiproc_executor.py:319/327` (timeout), `:389` (raise); `envs.py:210` (`=300`).
- Trigger point: `gpu_model_runner.py:4564-4565` (`eplb_step`), `:3300` (`eplb_step`→`step`), `:5216` (`start_async_loop`).
- EPLB group riêng: `parallel_state.py:1878` (comment "prevent deadlocks"), `get_eplb_group:1378`, `get_ep_group:1366`.
- Config EPLB: `config/parallel.py` — EPLBConfig `:56-105`, auto-select communicator `:907-924`, validate `:459-480`.

---

## Q. Nhật ký câu hỏi (RULE: mọi câu hỏi tương ứng phần nào trong docs đều được lưu vào docs)
| # | Câu hỏi | Trả lời tại |
|---|---|---|
| Q1 | 1 máy 8 GPU 8 tiến trình thì RDMA/UCX/NIXL/gloo chạy ra sao (không phải "2 máy")? | **§A.5** |
| Q2 | Giải thích EPLB kĩ hơn, flow-by-flow như `EPLB_MECHANISM.vi.md` (ví dụ + link code) | **§2.0, §2.1–§2.7** |
| Q3 | Sync vs async kĩ hơn — sơ đồ 2 luồng (per-step / per-layer) + link code | **§2.7** |
| Q4 | `log_balancedness_interval` có nên thêm vào preset? Nó log ra tham số gì? | **§2.0c** |
| Q5 | Xác nhận data-path: nixl=GPU B đọc VRAM GPU A; nccl=P2P A→B; gloo=D2H→CPU→H2D | **§2.8** |
| Q6 | nixl re-run bị lỗi — root cause + fix + verdict (init/transfer/crash?) | **§3.1** |
| Q7 | Vì sao bench nixl luôn null dù "bench thành công"? | **§3.1** (formal test chạy nhưng 0 request lọt cửa sổ đo 200s; `unit_test.py:435`) |
| Q8 | `data[:10]` phải đổi theo KV khi bật EPLB (prefix~100% cho fair TPOT); vì sao 10 không phải 11 | **§3.2** (X=⌊KV/ISL⌋−1; EPLB co KV ⇒ 10 tràn ⇒ prefix hit tụt) |

---

## R. TRẠNG THÁI & CÁCH RESUME (cập nhật 2026-07-03, pod bị tắt giữa chừng)

**Đã xong (nằm trên disk, không mất):**
- Chẩn đoán + docs đầy đủ (§1–§3.2, Q1–Q8). nixl init fix (bỏ `rc_x` → `UCX_TLS=tcp,self,sm,rocm_copy,rocm_ipc`).
- **Adaptive X đã implement + verify**: `unit_test.py --kv_cache_tokens` (X=⌊KV/ISL⌋−1) ← `multi_process_test.py`
  (env `REBENCH_KV_TOKENS`) ← `run_and_bench.sh` (grep `GPU KV cache size`). Verify: baseline→X=10, gloo→8, nixl(40G)→6.
- nixl r0 KV nâng **40→48 GiB** (`presets/glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml`) cho fair với gloo auto ~49.
- `setup_dev_pod.sh`: pod `phuc-nguyen-mv4571-2` @ mi300-7, tự set KUBECONFIG, editable overlay 5 file eplb. `k.sh` POD mặc định `-2`.

**Kết quả hiện có:** baseline = **416 req, p50_tpot 23.86, prefix 99.56%** (FAIR, X=10 — giữ nguyên).
Các số EPLB cũ (gloo 82/94.88%, nixl 0/68.85%) là **KHÔNG fair** (dùng `data[:10]` tràn KV) → đã xoá, **chờ re-run**.

**CÒN LẠI (re-run công bằng khi có node):** 4 EPLB config `nixl / gloo / pynccl / nccl` với adaptive X (prefix ~100%).
```bash
cd mv-4571/mv-valhalla-cluster/dev
bash k.sh up                 # tạo lại pod -2 trên mi300-7 (chờ node Ready + hết cordon)
bash k.sh x -- bash -lc 'cd <ticket>/scripts && setsid bash run_debug5.sh > <ticket>/logs_debugs/debug5.log 2>&1 </dev/null &'
# run_debug5.sh tự SKIP baseline (đã có summary), chạy nixl→gloo→pynccl→nccl với adaptive X.
# Theo dõi: tail <ticket>/logs_debugs/debug5.log ; done khi in "ALL DONE".
```
Kỳ vọng: prefix ~100% cho cả 4 → TPOT sạch, chỉ khác đúng biến EPLB communicator. Verdict chức năng đã biết:
nccl/pynccl **async** deadlock (dùng SYNC), gloo async OK, nixl async OK trên mi300-7 (init/transfer không crash).
