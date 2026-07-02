# Cơ chế EPLB theo config đang chạy — async-default (giải thích kỹ)

Config tham chiếu: **async-default**
`enable_eplb=true`, `eplb_config = {use_async: true, communicator: nixl|nccl|pynccl}`,
KHÔNG set window/step → mặc định **window_size=1000, step_interval=3000**.
GLM5.2: **256 logical expert, EP8, 32 physical expert/rank, ~46 MoE layer, num_redundant_experts=0**.

Tham chiếu code: `3rdparty/vllm/distributed/eplb/{eplb_state.py, async_worker.py, rebalance_execute.py, policy/default.py}`.

---

## 1. Bước "load" — mỗi step thu thập gì cho policy

Mỗi forward step, mỗi lớp MoE cộng dồn vào bộ đếm `expert_load_pass`: **mỗi physical expert nhận bao nhiêu
token** ở step này (`eplb_state.py:150`). Cuối step, copy vào một ring buffer (cửa sổ trượt):

```
expert_load_window[ step % 1000 ] = expert_load_pass          (eplb_state.py:563-571)
    shape = (window=1000, num_layers≈46, num_physical_expert=256)
```

→ Sau warmup, window luôn giữ **1000 step gần nhất**: "mỗi expert nóng/lạnh cỡ nào trong 1000 step qua".
Đây là thao tác **cục bộ trên GPU, KHÔNG có communication** (chỉ cộng dồn counter). Không log_balancedness
(mặc định off) nên cũng không có collective phụ.

---

## 2. Tại step trigger (mỗi step_interval=3000) — policy quyết định gì

Chuỗi ở `eplb_state.py:691-763`:
1. **Cộng dồn window** → tổng load mỗi physical expert qua 1000 step.
2. **Physical → logical** (`scatter_add_`, dòng 704): quy về load của từng *logical* expert (256 expert model
   thật có), vì nhiều physical slot có thể là replica của cùng logical expert.
3. **All-reduce qua 8 rank** (dòng 718): mọi rank có **bức tranh load TOÀN CỤC** của từng logical expert.
   (Đây là 1 collective — nhưng chỉ 1 lần mỗi 3000 step → rẻ.)
4. **Gọi policy** `policy.rebalance_experts(load, num_replicas, num_groups, num_nodes, num_gpus, current_map)`
   (dòng 756) → trả về **`new_physical_to_logical_map`**: logical expert nào ngồi slot physical nào trên rank
   nào, sao cho **tổng load mỗi rank cân nhau** (và nhân bản hot expert nếu `redundant>0`; ở đây =0 nên chỉ
   **hoán vị** vị trí, không nhân bản).

→ Đây thuần là **"quyết định"** — tính toán trên bảng số, **CHƯA đụng tới trọng số trên GPU**.

Ví dụ thu nhỏ (2 rank / 4 expert cho dễ nhìn):
```
Load đo trong cửa sổ:  E0=1000 (HOT)   E1=100   E2=120   E3=90
TRƯỚC:                                   → policy tính new_map → SAU (hoán vị, redundant=0):
  rank0 = [E0, E1]  tải 1100  ⬅ nghẽn      rank0 = [E0]        tải 1000
  rank1 = [E2, E3]  tải  210               rank1 = [E1,E2,E3]  tải  310
        max/min = 5.2x                             max/min = 3.2x  (phẳng hơn)
```
Với `redundant=0`, expert siêu-hot đơn lẻ (E0) vẫn là trần → muốn phẳng hơn phải bật `num_redundant_experts>0`
để **nhân bản E0 ra 2 rank** và chia đôi tải. (Đây là lý do EPLB trên MTP5 giảm imbalance rất ít.)

---

## 3. Rearrange diễn ra lúc nào — SYNC vs ASYNC

### SYNC (`use_async=false`) — dễ hiểu, chặn engine
Ngay tại step trigger, main thread làm hết trong lúc engine **đứng im** (`eplb_state.py:766-781`):
```
step 3000: [tính new_map] ─► [rearrange_expert_weights_inplace: all-to-all DỜI weight, 3–38s] ─► [commit map] ─► tiếp
                                        ▲ ENGINE TREO ở đây (spike latency; log "Rearranged experts in Xs")
```

### ASYNC (`use_async=true`) — không chặn; đây là phần hay bị hiểu nhầm
Tại step trigger, main thread **chỉ chụp stats + đánh thức worker rồi đi tiếp** (`eplb_state.py:794-808`: chỉ
`rearrange_event.record()`, KHÔNG dời weight). Việc dời do **1 thread nền** (`async_worker.py`) làm.

**Nghi vấn: "để compute cần biết placement; vừa rearrange vừa compute sao đúng?"** → 3 cơ chế bảo đảm đúng:

**(a) Quyết định TRƯỚC, dời SAU.** Worker tính xong `new_map` trên CPU rồi mới bắt đầu dời weight
(`async_worker.py:66-76`). "Biết cần rearrange thế nào" đã có trước khi động vào trọng số.

**(b) Double-buffer + dời TỪNG LAYER, 1 layer / forward pass** (`async_worker.py:109-146`):
```
worker (nền): copy weight MỚI của layer L ──► expert_buffer   (comm chạy ở background stream)
main         : forward pass kế → copy expert_buffer ──► weight THẬT layer L + cập nhật map[L]
               (đồng bộ bằng consumed_event: worker không ghi đè buffer khi main chưa lấy xong)
               → lặp cho layer L+1, L+2 ... trải trên ~46 forward pass
```
Trọng số MỚI nạp vào **buffer riêng**, không đụng weight đang dùng. Chỉ khi layer L sẵn trong buffer, main
thread mới **swap** buffer→weight-thật của layer L, **cùng lúc** cập nhật `map[L]`.

**(c) Weight và map của MỖI layer đổi ĐỒNG THỜI, ATOMIC, giữa 2 forward pass.** Trong 1 forward pass, mỗi layer
luôn nhất quán: token định tuyến theo `map[L]` luôn tìm đúng expert đang nằm ở slot đó. **Không bao giờ có trạng
thái "nửa cũ nửa mới" trong 1 layer.** Các layer độc lập → layer 3 đã dùng placement mới trong khi layer 5 còn
cũ vẫn đúng, vì mỗi layer tự khớp weight với map của chính nó.

→ Cái **overlap** là giữa *copy-vào-buffer* (nền) và *compute* (main); còn lúc **thực sự đổi weight sống** chỉ
là copy buffer→weight + đổi map, rẻ, đặt ở ranh giới giữa 2 forward pass. Nhờ đó rearrange gần như **ẩn hoàn toàn**.

### Timeline async-default cụ thể
```
(bộ đếm trigger nạp sẵn 2250 → trigger lần 1 ở run-step 750; "step" dưới đây = run-step forward)
step 0..749  : chỉ đếm load (đổ vào window). Không dời gì. (bộ đếm 2250→2999)
step ~750    : TRIGGER (bộ đếm chạm 3000). main: all-reduce load → wake worker → ĐI TIẾP phục vụ (placement CŨ).
               worker (nền): tính new_map trên CPU (~ms).
step 751     : worker copy weight layer0 → buffer  ‖  main forward (placement cũ) → cuối pass swap layer0 + map[0]
step 752     : worker copy layer1              ‖  main forward ... swap layer1
   ...       : (mỗi forward dời 1 layer; ~46 forward thì xong toàn bộ)
step ~796    : dời xong 46 layer → placement mới hoàn tất. Serving KHÔNG hề treo. (trigger kế: step 3750, mỗi 3000)
```
So với sync: sync gom cả 46 layer dời 1 phát → treo 3–38s; async rải mỗi layer 1 forward pass → mỗi lần tí xíu,
ẩn vào nền. Đây là lý do async-default (45–48ms) nhanh & ổn hơn sync-default (nixl 55.9±21.2 do 1 lần treo 38s).

---

## 3b. Double-buffering trong code (chi tiết luồng async)

"Double-buffering" ở đây = **hai vùng nhớ tách biệt**: (1) `expert_weights` — trọng số **SỐNG** đang được forward
dùng; (2) `expert_buffer` — vùng **STAGING** để hứng trọng số MỚI. Worker ghi vào staging (qua comm nền), main
thread mới copy staging→sống. Nhờ tách 2 vùng, forward không bao giờ đọc trúng trọng số đang bị ghi dở.

### Cấu trúc dữ liệu (`eplb_state.py`)
- `expert_buffer: list[torch.Tensor]` — cấp phát 1 lần lúc init, kích thước = trọng số của **ĐÚNG 1 layer**
  (`eplb_state.py:448` `[torch.empty_like(w) for w in model.expert_weights[0]]`). **Dùng lại cho từng layer.**
- `pending_result: AsyncEplbLayerResult | None` — ô "bàn giao" 1 layer giữa worker↔main (`eplb_state.py:199`),
  đồng bộ bằng GIL, **tối đa 1 layer chờ**.
- `rearrange_event` — main báo worker bắt đầu. `consumed_event` (mỗi layer) — worker chờ main tiêu thụ xong buffer.

### Luồng đầy đủ
```
MAIN (tại step trigger, async)  — eplb_state.py:794-808
  ├─ snapshot global_expert_load_window.clone(); rebalanced = True
  └─ rearrange_event.record()  → đánh thức worker; ĐI TIẾP phục vụ (KHÔNG reset step counter, dòng 593-602)

WORKER (thread nền, CUDA stream riêng)  — async_worker.py:79-148
  ├─ rearrange_event.wait()                          # ngủ tới khi được đánh thức
  ├─ physical_to_logical_map → CPU
  ├─ new_map = policy.rebalance_experts(...)         # TÍNH placement mới (CPU) — QUYẾT ĐỊNH trước
  └─ for layer L in 0..num_layers-1:                 # DỜI weight, mỗi vòng 1 layer
       ├─ transfer_layer(L): comm gather weight MỚI của layer L  →  expert_buffer   (move_to_buffer)
       ├─ cuda_stream.synchronize()                  # chắc chắn buffer ghi xong
       ├─ pending_result = AsyncEplbLayerResult(L, new_map[L], metadata, consumed_event)  # HIỆN cho main
       └─ consumed_event.wait()                      # CHẶN worker tới khi main tiêu thụ xong buffer (back-pressure)

MAIN (mỗi forward step kế tiếp)  — eplb_state.py:579-591
  └─ if rebalanced AND _all_ranks_result_ready(L):   # CẢ 8 rank đều có pending_result layer L
       _move_to_workspace():
         ├─ move_from_buffer: expert_buffer → expert_weights[L]  (copy staging→SỐNG cho slot đổi)
         ├─ _commit_eplb_maps_for_layer: cập nhật physical_to_logical_map[L] = new_map[L]   # map đổi CÙNG weight
         ├─ consumed_event.record()                  # MỞ KHOÁ worker → nó dời layer L+1
         └─ pending_result = None
  # → mỗi forward pass áp đúng 1 layer; ~46 pass thì xong. Sau layer cuối: rebalanced=False, reset step counter.
```

### Vì sao "vừa dời vừa compute" vẫn ĐÚNG — 4 chốt chặn
1. **Quyết định trước, dời sau**: `new_map` tính xong (CPU) rồi mới bắt đầu comm dời weight (`async_worker.py:101`).
2. **Buffer tách khỏi weight sống**: comm ghi vào `expert_buffer`, KHÔNG đụng `expert_weights` đang forward dùng.
3. **Weight + map của mỗi layer đổi ATOMIC cùng nhau** trong `_move_to_workspace` (main thread, giữa 2 forward
   pass): `move_from_buffer` (weight) + `_commit_eplb_maps_for_layer` (map) đi liền → routing luôn khớp weight.
4. **Đồng bộ TOÀN RANK theo từng layer**: `_all_ranks_result_ready(L)` (all-reduce cờ) bắt **cả 8 rank** cùng
   sẵn sàng layer L rồi mới swap đồng loạt → trạng thái phân tán nhất quán (EP là collective, mọi rank phải
   cùng placement cho 1 layer). Và `consumed_event` là **back-pressure**: worker không ghi đè buffer cho layer
   L+1 khi main chưa lấy xong layer L → 1 buffer dùng lại an toàn.

→ Các layer độc lập nên "layer 3 đã placement mới, layer 5 còn cũ" vẫn đúng: mỗi layer tự khớp (weight, map).
Overlap thật sự = *comm-ghi-buffer (nền)* chồng lên *compute (main)*; còn swap buffer→sống chỉ là copy + đổi map,
rẻ, đặt ở ranh giới giữa 2 forward pass.

### Điểm tinh tế
- Trong lúc đang rearrange, **step counter KHÔNG reset** (`eplb_state.py:593-602` bail out) → EPLB chờ dời hết
  ~46 layer rồi mới reset & cho phép trigger lần sau. Không có 2 đợt rearrange chồng nhau.
- `transfer_layer` chỉ gửi/nhận **weight của expert thực sự đổi rank** (kết hợp `preserve_intragpu_slots` ở §6.6);
  expert ở yên trong rank chỉ copy nội bộ `w[dst].copy_(w[src])` (rebalance_execute.py:424), không tốn comm.
- Warmup lúc khởi động chạy `step(is_profile=True)` → ép nhánh SYNC (dòng 681 `... or is_profile`) để cấp phát
  buffer comm tối đa; đó là 2 dòng "Rearranged experts (profile)" thấy trong log async.

### Ví dụ "buffer tách khỏi weight sống" theo từng step
Bối cảnh: **rank0**, layer L, slot `s` đang giữ expert **Y**; placement mới muốn slot `s` giữ **X** (X đang ở rank3).
```
Step t   (worker, stream nền): move_to_buffer
   nhận X từ rank3 (comm) ──► expert_buffer[s] = X
   expert_weights[L][s] VẪN = Y   ◄── weight SỐNG không bị đụng;  map[L] VẪN cũ (slot s = Y)
   → forward pass ở step này: token route theo map cũ → tới Y → đọc expert_weights[L][s]=Y  ✓ ĐÚNG
     (X chưa được route tới rank0 vì map[L] chưa đổi)

Step t+k (main, giữa 2 forward pass): _move_to_workspace → move_from_buffer   (khi cả 8 rank sẵn sàng layer L)
   expert_weights[L][s].copy_(expert_buffer[s])   # Y → X : swap weight SỐNG
   map[L][s] = X                                   # đổi map CÙNG LÚC (atomic với weight)
   consumed_event.record()                         # mở khoá worker → dời layer L+1
   → forward pass kế: token của X → rank0 slot s → đọc expert_weights[L][s]=X  ✓ ĐÚNG
```
Giữa step t và t+k, `expert_weights[L]` **chưa đổi** → mọi forward pass ở giữa vẫn nhất quán với map cũ. Comm chỉ
ghi vào `expert_buffer`, nên forward đọc `expert_weights` không bao giờ trúng dữ liệu ghi-dở.

### VRAM: async KHÔNG gấp đôi
`expert_buffer` chỉ to bằng **trọng số expert của ĐÚNG 1 LAYER** trên rank đó, cấp phát 1 lần và **dùng lại** cho
mọi layer (dời 1 layer/forward-pass): `expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]`
(`eplb_state.py:448` — chỉ layer 0).
- VRAM thêm ≈ (trọng số expert) / num_layers ≈ **1/46 ≈ ~2%**, KHÔNG phải 2×. Buffer là staging tạm 1 layer, không
  phải bản copy toàn model.
- **Sync cũng dùng đúng buffer này** → async không tốn VRAM hơn sync; "double" chỉ là 1 layer tại một thời điểm.
- Đừng nhầm với **`num_redundant_experts`**: cái đó mới thêm VRAM lâu dài (mỗi replica = 1 bản expert thật, tồn tại
  suốt — xem §6.5). Double-buffer async chỉ là staging tạm ~2%.

---

## 4. Ba communicator: nixl vs nccl vs pynccl (phân biệt)

Communicator = **transport dời trọng số expert giữa các rank khi rearrange** (`config/parallel.py:39`:
`torch_nccl | torch_gloo | nixl | pynccl`). `communicator=None` → auto-select → **build moreh này resolve về `nixl`** (§7).

| | transport | tốc độ / lần rearrange | sync? | async? | thực đo MTP5 10k_c36 |
|---|---|---|---|---|---|
| **nixl** | NIXL/RIXL qua **UCX** (RDMA + staging buffer host) | **CHẬM ~37s** (ROCm UCX staging) | ✓ (cần UCX_TLS fix) | ✓ **ổn định nhất** | sync-freq **123.8** (chậm thảm); async 45–48 (ổn) |
| **nccl** (`torch_nccl`) | `torch.distributed` NCCL/RCCL **trên device (GPU)** | **NHANH ~3s** | ✓ (default sống) | ✗ **bị né** (multi-stream hang) | default-sync sống; **freq-sync CRASH**; async-freq CRASH |
| **pynccl** | PyNCCL `send/recv` (wrapper NCCL) | nhanh ~0.4–1s | ✓ (ổn hơn torch_nccl) | ✗ (freq crash) | sync/default sống; **freq-async CRASH** |
| *(gloo)* | `torch.distributed` gloo, **CPU staging** | rất chậm (trung chuyển qua host) | ✓ | — | chỉ test noMTP; bền nhưng chậm nhất |

### Chi tiết & khác biệt cốt lõi
- **nixl**: dời qua UCX/RDMA host-side → **KHÔNG phải kernel GPU** (trace GPU không thấy). Trên ROCm mỗi lần rearrange
  **~37s** (staging buffer) — rất chậm; **sync-nixl gần như vô dụng** (freq 18×37s → 123.8ms). NHƯNG async đẩy 37s
  xuống nền → **async-nixl ổn định nhất** trong mọi backend. Cần `UCX_TLS=…,rocm_copy,rocm_ipc` + `UCX_MEMTYPE_CACHE=n`
  (không có → crash "VRAM detected as host", xem `EPLB_NIXL_FIX.md`).
- **nccl (torch_nccl)**: dời = `ncclDevKernel` send/recv **trên device** → nhanh nhất (~3s). NHƯNG: (a) **async né
  torch_nccl** (xung đột multi-stream / hang `batched isend-irecv`, §7); (b) **rearrange dày → worker-stall CRASH**
  (`RPC 'sample_tokens' timed out`) — freq-sync-nccl 72/68/0, async-freq-nccl 68/0/0. Chỉ **default-sync** (rearrange
  thưa) sống được (72/72/71).
- **pynccl**: NCCL qua Python `send/recv`, nhanh (~0.4–1s), **ổn hơn torch_nccl khi sync** (freq-sync-pynccl là config
  freq DUY NHẤT sống). Nhưng **freq-async vẫn CRASH** (36/0/0) — cùng lớp worker-stall.
- Cả **nccl/pynccl**: kernel dời là `ncclDevKernel` nhưng nằm NGOÀI vùng MoE forward → `analyze_time.py` (chỉ soi
  kernel MoE) **không tính** vào `comm`.

### torch_nccl vs pynccl — CÙNG NCCL, khác WRAPPER (không phải 2 transport)
Cả 2 dùng **chung NCCL/RCCL device↔device** (đều sinh `ncclDevKernel`); chỉ khác đường gọi:
| | torch_nccl (`TorchDistNcclEplbCommunicator`) | pynccl (`PyNcclEplbCommunicator`) |
|---|---|---|
| Đường gọi | `torch.distributed.isend/irecv` + `batch_isend_irecv` (eplb_communicator.py:120,148) | `PyNcclCommunicator.send/recv` = **ctypes thẳng `libnccl`** `ncclSend/Recv` (`:600,610`) |
| Qua ProcessGroup? | Có (torch quản stream) | **Không** — bỏ qua torch.distributed |
| Kiểm soát stream | torch tự quản | **explicit `stream=cuda_stream`** (vLLM chọn) + `group_start/end` |
| Hệ quả | `batch_isend_irecv` **hang khi async** (multi-stream) → async NÉ | nhẹ hơn, tự quản stream → **ổn hơn khi sync** |
→ Đó là lý do sync-pynccl sống mà sync-torch_nccl chết (cùng freq); nhưng async-freq cả 2 vẫn crash vì nút thắt
là **worker-stall RPC giữa DP worker** (shm_broadcast), không phải ở lớp NCCL. So với **nixl** thì cả 2 mới là
"cùng transport" — nixl là transport KHÁC hẳn (UCX/RDMA + staging host, không phải kernel GPU).

### Chọn cái nào (kết luận thực nghiệm)
- Cần **async** (ẩn overhead rearrange): chỉ **nixl** thật sự ổn (nccl/pynccl async crash khi rearrange dày).
- Cần **sync**: **pynccl** ổn nhất; **torch_nccl** chỉ sống khi rearrange thưa (default); **nixl-sync** né vì 37s/lần.
- **Rearrange dày (freq)** = tử thần cho nccl & pynccl (worker-stall) — chỉ async-nixl chịu được.
- Trên MTP5 mọi lựa chọn đều ≈ baseline về tpot (EPLB không lợi) → nếu buộc bật: **async-nixl** (an toàn) hoặc **tắt hẳn**.

---

## 5. Vì sao profile "không thấy" chi phí rearrange
- **default step_interval=3000** (bộ đếm nạp sẵn 2250 → trigger lần 1 ở forward 750), mà profile chỉ chạy ~270–500 forward (< 750) → **trigger không bao giờ
  tới → 0 rearrange**. Không có gì để thấy.
- Kể cả khi rearrange xảy ra, `analyze_time.py` phân loại comm = `"ncclDevKernel" in name` (`analyze_time.py:90`)
  và chỉ đếm trong vùng MoE mỗi step (24800 = gather+reduce-scatter steady-state) → **không** gồm weight-movement
  của rearrange (nixl còn chẳng phải kernel GPU).

⇒ Profile chỉ đo **compute + all-to-all steady-state**; chi phí rearrange **vô hình** với profile → chỉ **bench**
(chạy đủ dài, đo end-to-end) mới tính được cái giá đó. Đây là lý do freq trông "đẹp" ở profile-imbalance nhưng
tệ ở bench-tpot.

---

## 6. Thuật toán policy tính placement — chi tiết + ví dụ 8 rank

Code: `policy/default.py` (thuật toán chuyển thể từ [DeepSeek EPLB](https://github.com/deepseek-ai/eplb)).

### 6.1 Physical expert vs logical expert (nền tảng để hiểu policy)
- **Logical expert**: 256 expert model THỰC có. **Router luôn địa chỉ hoá theo logical id** (top-8 của 1 token là 8 logical id).
- **Physical expert**: **slot trọng số thật trên GPU**. `num_physical = num_logical + num_redundant`. Mỗi physical
  slot chứa **bản copy trọng số của một logical expert**.
- Hai bảng ánh xạ:
  - `physical_to_logical_map[layer, phy_slot] = logical_id` — slot vật lý này đang giữ logical expert nào. **Đây
    chính là thứ policy tính ra.**
  - `logical_to_physical_map[layer, logical_id, :] = [phy_slot,...]` — ngược lại, 1 logical expert có mặt ở
    những slot vật lý nào (để router gửi token tới 1 trong các replica).
- **redundant=0** (config ta): 256 physical = 256 logical, **song ánh 1-1** — mỗi logical đúng 1 physical.
  Rearrange = **chỉ hoán vị GPU nào giữ logical expert nào** (không nhân bản).
- **redundant>0**: hot logical expert được cấp **nhiều physical slot** trên nhiều GPU → tải của nó **chia ra**.
- **`num_redundant_experts` là GLOBAL** (tổng trên TOÀN nhóm EP) mỗi layer — **KHÔNG per-rank**
  (`model_executor/layers/fused_moe/layer.py:182`: `global_num_experts = num_experts + num_redundant_experts`;
  `num_physical_experts = num_logical + num_redundant`). Số slot mỗi rank = `num_physical / num_ranks`
  (`local_num_experts`). Ràng buộc: `num_physical` phải chia hết cho `num_ranks`.
  - Ví dụ GLM5.2 **redundant=8**: 256 + 8 = **264 physical/layer TOÀN CỤC** → 264 / 8 rank = **33 slot/rank**
    (mỗi rank **+1** slot, KHÔNG phải +8). 8 slot thừa được policy rải cho các expert nóng nhất trên toàn 8 rank.
  - Ví dụ tí hon §6.5 (4 logical + redundant=2): 6 physical **TOÀN CỤC** → 6/2 GPU = 3 slot/GPU (mỗi GPU +1, không +2).

redundant=0: 8 GPU × 32 slot = 256 physical. Router chọn logical → tra `logical_to_physical` → gửi token tới GPU giữ slot đó.

### 6.2 Thuật toán tổng thể (hierarchical, 3 bước) — `rebalance_experts_hierarchical`
Input: `weight[layer, 256]` = tải mỗi logical expert (đã all-reduce toàn cục). Với config ta **num_nodes=1**
(1 máy 8 GPU), **redundant=0**:
1. **Pack group → node**: chia expert-group cho các node. num_nodes=1 → tất cả vào 1 node (bước này **vô hiệu**).
2. **Replicate trong node**: thêm redundant replica. redundant=0 → **không nhân bản** (physical=logical).
3. **Pack physical expert → GPU**: `balanced_packing(tải, 8)` — **đây là bước DUY NHẤT có tác dụng với config ta**:
   xếp 256 expert vào 8 GPU (32/GPU) sao cho tổng tải mỗi GPU cân nhau nhất.

→ Nên với async-default hiện tại, **policy = balanced_packing 256 expert vào 8 rank**. Hai bước kia chỉ bật khi
có nhiều node / có redundant.

### 6.3 `balanced_packing` — thuật toán LPT (Longest-Processing-Time greedy)
```
Sắp xếp item theo tải GIẢM DẦN.
Duyệt từng item: gán vào PACK (GPU) đang NHẸ NHẤT mà chưa đầy (mỗi pack đúng n/m item).
```
Đây là heuristic bin-packing kinh điển: nhét vật nặng trước, luôn bỏ vào thùng nhẹ nhất → cân bằng tốt.

**Ví dụ nhỏ — 8 expert → 4 rank (2 slot/rank), redundant=0:**
```
Tải: E0=100 E1=90 E2=80 E3=70 E4=60 E5=50 E6=40 E7=30   (đã sort giảm dần)

Gán lần lượt vào rank nhẹ nhất (còn chỗ):
  E0(100)→R0   E1(90)→R1   E2(80)→R2   E3(70)→R3     (mỗi rank 1 item)
  E4(60)→R3 nhẹ nhất(70) → R3=[E3,E4]=130 (đầy)
  E5(50)→R2(80)          → R2=[E2,E5]=130 (đầy)
  E6(40)→R1(90)          → R1=[E1,E6]=130 (đầy)
  E7(30)→R0(100)         → R0=[E0,E7]=130 (đầy)

KẾT QUẢ: R0=130 R1=130 R2=130 R3=130   → max/min = 1.00  (cân bằng hoàn hảo)
```
So với xếp NGÂY THƠ (liền khối R0=[E0,E1]...):
```
R0=[E0,E1]=190  R1=[E2,E3]=150  R2=[E4,E5]=110  R3=[E6,E7]=70  → max/min=2.71
```
→ LPT ghép "nặng nhất + nhẹ nhất" cùng rank → phẳng hơn hẳn. Đây chính là cách EPLB hạ imbalance.

### 6.4 Ví dụ thực tế — 256 expert → 8 rank; và vì sao super-hot không cứu được (redundant=0)
**(a) Lệch vừa phải** — giả sử 8 expert nóng (tải 100), 248 expert nguội (tải 10):
```
Placement liền khối ban đầu: rank r giữ expert [32r .. 32r+31].
Nếu 8 hot expert dồn vào rank 0,1 → rank0,1 quá tải, rank khác nhàn → imbalance cao.
Sau balanced_packing: rải 8 hot ra 8 rank (mỗi rank 1 hot + 31 nguội)
   → mỗi rank ≈ 100 + 31×10 = 410, đều nhau → imbalance ≈ 1.0
```
**(b) Một expert SIÊU HOT (E5=2000, còn lại ~10):**
```
balanced_packing đặt E5 một mình "gánh" 1 rank; rank đó = 2000 + 31×10 ≈ 2310.
Các rank khác ≈ 32×10 = 320.  → max/min ≈ 2310/320 = 7.2  VẪN LỆCH NẶNG.
```
→ Với **redundant=0**, một logical expert siêu hot **không thể chia nhỏ** (chỉ 1 physical slot) → luôn là trần
của 1 rank. Đây là lý do EPLB default (redundant=0) chỉ san được lệch "phân tán", **không cứu được hot-spot đơn lẻ**,
và vì sao trên MTP5 (routing khá đều) EPLB giảm imbalance rất ít.

### 6.5 `replicate_experts` + ví dụ đầy đủ "Thế giới tí hon"
Thuật toán cấp replica:
```
Bắt đầu: mỗi logical expert 1 replica (logcnt=1).
Lặp cho mỗi slot thừa: chọn expert có tải-hiệu-dụng cao nhất = argmax(weight / logcnt) → thêm 1 replica cho nó.
```

#### Thế giới tí hon
- **2 GPU** (rank0, rank1), **4 logical expert**: A, B, C, D. Mỗi token đi tới **1 expert** (top-1).
- **600 token**: A **siêu hot = 300 token**; B, C, D mỗi cái 100 token.
- Mỗi GPU chứa `num_physical / 2` slot.

#### Trường hợp redundant=0 (4 physical = 4 logical)
```
physical_to_logical = [A, B,  C, D]      # slot0,1 ở GPU0 ; slot2,3 ở GPU1
logical_to_physical: A→[0]  B→[1]  C→[2]  D→[3]      (replica_count: tất cả = 1)
```
balanced_packing xếp 4 expert vào 2 GPU (2/GPU), ví dụ GPU0=[A,D], GPU1=[B,C]:
```
GPU0 = A(300) + D(100) = 400        ⬅ nghẽn
GPU1 = B(100) + C(100) = 200
→ Imbalance = 400/200 = 2.0 ✗   (A chỉ có 1 slot → KHÔNG chia được → kẹt 1 GPU)
```

#### Trường hợp redundant=2 (6 physical = 4 logical + 2 thừa; TOÀN CỤC → 3 slot/GPU)
**Bước 1 — replicate_experts cấp 2 slot thừa cho expert nóng nhất:**
```
count ban đầu: A=1 B=1 C=1 D=1   (tải hiệu dụng = tải / count)
slot thừa #1: argmax(tải/count) = A (300/1=300)   → A.count=2  (A hiệu dụng 150)
slot thừa #2: argmax = A (300/2=150) vs B/C/D(100) → A.count=3  (A hiệu dụng 300/3=100)
→ A có 3 replica; B,C,D mỗi cái 1. Tổng 3+1+1+1 = 6 physical ✓. Tải hiệu dụng: TẤT CẢ = 100 (đều tăm tắp).
```
**Bước 2 — balanced_packing xếp 6 physical vào 2 GPU (3 slot/GPU):** 6 món đều =100 → chia kiểu gì cũng đều:
```
GPU0 (slot 0,1,2) = [A, A, B]
GPU1 (slot 3,4,5) = [A, C, D]
physical_to_logical = [A, A, B,  A, C, D]
logical_to_physical: A→[0, 1, 3]   B→[2]   C→[4]   D→[5]        replica_count: A=3, B=C=D=1
```
**Bước 3 — định tuyến lúc forward (nơi replica được DÙNG):** token muốn A chọn 1 trong 3 replica bằng Knuth hash:
```
replica_idx = (token_idx * 2654435769 & 0xFFFFFFFF) % replica_count(A=3)
physical_id = logical_to_physical[A][replica_idx]
→ hash rải ĐỀU token qua {0,1,2} → ~100 token mỗi replica của A:
     ~100 token → slot0 (GPU0)
     ~100 token → slot1 (GPU0)
     ~100 token → slot3 (GPU1)
(token muốn B/C/D: count=1 → replica_idx = hash%1 = 0 → luôn slot duy nhất)
```
**Bước 4 — tải thực tế mỗi GPU:**
```
GPU0 = A-slot0(100) + A-slot1(100) + B(100) = 300
GPU1 = A-slot3(100) + C(100)       + D(100) = 300
→ Imbalance = 300/300 = 1.0 ✓   (A siêu-hot đã bị XÉ thành 3 mảnh 100, rải 2 GPU → hết nghẽn)
```
**Bước 5 — ghi load & vòng lặp policy:** forward đếm load **theo physical slot** (slot0=slot1=slot3=100...); tới
trigger `scatter_add` **gộp replica về logical** → A = slot0+slot1+slot3 = 300 → policy thấy tổng tải thật của A =
300 → giữ 3 replica.

| | A chia được? | Imbalance |
|---|---|---|
| redundant=0 | Không (A chỉ 1 slot) | **2.0** (A kẹt 1 GPU) |
| redundant=2 | Có (A → 3 replica) | **1.0** (A xé đều) |

**Đổi lại**: mỗi replica tốn thêm 1 bản copy trọng số (VRAM) + tăng khối lượng dời khi rearrange.

#### Vì sao MTP5 redundant=8 KHÔNG ăn thua (thực nghiệm)
MTP5 routing **vốn đã đều** — không có "A siêu hot" như ví dụ; tải các expert xấp xỉ nhau (imbalance chỉ ~1.32).
Không có gì để xé → 8 replica cấp cho expert "hơi nóng" gần như vô dụng, mà vẫn tốn VRAM + hash + rearrange nhiều
slot. Thực đo cả 3 communicator (async, default interval, MTP5 10k_c36):

| comm | imbalance red0 → red8 | mean_tpot red0 → red8 | completed red8 |
|---|---|---|---|
| nixl | 1.341 → 1.321 | 45.2 → **52.8 ± 9.1** | 72/72/72 |
| nccl | 1.322 → **1.280** | 48.4 → **60.1 ± 3.0** | 72/72/72 |
| pynccl | 1.340 → 1.317 | 48.3 → **48.9 ± 13.0** | 72/72/**36** (1 run timeout) |

→ redundant=8 giảm imbalance **rất nhẹ** (1.28–1.32; nhiều nhất là nccl −3%) nhưng **tpot tệ hơn toàn bộ** (nccl
+24%, nixl +17%) + pynccl có 1 run chỉ 36/72 (timeout). Redundant chỉ đáng khi có **hot-spot thật** (imbalance
cao, như noMTP hoặc routing lệch mạnh) — trên MTP5 là **mất không**.

### 6.6 `preserve_intragpu_slots` — giảm copy không cần thiết
Sau khi có placement mới, policy **sắp lại thứ tự slot TRONG mỗi GPU** để expert nào **vẫn ở lại GPU cũ thì giữ
nguyên slot cũ** (pass 1), chỉ expert mới đến mới lấp slot trống (pass 2). Nhờ đó khi rearrange chỉ phải **copy
trọng số của expert thực sự đổi GPU**, không copy expert ở yên → giảm communication của bước dời.

### 6.7 Tóm tắt luồng policy (config async-default hiện tại)
```
weight[layer,256]  (tải logical, đã all-reduce)
   │  num_nodes=1, redundant=0
   ▼
balanced_packing(weight, 8 GPU)     ← LPT: sort giảm dần, nhét vào GPU nhẹ nhất, 32/GPU
   │
   ▼
new_physical_to_logical_map[layer, 256]   ← GPU nào giữ logical expert nào
   │
   ▼
preserve_intragpu_slots(...)        ← giữ slot cũ cho expert ở lại → ít copy
   │
   ▼
so với map cũ → chỉ dời weight của expert đổi GPU  (sync: blocking / async: nền, xem §3)
```

### 6.8 Luồng thực thi RUNTIME khi redundant>0 (định tuyến replica lúc forward)

redundant=0: mỗi logical expert có đúng 1 physical → **không có lựa chọn** (token luôn tới slot duy nhất).
redundant>0: hot expert có **nhiều physical replica trên nhiều rank** → mỗi token phải **CHỌN 1 replica**. Cơ chế
ở kernel triton `_eplb_map_and_record_triton` (`model_executor/layers/fused_moe/router/base_router.py`).

**Maps liên quan (mỗi layer):**
- `logical_to_physical_map[E] = [slot_0, …, slot_{k-1}, -1, …]` — k physical slot giữ E (trên các rank khác nhau).
- `logical_replica_count[E] = k` — số replica của E.

**Mỗi token × mỗi logical expert E trong top-8 (lúc forward):**
```
k          = logical_replica_count[E]                    # số replica của E
token_idx  = offs // top_k                               # token này là token thứ mấy
hashed     = (token_idx * 2654435769) & 0xFFFFFFFF       # Knuth multiplicative hash
replica_idx= hashed % k                                  # CHỌN 1 trong k replica (rải đều theo token)
physical_id= logical_to_physical_map[E, replica_idx]     # slot vật lý thật
→ dispatch token tới rank chứa physical_id
→ expert_load_view[physical_id] += 1                     # ghi load THEO PHYSICAL (mỗi replica đếm riêng)
```
Cốt lõi: **token được hash (Knuth) rồi mod số replica → phân tán token của E ĐỀU ra k replica** → k rank chia nhau
tải của E. Với redundant=0 ⇒ k=1 ⇒ `replica_idx=0` luôn ⇒ không phân tán (đúng như bản đã hiểu).

**Ghi load theo PHYSICAL, policy quy về LOGICAL** (vòng khép kín):
```
forward : token --Knuth hash--> 1 trong k replica --> load[physical]++      (tải E rải đều k rank)
   │
trigger : load[physical] --scatter_add (eplb_state.py:704)--> load[logical E]  (GỘP tải mọi replica của E)
   │
policy  : replicate_experts (§6.5) → E nóng được thêm replica;  balanced_packing → rải k replica ra k rank
   │
rearrange (double-buffer §3b): tạo/dời các bản replica tới rank mới
```

**Ví dụ E5 có 3 replica (đặt ở rank 2, 5, 7):**
- Token vào E5: `replica_idx = hash(token) % 3` → ~1/3 token tới rank2, 1/3 rank5, 1/3 rank7.
- Mỗi rank chỉ gánh ~1/3 tải E5 → hết nghẽn (thay vì 1 rank gánh 100%).
- Load ghi riêng cho 3 physical slot; trigger sau gộp lại = tổng tải E5 → nếu vẫn nóng, policy giữ/tăng replica.

**Chi phí thêm so redundant=0:** mỗi token +1 hash+mod+lookup (rẻ, trong kernel); **VRAM +k bản weight** cho expert
nhân bản (lâu dài — khác staging buffer async §3b chỉ ~2%); rearrange phải dời nhiều slot hơn.

---

Kết quả thực nghiệm đầy đủ: xem `EPLB_BENCH_DOC.md`. Cơ chế chi tiết line-by-line: `EPLB_MECHANISM.vi.md`.
