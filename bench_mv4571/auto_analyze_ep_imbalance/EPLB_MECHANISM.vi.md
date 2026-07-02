# vLLM EPLB (Expert Parallel Load Balancing) — Phân tích cơ chế chuyên sâu

Tài liệu mô tả từng dòng cách EPLB hoạt động trong cây vLLM được vendored này
(3rdparty/vllm), được tạo ra cho công việc EP-imbalance của GLM5.2 (MV-4571). Các tham chiếu code
đều tương đối theo repo dưới 3rdparty/vllm. Xem EPLB_PROGRESS.md để biết kết quả thực nghiệm
và EPLB_NIXL_FIX.md để biết bản vá NIXL/UCX.

---

# 0. Demo trực quan (đọc cái này trước)

Phần này minh hoạ EPLB bằng ví dụ cụ thể với **cấu hình mặc định** `window_size=1000, step_interval=3000`,
tách rõ **sync** vs **async**. Chi tiết code ở các phần sau.

## 0.1 Bối cảnh ví dụ
GLM5.2, EP8: **8 GPU (rank 0..7)**, **256 logical expert**, mỗi token router chọn **top-8** expert.
Mỗi rank giữ 256/8 = **32 expert**. `num_redundant_experts=0` (không nhân bản).
EPLB = một vòng lặp nền bám theo từng **step** (mỗi step = 1 forward pass của engine).

## 0.2 EPLB làm gì tại MỖI step (rẻ — chỉ đếm)
Mỗi step, EPLB chỉ **cộng dồn bộ đếm**: expert nào vừa nhận bao nhiêu token. Nó giữ **cửa sổ trượt
1000 step gần nhất** (`window_size`). **Không di chuyển gì cả.**

```
run-step  :   0     1    ...    750               751  ...   3750            ...
đếm-trigger: 2250  2251  ...   3000 → TRIGGER→ reset 0  ...  3000 → TRIGGER   ...
             └── mỗi step: load[expert] += token (giữ cửa sổ 1000 step gần nhất) ──┘
                                ▲                              ▲
                          TRIGGER lần 1                   TRIGGER kế
                   (đếm-trigger NẠP SẴN 2250 = ¾·3000       (mỗi 3000 step)
                    → chạm 3000 sau chỉ 750 step)
```
Lưu ý: **2250 là giá trị KHỞI TẠO** của bộ đếm trigger (`eplb_state.py:431-436`, "initial progress = 3/4"), KHÔNG
phải bước trigger. Trigger khi bộ đếm ≥ `step_interval`=3000 → **lần đầu ở run-step 750** (2250→3000), rồi mỗi 3000.

## 0.3 EPLB làm gì tại step TRIGGER (đắt — có thể di chuyển weight)
Cứ **mỗi `step_interval`=3000 step** (lần đầu sớm hơn — ở run-step 750 do bộ đếm nạp sẵn 2250), EPLB:
1. Lấy load trung bình mỗi expert qua **1000 step gần nhất**.
2. Chạy **policy** → tính vị trí đặt mới (expert nào nằm rank nào).
3. Nếu vị trí mới ≠ hiện tại → **DI CHUYỂN trọng số expert giữa các rank** (qua communicator nccl/nixl/pynccl).
   Đây chính là **"rearrange"** — bước tốn kém.

**Cách nó cân bằng (ví dụ thu nhỏ 2 rank / 4 expert cho dễ nhìn):**
```
Load đo trong cửa sổ:  E0=1000 (HOT)   E1=100   E2=120   E3=90
TRƯỚC rearrange:                          SAU rearrange (re-permutation, redundant=0):
  rank0 = [E0, E1]  -> tải 1100  ⬅ nghẽn    rank0 = [E0]        -> tải 1000  (vẫn hot)
  rank1 = [E2, E3]  -> tải  210             rank1 = [E1,E2,E3]  -> tải  310
        max/min = 1100/210 = 5.2x                    max/min = 1000/310 = 3.2x  (phẳng hơn)
```
→ MoE layer chỉ xong khi rank chậm nhất xong; kéo max xuống = layer nhanh hơn. Với `redundant=0`, một
expert **siêu hot đơn lẻ** (E0) vẫn là trần → muốn phẳng hơn nữa phải bật `num_redundant_experts>0` để
**nhân bản E0 ra 2 rank** và chia đôi tải của nó. (Đây là lý do default trên MTP5 giảm imbalance rất ít.)

## 0.4 SYNC vs ASYNC — khác nhau CHÍNH ở bước di chuyển weight (bước 3)

```
SYNC  (use_async=false):
  ...step 2249 ──► [ step 2250: ENGINE DỪNG SERVING, copy weight 3–38s ] ──► step 2251 ──► ...
                          ▲                                                        
                    request TREO ở đây (spike latency; log "Rearranged experts in Xs")
                    nếu rearrange dày (freq) → treo lặp lại → có thể worker-stall CRASH

ASYNC (use_async=true):
  ...step 2249 ──► step 2250 ──► step 2251 ──► step 2252 ──► ...   (SERVING LIÊN TỤC, placement CŨ)
                       │
                       └──► [background thread copy weight] ───► xong → SWAP sang placement mới ở step kế
                            (overlap với compute; serving KHÔNG treo; log "Rearranging (async mode)")
```

- **sync**: đơn giản, dùng được `torch_nccl`; nhưng forward pass **đứng im** trong lúc copy weight → spike latency.
- **async**: giao việc copy cho **1 luồng nền** rồi phục vụ tiếp bằng placement cũ; copy xong mới đổi sang mới
  → **ẩn** gần hết stall. Đổi lại: communication phức tạp hơn và **không dùng được `torch_nccl` thuần** cho async
  (xung đột multi-stream — xem §7), nên async hay đi với nixl/pynccl.

## 0.5 Nối với kết quả thực nghiệm (vì sao con số như đã thấy)
- **Profile ngắn (~270–500 forward) < 750** (trigger lần 1) → **0 rearrange** → EPLB chỉ đếm, không cân bằng →
  **y hệt baseline** (đó là lý do imbalance profile của default ≈ baseline ~1.32).
- **Bench dài hơn (vượt 750, rồi mỗi 3000)** → **1–2 rearrange** thật → có cân bằng nhẹ + **trả overhead**:
  - sync-nixl-default: 1 lần rearrange 38s **block serving** → tpot 55.9±21.2 (variance cao vì 1 run dính 38s).
  - async-*-default: 38s đó chạy nền → tpot 45–48ms, hết variance.
- **freq (`step_interval=100`)**: trigger **rất dày** → rearrange lặp lại liên tục → sync-nixl chậm thảm
  (18×37s → 123.8ms), nccl/pynccl worker-stall → **CRASH**. Đây là lý do freq tệ hơn default khi bench.

---


# 1. Khái niệm & Ngữ nghĩa cấu hình

Phiên bản chính xác đến từ một `_version.py` được sinh tự động. Tôi có toàn bộ code cần thiết để tạo ra phần này.

## EPLB: Expert Parallel Load Balancing — Khái niệm và Ngữ nghĩa cấu hình

> Các tham chiếu code là tương đối theo repo `3rdparty/vllm`. Hai tệp có thẩm quyền là `config/parallel.py` (các dataclass cấu hình + validation) và `engine/arg_utils.py` (cách CLI/YAML điền vào chúng). Cây vendored này ghim phiên bản của nó qua một `_version.py` được sinh tự động (`vllm/version.py:5`), nên chuỗi phiên bản chính xác phụ thuộc thời điểm build; ngữ nghĩa bên dưới được đọc trực tiếp từ mã nguồn trong cây này và chính xác đối với nó. Ở những chỗ hành vi phụ thuộc phiên bản/phần cứng, tôi sẽ chỉ rõ ràng.

### 1. Bài toán mà EPLB giải quyết

Dưới **Expert Parallelism (EP)**, các expert của một lớp MoE được shard vật lý trên các rank (GPU). Router của một token chọn ra một tập con top-k nhỏ các expert cho mỗi lớp; token sau đó được dispatch (all-to-all) đến rank (hay các rank) đang giữ những expert đó, được tính toán ở đó, rồi được combine trở lại. Điều này được bật bởi `enable_expert_parallel` (`config/parallel.py:154`: *"Use expert parallelism instead of tensor parallelism for MoE layers."*).

Chế độ hỏng: **lưu lượng routing thực tế không đồng đều**. Một số expert ("hot" expert) nhận được nhiều token hơn hẳn các expert khác. Vì expert bị ghim vào rank, một rank giữ các hot expert trở thành kẻ chậm chân (straggler) — nó phải tính toán nhiều hơn và tạo ra nhiều lưu lượng all-to-all hơn các rank ngang hàng. Vì lớp MoE không thể hoàn thành cho đến khi rank chậm nhất hoàn thành, sự mất cân bằng load giữa các rank trực tiếp giới hạn throughput và làm tăng latency. Càng scale ra nhiều rank (wide-EP), một hot rank đơn lẻ càng gây tổn hại nặng.

**EPLB** tấn công vấn đề này bằng cách (a) cho phép một số expert được **replicate** (nhân bản) trên các rank (redundant expert) và (b) **định kỳ quyết định lại vị trí đặt vật lý** của các expert lên các rank dựa trên load quan sát được, để các hot expert nhận được nhiều replica hơn / được trải ra các rank đang under-load. Đây là một vòng lặp rebalancing động, tại runtime, đặt chồng lên trên EP tĩnh.

### 2. Logical expert so với physical expert, redundant expert

- **Logical expert**: các expert mà model thực sự định nghĩa (ví dụ 256 routed expert trong một MoE kiểu DeepSeek). Router luôn địa chỉ hóa (addresses) đến các logical expert.
- **Physical expert**: các slot trọng số cụ thể trên từng rank, chứa trọng số của expert. Với EPLB, `num_physical_experts = num_logical_experts + num_redundant_experts`. Các slot thừa là các **replica** của các logical expert (thường là hot).
- **Redundant expert** (`num_redundant_experts`, `config/parallel.py:69`): số lượng slot physical *thừa* vượt quá số lượng logical. `0` (mặc định) nghĩa là một hoán vị (permutation) thuần túy — vị trí đặt có thể được rearrange nhưng không expert nào bị nhân đôi, nên sự giảm tải chỉ đến từ việc di chuyển expert sang các rank tốt hơn. `>0` nghĩa là các hot logical expert có thể có nhiều bản copy vật lý trên các rank khác nhau, cho phép chia nhỏ tải token của chúng. Có một map gián tiếp từ logical→physical expert, và việc dispatch của router được ánh xạ lại qua nó. Map này chính là thứ mà bước rearrange tính toán lại.

Liên quan nhưng khác biệt là **chiến lược đặt tĩnh** (`expert_placement_strategy`, `config/parallel.py:167`), thứ điều khiển layout logical→physical *ban đầu*:
- `"linear"` (mặc định): các expert được đặt liền kề nhau — 4 expert / 2 rank → rank0 `[0,1]`, rank1 `[2,3]`.
- `"round_robin"`: xen kẽ (interleaved) — rank0 `[0,2]`, rank1 `[1,3]`. Docstring lưu ý rằng cách này "can help improve load balancing for grouped expert models with no redundant experts" — tức là nó là một biện pháp giảm nhẹ tĩnh rẻ tiền khi bạn *không* phải trả giá cho redundant expert.

EPLB là lớp động đặt lên trên bất kỳ chiến lược tĩnh nào bạn khởi đầu.

### 3. Chu kỳ rearrange (vòng lặp chạy như thế nào)

EPLB chạy một vòng lặp nền gắn với các bước forward của model:
1. **Ghi nhận load**: mỗi bước, mỗi rank tích lũy số token đã đi vào từng (physical) expert. Chỉ giữ lại `window_size` bước gần nhất (một sliding window — cửa sổ trượt — của thống kê load).
2. **Rearrange**: cứ mỗi `step_interval` bước, load đã ghi nhận được đưa vào **policy** cân bằng, thứ tính toán một vị trí đặt logical→physical mới. Layout vật lý mới hàm ý phải di chuyển các tensor trọng số của expert giữa các rank — sự di chuyển đó đi qua backend **communicator** của EPLB. Khi `use_async=True`, sự di chuyển này là không chặn (non-blocking, được overlap với compute); khi `False` nó là một rearrange chặn (blocking), đồng bộ.
3. **(Tùy chọn) log balancedness**: nếu `log_balancedness` bật, một metric balancedness (độ cân bằng) được tính và ghi log (cứ mỗi `log_balancedness_interval` lần rearrange/bước), với cái giá là communication thêm.

Mối quan hệ giữa hai interval này rất quan trọng và được chính mã nguồn nêu rõ (`config/parallel.py:62-67`): nếu `step_interval > window_size`, thì rearrange chỉ bao giờ thấy được `window_size` bước dữ liệu gần nhất — chính window, chứ không phải interval, mới giới hạn lượng lịch sử được đưa vào một quyết định.

### 4. Dataclass `EPLBConfig` — từng trường một

Được định nghĩa tại `config/parallel.py:55-105`, được decorate bằng `@config` (`config/parallel.py:55`), thứ đăng ký nó như một dataclass cấu hình của vLLM (decorator `@config` nằm tại `config/utils.py:41+`; nó làm cho lớp có thể introspect được bởi cơ chế phân tích tham số ở §6). Mọi trường đều dùng một default `Field(...)` của pydantic kèm ràng buộc (bounds), nên các giá trị ngoài khoảng bị từ chối ngay lúc khởi tạo, chứ không bị clamp một cách im lặng.

#### `window_size: int = Field(default=1000, gt=0)` — `config/parallel.py:59`
- **Điều nó điều khiển**: độ dài của sliding-window (tính theo số bước forward) mà trên đó tải token theo từng expert được tích lũy trước khi được trao cho policy rearrange.
- **Đơn vị**: số bước forward của model.
- **Mặc định**: `1000`. Ràng buộc `gt=0` (phải dương thực sự; `0` hoặc âm bị từ chối).
- **Tác động / đánh đổi**: window lớn hơn = ước lượng load mượt hơn, ổn định hơn (ít phản ứng với các đợt bùng phát tạm thời, ít xáo trộn vị trí đặt) nhưng thích nghi chậm hơn với một sự dịch chuyển thực sự trong mẫu lưu lượng. Window nhỏ hơn = phản ứng nhanh hơn nhưng nhiễu hơn, có nguy cơ thrash (dao động liên tục) vị trí đặt. Như đã lưu ở trên, nếu `window_size < step_interval` thì các bước thừa giữa các window thực chất bị bỏ đi đối với quyết định đó.

#### `step_interval: int = Field(default=3000, gt=0)` — `config/parallel.py:61`
- **Điều nó điều khiển**: mức độ thường xuyên (tính theo bước) mà rearrange thực sự được kích hoạt. Docstring của chính nó (`config/parallel.py:62-67`): *"Interval for rearranging experts in expert parallelism."*
- **Đơn vị**: số bước forward của model.
- **Mặc định**: `3000`. Ràng buộc `gt=0`.
- **Tác động / đánh đổi**: đây là núm vặn đánh đổi giữa **độ trễ thích nghi và chi phí rearrange**. Một lần rearrange di chuyển các tensor trọng số qua các rank (communication + có thể stall nếu đồng bộ), nên nó không miễn phí. Interval nhỏ → vị trí đặt bám sát load nhanh nhưng bạn phải trả chi phí rearrange thường xuyên; interval lớn → chi phí phân bổ (amortized) rẻ nhưng vị trí đặt có thể lỗi thời trong một thời gian dài. Mặc định `3000` so với window `1000` nghĩa là: theo mặc định, window *nhỏ hơn* interval, nên mỗi lần rearrange dùng 1000 bước gần nhất.

#### `num_redundant_experts: int = Field(default=0, ge=0)` — `config/parallel.py:69`
- **Điều nó điều khiển**: số lượng slot physical expert *thừa* (replica) vượt quá số lượng logical expert. Docstring `config/parallel.py:70`: *"Number of redundant experts to use for expert parallelism."*
- **Đơn vị**: số lượng slot physical expert (toàn cục, trên toàn bộ nhóm EP).
- **Mặc định**: `0`. Ràng buộc `ge=0`.
- **Tác động / đánh đổi**: `0` = re-permutation thuần túy (vị trí đặt có thể di chuyển nhưng không expert nào bị nhân đôi → một expert siêu-hot đơn lẻ vẫn có thể là nút thắt cổ chai trên rank duy nhất của nó). `>0` = các hot expert có thể được replicate để tải của chúng được chia trên các rank, cho policy có khoảng trống thực sự để cân bằng — nhưng mỗi slot thừa tốn thêm bộ nhớ GPU (một bản copy bổ sung của trọng số expert đó) và làm tăng khối lượng di chuyển trọng số trong lúc rearrange. **Được bảo vệ (Guarded)**: đặt giá trị này `!= 0` trong khi EPLB bị tắt là một lỗi cứng (xem §5) — redundant expert vô nghĩa nếu không có bộ balancer.

#### `log_balancedness: bool = False` — `config/parallel.py:72`
- **Điều nó điều khiển**: có tính và ghi log một metric "balancedness" theo từng bước hay không. Docstring `config/parallel.py:73-76` nói rõ: *"This is turned off by default since it will cause communication overhead."*
- **Mặc định**: `False`.
- **Tác động / đánh đổi**: thuần túy về khả năng quan sát (observability). Bật nó lên cho phép bạn *thấy* các expert hiện đang cân bằng tốt đến mức nào (hữu ích để tinh chỉnh `window_size`/`step_interval`/`num_redundant_experts`), nhưng tính metric đó đòi hỏi một collective, nên nó thêm chi phí communication trên đường nóng (hot path). Để tắt trong production; bật lên khi tinh chỉnh.

#### `log_balancedness_interval: int = Field(default=1, gt=0)` — `config/parallel.py:77`
- **Điều nó điều khiển**: mức độ thường xuyên metric balancedness được ghi log (chỉ liên quan khi `log_balancedness=True`).
- **Đơn vị**: bước/chu kỳ-log.
- **Mặc định**: `1` (log mỗi lần). Ràng buộc `gt=0`.
- **Tác động / đánh đổi**: tăng nó lên để điều tiết (throttle) chi phí/khối lượng log khi bạn vẫn muốn metric nhưng không phải trên mọi bước đơn lẻ. Nó chỉ có ý nghĩa khi `log_balancedness` bật — bộ validator ràng buộc chúng lại với nhau (§4.1).

#### `use_async: bool = True` — `config/parallel.py:81`
- **Điều nó điều khiển**: EPLB không chặn so với chặn. Docstring `config/parallel.py:82-84`: *"Whether to use non-blocking EPLB."*
- **Mặc định**: `True`.
- **Tác động / đánh đổi**: async (`True`) overlap sự di chuyển trọng số expert của một lần rearrange với compute đang diễn ra, che giấu phần lớn stall của rearrange — với cái giá là communication phức tạp hơn và một lựa chọn backend bị ràng buộc (NCCL bị tránh một cách rõ ràng cho async — xem §7). Sync (`False`) thực hiện rearrange như một thao tác chặn: đơn giản hơn, có thể dùng `torch_nccl`, nhưng forward pass bị stall trong lúc trọng số di chuyển. **Ràng buộc**: async chỉ hợp lệ với policy `"default"` (được thực thi bởi validator, §4.1).

#### `policy: EPLBPolicyOption = "default"` — `config/parallel.py:86`
- **Điều nó điều khiển**: thuật toán cân bằng tải nào sẽ tính toán vị trí đặt mới. Kiểu là `EPLBPolicyOption = Literal["default"]` (`config/parallel.py:37`).
- **Mặc định / cho phép**: `"default"` — và trong cây *này* đó là giá trị **duy nhất** được cho phép (`Literal` chỉ có một thành viên). Nên hiện tại núm này thực chất là cố định; nó tồn tại như một điểm mở rộng cho các policy tương lai. Nếu bạn truyền bất cứ thứ gì khác, validation `choices` của argparse (được suy ra từ `Literal`, §6) sẽ từ chối nó.
- **Tương tác**: async EPLB yêu cầu giá trị này phải giữ ở `"default"` (§4.1).

#### `communicator: EPLBCommunicatorBackend | None = None` — `config/parallel.py:89`
- **Điều nó điều khiển**: transport được dùng để di chuyển trọng số expert giữa các rank trong lúc rearrange. Kiểu `EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]` (`config/parallel.py:39`). Docstring `config/parallel.py:90-97`:
  - `"torch_nccl"` — `torch.distributed` trên process group của thiết bị (GPU).
  - `"torch_gloo"` — `torch.distributed` gloo với **CPU staging** (trọng số được trung chuyển (staged) qua bộ nhớ host).
  - `"nixl"` — NIXL/RIXL với các buffer send/recv được staged.
  - `"pynccl"` — PyNCCL send/recv.
  - `None` — **tự động chọn** (được resolve trong `__post_init__`, xem §7).
- **Mặc định**: `None` (auto).
- **Tác động / đánh đổi**: đây là núm hiệu năng/tương thích cho bước chuyển trọng số. Chuyển dựa trên NCCL là nhanh nhất device-to-device nhưng **không tương thích với async EPLB** (xung đột multi-stream / treo (hang) khi batched isend-irecv — xem §7); gloo/CPU-staging thì bền vững nhưng chậm hơn; nixl được ưa chuộng khi có sẵn. Đa số người dùng để nó `None` và để vLLM tự chọn.

#### 4.1 Bộ validator của `EPLBConfig` — `config/parallel.py:99-105`
```python
@model_validator(mode="after")
def _validate_eplb_config(self) -> Self:
    if self.use_async and self.policy != "default":
        raise ValueError("Async EPLB is only supported with the default policy.")
    if self.log_balancedness and self.log_balancedness_interval <= 0:
        raise ValueError("log_balancedness_interval must be greater than 0.")
    return self
```
Chạy sau khi tất cả các trường được thiết lập. Hai bất biến (invariant) liên-trường: (1) async yêu cầu policy default; (2) nếu logging balancedness bật, interval của nó phải dương (một kiểm tra dây-lưng-và-đai-quần chồng lên trên `gt=0` của trường, bảo vệ trường hợp ràng buộc của trường bị bỏ qua). Vi phạm sẽ raise ngay lúc khởi tạo cấu hình — trước bất kỳ công việc GPU nào.

### 5. Nơi EPLB cắm vào `ParallelConfig`

`EPLBConfig` không độc lập; nó treo dưới `ParallelConfig`:

- `enable_eplb: bool = False` — `config/parallel.py:163-164`: *"Enable expert parallelism load balancing for MoE layers."* Đây là công tắc bật/tắt tổng.
- `eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)` — `config/parallel.py:165`: cấu hình lồng bên trong. `default_factory` nghĩa là mỗi `ParallelConfig` nhận một `EPLBConfig` mới toanh của riêng nó với các mặc định ở trên (tránh lỗi shared-mutable-default).

Việc validation xuyên suốt (cross-cutting) nằm trong `ParallelConfig._validate_parallel_config` (`@model_validator(mode="after")`, `config/parallel.py:424`). Khối liên quan đến EPLB là `config/parallel.py:459-480`:

```python
if self.enable_eplb:
    if not current_platform.is_cuda_alike():
        raise ValueError("Expert parallelism load balancing is only supported on "
                         "CUDA devices or ROCm devices now.")
    if not self.enable_expert_parallel:
        raise ValueError("enable_expert_parallel must be True to use EPLB.")
    if self.tensor_parallel_size * self.data_parallel_size <= 1:
        raise ValueError("EPLB requires tensor_parallel_size or data_parallel_size "
                         f"to be greater than 1, but got "
                         f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}.")
else:
    if self.eplb_config.num_redundant_experts != 0:
        raise ValueError("num_redundant_experts is set to "
                         f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                         "enabled. Either enable EPLB or unset num_redundant_experts.")
```

Từng khối một:
- `config/parallel.py:460-464` — EPLB chỉ được hỗ trợ trên các thiết bị kiểu-CUDA (NVIDIA CUDA **hoặc** AMD ROCm; `is_cuda_alike()` bao trùm cả hai). Trên bất kỳ backend nào khác nó lỗi cứng. Đây là cổng chặn liên quan đến ROCm cho cây này.
- `config/parallel.py:465-466` — EPLB *yêu cầu* EP phải được bật. Cân bằng tải vị trí đặt expert là vô nghĩa nếu ngay từ đầu các expert không được shard theo expert-parallel.
- `config/parallel.py:467-472` — EPLB yêu cầu `TP * DP > 1`, tức là phải có nhiều hơn một rank để cân bằng qua lại. Trên một rank đơn lẻ không có gì để rebalance.
- `config/parallel.py:473-480` (nhánh `else`) — nếu EPLB **tắt**, `num_redundant_experts` phải là `0`. Đây là guard được nhắc tới ở §4: redundant expert là một no-op khi không có bộ balancer, nên một giá trị khác không được coi là lỗi của người dùng và bị từ chối kèm thông báo rõ ràng.

Còn có một tương tác **elastic-EP** trong `ParallelConfig.__post_init__` (`config/parallel.py:786-788`): `enable_elastic_ep` yêu cầu `enable_eplb=True` (việc scale co giãn lên/xuống được xây trên cơ chế EPLB).

### 6. Cách `enable_eplb` / `eplb_config` chảy từ CLI / YAML `--config` vào `ParallelConfig`

Đường đi là: **CLI args / YAML → các trường của dataclass `EngineArgs` → các tham số argparse → `create_engine_config` → `ParallelConfig(...)`**.

**(a) `EngineArgs` phản chiếu các trường cấu hình.** Trong `engine/arg_utils.py`:
```python
eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")   # arg_utils.py:494
enable_eplb: bool = ParallelConfig.enable_eplb                        # arg_utils.py:495
```
`enable_eplb` copy giá trị mặc định thuần (`False`). `eplb_config` dùng `get_field(ParallelConfig, "eplb_config")` (`config/utils.py:83-112`) thay vì gán trực tiếp. **Vì sao**: `eplb_config` trong `ParallelConfig` là một `Field(default_factory=EPLBConfig)` của pydantic; `get_field` trích xuất cái `default_factory` đó và tái dựng một `field(...)` dataclass đúng đắn với factory còn nguyên vẹn (`config/utils.py:99-106` unwrap cái `FieldInfo`). Điều này bảo toàn ngữ nghĩa "instance mới cho mỗi EngineArgs" và tránh một shared mutable default xuyên các instance engine-args.

**(b) Các kwargs của argparse được sinh tự động từ kiểu của trường.** `parallel_kwargs = get_kwargs(ParallelConfig)` (`arg_utils.py:941`) → `_compute_kwargs` (`arg_utils.py:286`). Với mỗi trường, `_compute_kwargs` xem xét các type hint (`arg_utils.py:292`) và chọn một `type`/`action` argparse:
- `enable_eplb` là một `bool`, nên nó rơi vào `arg_utils.py:343-345` → `action = argparse.BooleanOptionalAction`. Điều đó tự động tạo ra **cả** `--enable-eplb` lẫn `--no-enable-eplb`.
- `eplb_config` là một dataclass, được phát hiện tại `arg_utils.py:295-296` (`is_dataclass`), nên nó đi vào nhánh `dataclass_cls is not None` (`arg_utils.py:326-337`). `type` của argparse cho nó trở thành closure `parse_dataclass` lồng bên trong (`arg_utils.py:328-335`):
  ```python
  def parse_dataclass(val, cls=dataclass_cls):
      val = _expand_json_human_readable_numbers(val)
      return TypeAdapter(cls).validate_json(val)
  ```
  Nên giá trị CLI cho `--eplb-config` được parse dưới dạng **JSON** và được validate thẳng vào một `EPLBConfig` qua `TypeAdapter` của pydantic. `_expand_json_human_readable_numbers` (`arg_utils.py:263-282`) trước tiên mở rộng các hậu tố như `1k`/`3k` bên trong JSON để bạn có thể viết chẳng hạn `{"window_size": 1k, "step_interval": 3k}`. Bất kỳ `ValidationError` nào của pydantic đều được raise lại thành một `argparse.ArgumentTypeError` (`arg_utils.py:332-333`), nên một trường sai sẽ hiện ra như một lỗi CLI thông thường. Phần help text cũng nhận được gợi ý JSON (`arg_utils.py:337`): *"Should either be a valid JSON string or JSON keys passed individually."*

**(c) Các tham số được đăng ký.** `arg_utils.py:1093-1094`:
```python
parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
```
`enable_expert_parallel` / `-ep` được đăng ký ngay phía trên tại `arg_utils.py:1061-1065`, và `--expert-placement-strategy` tại `arg_utils.py:1095-1098`.

**(d) YAML `--config` và dạng dict được chuẩn hóa.** Một giá trị `--config foo.yaml` hoặc `EngineArgs(eplb_config={...})` theo kiểu lập trình sẽ đến dưới dạng một `dict` thuần. `EngineArgs.__post_init__` ép kiểu nó (`arg_utils.py:729-730`):
```python
if isinstance(self.eplb_config, dict):
    self.eplb_config = EPLBConfig(**self.eplb_config)
```
Nên dù cấu hình đến dưới dạng một chuỗi JSON trên CLI (→ `parse_dataclass`) hay dưới dạng một mapping lồng nhau từ YAML/kwargs (→ ép kiểu này), cuối cùng nó đều trở thành một instance `EPLBConfig` thực sự. Lưu ý đường đi này (`EPLBConfig(**dict)`) chạy các ràng buộc trường của pydantic + validator `_validate_eplb_config` (§4.1), nên các giá trị không hợp lệ cũng bị bắt ở đây.

**(e) Bàn giao vào `ParallelConfig`.** Trong `create_engine_config`, hàm khởi tạo `ParallelConfig(...)` được nạp các trường phản chiếu (`arg_utils.py:1981-1992`):
```python
enable_expert_parallel=self.enable_expert_parallel,   # arg_utils.py:1981
...
enable_eplb=self.enable_eplb,                          # arg_utils.py:1990
eplb_config=self.eplb_config,                          # arg_utils.py:1991
expert_placement_strategy=self.expert_placement_strategy,  # arg_utils.py:1992
```
Tại điểm này các validator của chính `ParallelConfig` chạy: `_validate_parallel_config` (§5, các cổng chặn platform/EP/world-size/redundant-expert) và `__post_init__` (§7, tự chọn communicator + cổng chặn elastic-EP). Chỉ sau khi tất cả những cái này vượt qua thì cấu hình mới đến được các worker.

**Ghi chú về thứ tự ưu tiên**: vì YAML được nạp vào chính các trường `EngineArgs` đó và các cờ CLI tường minh ghi đè chúng qua argparse, thứ tự hiệu lực là các mặc định (mặc định trường của `EPLBConfig`) → YAML `--config` → các cờ CLI tường minh. Một `--eplb-config '{"num_redundant_experts": 16}'` từng phần chỉ ghi đè đúng khóa đó; phần còn lại giữ nguyên mặc định (pydantic điền vào các trường không được chỉ định).

### 7. Tự chọn communicator (`communicator=None`) — `config/parallel.py:907-924`

Khi EPLB được bật và `communicator` được để `None`, `ParallelConfig.__post_init__` resolve nó:
```python
if self.enable_eplb and self.eplb_config.communicator is None:
    if self.enable_elastic_ep:
        self.eplb_config.communicator = "pynccl"          # parallel.py:912
    else:
        from vllm.distributed.nixl_utils import is_nixl_available
        if is_nixl_available():
            self.eplb_config.communicator = "nixl"        # parallel.py:922
        else:
            self.eplb_config.communicator = "torch_gloo"  # parallel.py:924
```
Logic, từng khối một:
- **Elastic EP → `"pynccl"`** (`parallel.py:908-912`): elastic EP yêu cầu process group không trạng thái (stateless), và `torch.distributed.batch_isend_irecv` không hỗ trợ chế độ stateless, nên PyNCCL bị ép dùng.
- **Ngược lại, ưu tiên `"nixl"` nếu có sẵn, nếu không thì `"torch_gloo"`** (`parallel.py:913-924`). Bình luận (`parallel.py:914-917`) rất quan trọng: `torch_nccl` bị **cố ý tránh** trong tự chọn vì "NCCL is fundamentally incompatible with async EPLB due to multi-stream conflicts, and batched isend/irecv hangs under high load" (tham chiếu pytorch/pytorch#174288). Nên đường tự chọn không bao giờ chọn NCCL. Vì `use_async` mặc định là `True`, communicator runtime mặc định là nixl-hoặc-gloo, không phải NCCL.

Phần tóm tắt của docstring — *"None: Auto-select backend ('torch_gloo' for async, 'torch_nccl' for sync)"* (`config/parallel.py:96`) — là một mô tả **đơn giản hóa/lỗi thời**: mã thực tế ưu tiên `nixl` hơn `torch_gloo` khi NIXL có sẵn và xử lý đặc biệt elastic EP thành `pynccl`. Hãy coi mã (`parallel.py:907-924`) là có thẩm quyền hơn dòng docstring đó. Nếu bạn cần chuyển dựa trên NCCL, bạn phải đặt `communicator="torch_nccl"` một cách tường minh (và chỉ hợp lý với `use_async=False`).

### 8. Tóm tắt cách dùng tối thiểu

- Bật nó lên: `--enable-expert-parallel` (bắt buộc) + `--enable-eplb`, với `TP*DP > 1`, trên CUDA/ROCm.
- Tinh chỉnh nó: `--eplb-config '{"num_redundant_experts": 16, "window_size": 1k, "step_interval": 3k}'` (JSON; cho phép hậu tố dễ đọc cho con người).
- Quan sát nó: thêm `"log_balancedness": true` (chấp nhận chi phí communication), tùy chọn `"log_balancedness_interval": N` để điều tiết.
- Để `policy` (`"default"` là lựa chọn duy nhất ở đây), `use_async` (mặc định `True`), và `communicator` (`None`/auto) yên đó trừ khi bạn có một yêu cầu transport cụ thể.

---

# 2. Máy trạng thái EPLB (eplb_state.py)

## Máy trạng thái EPLB (`distributed/eplb/eplb_state.py`, `distributed/eplb/async_worker.py`)

### Mục đích

EPLB liên tục tái cân bằng các expert MoE trên các EP (expert-parallel) rank sao cho không có GPU đơn lẻ nào bị quá tải bởi các expert "nóng". Máy trạng thái có ba trách nhiệm:

1. **Ghi lại** (record) tải expert theo từng forward pass (số lượng token) vào một cửa sổ trượt (sliding window).
2. **Quyết định** (decide) khi nào đã tích lũy đủ số bước để kích hoạt một lần rearrange.
3. **Thực thi** (execute) việc rearrange — tính toán một ánh xạ physical→logical expert mới và di chuyển vật lý các trọng số expert giữa các rank — hoặc là đồng bộ (chặn vòng lặp forward) hoặc bất đồng bộ (trên một luồng nền).

Ràng buộc hệ thống phân tán then chốt xuyên suốt toàn bộ thiết kế: **mọi EP rank đều phải kích hoạt rearrange ở đúng cùng một bước**, bởi vì rearrange thực hiện giao tiếp tập thể (collective communication) (`all_reduce`, các phép xáo trộn trọng số all-to-all). Nếu các rank bất đồng về thời điểm rearrange, các collective sẽ deadlock. Đây là lý do trigger được điều khiển bởi một *bộ đếm bước có tính tất định* (`expert_rearrangement_step`) được đảm bảo giống hệt nhau trên tất cả các rank — chứ không phải bởi bất kỳ đại lượng đo được theo từng rank nào.

---

### Các cấu trúc dữ liệu chính

Hai dataclass cộng với bộ điều khiển `EplbState`.

**`EplbModelState`** (`eplb_state.py:90-207`) — các tensor theo từng model (có thể có nhiều hơn một model, ví dụ model chính + drafter cho speculative-decode). Các trường cốt lõi:

- `physical_to_logical_map` (`eplb_state.py:94`): shape `(num_moe_layers, num_physical_experts)`. Với mỗi physical slot trên mỗi layer, logical expert nào nằm ở đó. Đây là *nguồn chân lý* (source of truth) cho arrangement hiện tại.
- `logical_to_physical_map` (`eplb_state.py:110`): shape `(num_moe_layers, num_logical_experts, num_redundant_experts + 1)`. Ánh xạ ngược — các physical slot đang giữ mỗi logical expert, được padding bằng `-1`. Thưa (sparse) vì một logical expert có thể được nhân bản (replicate) 1..N lần. Đây là thứ mà router đọc lúc inference để chọn một physical replica.
- `logical_replica_count` (`eplb_state.py:134`): shape `(num_moe_layers, num_logical_experts)`. Số lượng các mục khác `-1` cho mỗi logical expert — tức là mỗi logical expert có bao nhiêu bản sao vật lý.
- `expert_load_pass` (`eplb_state.py:150`): shape `(num_moe_layers, num_physical_experts)`, `int32`. Tích lũy số lượng token mà mỗi physical expert xử lý *trong forward pass hiện tại*. Được ghi bởi các layer MoE thông qua một view (xem `EplbLayerState.expert_load_view`, `eplb_state.py:952`).
- `expert_load_window` (`eplb_state.py:157`): shape `(window_size, num_moe_layers, num_physical_experts)`, `int32`. Một bộ đệm vòng (circular buffer) của `window_size` pass được ghi lại gần nhất. Chú ý comment tại `eplb_state.py:163-169`: tải được ghi lại cho **tất cả** physical expert (không chỉ những cái cục bộ) để thống kê nhất quán giữa các dispatch backend (all-to-all thô sơ so với DeepEP); với all-to-all thô sơ, các con số được scale bởi `dp_size`.
- `rebalanced` (`eplb_state.py:177`): cờ **chỉ dùng cho async**. Được đặt thành `True` bởi luồng chính một khi các map mới đã được tính xong → báo cho async worker bắt đầu truyền trọng số. Được đặt lại về `False` một khi tất cả các layer đã được commit. Việc đồng bộ giữa luồng chính và worker dựa vào GIL (`eplb_state.py:184-186`).
- `pending_result` (`eplb_state.py:199`): **chỉ dùng cho async**. Async worker công bố một `AsyncEplbLayerResult` ở đây sau khi đã điền `expert_buffer` cho một layer; luồng chính tiêu thụ nó trong `_move_to_workspace`. "At most one result is pending at a time" (`eplb_state.py:203`) — đây là một cuộc bàn giao (hand-off), được đồng bộ bằng GIL.

**`EplbState`** — bộ điều khiển (`eplb_state.py:210`) — chính là máy trạng thái thực sự. `__init__` của nó (`eplb_state.py:215-284`) thiết lập các bộ đếm điều khiển mọi thứ:

- `expert_load_window_step` (`eplb_state.py:223`): chỉ số ghi hiện tại vào circular `expert_load_window`. **Theo từng rank** — comment tại `eplb_state.py:226-228` lưu ý mỗi rank có thể có giá trị riêng của nó (đây là ghi sổ cục bộ, không phải điểm đồng bộ).
- `expert_load_window_size` (`eplb_state.py:230`): hằng số, lấy từ config.
- `expert_rearrangement_step` (`eplb_state.py:235`): số bước kể từ lần rearrange trước — bộ đếm trigger. Docstring tại `eplb_state.py:240-243` chính là bất biến then chốt: *"all EP ranks need to have the same `expert_rearrangement_step` value to ensure synchronization. Otherwise, the rearrangement will hang at collective communication calls."*
- `expert_rearrangement_step_interval` (`eplb_state.py:245`): ngưỡng hằng số, lấy từ config.
- `should_record_tensor` (`eplb_state.py:250`): một tensor bool vô hướng dùng chung duy nhất. Mỗi layer đều giữ tham chiếu đến *cùng một* đối tượng, nên một lệnh `.fill_()` bật/tắt việc ghi cho tất cả các layer cùng một lúc (`eplb_state.py:253-256`).
- `is_async` (`eplb_state.py:257`): chế độ sync so với async.
- `rearrange_event` (`eplb_state.py:261`): một `CpuGpuEvent` được dùng để đánh thức luồng async worker.
- `async_worker` (`eplb_state.py:265`): handle của luồng nền.
- `num_valid_physical_experts` (`eplb_state.py:273`): số physical slot thực sự được ánh xạ tới các logical expert (liên quan đến elastic EP, nơi các rank mới thêm vào có thể có các slot chưa được ánh xạ).

Phần cuối của `__init__` (`eplb_state.py:281-284`) nắm bắt chỉ số thiết bị CUDA cho luồng worker.

---

### Xây dựng / khởi tạo: `add_model` (`eplb_state.py:342-472`)

Được gọi một lần cho mỗi model để xây dựng arrangement ban đầu và cấp phát tất cả các tensor.

- `eplb_state.py:350` xác thực rằng EP config của model mới khớp với bất kỳ model nào đã đăng ký (`validate_ep_configuration`, `eplb_state.py:306`).
- `eplb_state.py:351` chốt (latch) `is_async` từ config (`eplb_config.use_async`).
- `eplb_state.py:353-362`: xây dựng physical→logical map ban đầu qua `build_initial_global_physical_to_logical_map` (`eplb_state.py:286-304`). Hàm phụ trợ đó tạo ra `[0,1,...,num_routed-1]` cho các expert cơ sở, rồi nối thêm các redundant slot dưới dạng `i % num_routed_experts` — tức là `num_redundant_experts` logical expert đầu tiên, mỗi cái nhận thêm một replica. Gieo mầm (seeding) round-robin đơn giản; việc cân bằng thực sự xảy ra về sau.
- `eplb_state.py:366-370`: khẳng định (assert) `num_redundant_experts <= 1023` (`MAX_EXPERT_REDUNDANCY`), định kích thước cho chiều replica. Comment (`eplb_state.py:363-364`) giải thích trần 1024/8 = 128 node; được đánh dấu `TODO(rui): make this configurable`.
- `eplb_state.py:372-386`: cấp phát `logical_to_physical_map` toàn `-1`, rồi duyệt qua từng physical slot và điền ánh xạ ngược + `logical_replica_count` — tính nhất quán forward/inverse ban đầu được thiết lập tại đây.
- `eplb_state.py:388-413`: quảng bá (broadcast) các map một-layer ra tất cả `num_moe_layers` qua `unsqueeze(0).expand(...).contiguous()` — tất cả các layer khởi đầu giống hệt nhau.
- `eplb_state.py:415-429`: cấp phát `expert_load_pass` đã được zero và circular buffer `expert_load_window`; `expert_load_window_size` được đọc từ `eplb_config.window_size` (`eplb_state.py:420`).

**Việc khởi tạo 3/4** (`eplb_state.py:431-436`) — đoạn mà tác vụ chỉ định gọi ra:

```python
# Set the initial progress of rearrangement to 3/4
eplb_step_interval = self.parallel_config.eplb_config.step_interval
self.expert_rearrangement_step = max(
    0, eplb_step_interval - eplb_step_interval // 4
)
self.expert_rearrangement_step_interval = eplb_step_interval
```

Bộ đếm *không* được khởi tạo về 0. Nó bắt đầu tại `step_interval - step_interval//4` — tức là **đã đi được 3/4 quãng đường tới ngưỡng**. Hệ quả: lần rearrange *đầu tiên* kích hoạt chỉ sau khoảng ~`step_interval/4` bước thay vì cả một interval đầy đủ. Ý đồ là để phản ứng với sự mất cân bằng tải sớm trong lúc warm-up thay vì chờ hết cả một cửa sổ. Tất cả các interval tiếp theo đều là `step_interval` đầy đủ (bộ đếm reset về 0 sau mỗi lần rearrange — `eplb_state.py:603`). Bởi vì `step_interval` là một hằng số config giống hệt nhau trên mọi rank, giá trị khởi tạo này giống hệt nhau trên mọi rank, giữ nguyên bất biến đồng bộ giữa các rank. (Lưu ý về phiên bản: phân số "3/4" được hard-code ở đây qua `// 4`; các bản vLLM cũ hơn/mới hơn có thể khác.)

- `eplb_state.py:438-440`: chọn lớp policy từ `EPLB_POLICIES` theo tên (mặc định `DefaultEplbPolicy`).
- `eplb_state.py:442-446`: bàn giao `expert_load_pass`, `logical_to_physical_map`, `logical_replica_count` cho model để các layer MoE của nó có được các view vào các tensor này.
- `eplb_state.py:447`: `_init_should_record_tensor(model)` — phải chạy *sau* `set_eplb_state` vì nó đọc `eplb_state` của mỗi layer (`eplb_state.py:640-642`). Nó cấp phát tensor bool dùng chung duy nhất một lần (`eplb_state.py:650-653`) và trỏ mọi layer vào nó (`eplb_state.py:655-656`).
- `eplb_state.py:448-455`: cấp phát `expert_buffer` (khu vực staging cho các trọng số đang được truyền) và xây dựng `communicator` cho việc truyền trọng số.
- `eplb_state.py:457-471`: lắp ráp `EplbModelState` và lưu nó với khóa là `model_config.compute_hash()`.

---

### Theo từng bước: `step()` (`eplb_state.py:474-606`)

Được gọi một lần cho mỗi forward pass. Duyệt toàn bộ máy trạng thái.

**Short-circuit khi profile** (`eplb_state.py:500-502`): nếu `is_profile`, ngay lập tức thực hiện `self.rearrange(is_profile=True)` và trả về. Một bước profile thực hiện một lần rearrange *giả* (dummy) với chi phí giao tiếp tối đa để buộc cấp phát comm buffer, nhờ đó các lần rearrange thực sau này không bị OOM (docstring `eplb_state.py:487-490`). Không có trọng số thực nào di chuyển.

**Bước dummy** (`eplb_state.py:504-507`): nếu `is_dummy`, zero `expert_load_pass` cho mọi model — tải từ các pass dummy không được tính.

**Ghi log tính cân bằng (balancedness)** (`eplb_state.py:509-556`): cứ mỗi `log_balancedness_interval` bước (và chỉ khi `log_stats`), nó `_sync_load_pass()` (all-reduce một *bản clone* của load pass — `eplb_state.py:872-880`) rồi tính, theo từng layer, `avg_tokens = mean over ranks`, `max_tokens = max over ranks`, và `balancedness = avg/max` (`eplb_state.py:523-542`). Rank 0 ghi log nó cùng với số bước-đến-lần-rearrange-tiếp-theo (`eplb_state.py:544-556`). Thuần túy là telemetry; không ảnh hưởng đến máy trạng thái.

**Ghi vào cửa sổ trượt** (`eplb_state.py:558-571`): nếu không phải bước dummy, `should_record = self._should_record_current_step(...)`. Khi ghi, với mỗi model nó copy `expert_load_pass` vào `expert_load_window[expert_load_window_step]` rồi zero pass đó (`eplb_state.py:562-566`). Chỉ số cửa sổ sau đó tiến lên và cuộn vòng (wrap) (`eplb_state.py:568-571`).

`_should_record_current_step` (`eplb_state.py:608-628`) là tối ưu hóa tránh lãng phí công sức GPU: việc ghi chỉ được bật khi `steps_remaining <= window_size` (`eplb_state.py:615-618`) — tức là chúng ta đang ở trong phạm vi một cửa sổ tính từ lần rearrange tiếp theo, nên các pass được ghi này vẫn còn nằm trong buffer khi policy đọc nó. Nếu `log_stats`, nó cũng bật ghi khi gần bước log tiếp theo (`eplb_state.py:623-628`). Bất kỳ bước nào sớm hơn sẽ chỉ bị ghi đè trong circular buffer trước khi được dùng (xem giải thích trong `EplbLayerState.should_record_tensor`, `eplb_state.py:939-943`).

**Tiến bộ đếm trigger** (`eplb_state.py:573-577`):

```python
self.expert_rearrangement_step += 1
```

Comment tại `eplb_state.py:574-576` rất quan trọng: cái này tăng **ngay cả trên các bước dummy**, và rearrange vẫn kích hoạt — bởi vì tất cả các rank phải giữ nhịp bước (lockstep) trên các collective. Bỏ qua việc tăng trên một số rank sẽ phá vỡ đồng bộ.

**Commit workspace ở chế độ async** (`eplb_state.py:579-591`): ở chế độ async, với mỗi model, nếu `rebalanced` được đặt *và* `_all_ranks_result_ready(...)` trả về true, gọi `_move_to_workspace(...)`. Việc kiểm tra `rebalanced` phải *nhất quán giữa các rank* (comment `eplb_state.py:583-584`) nếu không `all_reduce` bên trong `_all_ranks_result_ready` sẽ treo.

`_all_ranks_result_ready` (`eplb_state.py:824-843`) là hàng rào (barrier) giữa các rank cho async: mỗi rank đặt `has_result = int(pending_result is not None)`, rồi `all_reduce` cờ đó (nhóm CPU nếu có, không thì nhóm thiết bị) và trả về true chỉ khi tổng bằng kích thước nhóm — tức là **mọi rank đều đã có việc truyền của một layer sẵn sàng**. Điều này đảm bảo tất cả các rank commit cùng một layer với nhau, giữ arrangement nhất quán toàn cục.

**Trigger** (`eplb_state.py:593-604`):

```python
if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
    if self.is_async and any(rebalanced for ... in model_states):
        self._update_layer_should_record(log_stats=log_stats)
        return
    self.expert_rearrangement_step = 0
    self.rearrange()
```

- Khi bộ đếm đạt tới interval, chúng ta muốn rearrange.
- **Chốt chặn async (async guard)** (`eplb_state.py:594-602`): nếu là async *và một lần rearrange trước đó vẫn đang trong quá trình thực hiện* (`any(... rebalanced ...)`), chúng ta **không** reset bộ đếm hoặc bắt đầu một lần rearrange mới — chúng ta chỉ làm mới `should_record` (luôn là True ở đây vì step ≥ interval) và trả về sớm. Điều này ngăn các lần rearrange async chồng lấn nhau; bộ đếm tiếp tục đếm vượt qua interval cho đến khi việc truyền còn dang dở hoàn tất.
- Ngược lại (`eplb_state.py:603-604`): reset bộ đếm về 0 và gọi `rearrange()`. Việc reset-về-0 (khác với khởi tạo 3/4) là lý do chỉ có chu kỳ *đầu tiên* là ngắn.

Cuối cùng `_update_layer_should_record(...)` (`eplb_state.py:606`) đẩy cờ record vừa tính lại vào tensor dùng chung cho tất cả các layer (`eplb_state.py:630-635`).

---

### `rearrange()` (`eplb_state.py:658-809`)

Tính toán và áp dụng một arrangement expert mới. Chạy trên luồng chính; hành vi rẽ nhánh theo `is_async`/`is_profile`.

- `eplb_state.py:674-675`: lấy nhóm tiến trình EP và rank của rank này.
- `eplb_state.py:679-689`: chỉ rank 0 (`is_main_rank`) thiết lập các CUDA timing event (chỉ dùng cho đường sync/profile) và ghi log chế độ.

**Tổng hợp tải vào không gian logical** (`eplb_state.py:691-716`): với mỗi model, nó cắt (slice) cửa sổ tới các physical expert hợp lệ (`eplb_state.py:694-696`), rồi `scatter_add_` tích lũy tải của physical expert vào một `logical_expert_load_window` dùng `physical_to_logical_map` làm chỉ số (`eplb_state.py:704-713`). Điều này gộp các replica của cùng một logical expert lại với nhau. Sau đó nó tính tổng trên chiều cửa sổ (`eplb_state.py:715`) để có tổng tải logical theo từng rank.

**All-reduce trên các rank** (`eplb_state.py:718`): `_allreduce_list(...)` (`eplb_state.py:845-870`) cộng tổng cửa sổ tải logical trên tất cả các EP rank sao cho mọi rank đều thấy cùng một tải *toàn cục* — policy phải chạy trên đầu vào giống hệt nhau ở khắp nơi để tạo ra một arrangement giống hệt nhau. Với nhiều model, nó nối (concatenate), `all_reduce` một lần, rồi tách trở lại (`eplb_state.py:856-869`).

**Định kích thước topology** (`eplb_state.py:720-748`): tính `num_replicas`, `num_groups`, `num_nodes`, `num_gpus`. Nhánh `rank_mapping` (`eplb_state.py:726-737`) xử lý việc thu nhỏ (scale-down) elastic-EP (tái cân bằng lên các GPU còn lại trước khi giải phóng một số cái). Nếu `num_gpus % num_nodes != 0` nó rơi về (fallback) `num_nodes = 1` và vô hiệu hóa thuật toán phân cấp (`eplb_state.py:742-748`).

**Rẽ nhánh sync so với async** (`eplb_state.py:751-805`) — theo từng model:

*Đường sync* (`if not self.is_async or is_profile`, `eplb_state.py:754-793`):
1. `self.policy.rebalance_experts(...)` (`eplb_state.py:756-763`) chạy thuật toán cân bằng trên `global_expert_load_window.cpu()` và trả về một `physical_to_logical_map` mới.
2. `rearrange_expert_weights_inplace(...)` (`eplb_state.py:766-775`) xáo trộn vật lý các trọng số expert giữa các rank (all-to-all qua communicator). **Đây là collective có tính chặn (blocking)** — nó diễn ra inline trong vòng lặp forward.
3. Nếu không phải đang profile, `_commit_eplb_maps(...)` (`eplb_state.py:777-781`) làm cho các map mới có hiệu lực (live).
4. Rank 0 ghi lại/ghi log thời gian đã trôi qua (`eplb_state.py:783-793`).

*Đường async* (`else`, `eplb_state.py:794-805`): nó **không** tính toán hay di chuyển bất cứ thứ gì ở đây. Nó chỉ chụp nhanh (snapshot) cửa sổ tải toàn cục (`global_expert_load_window.clone()` tại `eplb_state.py:799` — được clone để worker có thể đọc nó an toàn trong khi luồng chính tiếp tục), nhồi nó cộng với các kích thước topology vào `eplb_stats` (`eplb_state.py:795-804`), và đặt `rebalanced = True` (`eplb_state.py:805`). Việc tính toán policy thực sự và truyền trọng số được hoãn lại cho worker nền.

**Đánh thức worker** (`eplb_state.py:807-808`): ở chế độ async, không phải profile, `self.rearrange_event.record()` báo hiệu cho luồng async rằng có việc mới sẵn sàng.

---

### `_commit_eplb_maps` (`eplb_state.py:1117-1151`) và biến thể theo từng layer

Làm cho một `physical_to_logical_map` mới trở thành arrangement có hiệu lực:
- Copy physical→logical map mới vào `model_state.physical_to_logical_map` (`eplb_state.py:1127-1138`). Nhánh `src.shape[1] != dst.shape[1]` xử lý trường hợp hiếm khi số lượng physical expert thay đổi (số lượng GPU thay đổi lúc runtime) bằng cách *thay thế* tensor thay vì copy vào nó.
- Tính lại ánh xạ ngược + replica count qua `compute_logical_maps(...)` (`eplb_state.py:1140-1141`, được định nghĩa tại `eplb_state.py:994-1070`) và commit chúng (`_pad_out_tensor` padding map thưa trở về chiều rộng đầy đủ bằng `-1`).

`_commit_eplb_maps_for_layer` (`eplb_state.py:1080-1114`) là phiên bản một-layer cho async, được dùng vì async worker commit mỗi lần một layer. Nó khẳng định (assert) rằng số lượng physical expert không thay đổi (`eplb_state.py:1095-1099`) — async EPLB không hỗ trợ thay đổi kích thước (resize) elastic giữa chừng khi đang truyền.

---

### Thực thi async: `_move_to_workspace` + `async_worker.py`

**`_move_to_workspace`** (`eplb_state.py:1154-1179`) — chạy trên **luồng chính**, được gọi từ `step()` khi việc truyền của một layer đã sẵn sàng:
1. Đọc `pending_result` của worker (`eplb_state.py:1158-1159`).
2. `move_from_buffer(...)` (`eplb_state.py:1160-1166`) copy các trọng số đã staging từ `expert_buffer` vào `expert_weights` có hiệu lực cho layer đó.
3. `_commit_eplb_maps_for_layer(...)` (`eplb_state.py:1168-1172`) commit các map mới của layer đó.
4. Nếu đó là layer cuối cùng, xóa `rebalanced` (`eplb_state.py:1174-1175`) — báo hiệu toàn bộ lần rearrange đã xong và một lần mới có thể được lên lịch.
5. Reset `pending_result = None` và ghi lại `result.consumed_event` (`eplb_state.py:1178-1179`) để bỏ chặn worker cho phép nó tiến tới layer tiếp theo.

**`async_worker.py`** — luồng nền.

- `start_async_worker` (`async_worker.py:25-50`): tạo một daemon thread. `thread_target` (`async_worker.py:34-47`) ghim (pin) thiết bị CUDA (`async_worker.py:36`), tạo một CUDA stream chuyên dụng (`async_worker.py:37`) để các lần truyền không tranh chấp với stream tính toán chính, và chạy `transfer_run_periodically`. Được khởi động lười (lazily) bởi `EplbState.start_async_loop` (`eplb_state.py:811-822`), vốn không làm gì (no-op) trừ khi `is_async` và chỉ tạo một worker.

- `transfer_run_periodically` (`async_worker.py:79-148`) là vòng lặp worker:
  - `state.rearrange_event.wait(stream=cuda_stream)` (`async_worker.py:86`) chặn cho đến khi `rearrange()` của luồng chính ghi lại event.
  - Với mỗi model: chụp nhanh `physical_to_logical_map` sang CPU trên stream của worker (`async_worker.py:98-99`) — được đồng bộ theo `rearrange_event` để nó thấy map trước-khi-rearrange.
  - `run_rebalance_experts(...)` (`async_worker.py:101-103`, được định nghĩa tại `async_worker.py:53-76`): di chuyển cửa sổ tải toàn cục đã clone sang CPU và chạy `policy.rebalance_experts(...)` — **đây là nơi việc tính toán policy xảy ra ở chế độ async** (được hoãn lại từ `rearrange()`). Trả về map CPU mới.
  - Vòng lặp trong duyệt qua các layer (`async_worker.py:113-148`), có điều kiện là `model_state.rebalanced and layer_idx < num_layers`:
    - `transfer_layer(...)` (`async_worker.py:114-124`) copy các trọng số mới của layer đó vào `expert_buffer`.
    - `cuda_stream.synchronize()` (`async_worker.py:128`) chờ tất cả các lần ghi buffer hoàn tất trước khi công bố.
    - Tạo một `consumed_event` (`async_worker.py:133`), rồi công bố `pending_result` (`async_worker.py:135-140`) cho luồng chính nhặt lên trong `_move_to_workspace`.
    - `consumed_event.wait(stream=cuda_stream)` (`async_worker.py:145`) chặn cho đến khi luồng chính hoàn tất việc di chuyển buffer vào các trọng số có hiệu lực và ghi lại event đó — đây là back-pressure đảm bảo `expert_buffer` không bị ghi đè trước khi nó được tiêu thụ.
    - Tiến `layer_idx`.

Vậy nên thiết kế async là một cuộc bàn giao producer/consumer, mỗi forward pass một layer: worker sản xuất (tính map mới + staging trọng số vào buffer), `step()` của luồng chính tiêu thụ (di chuyển buffer → trọng số, commit map) nhưng chỉ khi `_all_ranks_result_ready` xác nhận *mọi* rank đều đã staging layer đó. Điều này giữ việc di chuyển trọng số bằng collective tốn kém nằm ngoài đường forward tới hạn (critical path) trong khi vẫn bảo toàn nhịp bước lockstep giữa các rank cần thiết cho chính các collective.

---

### Sync so với async — tóm tắt hai đường

| Khía cạnh | Sync (`is_async=False`) | Async (`is_async=True`) |
|---|---|---|
| Tính policy | Inline trong `rearrange()` (`eplb_state.py:756`) | Luồng worker, `run_rebalance_experts` (`async_worker.py:66`) |
| Di chuyển trọng số | Inline, chặn (`rearrange_expert_weights_inplace`, `eplb_state.py:766`) | Worker, theo từng layer `transfer_layer` (`async_worker.py:114`) |
| Commit | `_commit_eplb_maps` (toàn bộ model, `eplb_state.py:778`) | `_commit_eplb_maps_for_layer` trên luồng chính trong `_move_to_workspace` (`eplb_state.py:1168`) |
| Đồng bộ giữa các rank | Chính các collective all-to-all inline | `expert_rearrangement_step` cho trigger + barrier theo từng layer `_all_ranks_result_ready` (`eplb_state.py:824`) |
| Ngăn chồng lấn | Không áp dụng (chặn) | Chốt chặn `rebalanced` trong `step()` (`eplb_state.py:594-602`) |
| Resize elastic | Được hỗ trợ (nhánh thay đổi shape của `_commit_eplb_maps`) | **Không** được hỗ trợ (assertion `eplb_state.py:1095-1099`) |

Ở cả hai chế độ, *trigger* là giống hệt nhau và có tính tất định theo rank: bộ đếm `expert_rearrangement_step` bắt đầu tại 3/4 của `step_interval` và reset về 0 sau đó. Chính bộ đếm dùng chung duy nhất đó — không phải bất kỳ tải đo được nào — là thứ đảm bảo tất cả các rank đi vào đường rearrange bằng collective ở cùng một bước.

---

Các file nguồn (đường dẫn tuyệt đối):
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/async_worker.py`

Lưu ý về phiên bản: các số dòng và việc khởi tạo 3/4 hard-code (`// 4`) cùng `MAX_EXPERT_REDUNDANCY = 1023` là đặc thù của bản sao vLLM được vendor này; vLLM upstream ở các revision khác có thể khác.

---

# 3. Rebalance Algorithm (eplb_utils.py)

Tôi đã có mọi thứ cần thiết. Dưới đây là phần tài liệu.

---

# Thuật toán Rebalance của EPLB

## Lưu ý quan trọng về vị trí file (phụ thuộc phiên bản)

Task chỉ tới `distributed/eplb/eplb_utils.py`, nhưng **trong bản checkout này file đó *không* chứa thuật toán rebalance**. `eplb_utils.py` ở đây chỉ giữ hai helper: `CpuGpuEvent` (một cặp CUDA-event + `threading.Event` bắt buộc thứ tự `record()`→`wait()` giữa main thread và thread async-EPLB, `eplb_utils.py:16-61`) và `override_envs_for_eplb` (ép `NCCL_MAX_CTAS=8` để tránh deadlock do tranh chấp SM giữa việc trao đổi weight của NCCL và các cooperative MoE kernel, `eplb_utils.py:64-125`). Cả hai đều không đụng tới placement.

Trong phiên bản vLLM này, phần toán học của việc rearrange đã được tách ra thành một **policy package**:

- `distributed/eplb/policy/abstract.py` — interface `AbstractEplbPolicy.rebalance_experts`.
- `distributed/eplb/policy/default.py` — `DefaultEplbPolicy`, thuật toán thực (phỏng theo [DeepSeek EPLB](https://github.com/deepseek-ai/eplb)).
- `distributed/eplb/policy/__init__.py` — registry `EPLB_POLICIES = {"default": DefaultEplbPolicy}` (`policy/__init__.py:10`), được kiểm tra hợp lệ dựa trên config enum `EPLBPolicyOption`.

Mọi thứ bên dưới đều tham chiếu tới `distributed/eplb/policy/default.py` trừ khi có ghi chú khác. (Trong các bản phát hành vLLM cũ hơn, chính đoạn code này nằm trong `eplb_utils.py` / `rebalance_algo.py`; nếu bạn đang đọc một bản checkout khác thì các hàm có thể nằm ở đó thay vào đó.)

## Khái niệm và mục đích

Trong một mô hình MoE, mỗi token được định tuyến (route) tới một tập nhỏ các expert *logical*. Khi các expert được shard trên nhiều GPU (expert parallelism), một phép gán `expert_id → GPU` ngây thơ sẽ tạo ra **mất cân bằng tải (load imbalance)**: một số expert bị "nóng" (được route tới thường xuyên hơn nhiều), khiến GPU của chúng trở thành nút thắt cổ chai trong khi các GPU khác nhàn rỗi. EPLB khắc phục điều này bằng cách:

1. **Ghi nhận (Recording)** tải token trên từng expert qua một cửa sổ (window) gồm nhiều step.
2. **Nhân bản (Replicating)** các logical expert đang nóng thành nhiều bản sao (copy) *physical* (dùng các slot physical "redundant"), để lưu lượng của chúng có thể được chia nhỏ.
3. **Đóng gói (Packing)** các physical expert lên các GPU sao cho tổng tải của mỗi GPU đều nhau nhất có thể, đồng thời tôn trọng phân cấp mạng hai tầng (node → GPU) để các nhóm expert nói chuyện với nhau vẫn ở trên cùng một node (đường liên kết trong-node nhanh, ví dụ NVLink).

Đầu ra là một ánh xạ `phy2log` có shape `[layers, num_replicas]`: với mỗi physical expert slot, nó cho biết id của logical expert mà slot đó phục vụ. Các physical slot được bố trí liền kề theo từng GPU, nên `phy2log` đồng thời mã hóa cả *expert nào nằm trên GPU nào* lẫn *mỗi logical expert có bao nhiêu replica*.

Hai đại lượng đếm quan trọng xuyên suốt:
- `num_log` = số logical expert (cố định bởi mô hình).
- `num_phy` = `num_replicas` = số physical slot = `slots_per_gpu * num_gpus`. Hiệu `num_phy − num_log` là số slot **redundant** sẵn có để nhân bản.

## Cách tính tải ("weight") trước khi thuật toán chạy

Thuật toán là thuần khiết và không trạng thái (pure, stateless) — nó tiêu thụ một tensor `weight` có shape `[layers, num_logical_experts]`. Tensor đó được caller lắp ráp trong `eplb_state.py` trước khi `rebalance_experts` được gọi:

- Tải physical-expert theo từng step được tích lũy vào một cửa sổ trượt (rolling window) `expert_load_window`.
- `logical_expert_load_window.scatter_add_(dim=-1, index=physical_to_logical_map…, src=expert_load_window)` (`eplb_state.py:704-713`) gộp tải physical trở lại lên các logical expert — tức là nó cộng tổng tải của tất cả các replica hiện tại của một logical expert. Đây là bước then chốt: thuật toán suy luận theo tải **logical**, độc lập với sự nhân bản *hiện tại*.
- `global_expert_load_window = logical_expert_load_window.sum(dim=0)` (`eplb_state.py:715`) cộng tổng qua cửa sổ (loại bỏ chiều thời gian), sau đó `_allreduce_list` (`eplb_state.py:718`) cộng tổng trên tất cả các EP rank để mọi rank thống nhất về tải logical toàn cục.

Tensor đã được tổng hợp đó, sau khi chuyển sang CPU, chính là `weight`. Điểm gọi là `eplb_state.py:756-763`:

```python
new_physical_to_logical_map = self.policy.rebalance_experts(
    global_expert_load_window.cpu(),
    num_replicas,      # = model.num_physical_experts
    num_groups,        # = model.num_expert_groups
    num_nodes,         # from get_node_count() (or rank_mapping on scale-down)
    num_gpus,          # = ep_group.size()
    eplb_model_state.physical_to_logical_map.cpu(),  # old map, for slot preservation
)
```

## Ba primitive

Thuật toán được xây từ ba hàm có thể tổ hợp (composable). Tôi sẽ đi qua từng hàm, rồi tới bộ điều khiển phân cấp (hierarchical driver) nối chúng lại với nhau.

### 1. `balanced_packing` — phân hoạch đa hướng đều theo weight

`default.py:22-73`. Mục đích (docstring `default.py:26-28`): *"Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs are as balanced as possible."* Đây là một phép phân hoạch tham lam **bị ràng buộc về lực lượng (cardinality-constrained)** — mỗi pack cuối cùng có *đúng* cùng số phần tử, và trong số các phân hoạch hợp lệ đó, nó tham lam giảm thiểu weight lớn nhất của pack.

- `default.py:38-40`:
  ```python
  num_layers, num_groups = weight.shape
  assert num_groups % num_packs == 0
  groups_per_pack = num_groups // num_packs
  ```
  Các phần tử được xử lý theo batch trên `num_layers` (mỗi layer được pack độc lập). Ràng buộc về lực lượng chính xác đòi hỏi `num_groups` chia hết cho `num_packs`; `groups_per_pack` là sức chứa cố định của mỗi pack.

- `default.py:42-45` — đường nhanh (fast path):
  ```python
  if groups_per_pack == 1:
      pack_index = np.tile(np.arange(num_groups …), (num_layers, 1))
      rank_in_pack = np.zeros_like(pack_index …)
      return pack_index, rank_in_pack
  ```
  Nếu mỗi pack giữ đúng một phần tử, việc pack là tầm thường: phần tử *i* → pack *i*, rank 0. Không cần tối ưu hóa.

- `default.py:48` — `indices = np.argsort(-weight, axis=-1)`: sắp xếp các phần tử **giảm dần theo weight** cho từng layer. Đây là heuristic tham lam kinh điển "lớn-trước / LPT (longest processing time)" — đặt các phần tử lớn nhất trước sẽ cho độ cân bằng gần tối ưu.

- `default.py:50-54`: cấp phát các đầu ra `pack_index`/`rank_in_pack` (cả hai đều được điền `-1`) và trạng thái đang chạy `pack_weights` (weight tích lũy trên mỗi pack) cùng `pack_items` (số đếm trên mỗi pack).

- `default.py:57-71` — vòng lặp tham lam, cho từng layer, duyệt các phần tử theo thứ tự weight giảm dần:
  ```python
  pack = int(np.argmin(weights_row))          # lightest currently-open pack
  pack_index[layer_idx, group] = pack
  rank_in_pack[layer_idx, group] = items_row[pack]
  weights_row[pack] += weight[layer_idx, group]
  items_row[pack] += 1
  if items_row[pack] == groups_per_pack:
      weights_row[pack] = np.inf              # pack full → never chosen again
  ```
  Mỗi phần tử đi vào **pack nhẹ nhất chưa đầy**. `rank_in_pack` ghi lại vị trí của nó (0…groups_per_pack−1) bên trong pack đó. Khi một pack đạt sức chứa, weight của nó được đặt thành `inf` để `argmin` không bao giờ chọn lại nó nữa — đây là cách ràng buộc lực lượng chính xác được thực thi (`default.py:62,69-71` và comment "*full packs are masked out by inf*").

Trả về `(pack_index, rank_in_pack)`: mỗi phần tử rơi vào pack nào, và slot của nó trong pack đó.

### 2. `replicate_experts` — đặt các replica redundant

`default.py:75-101`. Mục đích (docstring `default.py:79-81`): *"Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized."* Hàm này quyết định **mỗi logical expert nhận được bao nhiêu bản physical**, dùng các slot dư (redundant).

- `default.py:91-93`:
  ```python
  n, num_log = weight.shape
  num_redundant = num_phy - num_log
  assert num_redundant >= 0
  ```
  `num_redundant` = số slot physical dư ngoài mức một-cho-mỗi-logical. Phải không âm (bạn không thể có ít physical hơn logical expert).

- `default.py:94-96` — khởi tạo:
  ```python
  phy2log = np.tile(np.arange(num_phy …), (n, 1))   # slots 0..num_log-1 map to logical 0..num_log-1
  logcnt  = np.ones((n, num_log) …)                  # every logical expert starts with 1 replica
  arangen = np.arange(n …)
  ```
  `num_log` slot physical đầu tiên được gán trước theo ánh xạ đồng nhất (một bản đảm bảo cho mỗi logical expert). `logcnt[l]` theo dõi số replica hiện tại của logical expert *l*.

- `default.py:97-100` — vòng lặp redundancy, một lần lặp cho mỗi slot dư `i` trong `[num_log, num_phy)`:
  ```python
  redundant_indices = np.argmax(weight / logcnt, axis=-1)
  phy2log[:, i] = redundant_indices
  logcnt[arangen, redundant_indices] += 1
  ```
  **Đây là heuristic nhân bản cốt lõi.** `weight / logcnt` là tải *trên-mỗi-replica* của từng logical expert nếu lưu lượng của nó được chia đều trên các replica hiện tại. `argmax` chọn logical expert có tải trên-mỗi-replica cao nhất — tức là nút thắt hiện tại — và cấp cho nó thêm một replica (slot `i`), rồi tăng số đếm của nó. Việc thêm một replica cho expert *l* ngay lập tức kéo tải hiệu dụng của nó từ `w_l/c` xuống `w_l/(c+1)`, nên lần lặp tiếp theo sẽ xem xét lại ai đang nóng nhất. Việc tham lam gán mỗi slot dư cho expert có max-per-replica hiện tại chính là điều "minimizes the maximum replica load."

Trả về `(phy2log, logcnt)`: với mỗi trong số `num_phy` slot là logical id của nó, và số đếm replica trên từng logical.

### 3. `rebalance_experts_hierarchical` — toàn bộ pipeline node→GPU

`default.py:103-189`. Nối hai primitive lại với nhau qua ba giai đoạn, tôn trọng phân cấp mạng. Thiết lập và các assert về tính chia hết (`default.py:125-132`):

```python
group_size       = num_logical_experts // num_groups   # experts per group
groups_per_node  = num_groups // num_nodes
phy_experts_per_gpu = num_physical_experts // num_gpus  # physical slots per GPU
```

*Group* expert là đơn vị routing cần được giữ cùng nhau trên một node. Một helper cục bộ `inverse(perm)` (`default.py:134-139`) xây nghịch đảo của một hoán vị theo từng hàng thông qua scatter (`inv[row, perm] = col`); nó được dùng lặp đi lặp lại để chuyển đổi các ánh xạ "cũ→vị trí mới" thành ánh xạ "mới→vị trí cũ".

**Bước 1 — pack các group vào các node** (`default.py:141-156`):
```python
tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(axis=-1)
group_pack_index, group_rank_in_pack = cls.balanced_packing(tokens_per_group, num_nodes)
log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack)[..., None]
             * group_size) + np.arange(group_size)).reshape(num_layers, num_logical_experts)
mlog2log = inverse(log2mlog)
```
Tải của một group là tổng tải của các thành viên. `balanced_packing(..., num_nodes)` phân phối các group nguyên vẹn đều khắp các node (mỗi node nhận `groups_per_node` group). `log2mlog` sau đó gán nhãn lại mọi logical expert vào một **thứ tự node-cục-bộ ("mlog")**: node index của một group × groups_per_node + rank-trong-node của nó cho ra slot group node-cục-bộ, nhân `group_size` cộng offset bên trong group cho ra chỉ số liền kề mới của expert. `mlog2log` là nghịch đảo (node-cục-bộ → global logical). Sau bước này, các khối liền kề trên trục mlog thuộc về một node.

**Bước 2 — dựng các redundant expert bên trong mỗi node** (`default.py:158-165`):
```python
tokens_per_mlog = np.take_along_axis(weight, mlog2log, axis=1).reshape(
    -1, num_logical_experts // num_nodes)
phy2mlog, mlogcnt = cls.replicate_experts(
    tokens_per_mlog, num_physical_experts // num_nodes)
```
Các weight được sắp xếp lại theo bố cục node-cục-bộ (`take_along_axis` với `mlog2log`) rồi **reshape để gấp chiều node vào trục batch** (`-1, experts_per_node`). Nhờ đó `replicate_experts` chạy *độc lập theo từng node*, phân phối phần slot redundant của node đó (`num_physical_experts // num_nodes`) cho các expert của node đó. Kết quả: `phy2mlog` (physical slot → node-cục-bộ logical) và `mlogcnt` (số đếm replica trên từng node-cục-bộ logical).

**Bước 3 — pack các physical expert vào các GPU bên trong mỗi node** (`default.py:167-189`):
```python
tokens_per_phy = np.take_along_axis(tokens_per_mlog / mlogcnt, phy2mlog, axis=1)
pack_index, rank_in_pack = cls.balanced_packing(tokens_per_phy, num_gpus // num_nodes)
phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
pphy2phy = inverse(phy2pphy)
```
`tokens_per_mlog / mlogcnt` là tải trên-mỗi-replica của từng logical expert; `take_along_axis(..., phy2mlog)` phát tán (broadcast) nó lên các physical slot, nên `tokens_per_phy` là **tải hiệu dụng của mỗi physical replica** (`default.py:168` comment: *"Effective per-physical load = logical load divided by replica count"*). `balanced_packing(..., num_gpus // num_nodes)` sau đó phân phối đều các physical replica đó khắp các GPU *bên trong một node*. `phy2pphy` ánh xạ một physical slot tới vị trí cuối cùng của nó (GPU index × slots_per_gpu + slot), và `pphy2phy` nghịch đảo nó để ta có thể đọc ra các slot theo thứ tự GPU cuối cùng.

Phần đuôi (`default.py:176-188`) chuyển thứ tự physical node-cục-bộ đã được pack trở lại **global logical id**:
```python
pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, axis=1)
pphy2mlog = (pphy2mlog.reshape(num_layers, num_nodes, -1)
             + np.arange(0, num_logical_experts, num_logical_experts // num_nodes)[None,:,None]
            ).reshape(num_layers, -1)
pphy2log = np.take_along_axis(mlog2log, pphy2mlog, axis=1)
return pphy2log
```
`pphy2mlog` thu thập id logical node-cục-bộ của từng physical slot cuối cùng. Số hạng `+ np.arange(0, …, experts_per_node)` cộng lại offset gốc theo-từng-node mà bước reshape ở Bước 2 đã lược bỏ (id mlog của mỗi node đã bị gấp về `[0, experts_per_node)`), khôi phục các id mlog global. Cuối cùng `take_along_axis(mlog2log, …)` chuyển mlog → global logical. `pphy2log` được trả về có shape `[layers, num_replicas]`: logical expert cho từng physical slot, được bố trí theo từng GPU một.

### `rebalance_experts` — điểm vào công khai

`default.py:274-332` (interface trong `abstract.py:12-20`). Điều phối những phần trên:

- `default.py:303-308`: chuyển `weight` sang `float().cpu().numpy()` (thuật toán chạy NumPy/CPU), và `old_global_expert_indices` tùy chọn cũng tương tự.
- `default.py:310-319` — fallback về phân cấp:
  ```python
  if num_groups % num_nodes == 0:
      phy2log_np = cls.rebalance_experts_hierarchical(weight_np, num_replicas, num_groups, num_nodes, num_ranks)
  else:
      phy2log_np = cls.rebalance_experts_hierarchical(weight_np, num_replicas, 1, 1, num_ranks)
  ```
  Nếu các group không chia đều được khắp các node, nó suy biến thành một policy **phẳng/toàn cục (flat/global)** bằng cách truyền `num_groups=1, num_nodes=1` (một node lớn, một group) — cùng đường code, không có phân cấp. Caller cũng bảo vệ điều này: `eplb_state.py:742-748` ép `num_nodes = 1` khi `num_gpus % num_nodes != 0`.
- `default.py:326-329`: nếu có một old map được cung cấp, chạy `preserve_intragpu_slots` (bên dưới).
- `default.py:331`: `torch.from_numpy(phy2log_np)` — quay về một tensor cho caller.

### `preserve_intragpu_slots` — giảm thiểu việc di chuyển weight

`default.py:191-272`. Mục đích (docstring `default.py:198-203`): sắp xếp lại phép gán slot *trong-GPU* của map mới sao cho một logical expert vẫn ở lại trên cùng một GPU giữ nguyên **physical slot trước đó** của nó, tránh một lần copy weight không cần thiết. Đây là một phép hoán vị slot thuần túy về mặt hình thức bên trong mỗi GPU — nó **không** thay đổi expert nào nằm trên GPU nào (điều đó đã được quyết định bởi packing), chỉ thay đổi thứ tự slot của chúng, nên không ảnh hưởng tới cân bằng tải. Nó chỉ chạy khi số GPU và số slot-trên-mỗi-GPU không đổi (`default.py:206-207` guard: `num_phy_experts % num_ranks != 0` → trả về không đổi).

Với từng GPU (`default.py:215-270`), trên dải slot `[start, end)` của nó:
- **Lượt một (First pass)** (`default.py:225-241`): với mỗi old slot, tìm một slot mới-cục-bộ đang giữ *cùng logical id* mà chưa bị chiếm (`matches = (new_local == old_local[:, slot_idx]) & ~used_new_indices`), rồi đặt nó vào vị trí old slot (`post_phy2log[..., start + slot_idx] = …`). Đánh dấu nguồn đó là `used` và đích là `preserved`. Việc này được vector hóa qua các layer với `argmax(matches)` chọn match đầu tiên cho mỗi layer.
- **Lượt hai (Second pass)** (`default.py:243-270`): các expert đến (không được preserve) lấp vào các slot còn thừa. Nó xây các mảng ưu tiên (`np.where(mask, idx_base, sentinel)`), `argsort` chúng để có được các vị trí nguồn/đích khả dụng theo thứ tự, lấy `min(remaining, fill)` trong số đó cho mỗi layer, và gán `post_phy2log[layer, start + dst_pos] = new_local[layer, src_pos]`. Bước này chỉ đơn giản là nhét các expert mới còn lại vào các lỗ trống còn lại theo một thứ tự ổn định.

Kết quả là cùng một phép gán GPU→experts mà thuật toán đã tạo ra, nhưng với các vị trí slot được chọn để tối đa hóa mức trùng lặp với bố cục cũ — nhờ đó `rearrange_expert_weights_inplace` (`eplb_state.py:766`) có thể bỏ qua việc copy weight cho những expert thực sự không di chuyển.

## Tổng kết đầu-cuối

Cho tải trên-từng-logical-expert `weight[layers, num_log]`:

1. **Groups → nodes**: `balanced_packing` trên tải của group phân bố các expert group đều khắp các node, giữ nguyên vẹn mỗi group (locality của phân cấp). Tạo ra một phép gán nhãn lại node-cục-bộ `mlog2log`.
2. **Replicate within node**: `replicate_experts` trao các slot physical dư của mỗi node cho các expert nóng nhất của nó (theo `weight/logcnt`), giảm thiểu tải trên-mỗi-replica lớn nhất. Tạo ra `phy2mlog`, `mlogcnt`.
3. **Physical → GPUs**: tải hiệu dụng trên-mỗi-replica `weight/mlogcnt` được `balanced_packing` pack khắp các GPU bên trong mỗi node, để mỗi GPU gánh tải gần bằng nhau. Các chỉ số được ánh xạ trở lại global logical id để cho ra `phy2log[layers, num_replicas]`.
4. **(Tùy chọn)** `preserve_intragpu_slots` sắp xếp lại các slot trong-GPU để khớp với old map và tránh các lần copy weight.

Hai ý tưởng thiết kế gánh vác toàn bộ phần toán học: **packing tham lam lớn-trước vào bin nhẹ nhất chưa đầy** (để phân hoạch đều dưới ràng buộc lực lượng chính xác), và **tham lam trao mỗi replica dư cho expert có `load/replica_count` lớn nhất** (để nhân bản theo hướng giảm thiểu tải). Toàn bộ công việc là NumPy được batch theo từng layer trên CPU; đầu vào ngẫu nhiên duy nhất là cửa sổ tải đã được ghi nhận.

Các đường dẫn file liên quan (tuyệt đối):
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/default.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/abstract.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/__init__.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py` (tổng hợp tải + điểm gọi)
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_utils.py` (KHÔNG chứa thuật toán trong phiên bản này)

---

# 4. Communicator Backends (eplb_communicator.py)

# EPLB Communicators (`distributed/eplb/eplb_communicator.py`)

## Khái niệm và mục đích

EPLB (Expert Parallel Load Balancing) định kỳ cân bằng lại việc physical MoE expert nào nằm trên rank nào. Khi bộ cân bằng quyết định rằng expert *E* nên chuyển từ rank *A* sang rank *B*, các **tensor trọng số của expert** cho *E* phải được chuyển vật lý từ bộ nhớ GPU của *A* sang bộ nhớ GPU của *B*. EPLB *communicator* là lớp trừu tượng thực hiện các lần chuyển đó.

Tệp này định nghĩa một lớp cơ sở trừu tượng (`EplbCommunicator`) và bốn backend cụ thể, cộng thêm một factory (`create_eplb_communicator`) chọn backend dựa trên vị trí thiết bị (device placement), loại group (static so với elastic), và một tên backend tường minh:

| Backend | Class | Transport | Nơi diễn ra giao tiếp |
|---|---|---|---|
| `torch_nccl` | `TorchDistNcclEplbCommunicator` | `torch.distributed` isend/irecv (NCCL) | trong `execute()` |
| `torch_gloo` | `TorchDistGlooStagedEplbCommunicator` | gloo P2P qua CPU staging | trong `execute()` |
| `nixl` | `NixlEplbCommunicator` | NIXL/RIXL RDMA READ (zero-copy) | thực hiện sớm (eager) trong `add_recv()`, chờ trong `execute()` |
| `pynccl` | `PyNcclEplbCommunicator` | PyNCCL ncclSend/ncclRecv | streamed qua `add_send`/`add_recv`, đóng lại trong `execute()` |

Giao thức gọi chung mà bộ rebalancer sử dụng (trong code kiểu `move_to_buffer`) là:

1. Tùy chọn `set_transfer_context(old_indices, layer_idx)` — chỉ NIXL dùng đến.
2. Xếp hàng các lần chuyển bằng `add_send(...)` (trên rank nguồn) và `add_recv(...)` (trên rank đích).
3. Gọi `execute()` — sau khi nó trả về, mọi dữ liệu được đảm bảo đã hiện diện trong các buffer đích.

Lưu ý sự bất đối xứng send/recv giữa các backend: với NCCL/gloo/PyNCCL, cả hai phía đều xếp hàng các op tương ứng; với NIXL, chỉ bên nhận hành động (`add_recv`), vì NIXL dùng RDMA READ do bên nhận khởi tạo (receiver-initiated) — `add_send` là một no-op.

---

## Imports và bước dò khả dụng của NIXL

`3rdparty/vllm/distributed/eplb/eplb_communicator.py:16-20` import các primitive P2P của torch.distributed (`P2POp`, `ProcessGroup`, `batch_isend_irecv`) được các backend NCCL/gloo dùng. `:22-33` kéo vào các helper NIXL, wrapper PyNCCL, các helper trạng thái group (`GroupCoordinator`, `get_pp_group`, `is_local_first_rank`), `StatelessGroupCoordinator` (elastic EP), và `is_weak_contiguous` (một phép kiểm tra tính liên tục cho phép các dim kích thước 1, v.v.).

`:40-42`:
```python
def has_nixl() -> bool:
    return nixl_utils.NixlWrapper is not None
```
NIXL là tùy chọn; `NixlWrapper` là `None` khi gói chưa được cài đặt. Factory gọi `has_nixl()` trước khi thử backend NIXL.

---

## `EplbCommunicator` (lớp cơ sở trừu tượng) — `:45-95`

`:48-64` khai báo hai phương thức xếp hàng trừu tượng. Cả hai đều nhận một danh sách tensor (tất cả tensor trọng số cho một expert đơn lẻ trên một layer đơn lẻ), rank đối tác, và một `expert_id`. Lưu ý rằng `expert_id` được ghi chú là **không dùng đến** bởi các backend NCCL/gloo/PyNCCL (xem chữ ký của chúng) — nó tồn tại trong interface chỉ vì NIXL cần nó để tra cứu physical source row.

`:66-73` — docstring của `execute()` nêu rõ hợp đồng ngữ nghĩa mấu chốt: *"Some backends perform communication here; others (e.g. NIXL) issue transfers eagerly in add_recv and only wait here. On return, all data is available in the destination buffers."* Đây là bất biến mà mọi backend phải giữ vững.

`:75-82` — `set_transfer_context(old_indices, layer_idx)` là một **no-op mặc định** cụ thể (`# noqa: B027` chặn cảnh báo flake8 về một phương thức không trừu tượng rỗng). Chỉ NIXL override nó; nó cung cấp ngữ cảnh theo từng layer "physical row nào giữ expert nào" mà NIXL cần để tính các địa chỉ nguồn RDMA bên trong `add_recv`.

`:84-88` — `needs_profile_buffer_reservation` mặc định là `True`. Trong lượt profiling bộ nhớ của vLLM, hầu hết các backend cần một collective giả để đặt trước (reserve) các buffer giao tiếp của chúng để profiling tính đến phần bộ nhớ đó. NIXL override thành `False` (`:310-312`) vì nó thực hiện RDMA zero-copy và không đặt trước buffer nào như vậy.

`:90-91` — `set_stream()` lưu một CUDA stream vào `self._cuda_stream`; các backend CUDA chạy op của chúng bên trong `torch.cuda.stream(self._cuda_stream)`. NIXL override thành no-op (`:328-329`) vì nó không có CUDA stream để điều khiển.

`:93-95` — `_log_initialized()` ghi log tên lớp một lần mỗi node (được kiểm soát bởi `is_local_first_rank()`).

---

## `TorchDistNcclEplbCommunicator` — `:98-152`

Backend đơn giản nhất: nó chỉ tích lũy các đối tượng `P2POp` và xả (flush) chúng bằng `batch_isend_irecv`.

- `:101-109` `__init__` lưu `ep_group` (device/NCCL process group), CUDA stream tùy chọn, và một danh sách `self._p2p_ops` rỗng.
- `:111-125` `add_send` — với mỗi tensor, thêm một `P2POp(torch.distributed.isend, tensor, dst_rank, self._ep_group)`. Chưa có gì được gửi đi; op bị hoãn lại.
- `:127-141` `add_recv` — đối xứng; thêm một `P2POp(torch.distributed.irecv, ...)`.
- `:143-152` `execute` — trả về sớm nếu không có op nào. Ngược lại, dưới CUDA stream đã cấu hình, `batch_isend_irecv(self._p2p_ops)` phát hết chúng như một NCCL batch, rồi `req.wait()` trên mỗi request trả về chặn cho đến khi hoàn tất. Khối `finally` xóa danh sách op để communicator có thể tái sử dụng. Việc gom tất cả send+recv lại với nhau chính là điều cho phép NCCL tránh deadlock trên P2P theo cặp có thứ tự.

---

## `TorchDistGlooStagedEplbCommunicator` — `:155-238`

gloo không có transport GPU, nên backend này **staging qua CPU**: copy GPU→CPU, làm gloo P2P trên các tensor CPU, copy CPU→GPU.

- `:158-166` `__init__` lưu `cpu_group` (một gloo process group), stream tùy chọn, và `self._ops` như một danh sách các bộ ba `(op_name, tensor, peer_rank)`. Lưu ý nó ghi lại op trừu tượng, không phải `P2POp` — việc staging thực tế diễn ra lazily trong `execute`.
- `:168-184` `add_send`/`add_recv` chỉ ghi lại `("send", tensor, dst_rank)` / `("recv", tensor, src_rank)`.
- `:186-238` `execute`:
  - `:193-215` `build_ops()` duyệt qua các op đã ghi. Với một **send**, nó làm `cpu_tensor = tensor.to(device="cpu", non_blocking=True)` (copy D2H bất đồng bộ) và thêm một isend của tensor CPU đó. Với một **recv**, nó cấp phát `torch.empty_like(tensor, device="cpu")`, thêm một irecv vào đó, và ghi `(dst_gpu_tensor, cpu_tensor)` vào `recv_staging` để có thể copy trở lại sau này.
  - `:217-219` chạy `build_ops()` bên trong CUDA stream (nên các copy D2H được xếp hàng ở đó).
  - `:223-228` — **đồng bộ hóa mấu chốt**: trước khi phát các op gloo nó đồng bộ CUDA stream (hoặc stream hiện tại) để tất cả copy D2H đã thực sự đáp vào các buffer CPU. Gửi một staging buffer chưa được điền vào sẽ truyền đi rác.
  - `:230-232` `batch_isend_irecv(p2p_ops)` trên gloo group, rồi chờ trên mỗi request.
  - `:234-238` — với mọi tensor CPU nhận được, `dst_tensor.copy_(cpu_tensor, non_blocking=True)` copy nó trở lại GPU dưới CUDA stream. (Ở đây không có sync tường minh sau copy H2D này — tính đúng đắn dựa vào việc caller sắp thứ tự công việc GPU tiếp theo trên cùng stream. Đây là một chi tiết tinh tế phụ thuộc phiên bản đáng lưu ý.)

---

## `NixlEplbCommunicator` — `:241-571`

Đây là backend phức tạp nhất và là trọng tâm của tác vụ. Nó dùng **NIXL/RIXL zero-copy RDMA READ**: mỗi rank pre-register toàn bộ bộ nhớ trọng số expert của nó với NIXL một lần; tại thời điểm rebalance, một rank nhận phát một RDMA READ kéo expert row cần thiết trực tiếp ra khỏi bộ nhớ trọng số đã đăng ký của một rank từ xa vào buffer nhận cục bộ của nó. Không có send tường minh — do đó `add_send` là một no-op.

### `__init__` — `:244-308`

- `:250-256` — assert rằng cả `all_expert_weights` và `expert_buffer` đều không rỗng, và rằng `NixlWrapper` import được (nếu không thì `RuntimeError("NIXL/ RIXL is unavailable.")`).
- `:258-260` lưu **CPU** process group (dùng cho all-gather metadata và barrier sau chuyển), cộng với `world_size`/`rank` suy ra từ nó.
- `:262-265`:
  - `self._all_expert_weights` — toàn bộ các tensor trọng số `(num_layers)(num_tensors_per_layer)`.
  - `self._expert_buffer` — các buffer nhận được cấp phát trước (một cho mỗi tensor trọng số của *một* layer).
  - `self._num_local_experts = all_expert_weights[0][0].shape[0]` — suy ra từ dim 0 của tensor trọng số đầu tiên (dim dẫn đầu đánh chỉ mục các local expert).
  - `self._device` — thiết bị của tensor đầu tiên.
- `:267-279` — vòng lặp validation: mọi tensor trọng số expert phải là `is_weak_contiguous` (RDMA cần một bố cục byte tuyến tính) và trên cùng `self._device`; mọi tensor `expert_buffer` cũng phải contiguous.
- `:281-286` — trạng thái chuyển:
  - `self._xfer_entries: list[tuple[int, int, int]]` — các READ đang bay (in-flight) dưới dạng `(local_dlist, remote_dlist, xfer_handle)`; được điền bởi `add_recv`, được rút cạn bởi `execute`.
  - `self._expert_to_src_row: list[dict[int, int]] | None` — `{expert_id: physical_row}` theo từng rank, được đặt bởi `set_transfer_context`.
  - `self._layer_idx: int | None` — layer hiện đang được chuyển.
- `:288-294` — dựng một cấu hình NIXL agent (`capture_telemetry=False` nếu helper cấu hình tồn tại) và khởi tạo wrapper: `self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)`. Mỗi rank là một NIXL agent riêng biệt.
- `:295-302` — `self._nixl_memory_type = "VRAM"` (trọng số nằm trong GPU VRAM); `self._registered_descs` (các handle đăng ký, được giải phóng trong `__del__`); `self._remote_agents: dict[int, str]` (peer rank → tên NIXL agent); và `self._remote_send_meta` — `peer -> (layer, tensor_idx) -> (base_ptr, bytes_per_expert, dev_id)`, sổ địa chỉ từ xa dùng để tính các địa chỉ nguồn RDMA.
- `:304-308` — chuỗi init có thứ tự, mỗi bước được bọc trong `_init_step` để gán nhãn lỗi đồng nhất:
  ```python
  self._cuda_device_id = int(self._device.index or 0)
  self._init_step("buffers", self._init_registered_buffers)
  self._init_step("agents", self._init_remote_agents)
  self._init_step("send meta", self._exchange_remote_send_meta)
  self._log_initialized()
  ```
  Thứ tự quan trọng: buffer phải được đăng ký trước khi các agent trao đổi metadata, và các agent phải được biết đến trước khi send-meta được trao đổi (send-meta lặp qua `self._remote_agents`).

### `_init_step` — `:314-319`

Một wrapper tĩnh chạy `fn(*args, **kwargs)` và, khi có bất kỳ ngoại lệ nào, re-raise thành `RuntimeError(f"NIXL EPLB init failed: {name}")` với `from exc` để bảo toàn nguyên nhân. Đây là lý do các lỗi init hiện ra dưới dạng, ví dụ, `"NIXL EPLB init failed: buffers"`.

### `_make_agent_name` — `:321-326`

Dựng một tên agent duy nhất theo triển khai: `f"eplb-{self._rank}{pp_suffix}-{uid}"`, trong đó `pp_suffix` là `-pp{pp_rank}` chỉ khi pipeline-parallel size > 1, và `uid` là 8 ký tự hex từ `uuid.uuid4()`. Rank phân biệt các peer; hậu tố PP phân biệt các giai đoạn PP; uuid ngăn xung đột tên qua các lần restart/re-init.

### `_init_registered_buffers` — `:417-425` (`register_memory` tại `:424`)

```python
all_tensors: list[torch.Tensor] = []
for layer_tensors in self._all_expert_weights:
    all_tensors.extend(layer_tensors)
all_tensors.extend(self._expert_buffer)
descs = self._nixl_wrapper.get_reg_descs(all_tensors)
self._nixl_wrapper.register_memory(descs)
self._registered_descs.append(descs)
```
Nó làm phẳng **tất cả tensor trọng số của mọi layer cộng với các buffer nhận** thành một danh sách, biến chúng thành các descriptor đăng ký NIXL (`get_reg_descs`), và đăng ký bộ nhớ đó với NIXL (`register_memory` tại `:424`). Việc đăng ký ghim/phơi bày (pin/expose) bộ nhớ cho RDMA — trọng số để các rank từ xa có thể READ chúng, buffer để các READ cục bộ có thể đáp vào chúng. Handle được giữ trong `self._registered_descs` để hủy đăng ký sau này trong `__del__`.

### `_init_remote_agents` — `:402-415`

```python
local_metadata = self._nixl_wrapper.get_agent_metadata()
gathered_metadata = [None] * self._world_size
torch.distributed.all_gather_object(gathered_metadata, local_metadata, group=self._cpu_group)
for peer in range(self._world_size):
    if peer == self._rank:
        continue
    peer_metadata = gathered_metadata[peer]
    assert peer_metadata is not None
    self._remote_agents[peer] = self._nixl_wrapper.add_remote_agent(peer_metadata)
```
Mỗi rank tuần tự hóa metadata NIXL agent của chính nó (thông tin kết nối) và all-gather nó trên CPU group. Với mọi peer trừ chính nó, nó đăng ký agent của peer qua `add_remote_agent`, lưu tên agent trả về vào `self._remote_agents[peer]`. Sau bước này, NIXL agent cục bộ biết cách tiếp cận agent của mọi rank khác.

### `_exchange_remote_send_meta` — `:427-469`

Bước này dựng sổ địa chỉ cho phép một bên nhận tính chính xác nơi các byte của một expert nhất định nằm trong bộ nhớ trọng số đã đăng ký của một rank từ xa.

- `:430-438` — dựng `local_meta`: với mỗi `(layer_idx, tensor_idx)`, ghi `(t.data_ptr(), nbytes_per_expert, cuda_device_id)`, trong đó `nbytes_per_expert = t.nbytes // self._num_local_experts`. Vì dim 0 của mỗi tensor trọng số đánh chỉ mục các local expert và tensor là contiguous, các byte của expert *r* bắt đầu tại `data_ptr() + r * nbytes_per_expert`. Bước sải (stride) theo từng expert này là đại lượng mấu chốt.
- `:443-448` — all-gather `local_meta` trên CPU group vào `gathered_meta`.
- `:450-468` — validation cho meta của mỗi peer của remote agent:
  - `:454-459` — tập hợp các khóa `(layer, tensor)` phải khớp chính xác giữa các rank; một sự không khớp raise `"NIXL EPLB metadata key mismatch with rank {peer}..."`. Điều này bảo vệ chống lại các rank có topology layer/tensor khác nhau.
  - `:460-468` — với mỗi khóa, stride theo từng expert (`nbytes_per_expert`) phải khớp giữa cục bộ và peer; một sự không khớp raise `"NIXL EPLB nbytes_per_expert mismatch..."`. Điều này đảm bảo số học row của bên nhận là hợp lệ đối với bố cục của bên gửi.
  - `:469` — lưu meta peer đã được validate vào `self._remote_send_meta[peer]`.

### `set_transfer_context` — `:341-355`

Được gọi một lần mỗi layer trước các lần gọi `add_recv` của layer:
```python
assert not self._xfer_entries, (... "execute() was not called after previous add_recv() calls")
self._layer_idx = layer_idx
n = self._num_local_experts
rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
self._expert_to_src_row = [
    {int(eid): i for i, eid in enumerate(row) if eid != -1}
    for row in rank_experts
]
```
- `:344-348` — assert không còn lần chuyển nào đang chờ (tức là, layer trước đã được `execute()` đúng cách).
- `:349-355` — `old_indices` là vị trí vật lý *trước rebalance* được làm phẳng dưới dạng `(world_size * num_local_experts)`. Nó được reshape thành `(world_size, num_local_experts)` nên `rank_experts[r]` là danh sách các expert ID nằm vật lý trên rank *r*. Với mỗi rank nó dựng `{expert_id: row_index}`, bỏ qua `-1` (ô trống). Ánh xạ này cho phép `add_recv` dịch "Tôi muốn expert *E* từ rank *src*" thành "đọc row *i* của trọng số của rank *src*."

### `add_send` — `:331-339`

Một no-op. Chú thích `:337-338`: *"NIXL READ is receiver-initiated. The sender's expert weights are pre-registered and always readable in-place."* Bên gửi không làm gì tại thời điểm chuyển.

### `add_recv` — `:357-400` (phát RDMA READ một cách eager)

```python
assert self._expert_to_src_row is not None and self._layer_idx is not None, (...)
src_row = self._expert_to_src_row[src_rank][expert_id]
layer_idx = self._layer_idx
local_descs, remote_descs = [], []
for t_idx, t in enumerate(tensors):
    send_base, send_stride, remote_dev = self._remote_send_meta[src_rank][(layer_idx, t_idx)]
    assert t.nbytes == send_stride, (...)
    local_descs.append((t.data_ptr(), t.nbytes, self._cuda_device_id))
    remote_descs.append((send_base + src_row * send_stride, send_stride, remote_dev))
local_h, remote_h, xfer_h = self._create_peer_xfer(src_rank, local_descs, remote_descs)
self._nixl_wrapper.transfer(xfer_h)
self._xfer_entries.append((local_h, remote_h, xfer_h))
```
- `:366-368` — yêu cầu ngữ cảnh đã được đặt.
- `:369` — phân giải `src_row` = physical row trên `src_rank` giữ `expert_id`.
- `:374-394` — với mỗi tensor đích `t` (buffer nhận cho một tensor trọng số):
  - Tra cứu bố cục từ xa `(send_base, send_stride, remote_dev)` từ `_remote_send_meta`.
  - `:378-380` — assert tensor đích cục bộ đúng bằng số byte của một expert (`t.nbytes == send_stride`); tức là, buffer nhận giữ một expert row đơn lẻ.
  - `:381-387` — **local descriptor** là `(dst_data_ptr, nbytes, local_dev)`.
  - `:388-394` — **remote descriptor** là `(send_base + src_row * send_stride, send_stride, remote_dev)` — dải byte chính xác của expert mong muốn trong trọng số đã đăng ký của rank từ xa. Đây là số học địa chỉ row được `_exchange_remote_send_meta` cho phép.
- `:396-399` — dựng transfer đã prep qua `_create_peer_xfer` và **phát nó ngay lập tức** với `self._nixl_wrapper.transfer(xfer_h)`. Đây là hành vi eager được ghi chú trong docstring `execute` của lớp cơ sở — RDMA READ bắt đầu ở đây, chồng lấp (overlap) với phần còn lại của vòng lặp xếp hàng Python.
- `:400` — ghi lại ba handle để chờ/dọn dẹp.

### `_create_peer_xfer` — `:487-524`

Dựng một READ được gom theo batch đơn lẻ trên nhiều descriptor từ một peer:
- `:500-506` — biến `local_descs` thành một danh sách descriptor NIXL (`get_xfer_descs`) và prep nó dưới local init agent (`prep_xfer_dlist("NIXL_INIT_AGENT", ...)`).
- `:508-514` — tương tự cho `remote_descs`, prep đối với tên agent của peer `self._remote_agents[src]`.
- `:516-523` — `make_prepped_xfer("READ", local_handle, indices, remote_handle, indices)` tạo transfer handle; `indices = range(len(local_descs))` ghép mỗi local descriptor với remote descriptor cùng chỉ mục. Trả về `(local_handle, remote_handle, xfer_handle)`.

### `_wait_for_all_transfers` — `:471-485`

Poll cho đến khi tất cả READ đã phát hoàn tất:
```python
pending = set(handles)
while pending:
    completed = []
    for handle in pending:
        state = self._nixl_wrapper.check_xfer_state(handle)
        if state == "DONE":
            completed.append(handle)
        elif state != "PROC":
            raise RuntimeError(f"NIXL transfer failed with state={state}")
    for handle in completed:
        pending.remove(handle)
    if pending:
        time.sleep(0.0005)
```
Nó coi `"DONE"` là hoàn tất, `"PROC"` là vẫn đang tiến hành, và bất cứ thứ gì khác là thất bại. Nó ngủ 0.5 ms giữa các lượt poll để tránh busy spin. (Đây là một busy-poll phía CPU; không có cơ chế callback/event nào.)

### `execute` — `:526-551`

```python
assert self._layer_idx is not None or not self._xfer_entries, (...)
try:
    self._wait_for_all_transfers([x[2] for x in self._xfer_entries])
    torch.distributed.monitored_barrier(group=self._cpu_group, timeout=timedelta(minutes=5))
finally:
    for local_h, remote_h, xfer_h in self._xfer_entries:
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_xfer_handle(xfer_h)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(local_h)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(remote_h)
    self._xfer_entries.clear()
    self._expert_to_src_row = None
    self._layer_idx = None
```
- `:532` — chờ mọi READ đang bay (dùng phần tử thứ ba, xfer handle) đạt `"DONE"`.
- `:534-540` — một **`monitored_barrier` sau READ** trên CPU group. Chú thích: *"Correctness fence for zero-copy: prevents overwrite-while-remote-read race."* Vì các READ kéo trực tiếp từ bộ nhớ trọng số đã đăng ký từ xa, không rank nào được phép bắt đầu biến đổi (mutate) trọng số của nó (bước rebalance tiếp theo) cho đến khi *tất cả* các rank đã đọc xong từ mọi rank — barrier cưỡng chế thứ tự toàn cục đó. Timeout 5 phút bảo vệ chống lại một peer bị treo.
- `:541-551` — `finally` giải phóng mọi NIXL transfer handle và cả hai dlist handle (mỗi cái được bảo vệ bởi `contextlib.suppress(Exception)` để việc dọn dẹp không bao giờ che khuất lỗi thực sự), rồi reset `_xfer_entries`, `_expert_to_src_row`, và `_layer_idx` để communicator sẵn sàng cho layer tiếp theo.

### `__del__` — `:553-571`

Teardown nỗ lực tối đa (best-effort), mọi thứ được bọc trong các `contextlib.suppress(Exception)` lồng nhau:
- `:554-561` — giải phóng bất kỳ transfer/dlist handle còn sót lại trong `_xfer_entries`.
- `:562-566` — `deregister_memory` cho mỗi tập desc đã đăng ký, rồi xóa danh sách.
- `:567-571` — `remove_remote_agent` cho mỗi peer agent đã biết, rồi xóa map.
Điều này phản chiếu các tài nguyên được cấp phát trong `_init_registered_buffers` và `_init_remote_agents`.

---

## `PyNcclEplbCommunicator` — `:574-615`

Dùng `PyNcclCommunicator` riêng của vLLM và ngữ nghĩa NCCL group.
- `:577-585` `__init__` lưu `pynccl_comm`, stream tùy chọn, và `self._group_started = False`.
- `:587-590` `_ensure_group_started` — gọi lazily `self._pynccl_comm.group_start()` trên lần send/recv đầu tiên và lật cờ. `group_start`/`group_end` đóng khung một tập các lệnh gọi NCCL để chúng được hợp nhất thành một thao tác group được gom (coalesced).
- `:592-600` `add_send` — đảm bảo group đã bắt đầu, rồi `self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)` cho mỗi tensor. Các send được phát ngay lập tức (bên trong group đang mở), không bị hoãn.
- `:602-610` `add_recv` — đối xứng với `.recv(...)`.
- `:612-615` `execute` — nếu một group đã bắt đầu, `group_end()` đóng nó (đây là nơi NCCL thực sự hoàn tất batch send/recv được gom) và reset cờ. Nếu không có gì được xếp hàng, nó không làm gì.

Không giống backend NCCL `P2POp`, PyNCCL phát các op một cách eager trong một group đang mở và hoàn tất chúng tại `group_end()`.

---

## Factory `create_eplb_communicator` — `:618-734`

Chọn và khởi tạo một backend. Chữ ký `:618-623`:
```python
def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str | None,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
) -> EplbCommunicator:
```

**Backend mặc định** — `:644-645`: nếu `backend is None`, nó trở thành `"torch_nccl"`.

**Phát hiện thiết bị** — `:647-653`:
```python
first_layer = expert_weights[0] if expert_weights else []
tensor_device_type = first_layer[0].device.type if first_layer else "cpu"
torch_group = (
    group_coordinator.cpu_group
    if tensor_device_type == "cpu"
    else group_coordinator.device_group
)
```
Nó xem xét thiết bị của tensor trọng số đầu tiên. Nếu trọng số ở trên CPU nó dùng CPU process group; ngược lại dùng device (GPU/NCCL) group. `torch_group` chỉ được tiêu thụ bởi nhánh `torch_nccl`.

**Closure `_create_pynccl`** — `:655-689`: validate và dựng một `PyNcclEplbCommunicator`:
- `:656-660` — từ chối tensor CPU (PyNCCL chỉ dành cho CUDA).
- `:661-674` — kiểm tra mọi dtype của tensor có biểu diễn được trong NCCL (`ncclDataTypeEnum.supports_torch_dtype`); các dtype không được hỗ trợ raise cùng danh sách vi phạm.
- `:676-683` — kéo `pynccl_comm` ra khỏi device communicator và yêu cầu nó hiện diện, không `disabled`, và `available`.
- `:684-689` — khởi tạo communicator, bọc bất kỳ thất bại nào trong một `RuntimeError` mô tả.

**Quy tắc thăng cấp (promotion) cho Elastic-EP (stateless)** — `:691-704`:
```python
is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
if is_stateless:
    if backend not in ("torch_nccl", "pynccl"):
        raise ValueError("Elastic EP requires 'torch_nccl' or 'pynccl' ...")
    if backend == "torch_nccl":
        logger.warning("Stateless elastic EP requires PyNCCL backend. Forcing EPLB communicator to 'pynccl'.")
        backend = "pynccl"
    return _create_pynccl()
```
Với elastic EP (`StatelessGroupCoordinator`), chỉ `torch_nccl` và `pynccl` được cho phép; `nixl`/`torch_gloo` raise `ValueError`. Điểm mấu chốt, `torch_nccl` được **thăng cấp âm thầm thành `pynccl`** (kèm cảnh báo) — đường dẫn stateless chỉ hoạt động trên PyNCCL — và hàm trả về một PyNCCL communicator bất kể.

**Các nhánh static-group** — `:706-734`:
- `:706-725` `nixl` — yêu cầu `has_nixl()` (nếu không thì RuntimeError) và tensor kiểu CUDA, không phải CPU (nếu không thì RuntimeError). Dựng `NixlEplbCommunicator(cpu_group=..., all_expert_weights=expert_weights, expert_buffer=expert_buffer)`, bọc thất bại trong `RuntimeError(f"Failed to initialize NixlEplbCommunicator ({exc}).")`.
- `:726-729` `torch_gloo` — `TorchDistGlooStagedEplbCommunicator(cpu_group=group_coordinator.cpu_group)`.
- `:730-731` `torch_nccl` — `TorchDistNcclEplbCommunicator(ep_group=torch_group)` (CPU group nếu trọng số ở trên CPU, ngược lại là device group).
- `:732-733` `pynccl` — ủy quyền cho `_create_pynccl()`.
- `:734` — bất kỳ tên nào khác raise `ValueError(f"Unknown EPLB communicator backend: {backend}")`.

### Tóm tắt lựa chọn

- `backend=None` → `torch_nccl`.
- Stateless/elastic EP → bị buộc thành `pynccl` (`torch_nccl` được thăng cấp kèm cảnh báo; `nixl`/`torch_gloo` bị từ chối).
- Tensor CPU → `torch_nccl` dùng CPU group; `torch_gloo` cũng hoạt động; `nixl`/`pynccl` từ chối CPU.
- `nixl` → yêu cầu gói NIXL cộng với tensor GPU kiểu CUDA.

---

## Ghi chú / cảnh báo

- `expert_id` là một phần của chữ ký trừu tượng `add_send`/`add_recv` nhưng chỉ có ý nghĩa đối với NIXL; ba backend còn lại chú thích tường minh nó `# unused`.
- `set_transfer_context` và luồng `add_recv` chỉ dành cho bên nhận là đặc thù của NIXL; các backend khác bỏ qua ngữ cảnh và yêu cầu các cặp `add_send`/`add_recv` khớp nhau.
- Copy H2D cuối cùng của backend gloo (`:236-238`) được xếp hàng trên CUDA stream mà không có sync tường minh sau copy trong `execute`; tính đúng đắn phụ thuộc vào việc caller sắp thứ tự công việc GPU hạ nguồn trên cùng stream. Đây là một chi tiết triển khai có thể khác nhau giữa các phiên bản.
- `_wait_for_all_transfers` là một busy-poll phía CPU với giấc ngủ 0.5 ms; trên số lượng chuyển rất lớn, chi phí poll này khác không nhưng bị chặn trên (bounded).
- NIXL dựa vào việc các tensor trọng số là contiguous với dim 0 = trục local-expert và một per-expert byte stride đồng nhất giữa các rank; cả hai đều được cưỡng chế tại init (các phép kiểm tra `is_weak_contiguous` và phép kiểm tra bằng nhau `nbytes_per_expert` giữa các rank). Nếu một model bố trí trọng số expert theo cách khác, backend này sẽ không áp dụng được.

---

# 5. Tích hợp MoE & Process Group

Tôi đã có mọi thứ cần thiết để viết một mục tham chiếu chính xác, theo từng dòng. Dưới đây là tài liệu.

---

## Tích hợp EPLB ↔ MoE và EPLB Process Group

### Khái niệm và mục đích

EPLB (Expert Parallel Load Balancing) giải quyết một vấn đề cụ thể trong suy luận Mixture-of-Experts dưới Expert Parallelism (EP): với cách chia expert tĩnh, đều nhau giữa các GPU, lưu lượng thực tế lại bị lệch — một số ít expert logic "nóng" (hot) nhận phần lớn token, nên các GPU giữ chúng trở thành nút thắt cổ chai trong khi các GPU khác nhàn rỗi. EPLB khắc phục điều này bằng cách tách rời **logical expert** (E expert mà mô hình được huấn luyện) khỏi **physical expert** (E + R khe replica thực sự được bố trí trên các rank, R = redundant expert). Sau đó nó (a) *nhân bản* (replicate) các logical expert nóng vào các khe physical bổ sung, (b) *gán lại* khe physical nào nằm trên rank nào, và (c) định kỳ *rearrange* các tensor trọng số thực tế giữa các GPU để khớp với load đo được.

Có ba mảnh trạng thái riêng biệt làm cho điều này hoạt động, và lớp MoE chạm đến cả ba:

- **`expert_map`** — một bảng chỉ số physical global→local theo từng rank. Đây là cơ chế định tuyến (routing) của *EP* (tồn tại ngay cả khi không có EPLB) và cho kernel biết những physical expert nào đang cư trú cục bộ.
- **`logical_to_physical_map` / `logical_replica_count`** — các bảng chuyển hướng của *EPLB*. Routing tạo ra ID expert *logic*; EPLB viết lại chúng thành ID *physical* (chọn trong số các replica) trước khi dispatch.
- **`expert_load_view`** — một tensor bộ đếm mà forward pass tăng lên theo từng physical expert, cung cấp dữ liệu cho chính sách rebalance.

Việc di chuyển trọng số trong lúc rearrange chạy trên một **process group riêng** (`get_eplb_group()`), tách biệt với nhóm EP của forward pass, để các phép chuyển trọng số tập thể (collective) không thể xen kẽ/deadlock với các collective forward của MoE.

Bên dưới, các đường dẫn `file:line` là tương đối với repo dưới `3rdparty/vllm`.

---

### 1. Mô hình chỉ số hai cấp: physical vs logical

`distributed/eplb/eplb_state.py:90-148` định nghĩa `EplbModelState`, với các docstring là đặc tả có thẩm quyền:

- `physical_to_logical_map` — kích thước `(num_moe_layers, num_physical_experts)`. Phần tử `[layer, p]` = logical expert mà khe physical `p` đang giữ. Ví dụ cho 6 physical / 4 logical / 3 rank: `[[0,1,2,3,0,1],[0,2,0,1,0,3]]` — lưu ý logical expert 0 và 1 được nhân bản (xuất hiện hai lần) ở layer 0.
- `logical_to_physical_map` — kích thước `(num_moe_layers, num_logical_experts, num_redundant_experts + 1)`, một bảng **sparse** (thưa) được đệm bằng `-1`. Phần tử `[layer, l, :]` liệt kê mọi khe physical đang giữ logical expert `l`. Đây là bảng mà forward path đọc để chọn một replica.
- `logical_replica_count` — kích thước `(num_moe_layers, num_logical_experts)`, "đúng bằng số lượng phần tử khác `-1` trong `logical_to_physical_map`."

Layout physical→logical **ban đầu** (trước khi rearrange) được dựng bởi `FusedMoE.make_expert_params_mapping` tại thời điểm nạp trọng số — `model_executor/layers/fused_moe/layer.py:1336-1380`:

```python
physical_to_logical_map = (
    EplbState.build_initial_global_physical_to_logical_map(
        num_experts, num_redundant_experts
    )
)
...
f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{base_layer}",
```

- `layer.py:1346` `num_physical_experts = num_experts + num_redundant_experts`.
- `layer.py:1352-1356` map ban đầu là `[0,1,...,num_experts-1]` theo sau bởi các khe redundant (xem `eplb_state.py:300-304`, nơi nối thêm phần đệm kiểu `num_routed_experts + i` — các khe redundant khởi đầu như bản sao).
- `layer.py:1370` là hợp đồng tại thời điểm nạp: vòng lặp duyệt `expert_id` trên các khe **physical** (`layer.py:1374`), nhưng tên tensor checkpoint được khóa bởi id **logic** `physical_to_logical_map[expert_id]`. Vì vậy khe physical 5 vốn ban đầu phản chiếu logical expert 1 sẽ nạp trọng số checkpoint của logical-1. Đây chính là điều cho phép một khe physical redundant được khởi tạo như một replica thực sự.

---

### 2. `expert_map` — bảng định tuyến EP global→local

Lớp này tách biệt với EPLB và tồn tại bất cứ khi nào `ep_size > 1`. `model_executor/layers/fused_moe/expert_map_manager.py:22-113` (`determine_expert_map`) dựng nó:

- `expert_map_manager.py:63-64` `ep_size == 1` → trả về `(global_num_experts, None, None)`: không cần map, mọi expert đều cục bộ.
- `expert_map_manager.py:67-69` chia đều; phần dư dồn về các rank thấp (`ep_rank < remainder` nhận thêm một expert).
- `expert_map_manager.py:72` `expert_map = torch.full((global_num_experts,), -1, ...)` — một bảng đầy đủ chiều rộng gồm `-1` (nghĩa là "không nằm trên rank này").
- `expert_map_manager.py:75-79` cho cách bố trí `"linear"`: rank này sở hữu một khối liền kề `[start_idx : start_idx+local_num_experts]`, và các phần tử đó được điền bằng `arange(0, local_num_experts)` — tức chỉ số **local**. Vậy `expert_map[global_id]` = chỉ số physical local nếu cư trú, `-1` nếu không. Đây chính xác là ngữ nghĩa được tài liệu hóa tại `expert_map_manager.py:297-307`.
- `expert_map_manager.py:80-87` biến thể `"round_robin"` thay vào đó gán các khe `global_id % ep_size == ep_rank` cho rank này.
- `expert_map_manager.py:95-111` dựng biến thể `expert_mask` cho ROCm-AITER (một mask 0/1 cộng với một khe sentinel `-1`), chỉ dùng khi bật AITER fusion.

**Ràng buộc liên quan đến EPLB** (`expert_map_manager.py:116-149`, `determine_expert_placement_strategy`): round-robin âm thầm bị hạ cấp thành linear khi `enable_eplb` là True (`expert_map_manager.py:126-127` yêu cầu `num_redundant_experts == 0 and not enable_eplb`). Vậy nên **với EPLB, cách bố trí luôn là linear.** Điều này quan trọng: EPLB di chuyển expert bằng cách chỉnh sửa `physical_to_logical_map` và xáo trộn trọng số về mặt vật lý, chứ không phải bằng cách thay đổi layout khối EP.

`ExpertMapManager.expert_map` (`expert_map_manager.py:296-307`) phơi bày tensor này; `map_global_to_local` (`expert_map_manager.py:336-352`) là phép tra cứu vô hướng trả về các id local an toàn với `-1`.

---

### 3. Cách lớp FusedMoE nối tất cả những thứ này lại lúc khởi tạo

`model_executor/layers/fused_moe/layer.py`, `__init__`:

- `layer.py:182-183`
  ```python
  self.global_num_experts = num_experts + num_redundant_experts
  self.logical_num_experts = num_experts
  ```
  Đây là sự phân tách: `global_num_experts` đếm các khe **physical** (bao gồm redundant); `logical_num_experts` là số expert thực của mô hình. Mọi kích thước ở hạ nguồn đều dẫn xuất từ hai giá trị này.

- `layer.py:200-212` cổng kiểm soát (gating) EPLB:
  ```python
  self.eplb_state: EplbLayerState | None = None
  if enable_eplb:
      if self.use_ep and self.global_num_experts % self.ep_size != 0:
          raise ValueError(... "even distribution of experts ...")
      self.eplb_state = EplbLayerState()
  else:
      assert not self.use_ep or num_redundant_experts == 0, (
          "Redundant experts are only supported with EPLB.")
  ```
  `layer.py:202` cưỡng chế `global_num_experts % ep_size == 0` — EPLB yêu cầu một phân bố physical đều *sạch sẽ* (không có rank dư). `layer.py:208` tạo container theo từng layer `EplbLayerState` (rỗng cho đến khi `set_eplb_state` điền vào). `layer.py:210-211` là guard nghịch đảo: redundant expert chỉ có nghĩa khi có EPLB.

- `layer.py:246-259` dựng `ExpertMapManager` (truyền `enable_eplb` để buộc bố trí linear) và ngay lập tức gọi `self.update_expert_map_info()`.

- `layer.py:293-312` dựng router và trao cho nó `eplb_state=self.eplb_state` (`layer.py:296`) và `num_logical_experts=self.logical_num_experts` (`layer.py:310`). Đây là liên kết cho phép routing thực hiện việc viết lại logical→physical (Mục 6).

- `layer.py:374-383` — cổng tương thích quant-method: `if enable_eplb and not self.quant_method.supports_eplb: raise ... "EPLB is not supported {quant_method}"`. Không phải mọi quant backend đều có thể rearrange được trọng số của nó.

#### `update_expert_map_info` — công bố map dưới dạng buffer

`layer.py:512-526`:
```python
def update_expert_map_info(self):
    self.local_num_experts = self.expert_map_manager.local_num_experts
    self.expert_placement_strategy = self.expert_map_manager.placement_strategy
    self.register_buffer("_expert_map", self.expert_map_manager.expert_map)
    self.register_buffer("expert_mask", self.expert_map_manager.expert_mask)
    routing_tables = self.expert_map_manager.routing_tables
    if routing_tables is not None:
        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)
```
Việc đăng ký `_expert_map`/`expert_mask` dưới dạng **buffer** (không phải parameter) có nghĩa chúng di chuyển theo `.to(device)` và hiển thị với forward graph đã biên dịch, nhưng không bị coi là trọng số có thể huấn luyện/nạp. `local_num_experts` (`layer.py:514`) chi phối kích thước của mọi tensor trọng số expert và đúng bằng chiều rộng mà `get_expert_weights` của EPLB sẽ reshape tới (Mục 5).

**Property** `expert_map` (`layer.py:1330-1334`) là thứ mà đường kernel đọc:
```python
@property
def expert_map(self) -> torch.Tensor | None:
    return self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
```
Trên đường không-AITER (đường build fused CUDA/ROCm này), nó trả về bảng global→local `_expert_map` thuần túy; dưới AITER fusion nó trả về `expert_mask` 0/1 thay thế — các kernel tiêu thụ các định dạng khác nhau.

`update_expert_map` (`layer.py:543-555`) là điểm vào cấu hình lại elastic-EP: nó gọi `ExpertMapManager.update(...)` rồi tái công bố các buffer. `ExpertMapManager.update` (`expert_map_manager.py:367-397`) tính toán lại trong một khối `with device:` để các tensor mới nằm đúng trên GPU chính xác.

---

### 4. `set_eplb_state` — gắn các bảng EPLB runtime vào lớp

`layer.py:1278-1303`:
```python
def set_eplb_state(self, moe_layer_idx, expert_load_view,
                   logical_to_physical_map, logical_replica_count) -> None:
    if self.eplb_state is not None:
        self.eplb_state.set_layer_state(
            moe_layer_idx, expert_load_view,
            logical_to_physical_map, logical_replica_count)
```
Hàm này được gọi **một lần cho mỗi MoE layer** từ `EplbState.add_model` (`eplb_state.py:442-446`), nơi truyền vào các tensor **global, toàn-layer**. `EplbLayerState.set_layer_state` (`eplb_state.py:945-954`) sau đó cắt (slice) hàng của layer này và — điều quan trọng — lưu trữ **view**, không phải bản sao:
```python
self.expert_load_view = expert_load_view[moe_layer_idx]
self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
self.logical_replica_count = logical_replica_count[moe_layer_idx]
```
Vì đây là các index-view vào các tensor do `EplbState` sở hữu, khi bộ rebalance ghi đè các map global tại chỗ (Mục 7), **view của mọi layer đều tự động cập nhật** — không cần đăng ký lại. Tương tự, `expert_load_view` là một view mà forward pass tăng lên nguyên tử (atomically), và `EplbState` đọc cùng vùng nhớ đó.

`should_record_tensor` dùng chung (`eplb_state.py:933-943`, `EplbLayerState`) là một tensor bool-vô hướng đơn lẻ được *tất cả* các layer tham chiếu; `EplbState` lật nó một lần bằng `.fill_()` để bật/tắt việc ghi load trên toàn cục. Nó là False trong `step_interval - window_size` bước đầu tiên vì những mẫu đó dù sao cũng sẽ bị ghi đè trước lần rearrange kế tiếp (được tài liệu hóa tại `eplb_state.py:939-942`).

---

### 5. `get_expert_weights` — các view trọng số an toàn với bộ nhớ mà EPLB rearrange

`layer.py:1201-1276`. Quá trình rearrange của EPLB phải hoán đổi *vùng nhớ trọng số thực sự* giữa các GPU. Method này trả về, cho một layer, danh sách các tensor trọng số expert, mỗi tensor được reshape thành `(local_num_experts, -1)` để kernel rebalance có thể lập chỉ mục chúng theo khe physical:

- `layer.py:1270-1276`
  ```python
  return [
      weight.view(self.local_num_experts, -1)
      for name, weight in weights
      if name not in NON_EXPERT_WEIGHTS
      and weight.shape != torch.Size([])
      and not name.startswith(NON_EXPERT_PREFIXES)
  ]
  ```
  `.view(self.local_num_experts, -1)` là một **reshape không sao chép (zero-copy)** — các tensor trả về là bí danh (alias) của vùng lưu trữ parameter của layer, nên việc ghi vào chúng sẽ biến đổi trọng số đang hoạt động. `local_num_experts` ở đây là số lượng được công bố trong `update_expert_map_info`.

- Các trường hợp loại trừ quan trọng đối với tính đúng đắn:
  - `NON_EXPERT_WEIGHTS` (`layer.py:1248-1252`) — `e_score_correction_bias`, `w13_input_scale`, `w2_input_scale`. Các input scale là các view broadcast `.expand()` (stride 0) dùng chung giữa các expert, không phải theo từng expert, nên việc rearrange chúng sẽ làm hỏng bộ nhớ (`layer.py:1244-1247`).
  - `NON_EXPERT_PREFIXES` (`layer.py:1256-1261`) — các submodule shared-expert / gate / transform nằm dưới `runner.`; chúng được nhân bản (replicated), không sharded theo từng expert.

- `_maybe_make_contiguous` (`layer.py:1202-1239`) xử lý các tensor quant scale có hai chiều cuối bị hoán vị. `layer.py:1237-1238` trả về một `nn.Parameter` *mới* bao bọc `torch.transpose(p.data, 1, 2)` — chú thích tại `layer.py:1233-1236` nhấn mạnh rằng nó trỏ đến **cùng một vùng nhớ nền**, nên bản sao EPLB vẫn đang di chuyển các byte thực, chỉ là thông qua một view liền kề (contiguous). `layer.py:1263-1268` khẳng định mọi trọng số không bị loại trừ đều liền kề trước khi rearrange chạy.

`EplbState.add_model` tiêu thụ điều này tại `eplb_state.py:448`: `expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]` — một buffer trung chuyển (staging) có hình dạng như danh sách trọng số expert của một layer.

---

### 6. Đường forward: định tuyến token sau khi rearrange

Đây là trái tim của "cách lớp sử dụng các map để định tuyến token." Routing tạo ra ID top-k **logic**; EPLB viết lại chúng thành ID **physical** và ghi lại load, trong một kernel Triton hợp nhất (fused) duy nhất.

`model_executor/layers/fused_moe/router/base_router.py`:

- `_apply_eplb_mapping` (`base_router.py:198-213`) là hook được gọi trong lúc routing khi `eplb_state is not None`. Nó chuyển tiếp ba view + `should_record_tensor` vào `eplb_map_to_physical_and_record`.

- `_eplb_map_and_record_i32_kernel` (`base_router.py:18-78`) thực hiện công việc theo từng (token, khe top-k):
  1. `base_router.py:36` nạp `expert_id` **logic** đã được định tuyến.
  2. `base_router.py:41-52` chọn replica: đọc `logical_replica_count[expert_id]`, kẹp về ≥1, và chọn một replica qua một hash nhân Knuth của chỉ số token modulo số replica (`replica_idx = hashed % replica_count`, `base_router.py:52`). Điều này trải các token cho một logical expert nóng một cách *xác định và đều đặn* trên các physical replica của nó mà không cần bộ đếm toàn cục.
  3. `base_router.py:67-72` `map_index = expert_id * map_slots + replica_idx`, rồi `physical_id = logical_to_physical_map[map_index]`. Đây là phép chuyển hướng logical→physical. `base_router.py:73` ghi id physical trở lại vào `topk_ids`.
  4. `base_router.py:75-78` nếu bật ghi nhận, `tl.atomic_add(out_ptr + physical_id, 1)` — tăng `expert_load_view` **theo từng physical expert**. Theo `eplb_state.py:163-165`, load được ghi cho *tất cả* physical expert (không chỉ cục bộ) để thống kê không phụ thuộc vào phương thức dispatch.

Sau kernel này, `topk_ids` chứa ID expert **physical**. Kernel MoE sau đó áp dụng `expert_map` (Mục 2) để chuyển đổi mỗi ID physical global thành chỉ số local (hoặc `-1` để loại bỏ nó nếu nó nằm trên rank khác) và dispatch. Vậy thứ tự là: **router → logical→physical (EPLB) → physical→local (expert_map) → all-to-all dispatch.**

`_validate_eplb_state` (`base_router.py:179-190`) bảo vệ rằng cả bốn tensor đều không None trước forward đầu tiên, biến một lỗi nối dây (mis-wire) âm thầm thành một lỗi rõ ràng.

(Ghi chú phiên bản: build này dùng một kernel Triton hợp nhất; `base_router.py:126-133` cho thấy một nhánh fallback không-Triton của `eplb_map_to_physical_and_record` cho các môi trường không có Triton. Ngữ nghĩa ánh xạ là giống hệt nhau.)

---

### 7. Bước rearrange và process group nào mang trọng số

`EplbState.rearrange` (`eplb_state.py:658-780`) tính toán lại layout và di chuyển trọng số về mặt vật lý:

- `eplb_state.py:674-675`
  ```python
  ep_group = get_ep_group().device_group
  ep_rank = ep_group.rank()
  ```
  *Thống kê* load (`_allreduce_list`, `eplb_state.py:718`) và quyết định rebalance chạy trên nhóm **EP**.
- `eplb_state.py:691-716` phân tán (scatter) load theo từng physical trở lại lên các logical expert (`scatter_add_` trên `physical_to_logical_map`) và cộng cửa sổ trượt → `global_expert_load_window`.
- `eplb_state.py:718` all-reduce load giữa các rank; `eplb_state.py:756-763` gọi chính sách (`rebalance_experts`) để tạo ra `new_physical_to_logical_map`.
- `eplb_state.py:766-772` `rearrange_expert_weights_inplace(...)` thực hiện việc xáo trộn trọng số GPU-sang-GPU thực sự, và nó được trao **hai** thứ: `ep_group` *và* `eplb_model_state.communicator`. `communicator` (được dựng trên nhóm **EPLB**, mục kế tiếp) là thứ mang các byte về mặt vật lý; `ep_group` cung cấp metadata về rank/topology.

Thiết kế hai-nhóm này được tài liệu hóa tại nơi tạo nhóm, `distributed/parallel_state.py:1878-1881`:
> "Create EPLB group with the same ranks as EP if EPLB is enabled. This is a separate process group to isolate EPLB communications from MoE forward pass collectives and prevent deadlocks."

---

### 8. `get_eplb_group` và EPLB process group

`distributed/parallel_state.py`:

- `parallel_state.py:1375-1384` biến global cấp module và accessor:
  ```python
  _EPLB: GroupCoordinator | None = None
  def get_eplb_group() -> GroupCoordinator:
      assert _EPLB is not None, ("EPLB group is not initialized. ... "
          "Ensure parallel_config.enable_eplb is True.")
      return _EPLB
  ```
  Vậy nên gọi hàm này khi chưa bật EPLB là một thất bại assertion cứng — nhóm được tạo một cách lười biếng (lazily) và chỉ khi cần.

- Việc tạo, `parallel_state.py:1850-1899`, bên trong `initialize_model_parallel`. Nhóm EP được dựng trước (`parallel_state.py:1850-1876`) từ `group_ranks` được tính tại `parallel_state.py:1854-1864` (một phép transpose/reshape của lưới rank toàn cục để EP trải trên `dp × pcp × tp`). Sau đó:
  ```python
  global _EPLB
  assert _EPLB is None, "EPLB group is already initialized"
  if config.parallel_config.enable_eplb:
      if enable_elastic_ep:
          _EPLB = _init_stateless_group(group_ranks, "eplb", ...)   # 1886
      else:
          _EPLB = init_model_parallel_group(
              group_ranks, get_world_group().local_rank, backend,
              group_name="eplb")                                    # 1894-1898
  ```
  Các thực tế then chốt:
  - `_EPLB` dùng **đúng cùng `group_ranks` như `_EP`** (`parallel_state.py:1887`, `1895`) — cùng thành viên, nhưng một communicator NCCL/Gloo *riêng biệt*, đó chính là toàn bộ mục đích (sự cô lập).
  - Đường không-elastic (`parallel_state.py:1894`) gọi `init_model_parallel_group` (`parallel_state.py:1264-1279`), hàm này dựng một `GroupCoordinator` đầy đủ với `use_device_communicator=True` (mặc định). Điều đó có nghĩa coordinator nhận được cả `device_group` lẫn `cpu_group` và, nếu `world_size > 1`, một `device_communicator` thực (xem `GroupCoordinator.__init__`, `parallel_state.py:453-464`).
  - Đường elastic (`parallel_state.py:1886`) thay vào đó dựng một `StatelessGroupCoordinator`.

- Đấu nối vòng đời (lifecycle): `prepare_communication_buffer_for_model` (`parallel_state.py:1980-1981`) gọi `_EPLB.prepare_communication_buffer_for_model(model)`; `_replace_active_groups` (`parallel_state.py:1320-1328`) và phần teardown tại `parallel_state.py:2041-2044` hủy `_EPLB` cùng với các nhóm khác.

---

### 9. Giải quyết `communicator=None` trên build này

Yêu cầu này hỏi cụ thể về `communicator=None`. Có hai lớp cho vấn đề này; trên **build này** chúng được giải quyết như sau.

**(a) Giải quyết tại thời điểm config — câu trả lời thực cho build này.** `config/parallel.py:89` khai báo:
```python
communicator: EPLBCommunicatorBackend | None = None
```
tức backend communicator EPLB không được đặt theo mặc định. Nó được giải quyết trong post-init của parallel-config tại `config/parallel.py:907-924`:
```python
if self.enable_eplb and self.eplb_config.communicator is None:
    if self.enable_elastic_ep:
        self.eplb_config.communicator = "pynccl"                  # 912
    else:
        # Avoid torch_nccl: NCCL is fundamentally incompatible
        # with async EPLB due to multi-stream conflicts, and
        # batched isend/irecv hangs under high load.
        from vllm.distributed.nixl_utils import is_nixl_available
        if is_nixl_available():
            self.eplb_config.communicator = "nixl"                # 922
        else:
            self.eplb_config.communicator = "torch_gloo"          # 924
```
Vậy trên một build tiêu chuẩn (không-elastic) **không có NIXL**, `communicator=None` được giải quyết thành **`torch_gloo`**, không phải NCCL. Chú thích (`parallel.py:914-917`) nêu rõ rằng `torch_nccl` cố tình bị tránh vì các xung đột đa-luồng (multi-stream) của NCCL xung đột với async EPLB và `isend/irecv` theo lô treo dưới tải (tham chiếu pytorch/pytorch#174288). Nếu có NIXL thì nó ưu tiên `nixl`; elastic EP buộc dùng `pynccl`.

**(b) Fallback tại thời điểm factory.** `create_eplb_communicator` (`distributed/eplb/eplb_communicator.py:618-734`) là nơi chuỗi backend trở thành một communicator cụ thể. Nó được gọi từ `EplbState.add_model` (`eplb_state.py:450-455`):
```python
communicator = create_eplb_communicator(
    group_coordinator=get_eplb_group(),          # 451  <-- the EPLB group
    backend=self.parallel_config.eplb_config.communicator,
    expert_weights=model.expert_weights,
    expert_buffer=expert_buffer,
)
```
Bên trong factory:
- `eplb_communicator.py:644-645` `if backend is None: backend = "torch_nccl"`. Đây là một fallback *thứ cấp* — nhưng trên build này nó thực chất không bao giờ được chạm đến, vì post-init của config (phần a) đã biến `None` thành `torch_gloo`/`nixl`/`pynccl` trước điểm này rồi.
- `eplb_communicator.py:647-653` chọn torch process group từ **coordinator `get_eplb_group()`**: `cpu_group` nếu các trọng số expert là tensor CPU, còn không thì `device_group`. Đây là mối liên hệ cụ thể giữa "EPLB process group" và "đối tượng di chuyển trọng số."
- Dispatch backend: `"torch_gloo"` → `TorchDistGlooStagedEplbCommunicator(cpu_group=...)` (`eplb_communicator.py:726-729`, gloo P2P với staging trên CPU); `"torch_nccl"` → `TorchDistNcclEplbCommunicator(ep_group=torch_group)` (`eplb_communicator.py:730-731`); `"pynccl"`/`"nixl"` được xử lý tại `eplb_communicator.py:732-733` / `706-725`.
- Các coordinator stateless (elastic) bị giới hạn ở `torch_nccl`/`pynccl` và âm thầm nâng cấp `torch_nccl`→`pynccl` (`eplb_communicator.py:691-704`).

**Kết luận cho build này:** với EPLB được bật, không-elastic, và NIXL không khả dụng, một communicator không được đặt (`None`) trở thành **`torch_gloo`**, và communicator đó được gắn với **CPU group của `get_eplb_group()`** — một nhóm có cùng rank như EP nhưng với backend cô lập. Trường `EplbModelState.communicator` (`eplb_state.py:195-198`, được đặt tại `eplb_state.py:469`) là thứ mà `rearrange` truyền vào `rearrange_expert_weights_inplace` (Mục 7).

---

### 10. `v1/worker/gpu/eplb_utils.py` — `maybe_register_model` (khung stack của crash traceback)

`v1/worker/gpu/eplb_utils.py` định nghĩa `EPLBController`, đối tượng phía worker sở hữu `EplbState` và điều khiển nó. Trong crash traceback, `maybe_register_model` là khung nơi mô hình lần đầu được gắn vào EPLB.

Chuỗi lời gọi (từ `v1/worker/gpu/model_runner.py`):
- `model_runner.py:273` `self.eplb = EPLBController(self.parallel_config, self.device)`.
- `model_runner.py:294` `self.eplb.prepare_load()` — reset trạng thái; nếu EPLB được bật, dựng `EplbState` (`eplb_utils.py:58-62`).
- `model_runner.py:336-340` sau khi nạp mô hình và chuẩn bị communication-buffer:
  ```python
  eplb_models_added |= self.eplb.maybe_register_model(
      self.model, self.model_config, load_dummy_weights)
  ```
- `model_runner.py:341` `self.eplb.maybe_start_async_loop(eplb_models_added)`.

Bản thân `maybe_register_model` (`eplb_utils.py:96-113`):
```python
def maybe_register_model(self, model, model_config, load_dummy_weights) -> bool:
    if not self.parallel_config.enable_eplb or load_dummy_weights:
        return False                                              # 102
    model = _unwrap_moe(model)                                    # 105
    if not is_mixture_of_experts(model):
        return False                                              # 106-107
    logger.info_once("EPLB is enabled for model %s.", model_config.model)
    assert self.state is not None                                # 110
    self.state.add_model(model, model_config)                    # 111
    self._has_registered_models = True                           # 112
    return True
```
Từng khối một:
- `eplb_utils.py:102` — thoát sớm trừ khi EPLB được bật và đây là một lần nạp trọng số thực (không-dummy). Các lần nạp dummy bỏ qua EPLB vì các trọng số mà nó sẽ rearrange là rác.
- `eplb_utils.py:105` `_unwrap_moe(model)` (`eplb_utils.py:21-28`) — các wrapper VLM (ví dụ `KimiK25ForConditionalGeneration`) không tự cài đặt `MixtureOfExperts` nhưng giữ MoE LM dưới `.language_model`; hàm này đi xuống qua `get_language_model()`. Nếu crash xảy ra trên một mô hình đa phương thức (multimodal), thì phép unwrap này là khung liên quan.
- `eplb_utils.py:106-107` — nếu vẫn không phải mô hình MoE, thoát (mô hình dense không có expert để cân bằng).
- `eplb_utils.py:110-111` — `assert self.state is not None` rồi `self.state.add_model(model, model_config)`. **Đây là khung kích hoạt toàn bộ chuỗi đăng ký (registration cascade)** được tài liệu hóa ở trên: `add_model` dựng các map ban đầu, gọi `model.set_eplb_state(...)` trên mọi layer (Mục 4), chụp nhanh (snapshot) `expert_weights`, và gọi `create_eplb_communicator(group_coordinator=get_eplb_group(), ...)` (Mục 9b). Vì vậy một crash *bên trong* lời gọi này rất có thể hoặc là (i) assertion của `get_eplb_group()` (`parallel_state.py:1379`) nếu nhóm EPLB chưa từng được tạo — điều này xảy ra khi `enable_eplb` không nhất quán giữa lúc khởi tạo nhóm ở thời điểm config và tại đây — hoặc (ii) một thất bại khi dựng communicator trong `create_eplb_communicator` (ví dụ lỗi `pynccl`/`nixl`/dtype tại `eplb_communicator.py:657-724`). `assert self.state is not None` tại `eplb_utils.py:110` bảo vệ chống lại việc `prepare_load` chưa chạy.

`maybe_register_speculator` họ hàng (`eplb_utils.py:64-94`) làm điều tương tự cho một mô hình MoE draft/speculator nhưng khẳng định `not enable_elastic_ep` (`eplb_utils.py:83-85`). `setup_from_mapping` (`eplb_utils.py:138-156`) là đường elastic-EP tái dựng `EplbState` từ một map `expanded_physical_to_logical` được cung cấp từ bên ngoài qua `EplbState.from_mapping`.

Việc stepping tại thời điểm nạp được điều khiển bởi decorator `step_eplb_after` (`eplb_utils.py:31-47`), hàm này gọi `self.eplb.step(...)` sau một method của runner; `EPLBController.step` (`eplb_utils.py:119-136`) không làm gì (no-op) trừ khi `enable_eplb and not suppressed and state is not None and _has_registered_models` — tức là nó sẽ không chạy cho đến khi `maybe_register_model` đã đặt `_has_registered_models = True` (`eplb_utils.py:112`).

---

### Các file được tham chiếu
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/layer.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/expert_map_manager.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/router/base_router.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/parallel_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_communicator.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/config/parallel.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/v1/worker/gpu/eplb_utils.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/v1/worker/gpu/model_runner.py`

---
