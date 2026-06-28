# GLM-5.2-FP8 — MoE data flow (DP8/EP8, vLLM-moreh, ROCm/AITER)

Tài liệu luồng dữ liệu 1 layer MoE khi serve GLM-5.2-FP8 (DP8, EP8, TP1) trên vLLM-moreh.
Mọi đường dẫn trỏ tới code đã mount ở `bench_mv4571/3rdparty/`.

Config liên quan (xác nhận từ `serve.log` + `config.json`): `n_routed_experts=256`,
`num_experts_per_tok=8`, `first_k_dense_replace=3` (layer 0–2 dense, 3–77 MoE → 75 MoE layer),
EP8 → 32 expert/rank, `all2all_backend = AgRs` (AllGather + ReduceScatter).

---

## 1. Tổng quan 1 cụm MoE (như thấy trên profiler)

```
   ┌─ dispatch ─┐   ┌──────────── MoE compute (AITER) ───────────┐   ┌─ combine ──┐
   ncclDevKernel  → MoeSorting → quant → gemm s1 → quant → gemm s2 → ncclDevKernel
   (all_gatherv)                                                     (reduce_scatterv)
```

- **dispatch** = `all_gatherv`: gom token + topk_ids của TẤT CẢ DP rank. Sau bước này, mỗi
  rank giữ `sum(x_i)` token và concat topk_ids (rank i đóng góp `x_i` token).
- **MoE compute**: mỗi rank chạy gemm CHỈ cho 32 expert local của nó (token nào chọn expert
  ngoài rank bị `expert_map` mask) → khối lượng gemm trên rank r ∝ số (token,slot) trỏ vào
  block expert của r (= `load[r]` trong phân tích histogram).
- **combine** = `reduce_scatterv`: cộng dồn kết quả rồi chia lại, mỗi rank về `x_i` token như đầu.
- dispatch/combine là **collective đồng bộ cả 8 rank** ⇒ cụm MoE thứ `k` trên mọi rank ứng với
  cùng `(layer, step)`; combine là **barrier** ⇒ step bị chặn bởi rank có MoE-compute lâu nhất.

---

## 2. Call chain (top-down)

| # | Bước | Code |
|---|------|------|
| 1 | `DeepseekV2MoE.forward` → `self.experts(...)` | [models/deepseek_v2.py:373](../3rdparty/vllm/model_executor/models/deepseek_v2.py#L373) |
| 2 | `FusedMoE.forward` → runner → `moe_runner._forward_impl` | [fused_moe/layer.py:1324](../3rdparty/vllm/model_executor/layers/fused_moe/layer.py#L1324) |
| 3 | `Fp8MoEMethod.apply` ← **[EP_COLLECT] log topk_ids ở đây (trước all2all)** | [quantization/fp8.py:896](../3rdparty/vllm/model_executor/layers/quantization/fp8.py#L896) |
| 4 | `FusedMoEModularKernel.apply` → `_prepare` / `_fused_experts` / `_finalize` | [fused_moe/modular_kernel.py:1342](../3rdparty/vllm/model_executor/layers/fused_moe/modular_kernel.py#L1342) |
| 5a | ❶ dispatch | xem §3 |
| 5b | ❷ MoE compute | xem §4 |
| 5c | ❸ combine | xem §5 |

`FusedMoEModularKernel.apply` (modular_kernel.py) gọi tuần tự:
- `self._prepare(...)` → `prepare_finalize.prepare(...)`  → **❶ dispatch**
- `self._fused_experts(...)` → `experts.apply(...)`         → **❷ MoE compute**
- `self._finalize(...)` → `prepare_finalize.finalize(...)`  → **❸ combine**

---

## 3. ❶ `get_ep_group().dispatch(...)` gọi về đâu

```
naive_dp_ep.py  MoEPrepareAndFinalizeNaiveDPEPModular.prepare()
  ├─ _quantize_and_setup_dispatch()                 # quant FP8 hidden_states TRƯỚC khi gửi
  └─ get_ep_group().dispatch(a1q, topk_weights, topk_ids, ...)
        → parallel_state.py  GroupCoordinator.dispatch()
            → self.device_communicator.dispatch()                    # CudaCommunicator (ROCm dùng path này)
                → self.all2all_manager.dispatch()                    # backend=AgRs
                    → all2all.py  AgRsAll2AllManager.dispatch()
                        → dist_group.all_gatherv([hidden, topk_w, topk_ids], dim=0, sizes=chunk_sizes_across_dp)
```

- [naive_dp_ep.py:112](../3rdparty/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L112) `prepare()` → `get_ep_group().dispatch` (L158)
- [parallel_state.py:1206](../3rdparty/vllm/distributed/parallel_state.py#L1206) `GroupCoordinator.dispatch` → `device_communicator.dispatch`
- [cuda_communicator.py](../3rdparty/vllm/distributed/device_communicators/cuda_communicator.py#L122) chọn `all2all_manager` theo `all2all_backend` (`"naive"/"allgather_reducescatter"` → `AgRsAll2AllManager`)
- [all2all.py:84](../3rdparty/vllm/distributed/device_communicators/all2all.py#L84) `AgRsAll2AllManager.dispatch` → **`dist_group.all_gatherv(...)`** (L109)

→ Bản chất dispatch ở backend AgRs = **AllGather biến-kích-thước** trên DP group: `sizes` =
`dp_metadata.get_chunk_sizes_across_dp_rank()` (số token mỗi rank). Kernel GPU = `ncclDevKernel`
(RCCL all-gather). Đây chính là "rank i có x_i token → sau communication mọi rank có sum(x_i)
token + concat topk" như mô tả.

---

## 4. ❷ `MoeSorting → quant → gemm s1 → quant → gemm s2` gọi về đâu

```
modular_kernel.py  _fused_experts(...)
  → rocm_aiter_moe.py  AiterExperts.apply()
      → _aiter_ops.py  rocm_aiter_ops.fused_moe(...)
          → from aiter.fused_moe import fused_moe        # THƯ VIỆN aiter (ngoài vllm)
          → aiter.fused_moe.fused_moe(hidden, w1, w2, topk_weight, topk_ids, expert_mask, ...)
                → [CK kernels] MoeSorting → dynamic_per_group_scaled_quant
                               → kernel_moe_gemm (stage1) → quant → kernel_moe_gemm (stage2)
```

- [rocm_aiter_moe.py:474](../3rdparty/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py#L474) `AiterExperts.apply` → `rocm_aiter_ops.fused_moe(...)` (L515, kèm `expert_mask`, `moe_sorting_dispatch_policy`)
- [_aiter_ops.py:160](../3rdparty/vllm/_aiter_ops.py#L160) `from aiter.fused_moe import fused_moe` → gọi `fused_moe(...)` (L165)
- Các kernel CK thực thi nằm trong **thư viện `aiter`** (ROCm, đã compile — không ở trong repo này),
  tham chiếu upstream: `aiter/fused_moe.py` (xem comment L345-346 trong rocm_aiter_moe.py).

Phân biệt 2 stage gemm (cùng tên `kernel_moe_gemm`, khác template param):
- **stage1**: `(ck::InMemoryDataOperationEnum)0` + `ck::Sequence<1, 16, 1, 16>` (ghi đè kết quả).
- **stage2**: `(ck::InMemoryDataOperationEnum)1` + `ck::Sequence<1, 4, 1, 64>` (atomic-add/reduce).

Khối lượng (và thời gian) gemm trên rank r tỉ lệ với số token-routing trỏ vào 32 expert local của
r. Lưu ý: thời gian **không tuyến tính** với token do `MoeSorting` pad token mỗi expert lên bội số
`BLOCK_SIZE_M` (xem [moe_align_block_size.py](../3rdparty/vllm/model_executor/layers/fused_moe/moe_align_block_size.py)).

---

## 5. ❸ `get_ep_group().combine(...)` gọi về đâu

```
naive_dp_ep.py  MoEPrepareAndFinalizeNaiveDPEPModular.finalize()
  → get_ep_group().combine(out, ...)
      → parallel_state.py  GroupCoordinator.combine()
          → self.device_communicator.combine()                       # CudaCommunicator
              → self.all2all_manager.combine()                       # AgRs
                  → all2all.py  AgRsAll2AllManager.combine()
                      → dist_group.reduce_scatterv(hidden_states, dim=0, sizes=chunk_sizes_across_dp)
```

- [naive_dp_ep.py:187](../3rdparty/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L187) `finalize()` → `get_ep_group().combine` (L208)
- [parallel_state.py:1228](../3rdparty/vllm/distributed/parallel_state.py#L1228) `GroupCoordinator.combine` → `device_communicator.combine`
- [all2all.py:124](../3rdparty/vllm/distributed/device_communicators/all2all.py#L124) `AgRsAll2AllManager.combine` → **`dist_group.reduce_scatterv(...)`** (L136)

→ combine ở AgRs = **ReduceScatter biến-kích-thước**: cộng phần đóng góp từ mọi rank rồi chia lại
mỗi rank đúng `x_i` token ban đầu. Kernel GPU = `ncclDevKernel` (RCCL reduce-scatter).

---

## 6. Ý nghĩa cho phân tích EP imbalance

- **Token imbalance** (notebook `process_ep_logs_glm5.ipynb`): từ `[EP_COLLECT]` (topk_ids trước
  dispatch), cộng histogram 8 rank → `load[r]` = #token-routing rank r phải tính. `max/min` per
  `(layer, step)`.
- **Time imbalance** (`analyze_trace_pairs.py` / `process_trace_time_glm5.ipynb`): từ profiler, đo trực tiếp thời gian cụm MoE-compute
  mỗi rank; vì combine là barrier, time/step = `max_r(MoE_compute_time)`. So `Σ max_r` (critical path)
  với `Σ mean_r` (balanced) → **headroom thực tế nếu cân bằng**.
- token-count chỉ là proxy; time mới phản ánh tác động thật (do tiling/padding của MoeSorting).
