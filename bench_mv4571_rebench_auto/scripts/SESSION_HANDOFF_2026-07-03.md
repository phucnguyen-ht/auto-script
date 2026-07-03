# SESSION HANDOFF — mv-4571 EPLB bench (GLM-5.2, MI300 / valhalla) — 2026-07-03

> **ĐỌC FILE NÀY TRƯỚC.** Đây là điểm vào cho session mới. Chi tiết kỹ thuật + line-by-line ở
> [`EPLB_DEEPDIVE.md`](EPLB_DEEPDIVE.md) (doc chính, có §A nền tảng cho người mới → §3.2 + Q1–Q8 + §R resume).
> Bối cảnh cũ (MI325 + MI300 tw031): [`CONTEXT_HANDOFF.md`](CONTEXT_HANDOFF.md), [`progress.md`](progress.md),
> [`DEBUG_ASYNC_HANG.md`](DEBUG_ASYNC_HANG.md).

## 0. Mục tiêu ticket
Bench các cấu hình **EPLB** (Expert-Parallel Load Balancing) cho GLM-5.2-FP8 (dp8/ep8, 1P1D) trên node **mi300-7**,
so 4 communicator (nixl/gloo/pynccl/torch_nccl) + baseline no-EPLB. Workload: **100K ISL × concurrency 64**.
Việc phân tích cơ chế + THE BIG PROBLEM (async communicator deadlock) đã xong; đang ở khâu **lấy số throughput công bằng**.

## 1. HẠ TẦNG / CÁCH TRUY CẬP (quan trọng)
- **Host chạy lệnh**: `mi250-018` (jump host, KHÔNG GPU). Working dir: `/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571`.
- **kubectl**: đã cài user-local `~/.local/bin/kubectl` (không sudo). **Kubeconfig**: `<mv-4571>/config`
  (admin, cluster valhalla, 1 context `kubernetes-admin@cluster.local`, server `192.168.2.22:6443`).
  Mọi lệnh: `export KUBECONFIG=<mv-4571>/config; export PATH="$HOME/.local/bin:$PATH"`.
- **Node target**: `mi300-7` (valhalla) = MI300X gfx942 **192 GiB/GPU**, **CÓ RDMA** (`mlx5_0/1`) + `rocm_ipc` + full **xGMI** mesh.
  Fallback nếu mi300-7 cordon: **mi300-10** (MI300X Ready). (mi300-9, mi308x-2 hay bị cordon.)
- **Pod**: `phuc-nguyen-mv4571-2`, image `moreh-vllm:0.23.0-260626-rc1`, mount `/remote/vast0`, workingDir = mv-4571.
  Tạo/điều khiển bằng 2 script trong `mv-4571/mv-valhalla-cluster/dev/`:
  - `setup_dev_pod.sh` — pod spec (NAME=`phuc-nguyen-mv4571-2`, NODE=`mi300-7`); **tự set KUBECONFIG**;
    `APPLY_ONLY=1` = tạo+chờ (không mở shell); có **editable overlay** 5 file eplb (sửa host → container thấy).
  - `k.sh` — helper (POD mặc định `-2`): `k.sh up` (tạo pod), `k.sh shell`, `k.sh x -- <cmd>`, `k.sh pull`, `k.sh down`.
- **⚠️ Gotchas hạ tầng**:
  - Khi có `vllm-moreh serve` chạy, pod bão hoà → `kubectl exec -it` (k.sh shell) **hang**. Đọc log từ **host** (shared FS) thay vì exec.
  - **ĐỪNG force-delete pod** khi node có thể bị cordon → không tạo lại được. Muốn dừng run: **kill serve BÊN TRONG pod**
    (`kubectl exec -- bash -c "pkill -9 -f 'vllm-moreh serv[e]'; pkill -9 -f 'VLL[M]::'"` — dùng `[ ]` để pkill không tự giết exec shell).
  - **ĐỪNG sửa run script/preset khi run đang chạy** (làm hỏng run đang chạy).
  - Driver chạy bằng `setsid` **sống độc lập** qua các lần session gián đoạn; chỉ monitor phía agent bị dừng → chỉ cần cắm lại.

## 2. SOURCE EDITABLE (đọc/sửa vLLM source)
- Copy từ container ra host (host-owned, sửa được) bằng `kubectl cp`, đặt **NGOÀI** mv-4571 tại
  `/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571-vllm-src/` + **symlink** `auto-script/bench_mv4571_rebench_auto/vllm_src`.
- 5 file eplb được **overlay-mount live** vào container (sửa host → container dùng ở lần `serve` kế): `eplb_communicator.py`,
  `async_worker.py`, `eplb_state.py`, `rebalance_execute.py`, `eplb_utils.py` (danh sách trong `setup_dev_pod.sh` mảng `EDITABLE_FILES`).
- vLLM version: `0.23.1.dev0+g0fc695fc6`.

## 3. KẾT LUẬN KỸ THUẬT (đã chốt — chi tiết ở EPLB_DEEPDIVE.md)
- **THE BIG PROBLEM**: EPLB **async** với `torch_nccl`/`pynccl` → **deadlock**. `_EPLB` là NCCL group RIÊNG (parallel_state.py:1878)
  nhưng vẫn deadlock vì **2 luồng (main forward + async worker) cùng bơm NCCL kernel lên 1 GPU** → thứ tự rendezvous giữa 8 GPU
  lệch → vòng chờ chéo → treo → RPC timeout 300s (`multiproc_executor.py:389`, `envs.py:210`) → EngineDead. **Intra-node, không cần 2 máy.**
- **4 communicator**: `torch_nccl`/`pynccl` (NCCL trên GPU → **chỉ SYNC**); `torch_gloo` (CPU-staging qua cpu_group → **ASYNC OK**);
  `nixl` (RDMA READ; intra-node = **rocm_ipc** qua xGMI, KHÔNG dùng NIC mlx5 → **ASYNC OK**). NIC mlx5 chỉ cần khi **multi-node**.
- **nixl init fail (đã FIX)**: `rc_x` trong UCX_TLS → NIXL `ibv_reg_mr` đăng ký VRAM với mlx5 (GPUDirect-RDMA) → `Bad address`/`NIXL_ERR_BACKEND`
  (node không có amdgpu↔mlx5 peermem). **FIX = bỏ `rc_x`** → `UCX_TLS=tcp,self,sm,rocm_copy,rocm_ipc` (đăng ký qua rocm_ipc). Verified
  `nixl_probe.py`→REGISTER_OK; full serve init+transfer+**không crash**. (`gen_eplb_presets.sh` `UCX_TLS_VAL` mặc định đã bỏ rc_x.)
- **nixl 0-req KHÔNG phải crash**: formal test 240s chạy nhưng 0 request lọt cửa sổ đo `[20s,220s]` (`unit_test.py:435`); EngineDeadError = artifact §6 shutdown.
- **Bench fairness / adaptive X (mới, đã implement)**: `data[:10]` tune cho KV baseline 1.13M. EPLB co KV (gloo 0.94M, nixl 0.77M@40G)
  → 10 prompt (1.0M) **tràn** KV → prefix hit tụt (99.56%→94.88%→68.85%) → TTFT dominate → **đây là lý do chính nixl 0-req + gloo TTFT cao**.
  Công thức: **X = ⌊KV_tokens/ISL⌋ − 1** (baseline 10, gloo 8, nixl 6). "10 không phải 11" vì 11×100k=1.1M ăn trọn KV, hết headroom cho decode.

## 4. THAY ĐỔI CODE ĐÃ LÀM (nằm trên disk, verified)
- **Adaptive X** (prefix ~100% cho mọi KV): `unit_test.py` (`--kv_cache_tokens` → `X=⌊KV/ISL⌋−1`, fallback slice cũ) ←
  `multi_process_test.py` (forward env `REBENCH_KV_TOKENS` cho cả warmup+bench) ← `run_and_bench.sh` (grep `GPU KV cache size`, min qua 8 rank).
  Verified: baseline→X=10, gloo→8, nixl(40G)→6.
- **nixl r0 KV: 40 → 48 GiB** (`presets/glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml`) cho fair với gloo auto ~49 GiB.
- `gen_eplb_presets.sh`: `UCX_TLS_VAL` (bỏ rc_x), opt-in `LOG_BALANCEDNESS`, `NIXL_KV_CACHE_BYTES` (r16-safe 40G — chỉ dùng khi sweep đủ r0/r8/r16).
- `scripts/run_debug5.sh`: driver 5-config (baseline/nixl/gloo/pynccl/nccl), **resumable** (skip nếu đã có `scenario_summary.csv`),
  kill-trước-wait, chờ GPU rảnh (shared node). `scripts/nixl_probe.py`: test đăng ký NIXL (~5s).
- Docs: `EPLB_DEEPDIVE.md` (§A→§3.2, Q1–Q8, §R resume), memory RULE "lưu câu hỏi vào docs".

## 5. KẾT QUẢ HIỆN CÓ
| config | requests | p50_tpot | prefix hit | ghi chú |
|---|---|---|---|---|
| baseline (no-eplb) | **416** | 23.86 | **99.56%** | FAIR (X=10). Log: `logs_debugs/1-baseline-noeplb/` — GIỮ NGUYÊN |
| gloo async r0 (cũ) | 82 | 72.9 | 94.88% | KHÔNG fair (data[:10] tràn KV 0.94M) → đã xoá, chờ re-run |
| nixl async r0 (cũ) | 0 | — | 68.85% | KHÔNG fair (KV 0.77M) — không crash, chỉ 0 req lọt cửa sổ → đã xoá |
| pynccl/nccl sync | — | — | — | chưa từng chạy xong |

**Verdict CHỨC NĂNG (đã chắc, không cần chạy lại để biết):** nccl/pynccl **async = deadlock** (dùng SYNC); gloo async OK;
nixl async trên mi300-7 **init/transfer OK, không crash** (nhờ bỏ rc_x).

## 6. VIỆC CÒN LẠI — re-run 4 EPLB config CÔNG BẰNG (adaptive X, prefix ~100%)
Chờ node rảnh (mi300-7 hoặc mi300-10). Lệnh:
```bash
export KUBECONFIG=/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571/config
export PATH="$HOME/.local/bin:$PATH"
cd /remote/vast0/phuc-nguyen/workspace/tickets/mv-4571/mv-valhalla-cluster/dev
bash k.sh up                              # tạo pod -2 trên mi300-7 (đổi NODE=mi300-10 nếu mi300-7 cordon)
# kiểm tra GPU sạch: bash k.sh x -- bash -lc 'rocm-smi --showmemuse | grep VRAM%'
SC=/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571/auto-script/bench_mv4571_rebench_auto/scripts
LOGD=/remote/vast0/phuc-nguyen/workspace/tickets/mv-4571/auto-script/bench_mv4571_rebench_auto/logs_debugs
kubectl exec phuc-nguyen-mv4571-2 -- bash -c "cd '$SC'; setsid bash run_debug5.sh > '$LOGD/debug5.log' 2>&1 </dev/null & echo ok"
# run_debug5.sh tự SKIP baseline (đã có summary), chạy nixl→gloo→pynccl→nccl với adaptive X.
# Theo dõi (host, không exec): tail -f "$LOGD/debug5.log" ; xong khi in "[debug5] ALL DONE".
# Kết quả mỗi config: "$LOGD/<n>-<name>/scenario_summary.csv" + prefix_cache_hit.txt (kỳ vọng hit ~100%).
```
Kỳ vọng: prefix ~100% cả 4 → **TPOT sạch, chỉ khác đúng biến EPLB communicator**. Nếu 1 config vẫn 0/thấp req dù prefix ~100%
→ tăng `time_limit` trong `multi_process_test.py` (hiện 240s) hoặc kiểm tra contention (`rocm-smi` phải <10% trước khi serve).

## 7. Câu hỏi user đã hỏi (RULE: lưu vào docs) → đều ở `EPLB_DEEPDIVE.md` §Q (Q1–Q8)
Q1 single-node transports (§A.5) · Q2/Q3 EPLB flow + sync/async swimlane (§2.0–§2.8) · Q4 log_balancedness (§2.0c) ·
Q5 data-path 3 transport (§2.8) · Q6 nixl init fix (§3.1) · Q7 nixl 0-req = window (§3.1) · Q8 adaptive X / prefix cache (§3.2).
