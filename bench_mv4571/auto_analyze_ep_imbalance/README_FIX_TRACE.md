# MV-4571 — Fix "auto_profile chạy xong KHÔNG sinh trace / server treo" + chạy tiếp sweep EP imbalance

> Tài liệu mô tả **toàn bộ các bước** đã làm trong phiên 2026-06-28: chẩn đoán nguyên nhân thật,
> sửa code (KHÔNG đụng preset), khắc phục các lỗi môi trường của container, và smoke-test 2 phase
> (token + time) của sweep tới khi không còn lỗi.

---

## 0. TL;DR (tóm tắt nhanh)

- **Lỗi user gặp:** chạy profile (auto_profile.sh / run_all_full_presets_vllm.sh) với DP8EP8 → assertion +
  server "chết", **không thấy file trace**.
- **Nguyên nhân THẬT (đã xác nhận trên GPU):** 8 file trace **VẪN được ghi đầy đủ**, nhưng ngay sau
  `/stop_profile` các DP engine **deadlock** → `vllm bench` **treo vĩnh viễn** → orchestration cũ block
  vào bench nên không bao giờ thu (harvest) được trace → nhìn như "mất trace / server chết".
- **Cách sửa (đúng ý user: KHÔNG sửa preset, kệ assertion):** chạy bench Ở NỀN, **poll thư mục trace**
  tới khi đủ file & đã flush xong → rồi mới kill bench treo + server.
- **Kết quả:** cả 2 phase chạy OK, số liệu KHỚP reference cũ (time headroom **20.71%**, token
  **525 prefill / 153.675 decode** steps).

---

## 1. Bối cảnh & mục tiêu

Ticket MV-4571 phân tích EP8 imbalance cho GLM-5.2-FP8 (DP8/EP8, TP1, MI300/ROCm/AITER) theo 2 hướng:
- **TOKEN**: histogram #token-routing/expert trước all2all (từ log `[EP_COLLECT]` trong serve.log).
- **TIME**: thời gian MoE-compute/rank từ torch profiler traces (8 rank).

Yêu cầu của user phiên này:
1. Tự exec vào docker `phuc-nguyen-mv-4571`, chạy `run_all_full_presets_vllm.sh` (mode profile) để
   **tái hiện** hiện tượng mất trace, rồi **tìm cách sửa** tới khi: vẫn để assertion xảy ra, NHƯNG
   **bắt buộc phải có file trace** — và **KHÔNG sửa thêm gì vào preset** (không ép `api_server_count=1`,
   không bật `ignore_frontend`).
2. Làm tiếp task "sweep analyze EP imbalance" (orchestrator `auto_analyze_ep_imbalance.sh`), smoke-test
   vài case tới khi không còn lỗi.

---

## 2. Các bước đã thực hiện

### Bước 1 — Đọc & nắm dự án
Đọc `handoff.md`, các script: `run_all_full_presets_vllm.sh` → `run_all.sh` → `auto_profile.sh`
(symlink tới `bench_mv4526/auto_profile.sh` = `auto_bench.sh` với `MODE=profile`), `common/auto_bench_template.sh`,
`common/helper.sh`, `serve.sh`, và orchestrator `auto_analyze_ep_imbalance.sh`. Đọc luồng profile trong
vllm đã cài (`entrypoints/serve/profile/api_router.py`, `v1/engine/async_llm.py`, `v1/engine/core_client.py`,
`v1/worker/gpu_worker.py`, `entrypoints/cli/serve.py`).

Phát hiện quan trọng khi đọc code: `profile_async("profile")` của DP client **broadcast tới TẤT CẢ engine**
(`core_client.py:1428`), nên chỉ cần 1 `/start` + 1 `/stop` là đủ để ghi cả 8 trace ⇒ giả thuyết cũ
"start/stop lệch engine → thiếu cặp → mất trace" là **SAI**. Phải chạy thực tế để biết nguyên nhân thật.

### Bước 2 — Chạy tái hiện trên GPU (và gỡ 1 loạt lỗi môi trường)
Chạy `auto_profile.sh` trong container. Gặp lần lượt:

1. **Server chết lúc startup** — `RuntimeError: No such file or directory:
   '.../vllm-moreh/src/aiter_moreh/configs/a8w8_bpreshuffle_tuned_gemm_bruteforce.csv'`.
   → `AITER_MOREH_ROOT_DIR` (set bởi docker.sh) trỏ vào **dev checkout không tồn tại** trong container
   build mới (SETUP_DEV=0, không clone vllm-moreh). Package `aiter_moreh` đã cài (có `configs/`) nằm ở
   `/usr/local/lib/python3.12/dist-packages/aiter_moreh`. Lưu ý: KHÔNG được unset env này vì
   `aiter.jit.core` assert nó phải set.

2. **Bẫy "leftover script"** — `kill_server` chỉ `pkill -9 VLLM`, KHÔNG kill `auto_bench.sh`/`vllm bench`/
   `vllm serve`. Một `auto_bench.sh` cũ còn kẹt trong `wait_for_server` đã "sống lại" và bench đè lên
   server mới → 2 lần `--profile` chồng nhau. Phải dọn sạch process theo PID/pattern (tránh
   `pkill -f "vllm serve"` vì nó tự match command của chính mình → exit 137).

### Bước 3 — Xác nhận NGUYÊN NHÂN THẬT
Sau khi sửa env aiter và chạy 1 lần SẠCH, quan sát serve.log + tiến trình:
- `/start_profile` về ApiServer_7 (200 OK), `/stop_profile` về ApiServer_5 (do SO_REUSEPORT) — **2 frontend
  khác nhau**.
- **8 file trace `dp*_rank0.*.pt.trace.json.gz` ĐƯỢC GHI ĐẦY ĐỦ** (~645MB/rank cho eager), gzip OK, parse OK.
- NHƯNG `/stop_profile` **không bao giờ trả về 200**; worker spin ~100% CPU; serve.log spam
  `No available shared memory broadcast block found in 60 seconds` ⇒ **DP engine DEADLOCK sau khi đã ghi trace**.
- ⇒ `vllm bench` **treo mãi** → bản cũ `run_one`/`do_time_phase` block vào bench → KHÔNG bao giờ harvest/kill.

Thêm: **export trace rất chậm** — `profiler.stop()` mỗi rank mất vài phút (cửa sổ eager 4 phút → file đầu
xuất hiện ~8 phút sau `/stop_profile`). Nên logic "đợi trace" phải kiên nhẫn + chờ flush xong hẳn.

### Bước 4 — Sửa code (KHÔNG đụng preset)
Ý tưởng: **đừng block vào bench**. Chạy bench Ở NỀN → **poll thư mục trace** tới khi đủ N file VÀ tổng bytes
ngừng đổi (đã flush xong) → rồi kill cả bench treo lẫn server.

| File | Thay đổi |
|---|---|
| `serve.sh` | Guard `AITER_MOREH_ROOT_DIR`: nếu rỗng/không tồn tại → fallback về package `aiter_moreh` đã cài (server start được, không cần dev checkout). |
| `common/helper.sh` → `harvest_profiles` | Poll lại: chờ `TRACE_APPEAR_TIMEOUT` (mặc định 900s) cho file trace đầu, rồi settle theo **count + tổng bytes** (size-aware → không move file 645MB đang ghi dở). |
| `common/helper.sh` → `profiler_config_json` | **KHÔNG** inject `ignore_frontend` mặc định (giữ frontend ON ⇒ vẫn có 500 vô hại, đúng ý "kệ assertion"). Opt-in `PROFILE_IGNORE_FRONTEND=1` nếu muốn log sạch. |
| `bench_mv4571/auto_bench.sh` → `run_one` | Profile mode: chạy bench NỀN + `harvest_profiles` (poll) + kill bench treo (thay vì block). |
| `auto_analyze_ep_imbalance.sh` → `patch_time_preset` | **Bỏ ép `api_server_count`** (opt-in `API_SERVER_COUNT_PROFILE=1`). Giữ enforce_eager=false (cudagraph, timing thực) + EP_LOG=0. |
| `auto_analyze_ep_imbalance.sh` → `do_time_phase` | Chạy `run_bench` NỀN → `wait_for_traces` → kill bench treo + server. |
| `auto_analyze_ep_imbalance.sh` → `wait_for_traces` | Size-aware (đủ file + bytes đứng yên). `TRACE_WAIT_TIMEOUT` mặc định 900s. |

> 2 workaround cũ (ép `api_server_count=1`, bật `ignore_frontend`) vẫn còn **dưới dạng opt-in** cho ai
> muốn log sạch / né deadlock — KHÔNG bật mặc định.

### Bước 5 — Khắc phục 2 env regression còn lại của container
- **matplotlib chưa cài** → `analyze_time.py` / `analyze_tokens.py` lỗi import. `pip install matplotlib`.
- **Patch `[EP_COLLECT]` mất** — log token histogram trong vllm `fp8.py::Fp8MoEMethod.apply` (gated
  `VLLM_MOREH_EP_LOG`, cần enforce_eager) nằm ở dev checkout đã mất ⇒ phase TOKEN báo
  *"No [EP_COLLECT] lines found"*. Viết `apply_ep_collect_patch.py` (idempotent, có `.bak`) để **vá lại**
  vào installed `fp8.py`; orchestrator tự chạy preflight này khi PHASES có token. Format log khớp
  `analyze_tokens.py`: `[EP_COLLECT] layer=<name> it=<k> ntok=<n> E=256 counts=[c0..c255]`.

### Bước 6 — Smoke-test tới khi không còn lỗi (case 8k / conc8)
- **Validate analyze_time.py** (CPU, trên trace đã có): chạy OK cả trace eager lẫn cudagraph.
- **Orchestrator TIME phase** (cudagraph): serve OK (nhờ guard aiter) → bench nền → `wait_for_traces`
  bắt **8/8 trace DÙ deadlock** → analyze → **headroom 20.71%**, maxmin_mean 1.706, clusters 77.325
  ⇒ KHỚP reference (`TRACE_TIME_ANALYSIS.md` ~21.7% / 1.73). Teardown sạch (GPU về 0).
- **Orchestrator TOKEN phase** (eager): **1.235.400** dòng `[EP_COLLECT]`; R=8, E=256, 75 layer;
  **525 prefill / 153.675 decode** steps; auto-threshold 338.8; decode imbalance mean ~4.4; e2e rank
  max/min 1.038 ⇒ KHỚP reference. Đủ artifact (hist, heatmap, 75×2 per-layer, parquet, summary).
- **`auto_profile.sh` (đường run_all)**: harvest đủ **8 eager trace** vào `run1/8k_rinf_c8`, **KHÔNG treo**,
  teardown sạch.

---

## 3. File đã thay đổi / thêm mới

```
M  serve.sh                                              # guard AITER_MOREH_ROOT_DIR
M  common/helper.sh                                      # harvest_profiles (poll size-aware) + profiler_config_json (bỏ ignore_frontend mặc định)
M  bench_mv4571/auto_bench.sh                            # run_one: profile mode chạy bench nền + harvest poll + kill
M  bench_mv4571/auto_analyze_ep_imbalance/auto_analyze_ep_imbalance.sh
                                                         # patch_time_preset (bỏ ép api_server_count) + do_time_phase (bench nền) + wait_for_traces (size-aware) + preflight EP_COLLECT
A  bench_mv4571/auto_analyze_ep_imbalance/apply_ep_collect_patch.py   # vá lại [EP_COLLECT] vào installed fp8.py (idempotent)
M  bench_mv4571/auto_analyze_ep_imbalance/handoff.md     # cập nhật nguyên nhân thật + trạng thái
```

> Lưu ý: thay đổi trên installed vllm (`fp8.py`) và `pip install matplotlib` là trong **container** (ephemeral);
> nếu container bị tạo lại phải làm lại (xem §5 / §3b handoff). Thay đổi script là trong repo (bền vững).

---

## 4. Cách chạy

Mọi thứ chạy TRONG docker `phuc-nguyen-mv-4571`, repo mount tại `/home/phuc-nguyen/workspace/mv-4571/auto-script`.

```bash
# (chỉ cần 1 lần / sau khi container tạo lại)
docker exec phuc-nguyen-mv-4571 pip install matplotlib
docker exec phuc-nguyen-mv-4571 python3 \
  /home/phuc-nguyen/workspace/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/apply_ep_collect_patch.py

# Smoke-test 1 case (cả token + time), tự đợi GPU rảnh:
docker exec -ti phuc-nguyen-mv-4571 bash -lc \
 'cd /home/phuc-nguyen/workspace/mv-4571/auto-script && \
  ONLY=0 bash bench_mv4571/auto_analyze_ep_imbalance/auto_analyze_ep_imbalance.sh'

# Chỉ 1 phase:  PHASES=time ONLY=0 ...   |   PHASES=token ONLY=0 ...
# Full sweep:   bỏ ONLY (30 tổ hợp/preset — lâu, export trace eager ~8 phút/lần)

# Kiểm tra riêng đường auto_profile.sh (run_all):
docker exec -ti phuc-nguyen-mv-4571 bash -lc \
 'cd /home/phuc-nguyen/workspace/mv-4571/auto-script && \
  PRESET=glm5.2/dp8ep8/noMTP-bs64-dg.yaml bash bench_mv4571/auto_profile.sh'
```

**Env hữu ích:** `PHASES=token,time`, `ONLY=<idx>`, `SKIP_GPU_WAIT=1`, `GANTT_MAX_FIGS`, `ANALYZE=0`,
`DRY_RUN=1`, `TRACE_WAIT_TIMEOUT` (mặc định 900s). **Opt-in log sạch (không bắt buộc):**
`API_SERVER_COUNT_PROFILE=1`, `PROFILE_IGNORE_FRONTEND=1`.

> Để chạy phase profile qua `run_all_full_presets_vllm.sh`: bỏ comment `- profile` trong
> `bench_mv4571/env.yaml` (mục `phases:` đang chỉ bật `bench`). Chạy thẳng `auto_profile.sh` là tương đương.

---

## 5. Output layout

```
logs/<RUN_TS>/<model>/<preset_short>/<name>_r<rate>_c<conc>/
  ├── tokens/  serve.log (đầy [EP_COLLECT]) · bench/ · analysis/ (PNG + steps_imbalance.parquet + summary_tokens.json)
  └── time/    serve.log · traces/ (8× dp*_rank0.*.pt.trace.json.gz) · analysis/ (PNG + gantt/ + trace_pairs.npz + summary_time.json)
```

---

## 6. Việc còn lại
1. Chạy **full sweep** MV-4571 (bỏ `ONLY`).
2. Bật kimi cases (uncomment `scenario.yaml`) cho MV-4572 khi sẵn sàng.
3. Nếu container tạo lại: cài lại matplotlib + chạy lại `apply_ep_collect_patch.py` (orchestrator đã tự
   preflight cho token; guard aiter trong serve.sh là tự động).
