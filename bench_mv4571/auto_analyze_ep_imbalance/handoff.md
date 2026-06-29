# MV-4571 / MV-4572 — Handoff (resume in new session)

> Mục tiêu phiên này: gộp toàn bộ quy trình phân tích EP8 imbalance (token + time)
> thành **1 script auto** sweep mọi testcase theo ticket. **Build + validate offline ĐÃ XONG.
> Chỉ còn smoke-test 1 case trên GPU (đang bị chặn vì 8 GPU bận).**

---

## 0. Cách resume nhanh
1. Đọc file này.
2. Kiểm tra GPU rảnh chưa: `pgrep -fa VLLM` + `rocm-smi --showmemuse`.
3. Smoke-test 1 case (8k/conc8, cả token+time) TRONG docker `phuc-nguyen-mv-4571`:
   ```bash
   docker exec -ti phuc-nguyen-mv-4571 bash -lc \
     'cd /home/phuc-nguyen/workspaces/mv-4571/auto-script && \
      ONLY=0 bash bench_mv4571/auto_analyze_ep_imbalance/auto_analyze_ep_imbalance.sh'
   ```
   Script tự ĐỢI (retry 30s) nếu GPU bận — KHÔNG kill server người khác.
4. Sau khi smoke-test OK → chạy full sweep (bỏ `ONLY=0`).

---

## 1. Ticket & phạm vi
- **MV-4571** [GLM5.2] Analyze EP balancing impact (DP8/EP8, TP1, MI300/ROCm, AITER).
  Matrix (từ desc): ISL{8k,10k,100k,1M}; conc theo ISL (8k/10k: 8,22,31,36,52 · 100k: 8,22,31,36 · 1M: 8);
  MTP{0,3}=2 preset; dataset custom longbenchv2. Báo cáo imbalance theo %.
- **MV-4572** [Kimi K2.6] tương tự nhưng desc CHƯA define matrix → quy ước tạm **1 conc/ISL** (=8).
- 2 hướng phân tích:
  - **TOKEN**: histogram #token-routing/expert trước all2all → max/min load per (layer,step).
  - **TIME**: từ torch profiler traces 8 rank → MoE-compute time/rank/cụm, headroom nếu balance.

---

## 2. Deliverables (đã viết & validate) — thư mục `bench_mv4571/auto_analyze_ep_imbalance/`
| File | Vai trò |
|---|---|
| `scenario.yaml` | Matrix 2 ticket. `models` registry: model→{model_path, presets[]}. Case `{model,name,osl,rates[],concurrencies[]}` nở ra: presets × rates × concurrencies. defaults: ppc=2, num_prompts_floor=1, num_prompts_cap=256. **KHÔNG còn `decode_max_tokens`**. Kimi cases đang comment. |
| `analyze_tokens.py` | Port notebook `process_ep_logs_glm5_log8k.ipynb` CELL 5 (hist max/min all/prefill/decode), 12 (phân phối tải per-expert/per-rank + heatmap layer×{expert,rank}), 13 (per-layer load — TẤT CẢ layer), 14 (imbalance/layer + decode heatmap). **Tự dò ranh giới prefill/decode** (khe lưỡng cực num_tokens, hàm `auto_decode_threshold`) thay knob. Args: `--log --out [--concurrency] [--layers all] [--decode-max-tokens auto]`. |
| `analyze_time.py` | Port `process_trace_time_glm5.ipynb`: verify Bước0(CHECK1-4)/1b/1c, bảng theo RANK (ms)+breakdown.png, Metrics, hist sec3, Gantt 3×3 (~200 ảnh sample đều), 5b (REL_THR=0.2), 5c (THR_US=50µs), export summary_time.json. Args: `--trace-dir --out [--drop-head 0] [--rel-thr 0.2] [--thr-us 50] [--gantt-max-figs 200] [--no-gantt] [--no-verify-schema]`. Parse ~4.5 phút/8 traces. |
| `auto_analyze_ep_imbalance.sh` | **Orchestrator**. Vòng lặp case→preset→rate→conc; mỗi (preset,rate,conc) chạy 2 phase, mỗi phase serve RIÊNG (kill→serve→bench→analyze→kill). |

Output layout: `logs/<RUN_TS>/<model>/<preset_short>/<name>_r<rate>_c<conc>/{tokens,time}/`
- tokens/: serve.log, bench/, analysis/ (PNG + steps_imbalance.parquet + summary_tokens.json)
- time/: serve.log, traces/ (8×dp*_rank0.*.pt.trace.json.gz), analysis/ (PNG + gantt/ + trace_pairs.npz + summary_time.json)

### Phase config (patch preset bằng yq)
- **TOKEN**: `enforce_eager=true` (cần eager để side-effect log `[EP_COLLECT]` chạy), `VLLM_MOREH_EP_LOG='1'`, `VLLM_MOREH_EP_LOG_DEBUG='0'`, **del** profiler_config. num_prompts = clamp(ppc*conc, floor, cap) (ppc=2). Bench KHÔNG `--profile`. → serve.log đầy `[EP_COLLECT]`.
- **TIME**: `enforce_eager=false` (cudagraph FULL_DECODE_ONLY = timing thực), `VLLM_MOREH_EP_LOG='0'`, **`api_server_count=1`**, inject profiler_config (torch, 4 cờ False). num_prompts = clamp(conc, floor, cap) (ppc LUÔN=1 cho profile). Bench `--profile`. → 8 traces.

### Env override hữu ích
`SCENARIO_YAML, PHASES=token,time, ONLY=<idx tổ hợp 0-based>, ANALYZE=0, DRY_RUN=1,
GANTT_MAX_FIGS, REL_THR, THR_US, DROP_HEAD, SKIP_GPU_WAIT=1, GPU_POLL_INTERVAL=30,
TRACE_WAIT_TIMEOUT=600, EXPECT_RANKS=8, API_SERVER_COUNT_PROFILE=1`.

---

## 3. FIX GỐC: "auto_profile chạy xong KHÔNG sinh trace / server chết" — ĐÃ XÁC NHẬN TRÊN GPU (2026-06-28)
- **Nguyên nhân THẬT (đã chạy & quan sát):** dp8 → `api_server_count` mặc định = data_parallel_size_local = 8
  (serve.py:105-109). `vllm bench serve --profile` gửi `/start_profile` rồi `/stop_profile` qua socket
  SO_REUSEPORT nên rơi vào 2 ApiServer KHÁC nhau. **8 file trace VẪN được worker ghi đầy đủ** (mỗi
  rank ~645MB cho eager / ~95MB cho cudagraph). NHƯNG ngay sau đó các DP EngineCore **DEADLOCK/LIVELOCK**:
  `/stop_profile` không bao giờ trả về 200, worker spin ~100% CPU, serve.log spam "No available shared
  memory broadcast block found in 60 seconds". ⇒ `vllm bench` TREO vĩnh viễn → bản cũ block vào bench
  nên KHÔNG bao giờ harvest/kill → nhìn như "không có trace / server chết".
- Lưu ý: `call_utility_async("profile")` broadcast tới TẤT CẢ engine (core_client.py:1428) nên 1 start
  + 1 stop là ĐỦ để ghi cả 8 trace. Giải thích cũ "start/stop lệch engine → thiếu cặp" là SAI.
  Export trace RẤT CHẬM: `profiler.stop()` mỗi rank mất vài phút (cửa sổ eager 4 phút → ~8 phút flush).
- **FIX (đúng ý user: KHÔNG đụng preset, kệ deadlock/assertion, trace bắt buộc có):** chạy bench Ở NỀN,
  POLL thư mục trace tới khi đủ N file VÀ tổng bytes đứng yên (đã flush xong) RỒI kill cả bench treo lẫn
  server. KHÔNG ép `api_server_count=1`, KHÔNG thêm `ignore_frontend` (giữ preset nguyên si).
  - `common/helper.sh::harvest_profiles` — chờ `TRACE_APPEAR_TIMEOUT` (900s) cho trace đầu, rồi settle
    theo count+bytes (size-aware → không move file 645MB đang ghi dở).
  - `common/helper.sh::profiler_config_json` — KHÔNG inject `ignore_frontend` mặc định (opt-in `PROFILE_IGNORE_FRONTEND=1`).
  - `bench_mv4571/auto_bench.sh::run_one` — profile mode chạy bench nền + harvest poll + kill bench treo.
  - orchestrator: `patch_time_preset` KHÔNG ép `api_server_count` (opt-in `API_SERVER_COUNT_PROFILE=1`);
    `do_time_phase` chạy bench nền → `wait_for_traces` (size-aware) → kill; `TRACE_WAIT_TIMEOUT` mặc định 900s.
- **2 workaround cũ vẫn còn dưới dạng OPT-IN** (nếu muốn log sạch, không deadlock): `API_SERVER_COUNT_PROFILE=1`
  (start+stop về 1 frontend) và/hoặc `PROFILE_IGNORE_FRONTEND=1` (tắt frontend profiler → no 500).
- **Cổng GPU ban đầu** (giữ nguyên): nếu GPU bận thì ĐỢI & retry mỗi 30s (`wait_for_gpu_free`, GPU_POLL_INTERVAL=30).

### 3b. ENV REGRESSIONS trong container build mới (phải fix để chạy được)
- `AITER_MOREH_ROOT_DIR` trỏ vào dev checkout không tồn tại → mọi EngineCore chết lúc startup
  (thiếu `a8w8_bpreshuffle_tuned_gemm_bruteforce.csv`). **ĐÃ FIX** bằng guard trong `serve.sh`
  (fallback về package `aiter_moreh` đã cài).
- `matplotlib` chưa cài → analyze_*.py lỗi import. `pip install matplotlib` (đã cài).
- Patch `[EP_COLLECT]` trong installed vllm `fp8.py` KHÔNG có (nó nằm ở dev checkout đã mất) → phase
  TOKEN báo "No [EP_COLLECT] lines found". **ĐÃ FIX**: `apply_ep_collect_patch.py` (idempotent, có .bak);
  orchestrator tự chạy preflight khi PHASES có token.

---

## 4. Validate OFFLINE đã chạy (khớp số liệu đã thiết lập trước đó)
Chạy trong docker `phuc-nguyen-mv-4571` (host không có numpy):
- **tokens** trên serve.log thật (`logs/glm5.2/dp8ep8/noMTP-bs64-dg/auto_bench/20260625_040651_8k_c8/serve.log`):
  decode max/min mean **3.8**, auto-threshold=300.3, prefill 525 / decode 153672 steps, 150 ảnh per-layer. ✓
- **time** trên 8 traces thật (`.../auto_profile/20260625_091838/profiling_result/run1/8k_rinf_c8`):
  critical **10734ms** / balanced **8172ms** / **headroom 23.86%**, comm 42973ms, decode max/min mean 1.73,
  Gantt 3×3 sample đều. ✓ (khớp `EP_imbalance_analysis/TRACE_TIME_ANALYSIS.md`)
- **Dry-run orchestrator**: 30 tổ hợp (2 preset × 15 ISL-conc), preset patch đúng, np token=16/time=8,
  `bash -n` OK, `DRY_RUN`/`ONLY` hoạt động.

---

## 5. PENDING (việc còn lại)
- ✅ **Smoke-test 1 case (8k/conc8) trên GPU — XONG (2026-06-28), cả 2 phase OK:**
  - TIME (cudagraph): 8/8 trace bắt được DÙ /stop_profile deadlock; `analyze_time` → headroom **20.71%**,
    maxmin_mean 1.706, clusters 77325 — KHỚP reference (TRACE_TIME_ANALYSIS.md ~21.7% / 1.73).
  - TOKEN (eager): **1.235M** dòng [EP_COLLECT]; R=8 E=256 75 layer; **525 prefill / 153675 decode** steps,
    auto-threshold 338.8, decode imbalance mean ~4.4, e2e rank max/min 1.038 — KHỚP reference.
  - `auto_profile.sh` (đường run_all) cũng đã verify: harvest đủ 8 eager trace, KHÔNG treo.
1. Chạy FULL sweep MV-4571 (bỏ `ONLY`): `bash auto_analyze_ep_imbalance.sh` (token+time, 30 tổ hợp/preset).
   Lưu ý GPU/thời gian: mỗi (rate,conc) serve 2 lần; export trace eager ~8 phút/lần.
2. Bật kimi cases (uncomment scenario.yaml) cho MV-4572 khi sẵn sàng.
3. Nếu container bị tạo lại: chạy lại `apply_ep_collect_patch.py` + `pip install matplotlib` (xem §3b).

---

## 6. Bối cảnh kỹ thuật đã chốt (tham chiếu nhanh)
- **TOKEN vs TIME phase split**: TIME chia phase bằng **kernel** (`nstage`: 1 fused fmoe=prefill, 2 gemm=decode);
  TOKEN không có kernel info trong log nên tự dò ngưỡng num_tokens (lưỡng cực decode~conc vs prefill~nghìn).
- **MoE-time/cụm (TIME)** = `reduce_scatter.start − gather.end` (comm-bounded span), độc lập kiểu kernel.
  combine (reduce-scatter) là **barrier** → step bị chặn bởi rank chậm nhất.
- **Headroom math** (đã giải thích kỹ cho user): improvement = `1 − Σmean/Σmax` (giả định TỔNG work bảo toàn,
  chia đều về mean). KHÔNG dùng max/min để suy %. `max/min=1.6` ⇒ ~21-24% (không phải 60%) vì balance đưa về
  MEAN chứ không về MIN; rank chậm **luân phiên** nên per-rank total ~bằng nhau (~8.2K ms) nhưng critical-path
  `Σmax` (10.7K) > balanced `Σmean` (8.2K) = 23.9%. Layer đang **comm-bound** (comm 42K ≫ MoE 7-8K) nên lợi ích
  end-to-end nhỏ hơn nhiều.
- **MoE data flow / verify**: xem `bench_mv4571/EP_imbalance_analysis/{MOE_DATA_FLOW.md, TRACE_TIME_ANALYSIS.md, events.MD}`.
  Backend EP = AgRs (all_gatherv dispatch + reduce_scatterv combine), supports_async=False → đồng bộ, barrier.
  GLM5.2: 256 expert, topk8, 78 layer, first_k_dense_replace=3 → 75 MoE layer, EP8 → 32 expert/rank.

## 7. Notebook nguồn (đã port sang .py, vẫn giữ để đối chiếu)
- `bench_mv4571/EP_imbalance_analysis/process_ep_logs_glm5_log8k.ipynb` (token)
- `bench_mv4571/EP_imbalance_analysis/process_trace_time_glm5.ipynb` (time)
- `bench_mv4571/EP_imbalance_analysis/analyze_trace_pairs.py` (parser time gốc, comm-bounded — analyze_time.py kế thừa logic).

## 8. Lưu ý môi trường
- Mọi thứ chạy TRONG docker `phuc-nguyen-mv-4571` (host thiếu numpy/pandas/ijson/yq).
- Container mount repo tại path gốc; chạy qua `docker exec`. yq v4 có sẵn (`/usr/local/bin/yq`), vllm CLI có sẵn.
- Lưu ý perms: docker chạy như root nên đôi khi file trong repo bị đổi owner→root; nếu host ghi bị
  `EACCES`, chown lại bằng `docker exec phuc-nguyen-mv-4571 chown -R 1010:1006 <path>` (uid host phuc-nguyen=1010, gid moreh=1006).
- Patch vllm cho `[EP_COLLECT]`: `bench_mv4571/3rdparty/vllm/model_executor/layers/quantization/fp8.py`
  (gated bởi `VLLM_MOREH_EP_LOG`), đã có sẵn từ phiên trước.
