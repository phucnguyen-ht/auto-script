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

## 3. FIX GỐC: "auto_profile chạy xong KHÔNG sinh trace" (user nhấn mạnh)
- **Nguyên nhân thật**: dp8 → `api_server_count` mặc định = data_parallel_size_local = 8
  (`bench_mv4571/3rdparty/vllm/entrypoints/cli/serve.py:105-109`). `vllm bench serve --profile`
  gửi `/start_profile` rồi `/stop_profile`, bị **load-balance sang 2 frontend KHÁC nhau** →
  start/stop rơi vào engine khác → worker không nhận đủ cặp start+stop → **trace không được ghi**
  (lúc có lúc không — đúng ca user gặp dù đã tắt ignore_frontend).
- Cơ chế: worker ghi trace trong `profiler.stop()` (`gpu_worker.py:899-950`), trigger qua
  `engine_core.profile(False)` ← `async_llm.stop_profile` (`v1/engine/async_llm.py:911-915`).
  Frontend profiler chỉ tồn tại khi `ignore_frontend=False` (`async_llm.py:178-200`); nó là
  nguồn 500 "Profiler must be initialized" nhưng KHÔNG phải nguyên nhân thiếu trace.
- **FIX**: ép `api_server_count=1` (patch_time_preset) → 1 frontend nhận CẢ start+stop →
  broadcast nhất quán 8 DP engine → 8 trace tin cậy + sạch (no 500). User chọn "giữ frontend ON";
  với count=1 thì frontend ON cũng start/stop cùng process nên sạch luôn.
- **Bồi thêm**: `wait_for_traces()` chờ đủ 8 file `dp*_rank0.*.pt.trace.json.gz` (flush bất đồng bộ
  sau /stop) ổn định RỒI MỚI kill server (bản auto_profile cũ kill ngay → mất trace).
- **Cổng GPU ban đầu** (theo yêu cầu user): trước vòng lặp, nếu GPU bận thì ĐỢI & retry mỗi 30s
  (`wait_for_gpu_free`, GPU_POLL_INTERVAL=30) → KHÔNG kill nhầm server người khác; sau cổng GPU rảnh
  nên kill_server per-phase chỉ là no-op.

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
1. **Smoke-test 1 case trên GPU** — đang chặn: lúc kiểm tra, 8 GPU đều 97-98% VRAM (2 bộ VLLM worker
   đang chạy của process khác). Script đã có cổng đợi-30s nên chạy được an toàn (sẽ tự đợi). Chờ GPU rảnh
   rồi chạy lệnh ở mục 0.
2. Sau smoke-test OK → chạy full sweep MV-4571; bật kimi cases (uncomment) cho MV-4572 khi sẵn sàng.
3. (Tùy chọn) kiểm tra trace dưới cudagraph (enforce_eager=false) parse có ra cụm chuẩn như eager không
   — run validate cũ là cấu hình enforce_eager đã COMMENT (cudagraph) và OK, nên kỳ vọng ổn.

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
