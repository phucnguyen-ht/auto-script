# Benchmark serving: luồng chạy & request_rate vs concurrency

Tài liệu cho bộ bench dùng `vllm bench serve` (mv4526 — custom dataset
`longbenchv2`, và tương tự mv4476 — random dataset).

---

## 1. Luồng chạy bench (Task 2)

Một lần chạy bench/profile đi qua các bước sau (do `auto_bench_template.sh` +
concrete `auto_bench.sh` điều phối):

1. **Resolve cấu hình**: từ preset → `MODEL_PATH` (theo family trong
   `model.paths`), `BASE_URL`, và đọc `env.yaml .<mode>.<method>`.

2. **Serve server MỘT lần** (dùng chung cho mọi scenario):
   - bench: `vllm serve <model> <preset>`.
   - profile: như trên nhưng inject `profiler_config` (torch profiler) vào preset
     để server bật endpoint profiling.
   - Đợi `GET /health` OK.

3. **Sinh scenarios** = tích đề-các `datasets × rates × concurrencies`:
   - `num_prompts = clamp(concurrency × prompts_per_concurrency, floor, cap)`.
   - `OSL` (output len) theo từng dataset.
   - mỗi scenario có label `"<dataset>_r<rate>_c<conc>"`.

4. **Run-major loop**: `for run in 1..runs: for scenario:` →
   1. (bench) **reset prefix cache**: `POST /reset_prefix_cache` để xoá KV/prefix
      cache, đảm bảo scenario sau không "ăn ké" cache của scenario trước → so
      sánh công bằng.
   2. **`vllm bench serve`** (client) gửi `num_prompts` request tới server:
      - `--dataset-name custom --dataset-path <prompt.jsonl>` (đã rename
        `text→prompt`), `--custom-output-len OSL`,
      - `--max-concurrency C`, `--request-rate R`,
      - `--ignore-eos` (ép sinh đủ OSL để độ dài so sánh được),
        `--skip-chat-template` (prompt đã được template sẵn),
      - `--profile` (chỉ ở mode profile).
      - Client đo per-request **TTFT / TPOT / ITL / E2EL** + **throughput**, lưu
        `run<i>/<label>.json`.

5. **Aggregate** (`agg_bench.py`): gom các json → `run<i>.csv` (hàng = scenario,
   cột = metric) và `mean.csv` / `std.csv` qua `runs` lần.

6. **Kill server**.

> Điểm mấu chốt: **1 server phục vụ tất cả scenarios** (không serve lại mỗi
> scenario). `vllm bench serve` chỉ là *client* gửi tải; model chạy ở server.

---

## 2. request_rate vs concurrency (Task 3)

Đây là **hai trục tải khác nhau** của `vllm bench serve`:

| | Ý nghĩa | Điều khiển |
|---|---|---|
| `--request-rate R` (req/s) | Tốc độ **ĐẾN** của request mới (client phát request theo phân phối Poisson với rate R). `inf` = bắn hết ngay (closed-loop). | Bao lâu thì gửi 1 request mới |
| `--max-concurrency C` | Số request **ĐỒNG THỜI** (in-flight) tối đa. Client giữ ≤ C request chưa xong. | Bao nhiêu request chạy cùng lúc |

### Khác nhau
- `R` = **arrival rate** (mở — open-loop): quyết định tải đổ vào nhanh hay chậm.
- `C` = **trần song song** (đóng — closed-loop): chặn số request server xử lý
  cùng lúc, bất kể đến nhanh cỡ nào.

### Ảnh hưởng lẫn nhau
- **R thấp** (dưới năng lực server ở mức C): server xử lý kịp, số request in-flight
  < C nên **trần C không bao giờ chạm** → độ trễ thấp, throughput ≈ R. Tăng C lúc
  này gần như **không đổi gì** (flat).
- **R cao / inf**: request dồn lại đến mức C → **C trở thành nút thắt** → cho ra
  đường cong latency-vs-throughput cổ điển. `inf` chính là closed-loop: chỉ C
  quyết định tải.
- **Quanh điểm bão hoà (knee)**: C thấp thì queue dồn (latency tăng), C cao thì
  theo kịp (throughput cao hơn) — đây là vùng hai trục "phân kỳ", đáng quan sát.
  Vượt knee thì throughput **plateau** (≈ năng lực server), tăng R/C thêm chỉ làm
  hàng đợi & latency dài ra.

> Vì vậy sweep `R × C`: với mỗi R cố định, quét C để thấy đường cong; `R=inf` là
> mốc closed-loop. Một số ô (R thấp + C cao) sẽ "trùng" nhau (C không chạm) —
> đó là điều bình thường và có chủ đích để thấy mỗi rate bão hoà ở đâu.

### request_rate tăng thì TPOT bị ảnh hưởng thế nào?
`TPOT` (time per output token) = thời gian sinh **mỗi token decode** (sau token
đầu tiên).

- **R tăng → nhiều request đồng thời hơn → batch decode lớn hơn.** Mỗi bước decode
  của vLLM xử lý cả batch cùng lúc; batch lớn hơn ⇒ cạnh tranh **compute** và đặc
  biệt **memory bandwidth** (đọc KV-cache) ⇒ **TPOT TĂNG** (mỗi token chậm đi).
- **TTFT** cũng tăng (request phải chờ trong hàng đợi + prefill batch lớn hơn).
- **Output throughput tổng** (token/s toàn hệ) thì **tăng** theo R — tới khi
  **bão hoà** thì plateau.

Tóm tắt xu hướng khi `R↑`:

```
request_rate ↑  ─►  batch size ↑  ─►  TPOT ↑ , TTFT ↑   (per-request chậm hơn)
                                   └►  total throughput ↑ ... rồi plateau ở knee
```

Đây chính là đánh đổi **latency ↔ throughput**: đẩy rate cao để vắt kiệt
throughput thì phải chấp nhận TPOT/TTFT (độ trễ) cao hơn. Khi đã vượt knee, TPOT
tăng gần tuyến tính theo batch trong khi throughput gần như không tăng nữa.

> Lưu ý: khi `C` là trần, lúc `R ≥ knee` thì hành vi ≈ `R=inf` (C giới hạn tải),
> nên nhiều ô rate cao trong sweep sẽ cho kết quả gần giống `inf`.
