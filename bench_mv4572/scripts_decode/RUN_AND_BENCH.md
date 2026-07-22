# Run & Bench — GLM-5.2-FP8 (dp8-ep8)

Practical guide to bring up a `vllm-moreh` server, run the `benchmark_zhipu`
serving sweep, and extract per-scenario metrics. Reflects the current
`multi_process_test.py` workflow (dataset-outer loop, warmup per bench,
`bench_start`/`bench_end` markers). Run everything **inside the dev container**.

---

## 0. Prerequisites

- **Model**: `/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/`
- **Preset**: `presets/full/zai-org-glm-5.2-fp8-mtp-amd-mi300x-dp8-moe-ep8.yaml` (dp8 → needs **8 free GPUs**)
- **Datasets**: `/remote/vast0/share-mv/longbenchv2-custom/longbenchv2-<8k|10k|100k|1M>.jsonl`

Check the GPUs are idle first:

```bash
rocm-smi --showmeminfo vram | grep "Used Memory"    # each ~298 MB when idle
ps -eo pid,args | grep -E "vllm-moreh serve|EngineCore" | grep -v grep
```

---

## 1. Serve the model

Keep each run's logs together in one folder:

```bash
cd /workspace/vllm-moreh
RUN="benchmark_zhipu/run_logs/glm52-fp8-mtp-dp8-ep8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

nohup vllm-moreh serve /remote/vast0/share-mv/zai-org/GLM-5.2-FP8/ \
  presets/full/zai-org-glm-5.2-fp8-mtp-amd-mi300x-dp8-moe-ep8.yaml \
  > "$RUN/serve.log" 2>&1 &
echo $! > "$RUN/serve.pid.txt"
```

Wait until healthy (model load + CUDA-graph capture on 8 GPUs takes several minutes):

```bash
until curl -sf http://localhost:8000/health >/dev/null; do
  kill -0 "$(cat "$RUN/serve.pid.txt")" 2>/dev/null || { echo "server died"; tail -40 "$RUN/serve.log"; break; }
  sleep 15
done && echo "HEALTH OK"
```

> KV cache is **~1.14M tokens per DP rank** (`grep "GPU KV cache size" "$RUN/serve.log"`).
> A prompt's working set (num_distinct_prompts × prompt_len) must fit under that
> to stay resident — see §5.

---

## 2. Configure the sweep

Edit the range lines at the top of `benchmark_zhipu/multi_process_test.py`:

```python
parallel_threads_range   = [16, 32, 64, 96, 128, 192, 256]   # default concurrencies
datasets_range           = ["8k", "10k", "100k", "1M"]       # ISL buckets
dataset_parallel_threads = {"1M": [4, 8]}                    # per-dataset override
```

Each dataset uses `parallel_threads_range` unless it appears in
`dataset_parallel_threads`, in which case that dataset's own concurrency list
takes over. The 1M bucket has very long inputs, so it only runs at low
concurrency (4, 8); 8k/10k/100k keep the full default sweep. Add another entry
to `dataset_parallel_threads` to cap any other bucket the same way.

The per-bench load window is set by `base_command` (default `--time_limit 240
--ignore_time_start 20 --ignore_time_end 20` → a **200 s** steady-state window).

`unit_test.py` controls warmup + data caps:
- **Warmup copies** = `for i in range(16)` — 16 per prompt. On dp8 this is what
  gives every DP rank per-prompt coverage (see §5); **don't drop it below ~2×
  the DP size**.
- Distinct prompts: 8k/10k → `data[:100]`, 100k → `data[:10]`, 1M → `data[:1]`.
- `OUTPUT_LEN` = 1024 for 8k, else 500.
- The payload `"model"` must match the served model path (currently FP8).

---

## 3. Run the bench

```bash
cd /workspace/vllm-moreh
nohup python3 -u benchmark_zhipu/multi_process_test.py > "$RUN/mpt_run.log" 2>&1 &
echo $! > "$RUN/bench.pid.txt"
```

Workflow per dataset: reset prefix cache **once**, then for each concurrency
`run_warmup` (16 copies/prompt, no markers) → `run_bench` (formal 200 s window,
bracketed by `GET /v1/bench_start` … `/v1/bench_end` — they 404, the access-log
line is the marker). Results land in
`benchmark_zhipu/results/longbenchv2-<ds>_p<conc>_<ts>/`.

Track progress / wait for completion:

```bash
grep -c "\[Bench\]"    "$RUN/mpt_run.log"    # benches started
grep -c "\[Summary\]"  "$RUN/mpt_run.log"    # benches finished
# block until the driver exits:
while kill -0 "$(cat "$RUN/bench.pid.txt")" 2>/dev/null; do sleep 30; done; echo DONE
```

---

## 4. Extract & summarize

Fresh server → serve.log holds only this run's `bench_start/end` windows, in
`[Bench]` order (dataset-outer, concurrency-inner). Derive labels from the
driver log, extract per-step decode batch size, then aggregate.

```bash
MIN_TS=$(stat -c %Y "$RUN/bench.pid.txt")   # scope to this run; excludes older result dirs

# labels in driver order: "[Bench] concurrency N, dataset D" -> D_pN
mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "$RUN/mpt_run.log" \
  | awk '{gsub(/,/,""); print $4"_p"$2}')

mkdir -p "$RUN/decode_running"
python3 benchmark_zhipu/extract_decode_running.py \
  "$RUN/serve.log" "$RUN/decode_running/decode_running" --scenarios "${LABELS[@]}"

python3 benchmark_zhipu/summarize_scenarios.py \
  benchmark_zhipu/results "$RUN/scenario_summary.csv" \
  --totals-dir "$RUN/decode_running" --min-ts "$MIN_TS"

column -t -s, "$RUN/scenario_summary.csv"
```

`scenario_summary.csv` → one row per scenario: TTFT/TPOT percentiles,
`mean_decode_tps`, system `output_tps`, and `mean/sustained_max_decode_batch_size`.

- `output_tps` = `total_output_tokens / benchmark_duration_s`, where the duration
  is the real generation span from `output.jsonl` token timestamps (≈ the 200 s
  window for a sustained run) — **not** a single request's latency.
- Only requests that both start after `ignore_time_start` **and** finish before
  the window end are counted; if per-request latency ≳ window, the sample shrinks.

---

## 5. Validate prefix cache (important on dp8)

Each DP rank has its **own** prefix cache and requests are load-balanced without
affinity, so warming a prompt globally does **not** guarantee it's cached on every
rank. Check the per-scenario hit rate:

```bash
python3 - "$RUN/serve.log" "$RUN/mpt_run.log" <<'PY'
import re,sys
L=open(sys.argv[1],errors="replace").readlines()
ansi=re.compile(r"\x1b\[[0-9;]*m"); pc=re.compile(r"Prefix cache hit:\s*[0-9.]+%\s*\((\d+)/(\d+)\)")
st=[i for i,l in enumerate(L) if 'bench_start' in l]; en=[i for i,l in enumerate(L) if 'bench_end' in l]
labs=[f"{m.group(2)}_p{m.group(1)}" for l in open(sys.argv[2],errors="replace")
      for m in [re.search(r'concurrency (\d+), dataset (\S+)',l)] if m]
for i,(s,e) in enumerate(zip(st,en)):
    H=Q=full=0
    for ln in L[s+1:e]:
        m=pc.search(ansi.sub("",ln)); 
        if not m: continue
        h,q=int(m.group(1)),int(m.group(2))
        if q==0: continue
        H+=h; Q+=q; full+= (h==0)
    lab=labs[i] if i<len(labs) else f"w{i+1}"
    print(f"{lab:>10} hit={100*H/Q if Q else 0:.2f}%  full_miss={full}")
PY
```

- **~99% (99.1% for short ISL, 99.9% for long)** = fully warm. The last block
  always recomputes, so 100% is unreachable; 99.1% *is* the ceiling.
- **Well below (e.g. 89%, or `full_miss` > 0)** = warmup didn't cover all ranks.
  Fix: `range(16)`+ warmup copies (≈`R·ln(R/ε)` for R ranks), loop warmup a few
  rounds, or reduce the distinct-prompt working set so it fits the KV cache.

---

## 6. Stop the server

```bash
SPID=$(cat "$RUN/serve.pid.txt")
kill -TERM -"$SPID" 2>/dev/null; sleep 5; kill -KILL -"$SPID" 2>/dev/null; sleep 3
rocm-smi --showmeminfo vram | grep "Used Memory"    # expect ~298 MB each
```

---

## 7. Gotchas

- **8 free GPUs required.** If init dies with `WorkerProc initialization failed`,
  another container is holding the GPUs — re-check `rocm-smi`.
- **Warmup copies vs DP size.** Low copy counts leave some (prompt, rank) pairs
  cold → sub-100% prefix hit → inflated TTFT. Use ≥ ~2× DP ranks (16 for dp8).
- **`mean_tpot_ms` is client-measured** (SSE receive gaps) and inflates at high
  concurrency from client-side jitter; the server's true per-token latency is
  lower. `mean_decode_tps` (≈ `1000/tpot`) corroborates the real value.
- **`min-ts`** scopes the summary to one run — the `results/` folder is shared
  across runs; without it you'll pick up stale dirs (summarize also dedups
  (ISL, conc, seed) by latest timestamp).
- **Long ISL × high concurrency**: if prefill saturates or per-request latency
  exceeds the 200 s window, in-window request counts collapse — raise
  `--time_limit` for those cells.
```
