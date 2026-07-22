# Run & Bench (Prefill) — GLM-5.2-FP8 (dp8-ep8)

Practical guide to bring up a `vllm-moreh` server and measure **prefill**
performance with the `benchmark_prefill` sweep. This is the prefill counterpart
of `benchmark_zhipu/RUN_AND_BENCH.md` (which measures decode). The differences
that make this a prefill benchmark:

- **Short OSL (default 2)** — every request generates only a couple of output
  tokens (`max_tokens = min_tokens = OUTPUT_LEN`, default `2`, override via
  `PREFILL_BENCH_OSL`), so each request is dominated by a full prefill. TTFT and
  `completion_ts` are measured from the *first* token, so this stays a prefill
  measurement regardless of the exact OSL. **OSL=1 is NOT usable with this
  preset**: with MTP + PD-separation + async scheduling a request that finishes
  in the same step its prefill completes triggers a `KeyError` in the vLLM async
  scheduler and kills every EngineCore — see §6.
- **Prefix cache disabled** — passed as `--no-enable-prefix-caching` on the serve
  command (see §0/§1) so every request recomputes its full prompt; nothing is
  served from cache.
- **No warmup** — the driver goes straight to the measurement window. Warming
  would only populate a cache we deliberately turned off.
- **`prefill_throughput` is the headline metric** — system input-token
  throughput = total prompt tokens processed / window span, where the window span
  is `(first-token time of the last recorded request) − (send time of the first
  recorded request)`.

Run everything **inside the dev container**.

---

## 0. Prerequisites

- **Model**: `/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/`
- **Preset**: `presets/full/zai-org-glm-5.2-fp8-mtp-amd-mi300x-dp8-moe-ep8.yaml` (dp8 → needs **8 free GPUs**)
- **Datasets**: `/remote/vast0/share-mv/longbenchv2-custom/longbenchv2-<8k|10k|100k|1M>.jsonl`

**Disable prefix caching via the serve command.** ⚠️ Setting
`enable_prefix_caching: false` *in the preset yaml does NOT work* — the preset
loader feeds engine args to vLLM through `--config`, and vLLM silently ignores
`enable_prefix_caching: false` there (it is a `BooleanOptionalAction`). You must
pass `--no-enable-prefix-caching` on the `vllm-moreh serve` command line instead
(§1). Verify it actually took effect after the server starts — checking the CLI
echo is not enough, confirm the *effective* engine config and a real 0% hit rate:

```bash
grep -m1 -oE "enable_prefix_caching=(True|False)" "$RUN/serve.log"   # expect =False
# during a bench, no step should show a non-zero prefix-cache hit:
grep -oE "Prefix cache hit: [0-9.]+%" "$RUN/serve.log" | grep -v "0.0%" | head  # expect empty
```

If you see `enable_prefix_caching=True` or any non-zero hit, caching is on and the
prefill numbers are inflated/meaningless (§6).

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
RUN="benchmark_prefill/run_logs/glm52-fp8-mtp-dp8-ep8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

nohup vllm-moreh serve /remote/vast0/share-mv/zai-org/GLM-5.2-FP8/ \
  presets/full/zai-org-glm-5.2-fp8-mtp-amd-mi300x-dp8-moe-ep8.yaml \
  --no-enable-prefix-caching \
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

---

## 2. Configure the sweep

Edit the range lines at the top of `benchmark_prefill/multi_process_test.py`:

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

The per-bench load window is **derived, not hand-set** — `window_params()` sizes
each scenario's `time_limit` so at least the first wave of `concurrency`
concurrent prompts finishes prefill inside the window:

```
time_limit = max(MIN_TIME_LIMIT, ceil(WINDOW_HEADROOM * concurrency * L / P))
```

where `L = dataset_prompt_tokens[ds]` (mean prompt tokens) and `P =
dataset_prefill_tps[ds]` (measured cluster prefill throughput, toks/sec summed
across DP ranks — see §4). `WINDOW_HEADROOM` (1.25) leaves room for the wave to
land inside the record window; `MIN_TIME_LIMIT` (60 s) floors tiny low-concurrency
waves. This scales automatically with both concurrency and input length, so
long-ISL / high-concurrency cells get a much longer window (100k @ 256 ≈ 27 min,
by design). **Re-measure `dataset_prefill_tps` and update the table if the
model / preset / cluster changes.** Each `[Bench]` line prints the `time_limit`
it used. With the defaults the full 3×7 sweep is ≈100 min (100k dominates; 8k+10k
together ≈18 min).

A dataset listed in `dataset_min_time_limit` gets a per-dataset **floor** on
`time_limit` (overriding `MIN_TIME_LIMIT`): `time_limit = max(floor, derived)`.
1M is floored at `700` s because at conc 4/8 (≤ DP size) per-request TTFT is
~constant, not proportional to concurrency, so the `conc*L/P` formula under-sizes
its window and records 0 requests. The derived value still wins for any dataset
whose formula exceeds its floor; remove the entry to fall back to `MIN_TIME_LIMIT`.

### Calibrate `P` first (one-off, ~1 min)

`P` (cluster prefill throughput) is roughly **the same across ISL** on this setup
(8k/10k/100k all land near the same toks/sec), so one cheap probe is enough to
seed `dataset_prefill_tps` for every dataset. With the server from §1 up:

```bash
CAL="$RUN/calib"; mkdir -p "$CAL"
CAL_AT=$(wc -l < "$RUN/serve.log")          # mark where this window starts in the log

python3 benchmark_prefill/unit_test.py \
  --time_limit 60 --ignore_time_start 0 --ignore_time_end 0 --parallel_threads 16 \
  --data_path /remote/vast0/share-mv/longbenchv2-custom/longbenchv2-8k.jsonl \
  --log_file_name calib.log --port 8000 --encoding_size 8192 --output_path "$CAL"

tail -n +"$CAL_AT" "$RUN/serve.log" > "$CAL/serve_slice.log"
python3 benchmark_prefill/extract_prefill_running.py \
  "$CAL/serve_slice.log" "$CAL/pr" --scenarios 8k_p16
#   read "mean (active steps)" from the output -> that is P (toks/sec).
```

Put that number into `dataset_prefill_tps` (same value for 8k/10k/100k is fine).

> Note: `P` **rises with concurrency until the GPUs saturate** (here ≈12k toks/sec
> at conc 16 → ≈20–22k by conc ~96). Measuring at conc 16 is quick but gives the
> *low* end, so windows come out **generous/safe** (bigger than strictly needed).
> For tighter windows, calibrate at a saturating concurrency (e.g. `--parallel_threads 96`)
> and use that higher `P`.

**No warm-up by default (`IGNORE_FRACTION = 0`).** The whole window is measured.
This is fine for the headline `prefill_throughput`: workers jump to full
concurrency at t0 (no ramp) and `tokens/span` is self-normalizing. The only cost
is that the initial synchronized request wave (empty-queue → unrepresentatively
low TTFT) is included, so the **TTFT percentiles / `mean_prefill_tps` read
slightly optimistic**. If you need clean latency stats, set `IGNORE_FRACTION` > 0
to trim a warm-up/cool-down band.

`unit_test.py` controls OSL + data caps:
- **OSL = `OUTPUT_LEN`** (default `2`, override via env `PREFILL_BENCH_OSL`);
  never overridden per-dataset. Keep it ≥ 2 — OSL=1 crashes this preset (§6).
  TTFT/`completion_ts` are read from the first token, so the metric is unaffected
  by the exact OSL.
- Distinct prompts: 8k/10k → `data[:100]`, 100k → `data[:10]`, 1M → `data[:1]`.
- The payload `"model"` must match the served model path (currently FP8).
- **No warmup path** — the driver runs the formal window directly.

---

## 3. Run the bench

```bash
cd /workspace/vllm-moreh
nohup python3 -u benchmark_prefill/multi_process_test.py > "$RUN/mpt_run.log" 2>&1 &
echo $! > "$RUN/bench.pid.txt"
```

Workflow per dataset: for each concurrency `run_bench` (formal window sized by
`window_params()`, bracketed by `GET /v1/bench_start` … `/v1/bench_end` — they
404, the access-log line is the marker). **No prefix-cache reset and no warmup**
between scenarios — caching is disabled, so every bench measures cold prefill.
Results land in
`benchmark_prefill/results/longbenchv2-<ds>_p<conc>_<ts>/`.

Track progress / wait for completion:

```bash
grep -c "\[Bench\]"    "$RUN/mpt_run.log"    # benches started
grep -c "\[Summary\]"  "$RUN/mpt_run.log"    # benches finished
# block until the driver exits:
while kill -0 "$(cat "$RUN/bench.pid.txt")" 2>/dev/null; do sleep 30; done; echo DONE
```

Each `[Summary]` line carries the prefill metrics, e.g.:

```
[Summary] {"parallel_threads": 16, "encoding_size": 8192, "requests": 120,
  "mean_ttft_s": ..., "p90_ttft_s": ..., "p99_ttft_s": ...,
  "mean_prompt_tokens": ..., "total_prompt_tokens": ...,
  "window_span_s": ..., "mean_prefill_tps": ..., "prefill_throughput": ...}
```

- `prefill_throughput` (**headline**) = `total_prompt_tokens / window_span_s`,
  where `window_span_s = max(completion_ts) − min(send_ts)` = from when the first
  recorded request was *sent* to when the last recorded request got its *first
  token*. Every recorded request's prefill lies inside this span, so this is the
  system input-token throughput. (It is deliberately *not* `max−min` of
  completion times alone — that would drop the first request's own prefill from
  the denominator and inflate the number.)
- `mean_prefill_tps` = mean of per-request `prompt_tokens / TTFT`. Under
  concurrency, TTFT includes queue wait, so this reads *lower* than
  `prefill_throughput`; the two bracket the real per-request vs system view.
- `mean/p90/p99_ttft_s` = prefill latency percentiles.

---

## 4. Extract & summarize

Fresh server → serve.log holds only this run's `bench_start/end` windows, in
`[Bench]` order (dataset-outer, concurrency-inner). Derive labels from the
driver log, extract per-step **cluster prefill throughput** (per-rank prompt
throughput summed across DP ranks), then aggregate.

```bash
MIN_TS=$(stat -c %Y "$RUN/bench.pid.txt")   # scope to this run; excludes older result dirs

# labels in driver order: "[Bench] concurrency N, dataset D" -> D_pN
mapfile -t LABELS < <(grep -oP 'concurrency [0-9]+, dataset \S+' "$RUN/mpt_run.log" \
  | awk '{gsub(/,/,""); print $4"_p"$2}')

mkdir -p "$RUN/prefill_running"
python3 benchmark_prefill/extract_prefill_running.py \
  "$RUN/serve.log" "$RUN/prefill_running/prefill_running" --scenarios "${LABELS[@]}"

python3 benchmark_prefill/summarize_scenarios.py \
  benchmark_prefill/results "$RUN/scenario_summary.csv" \
  --totals-dir "$RUN/prefill_running" --min-ts "$MIN_TS"

cat "$RUN/scenario_summary.csv"
```

`scenario_summary.csv` → one row per scenario. Columns:

- Client-side: `requests`, `mean/p90/p99_ttft_ms`, `mean_prompt_tokens`,
  `total_prompt_tokens`, `window_span_s`, `mean_prefill_tps`,
  `prefill_throughput`.
- Server-side (from the serve log, corroborating): `server_mean_prefill_tps`
  (mean cluster prompt throughput over all steps), `server_active_mean_prefill_tps`
  (mean over prefill-active steps only), `server_max_prefill_tps`.

> The server-side `Avg prompt throughput` requires the preset to log every step
> (`VLLM_MOREH_SCHEDULER_LOGGING=1`, `VLLM_MOREH_SCHEDULER_LOG_INTERVAL=0` — the
> GLM-5.2 MTP dp8-ep8 preset already sets these). If it's off, only the
> client-side columns are populated.

---

## 5. Stop the server

```bash
SPID=$(cat "$RUN/serve.pid.txt")
kill -TERM -"$SPID" 2>/dev/null; sleep 5; kill -KILL -"$SPID" 2>/dev/null; sleep 3
rocm-smi --showmeminfo vram | grep "Used Memory"    # expect ~298 MB each
```

---

## 6. Gotchas

- **OSL=1 crashes this preset — keep `OUTPUT_LEN` ≥ 2.** With MTP +
  PD-separation + async scheduling, a `max_tokens=1` request finishes in the same
  step its prefill completes; the async scheduler then looks it up after it was
  freed → `KeyError` in `vllm/v1/core/sched/scheduler.py::_update_after_schedule`,
  killing every EngineCore (server-side `HTTP 500`s, then the driver spins on a
  dead server). Symptom in `serve.log`: `EngineCore encountered a fatal error` +
  `KeyError: 'chatcmpl-...'`. The bench defaults to OSL=2 for this reason.
- **8 free GPUs required.** If init dies with `WorkerProc initialization failed`,
  another container is holding the GPUs — re-check `rocm-smi`.
- **Transient GPU fault during CUDA-graph capture on startup.** Occasionally the
  server aborts mid-capture with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
  (`Capturing CUDA graphs (FULL): NN/32` then a GPU coredump). It is flaky — kill
  leftovers, wait for `rocm-smi` to return to idle (~298 MB/GPU), and just serve
  again.
- **Confirm prefix caching is OFF (CLI flag, not the preset).** The preset's
  `enable_prefix_caching: false` is silently ignored; pass
  `--no-enable-prefix-caching` on the serve command and verify per §0 (effective
  `enable_prefix_caching=False` **and** 0% hit rate). If left on, repeated prompts
  from the small prompt pool are served from cache and `prefill_throughput` is
  inflated / meaningless.
- **`mean_prefill_tps` < `prefill_throughput`** is expected under concurrency —
  the former divides by TTFT (which includes queue wait), the latter is the
  system aggregate. Report `prefill_throughput` as the prefill number.
- **`min-ts`** scopes the summary to one run — the `results/` folder is shared
  across runs; without it you'll pick up stale dirs (summarize also dedups
  (ISL, conc) by latest timestamp).
- **Long ISL × high concurrency**: a request is only recorded if it both starts
  and finishes inside the window, so the usable window is `(time_limit − TTFT)`;
  if TTFT approaches `time_limit`, the client-side in-window count collapses.
  `window_params()` sizes the window from the measured throughput so the first
  wave still fits (§2), so long-ISL / high-concurrency cells are just *slow*, not
  broken — 256 concurrent 100k prompts queue 25.6M tokens, so the window is ~27
  min by design. If a cell still records too few requests, the throughput table
  (`dataset_prefill_tps`) is likely stale — re-measure and update it. And note the
  **server-side** prefill throughput (§4) is read straight from the scheduler
  log, so it stays valid even when the client records 0 requests.
```
