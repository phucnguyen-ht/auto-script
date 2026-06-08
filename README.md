# Auto Scripts

Automated VLLM profiling and benchmarking across different cache-hit levels and index-cache configurations.


## Run docker
```
bash docker.sh <CONTAINER_NAME>
```
## Directory structure

```
customer-poc-delivery/
├── README.md
├── env.yaml                # Shared config: model path, profiler flags, eval datasets
├── serve.sh                # VLLM server startup script (shared by profile/, bench/, eval/)
├── presets/                # Shared VLLM preset yamls
│   ├── *-0ic-profile.yaml  # Index-cache disabled (VLLM_MOREH_INDEX_CACHE_ENABLED=0)
│   └── *-50ic-profile.yaml # Index-cache enabled  (VLLM_MOREH_INDEX_CACHE_ENABLED=1)
├── profile/
│   ├── auto_profile.sh     # Orchestrates all profiling runs
│   ├── profile_0cachehit/  # bench.sh + profiling_result/ for 0% cache-hit
│   ├── profile_40cachehit/
│   └── profile_90cachehit/
├── bench/
│   ├── auto_bench.sh       # Orchestrates all benchmark runs
│   ├── benchmark_pd_tps.py
│   └── flops_calculator.py
└── eval/
    ├── auto_eval.sh        # Orchestrates all eval runs
    ├── gsm8k.sh
    ├── longbench.sh
    ├── longbench2.sh
    └── mmlu.sh
```

## Configuration (`env.yaml`)

```yaml
model:
  path: /share-mv/zai-org/GLM-5-FP8

profiler:
  TORCH_PROFILER_WITH_STACK: "False"
  TORCH_PROFILER_RECORD_SHAPES: "False"
  TORCH_PROFILER_WITH_MEMORY: "False"
  TORCH_PROFILER_WITH_FLOPS: "False"

eval:
  datasets:
    longbench: 1
    longbench2: 1
    mmlu: 1
    gsm8k: 1
```

All scripts (`auto_profile.sh`, `auto_bench.sh`, `auto_eval.sh`) read values from `env.yaml` at startup via `yq`.

## Presets

The two presets in `presets/` differ only in index-cache configuration:

| Preset | `VLLM_MOREH_INDEX_CACHE_ENABLED` | `hf_overrides` |
|---|---|---|
| `*-0ic-profile.yaml` | `"0"` | disabled |
| `*-50ic-profile.yaml` | `"1"` | enabled |

`no_enable_prefix_caching` is **not** stored in the preset files — it is injected at runtime:

- **cache-hit = 0%** → `no_enable_prefix_caching: true`
- **cache-hit > 0%** → `no_enable_prefix_caching: false`

---

## Profiling (`profile/auto_profile.sh`)

### Run all 8 profiles

Uncomment all `run_profile` lines in `auto_profile.sh`:

```bash
# --- CACHEHIT=0 ---
run_profile profile_0cachehit  0  2
run_profile profile_0cachehit 50  2
run_profile profile_0cachehit  0  3
run_profile profile_0cachehit 50  3

# --- CACHEHIT=40 ---
run_profile profile_40cachehit  0  4
run_profile profile_40cachehit 50  4

# --- CACHEHIT=90 ---
run_profile profile_90cachehit  0  12
run_profile profile_90cachehit 50  12
```

Then run:

```bash
cd profile
bash auto_profile.sh
```

### Run a specific profile

```
run_profile <PROFILE_DIR> <IC_SUFFIX> <CONC>
```

| Argument | Description | Values |
|---|---|---|
| `PROFILE_DIR` | Profile subdirectory | `profile_0cachehit`, `profile_40cachehit`, `profile_90cachehit` |
| `IC_SUFFIX` | Index-cache value | `0` or `50` |
| `CONC` | Concurrency | `2`, `3`, `4`, `12` |

### Profiling output

| File | Location |
|---|---|
| Serve log | `profile/logs/auto_profile/serve_<label>_<ts>.log` |
| Bench log | `profile/logs/auto_profile/bench_<label>_<ts>.log` |
| Torch profiler traces | `profile/<profile_dir>/profiling_result/conc_<CONC>/<IC>/` |

`profiler_config` (torch profiler flags, output dir) is assembled from `env.yaml` and injected into the preset at runtime — do not edit it in the preset files.

---

## Benchmarking (`bench/auto_bench.sh`)

### Run all 6 benchmarks

```bash
cd bench
bash auto_bench.sh
```

All 6 cases run sequentially (server is started and killed between each):

| PREFIX_CACHE_HIT | INDEX_CACHE | Batch sizes |
|---|---|---|
| 0 | 0, 50 | 1..10 |
| 40 | 0, 50 | 1..30 |
| 90 | 0, 50 | 1..30 |

### Run a specific benchmark

Comment/uncomment the relevant `run_bench` lines in `auto_bench.sh`:

```
run_bench <PREFIX_CACHE_HIT> <INDEX_CACHE>
```

### Benchmark output

| File | Location |
|---|---|
| Serve log | `bench/logs/auto_bench/serve_<label>_<ts>.log` |
| Bench log | `bench/logs/auto_bench/bench_<label>_<ts>.log` |
| Results (JSON) | `bench/logs/auto_bench/prefix_cache_hit_<N>/index_cache_<N>/results/` |

---

## Evaluation (`eval/auto_eval.sh`)

### Run all evals

```bash
cd eval
bash auto_eval.sh
```

For each IC preset (`0ic`, `50ic`) × each dataset × each run, the script:
**starts server → runs `lm_eval` → kills server** (fresh server per run, even within repeated runs of the same dataset).

Datasets and number of runs are configured in `env.yaml`:

```yaml
eval:
  datasets:
    longbench: 1   # runs lm_eval task: longbench_single
    longbench2: 1  # runs lm_eval task: longbench2_single
    mmlu: 1        # runs lm_eval task: mmlu
    gsm8k: 1       # runs lm_eval task: gsm8k
```

### Run a specific eval

Comment/uncomment the relevant inner loop body or call `run_eval` directly:

```
run_eval <IC_SUFFIX> <DATASET> <RUN_IDX>
```

### Eval output

| File | Location |
|---|---|
| Serve log | `eval/logs/auto_eval/serve_<label>_<ts>.log` |
| lm_eval log | `eval/logs/auto_eval/eval_<label>_<ts>.log` |
| Results (JSON) | `eval/logs/auto_eval/ic<IC>/<dataset>/run<N>/` |
