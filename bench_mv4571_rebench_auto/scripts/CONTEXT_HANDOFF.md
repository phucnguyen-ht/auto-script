# CONTEXT HANDOFF — bench_mv4571_rebench_auto (GLM-5.2 EPLB bench)

Full context to resume in a new session. Repo:
`/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script` (branch `main`).
Ticket dir: `bench_mv4571_rebench_auto/`. Companion doc with the debug detail +
code refs: `scripts/DEBUG_ASYNC_HANG.md`.

---

## 1. What this ticket is
A **thin wrapper** that automates `benchmark_zhipu` (a custom python serving-bench
for GLM-5.2-FP8 dp8/ep8): **serve → run driver → extract metrics**. It does NOT
change the bench flow — it wraps it.
- `scripts/multi_process_test.py` — driver (dataset-outer × concurrency-inner sweep;
  reset prefix cache once/dataset; warmup 16 copies/prompt; 200s window). Edited only
  to: read sweep from env `REBENCH_CONC/DATASETS`, `unit_test_path` = beside itself,
  results dir = env `REBENCH_RESULTS_DIR`. Flow logic untouched.
- `scripts/unit_test.py` — load generator (hits `/v1/chat/completions`, `bench_start/end`
  markers). UNCHANGED. Payload `"model"` hardcoded `/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/`.
- `scripts/extract_decode_running.py`, `scripts/summarize_scenarios.py` — §4 extraction
  (decode batch size from serve.log; TTFT/TPOT/throughput from results/). UNCHANGED.
- `scripts/RUN_AND_BENCH.md` — the original manual runbook (§1 serve … §6 stop).
- Self-contained: everything lands under a `RUN` dir (serve.log, mpt_run.log, results/,
  decode_running/, scenario_summary.csv, prefix_cache_hit.txt).

## 2. How to run (all on GPU node, inside the container)
- **Node**: `ssh gpu-5` (host = slurm controller `aac11-slurm-controller-p`, NO GPU / NO
  `/workspace` / NO rocm-smi there — only read logs on the shared FS from the host).
  If access lost: `salloc -A ce6 -p 256C8G1H_MI325X_Ubuntu22 --reservation=gpu-5_gpu-6_reservation -w gpu-5 --exclusive --mem=0`. Node is **MI325X** (256 GiB/GPU).
- **Container**: `phuc-nguyen-mv4571-rebench`, image
  `255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.23.0-260626-rc1`
  (see `podman.sh`). Mounts the repo at the same absolute path, so ticket scripts/presets
  resolve identically inside.
- **One preset, end-to-end** (serve→bench→extract→stop):
  ```
  ssh gpu-5 "podman exec phuc-nguyen-mv4571-rebench bash -lc '
    PRESET=<abs preset.yaml> RUN=<abs run dir> bash <ticket>/scripts/run_and_bench.sh'"
  ```
  `run_and_bench.sh` env: MODEL, PRESET, PORT(8000), RUN, SERVE_SH(=1→repo serve.sh instead
  of `vllm-moreh serve`), KEEP_SERVER, SERVER_WAIT_TIMEOUT(3600). Serves via
  `nohup vllm-moreh serve MODEL PRESET`, waits /health, runs driver, §4 extract, §6 stop.
- **Full sweep**: `scripts/sweep_presets.sh` (loops the PRESETS list, `wait_gpu_free`
  between, runs run_and_bench per preset). Run it the same way (podman exec).
- **Manual verbatim RUN_AND_BENCH** (repo-independent): see `RUN_AND_BENCH.md` — but its
  `presets/full/…` path does NOT exist here; use `presets/glm5.2.rebench/…` + the ticket
  `scripts/*.py`.
- Preset family = **`presets/glm5.2.rebench/`**; base `MTP5-bs64-dg.yaml` is **1P1D**
  (`VLLM_MOREH_USE_PD_SEPARATION=1` + `scheduler_cls=vllm_moreh.sched.pds_scheduler.PDSLoggingScheduler`).
  Base also sets `VLLM_MOREH_SCHEDULER_LOGGING=1`/`_LOG_INTERVAL=0` (decode extract) and
  `VLLM_SERVER_DEV_MODE=1` (reset_prefix_cache). `vllm-moreh serve` DOES read these env_vars.

## 3. THE BIG PROBLEM (goal) + resolution
Symptom (sweep `logs/sweep/20260702_165527`): with the 260626 image + 1P1D, EPLB
**async** `torch_nccl`/`pynccl` deadlock even at default interval →
`RPC call to sample_tokens timed out` → engine dies. And `nixl` async failed engine init.

**Root cause (async nccl/pynccl) — vLLM limitation, code refs** (`/usr/local/lib/python3.12/dist-packages/vllm/`):
- `distributed/eplb/async_worker.py:transfer_run_periodically` — a background thread
  issues the expert-transfer over the eplb ProcessGroup.
- `distributed/eplb/eplb_communicator.py` `TorchDistNcclEplbCommunicator.execute()` (~L146):
  `batch_isend_irecv` + `req.wait()` = **NCCL P2P on the GPU device_group from the bg thread**,
  concurrent with the main decode thread's NCCL collectives on the SAME communicator.
- NCCL needs all ranks to post ops in the same order → the bg thread breaks it → hang →
  `v1/executor/multiproc_executor.py:389` raises the RPC timeout after
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default **300s**, `envs.py:210`).
- **vLLM's own hint**: `eplb_state.py:240-243`: *"all EP ranks need the same
  `expert_rearrangement_step` … Otherwise, the rearrangement will hang at collective
  communication calls."* Not fixable by config — vLLM built `nixl` (UCX RDMA, no collective)
  and `torch_gloo` (CPU-staged, `cpu_group`) as the non-collective async transports.

**nixl engine-init fail**: I had set `kv_cache_memory_bytes=40 GiB`, but 1M max_model_len
needs 54.62 GiB for one request → `ValueError`. Fixed → **60 GiB**.

**4 EPLB communicators** (factory `eplb_communicator.py:618`): `torch_nccl`, `pynccl`,
`nixl`, `torch_gloo`. Async-viable = **nixl, torch_gloo** (off the NCCL group).

## 4. Verified results (bench succeeds — spot-checks on gpu-5)
| config | log dir | req | p50_tpot | note |
|---|---|---|---|---|
| pynccl **SYNC** | `logs/debug/pynccl-sync-run1/` | 491 | 18.9 | EPLB works sync |
| **gloo ASYNC** | `logs/debug/gloo-async-run1/` | 646 | 18.9 | 24 async transfers, no hang |
| **nixl ASYNC** (kv 60GiB) | `logs/debug/nixl-async-run1/` | 537 | 18.0 | 24 transfers, KV fix confirmed |
| nccl/pynccl **ASYNC** | (165527) | — | — | ❌ RPC-timeout deadlock (vLLM limit) |
All viable runs: no RPC timeout, no worker death, prefix ~99.5%, §6 cleanup OK.

## 5. Changes made (files)
- `gen_eplb_presets.sh` — generates the **viable** matrix: nixl/gloo=ASYNC, nccl/pynccl=SYNC;
  ×{default,s250}×{r0,r8,r16}=24; nixl gets UCX env + `kv_cache_memory_bytes=60GiB`.
  Run `bash gen_eplb_presets.sh` to (re)create the 24 preset files.
- `scripts/sweep_presets.sh` — PRESETS list = base + those 24 (viable only).
- `scripts/run_and_bench.sh` — **§6 fix**: added `pkill -9 -f "vllm-moreh serve"` (old code
  only `pkill -9 VLLM` which misses the main serve proc → lingered, would break the sweep).
- `scripts/recreate_container.sh` — recreate the container with **vllm + vllm_moreh source
  mounted editable** (copied out of the running container) for source-level patching. Run on gpu-5.
- Presets: `presets/glm5.2.rebench/MTP5-bs64-dg-eplb-{nixl,gloo}-async-*` +
  `-{nccl,pynccl}-sync-*` (24). base `MTP5-bs64-dg.yaml` unchanged (1P1D).
- Docs: `scripts/DEBUG_ASYNC_HANG.md` (step-by-step + root cause + code + result links),
  this `scripts/CONTEXT_HANDOFF.md`.

## 6. FULL SWEEP — LAUNCHED & RUNNING
Launched detached on gpu-5 (survives sessions):
`SWEEP_ROOT=logs/sweep/fullsweep_20260702_151319` (log: `${SWEEP_ROOT}/sweep.log`).
Command used: `ssh gpu-5 "podman exec -d phuc-nguyen-mv4571-rebench bash -lc 'SWEEP_ROOT=<...> bash <ticket>/scripts/sweep_presets.sh > <...>/sweep.log 2>&1'"`
(**note `podman exec -d`** — plain `nohup &` inside `podman exec` gets reaped when the
exec session ends; `-d` detaches properly.)
- 25 presets, ~12-14 min each → ~5-6h total. Per-preset results:
  `${SWEEP_ROOT}/<preset>/scenario_summary.csv`.
- Check progress: `tail -f ${SWEEP_ROOT}/sweep.log`; done when it prints `[sweep] DONE`.
- Re-launch after a stop the same way (fresh SWEEP_ROOT). To stop:
  `podman exec phuc-nguyen-mv4571-rebench bash -lc 'pkill -f sweep_presets; pkill -f run_and_bench; pkill -9 -f "vllm-moreh serve"; pkill -9 VLLM'`.
- nccl/pynccl **async** is intentionally excluded (unfixable). If someone insists on async
  for them, expect the RPC-timeout hang documented above.

## 7. Gotchas
- `rocm-smi` works only INSIDE the container (host returns empty). `wait_gpu_free` runs in
  the container so it's fine. Format: `GPU[N] : GPU Memory Allocated (VRAM%): X`.
- Leftover `vllm-moreh serve` (comm=python3) can linger if killed with only `pkill VLLM`
  → GPU busy → next serve OOM. §6 fix handles it; if a run is stuck, on gpu-5:
  `podman exec phuc-nguyen-mv4571-rebench bash -lc 'pkill -9 -f "vllm-moreh serve"; pkill -9 VLLM'`.
- `mean_tpot_ms` is noisy run-to-run (client SSE tail + prefill congestion at 100k×conc64 /
  200s window); trust **`p50_tpot`, `mean_decode_tps`, `mean_batch`** (stable ~18-19 / ~50 / ~34).
- Workload = env.yaml `[64]×[100k]` (concurrency×dataset). 100k×conc64 in a 200s window is
  the noisy "long-ISL × high-conc" regime (RUN_AND_BENCH §7); raise `time_limit` in
  `multi_process_test.py` for tighter stats if needed.
- To debug the framework further: `bash scripts/recreate_container.sh` on gpu-5 → editable
  `bench_mv4571_rebench_auto/vllm_src/{vllm,vllm_moreh}` mounted into a `-dev` container.
