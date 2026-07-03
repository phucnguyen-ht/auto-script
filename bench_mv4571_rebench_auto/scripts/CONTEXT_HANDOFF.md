# CONTEXT HANDOFF — bench_mv4571_rebench_auto (GLM-5.2 EPLB bench)

> **⚠️ READ THIS FIRST — SESSION HANDOFF for the MI300 run (2026-07-02, current).**
> The body below (§1..§7) is the ORIGINAL **MI325** handoff. This top block is the
> **MI300** port. Blow-by-blow detail: **`scripts/progress.md`** (read it — it has every
> step, bug, fix, and log path).

## MI300 HANDOFF (current task)

### Environment (differs from MI325 body below)
- Node: **MI300, gfx942, 8× GPU, 192 GiB/GPU** (206,141,652,992 B). **NO RDMA NIC**
  (`ibv_devices` empty) — matters for nixl. **Shared node** — colleagues' containers
  (e.g. `thanh-nguyenxuan-glm5-dev`) run big jobs here; expect GPU contention.
- **docker** (not podman). Container **`phuc-nguyen-mv-4571`**, image `moreh-vllm:0.23.0-260626-rc1`.
  Repo mounted at same path: `/home/phuc-nguyen/workspaces/mv-4571-rebench/auto-script`.
- Run one preset: `docker exec phuc-nguyen-mv-4571 bash -lc "RUN=<abs> PRESET=<abs.yaml> bash <scripts>/run_and_bench.sh"`.
  Launch detached (survives session): add `-d` to `docker exec` OR use a background runner.
- Workload unchanged: 100K ISL × concurrency 64 (`multi_process_test.py` `[64]×[100k]`, 240s window).

### What was found on MI300 (evidence in progress.md; bench2 old logs in `logs/sweep2`)
Same image as MI325 ⇒ the async **deadlock** for nccl/pynccl is identical ⇒ **nccl/pynccl → SYNC**
carries over. NEW MI300-specific issues (all measured):
1. **max_model_len=1M does NOT fit with EPLB.** Baseline auto-KV = **59.1 GiB** (fits 1M, 1.08x).
   Enabling EPLB costs ~10 GiB (rearrange buffers): avail KV drops to **~49 GiB (r0) .. 43.1 GiB (r16)**
   < 54.62 GiB needed for a 1M request → engine init `ValueError`. **Fix: cap `max_model_len=512K`**
   for all EPLB presets (workload is 100K ISL, so benched requests are unaffected). Baseline keeps 1M.
2. **nixl is NOT viable on this node (no RDMA).** With the UCX fix (`UCX_TLS` +`tcp`) nixl *inits*
   fine, but the **first async transfer CRASHES the engine** (GPU-mem RDMA-read over tcp). On an
   RDMA node (MI325) nixl works. ⇒ nixl is **hardware-gated** (see next).
3. **gloo async works** (r16 probe fired 21 transfers, no crash). gloo is the async backend on MI300.
4. Model load: 107.63 GiB (baseline) / 108.76 (r0) / 114.17 (r16, +16 redundant experts).

### The final preset set (generator + values)
`gen_eplb_presets.sh` now emits MI300-safe values (env-overridable for MI325):
- `MAX_MODEL_LEN=524288` (512K) on ALL EPLB presets. `NIXL_KV_CACHE_BYTES=40 GiB` (fits r16;
  nixl KV is fixed). nixl `UCX_TLS` includes `tcp`. Regenerate: `bash bench_mv4571_rebench_auto/gen_eplb_presets.sh`.
- **MI325 (RDMA, 256 GiB)**: `MAX_MODEL_LEN= NIXL_KV_CACHE_BYTES=64424509440 bash gen_eplb_presets.sh`
  (empty MAX_MODEL_LEN keeps 1M; 60 GiB KV for nixl).

### nixl auto-check (added this session, per user request)
`scripts/check_nixl.sh` → exit 0 if node has RDMA (nixl viable), else exit 1. `sweep_presets.sh`
calls it and **auto-adds the 6 nixl presets only on RDMA nodes** (skipped here). Override: `NIXL_FORCE=1/0`.
So the SAME sweep script runs nixl on the user's other (RDMA) machine and skips it here — no edits needed.

### `sweep_presets.sh` — final state
Outputs to **`logs/sweep_results/<ts>/`** (per user: final results go in `sweep_results/`, not `sweep/`).
PRESETS = baseline(1M) + **gloo-async + nccl-sync + pynccl-sync** × {default,s250} × {r0,r8,r16}
= **19** (+ 6 nixl auto-added iff RDMA). `wait_gpu_free` between presets (VRAM>10% ⇒ wait — good for the
shared node). `run_and_bench.sh` §6 kills `vllm-moreh serve` + `VLLM` (container-local pkill).

### STATE / WHAT'S LEFT (resume here)
- ✅ Diagnosis done, presets + scripts finalized, nixl-check wired, docs written.
- ⏳ **BLOCKER: GPUs 92% busy** (colleague's job) since ~21:47 — could not get a *clean* throughput run.
  Contaminated runs: `logs/mi300_probe/gloo-async-default-r0-512k` (stuck in cudagraph @100s/graph = contention).
- **TODO when GPUs free** (`rocm-smi --showmemuse` all <~10%):
  1. Clean confirm ONE gloo run: `RUN=.../mi300_probe/confirm-gloo-r0 PRESET=.../MTP5-bs64-dg-eplb-gloo-async-default-r0.yaml bash .../run_and_bench.sh`
     — expect ~hundreds of req, prefix ~99% (like baseline 510 req / p50_tpot 21.86). Also confirm one `pynccl-sync-default-r0`.
  2. Launch full sweep detached:
     `docker exec -d phuc-nguyen-mv-4571 bash -lc 'SWEEP_ROOT=<abs>/logs/sweep_results/<ts> bash <scripts>/sweep_presets.sh > <that>/sweep.log 2>&1'`
     then `tail -f .../sweep.log` (done at `[sweep] DONE`). Per-preset: `<ts>/<preset>/scenario_summary.csv`.
- Stop a stuck run: `docker exec phuc-nguyen-mv-4571 bash -lc 'pkill -f sweep_presets; pkill -f run_and_bench; pkill -9 -f "vllm-moreh serve"; pkill -9 VLLM'`.
- Verified-good reference: baseline on MI300 = `logs/sweep2/20260702_172743/MTP5-bs64-dg/` (510 req, p50_tpot 21.86).

---

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

- **vLLM says so explicitly** (definitive): `config/parallel.py:918-924` auto-select comment:
  *"Avoid torch_nccl: NCCL is fundamentally incompatible with async EPLB due to multi-stream
  conflicts, and batched isend/irecv hangs under high load. See
  https://github.com/pytorch/pytorch/issues/174288. Prefer nixl … fall back to torch_gloo."*
  Docstring `parallel.py:96`: default None → *"torch_gloo for async, torch_nccl for sync"*.
  So DEFAULT (`communicator=None`, `use_async=True`) NEVER picks torch_nccl/pynccl — our failing
  presets FORCED that combo. Also `eplb_state.py:240-243`: mismatched `expert_rearrangement_step`
  across ranks → "will hang at collective communication calls."
  Upstream: pytorch/pytorch#174288 (`batch_isend_irecv`+NCCL hangs under high load — the exact
  call `TorchDistNcclEplbCommunicator.execute` makes), #108378 (NCCL isend blocks w/o matching irecv).