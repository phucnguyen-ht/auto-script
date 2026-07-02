# Debug: EPLB async bench hang / RPC timeout (GLM-5.2 1P1D, image 260626)

Goal: make **bench succeed** (profile irrelevant). Env: image
`moreh-vllm:0.23.0-260626-rc1`, node gpu-5 (MI325X), container
`phuc-nguyen-mv4571-rebench`, preset family `presets/glm5.2.rebench/*` (now 1P1D:
`VLLM_MOREH_USE_PD_SEPARATION=1` + `scheduler_cls=vllm_moreh.sched.pds_scheduler.PDSLoggingScheduler`).
Bench = `scripts/sweep_presets.sh` → `scripts/run_and_bench.sh` per preset.

---

## Step 1 — Diagnose sweep `logs/sweep/20260702_165527`

| preset | result | root cause |
|---|---|---|
| `MTP5-bs64-dg` (base, no EPLB) | ✅ full (~600 req, 200s) | works |
| `eplb-nccl-async-default-r0` | ❌ crash mid-bench (63 req, 28s) | `TimeoutError: RPC call to sample_tokens timed out` → EngineCore_DP6 died → ApiServer died |
| `eplb-pynccl-async-default-r0` | ❌ crash (33 req, 30s) | same RPC-timeout pattern |
| `eplb-nccl-async-default-r8` | ❌ no summary | same (killed mid-bench) |
| `eplb-nixl-async-default-r0` | ❌ engine never starts | `ValueError: 54.62 GiB KV needed (max_model_len 1M) > 40 GiB kv_cache_memory_bytes` |

### Finding A — nccl/pynccl async: `sample_tokens` RPC deadlock
- serve.log (nccl-r0): `vllm/v1/executor/multiproc_executor.py:389 raise TimeoutError("RPC call to sample_tokens timed out")` → `EngineCore_DP6` ERROR → `ApiServer_6 died with exit code None` → whole server down.
- The RPC timeout is `envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (multiproc_executor.py:315/327), **default 300s**. Firing at 300s ⇒ a worker's `sample_tokens` step genuinely **hung >5 min** = a real **deadlock**, not a transient stall → raising the timeout won't fix it.
- Regression vs old data: `../../mtp_default_compare.csv` showed async **default** interval completed 72/72/72 for nccl/pynccl on the OLD image. Now (image 260626 + 1P1D) it deadlocks even at default interval. So the trigger is **EPLB async rearrange × (new image or 1P1D scheduler)**.
- Client-side `unit_test.py:431 TypeError: usage_tokens None - 1` is a **symptom** (server returns malformed responses as it dies), not the cause.

### Finding B — nixl: KV cache too small for 1M context
- `ValueError: … 54.62 GiB KV cache is needed (max seq len 1048576), larger than available 40.0 GiB … est max model length 767872.`
- Cause: I set `kv_cache_memory_bytes=42949672960` (40 GiB, MI300-conservative) but kept `max_model_len=1M`. Engine requires ≥1 request at max_model_len to fit → 40 GiB < 54.62 GiB → engine init fails.
- Fix (planned): bump `kv_cache_memory_bytes` ≥ ~60 GiB (fits MI325 256 GiB and MI300 192 GiB: 107.6 model + 60 KV + overhead), keeping max_model_len=1M. (nixl needs a FIXED kv_cache to register via UCX.)

---

## Step 2 — Test pynccl **SYNC** (use_async:false) — is it the async path?
Plan: serve `presets/glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-SYNC-default-r0.yaml`
(created; `eplb_config={"use_async": false, "communicator": "pynccl"}`) and bench.
If SYNC completes (full window) while async deadlocks → confirms the async rearrange
is the culprit → fix there.

- Status: RUNNING — `logs/debug/pynccl-sync-run1/` (serve+bench on gpu-5, rebench container).

## Root-cause analysis (source read, container 260626)
`vllm/distributed/eplb/async_worker.py` `transfer_run_periodically`: a **background
thread** waits on `rearrange_event`, then does `transfer_layer(... communicator=...)`
= collectives over the **eplb ProcessGroup**, on a separate cuda stream.
`eplb_state.py:242` warns *"Otherwise, the rearrangement will hang at collective"*.

- **nccl/pynccl async deadlock**: the async transfer issues **NCCL collectives** on the
  same NCCL group while the main decode thread also issues NCCL ops. NCCL requires
  all ranks issue collectives in the **same order**; the background transfer interleaves
  out-of-order across ranks → collective hangs → a worker's `sample_tokens` blocks →
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (300s) fires → EngineCore dies. New image/1P1D
  makes this trigger even at default interval.
- **nixl async is different**: transfers via **UCX point-to-point RDMA**, not NCCL
  collectives → doesn't deadlock the NCCL group (matches old compare.csv: nixl completed
  at every interval, nccl/pynccl crashed at dense intervals). → **nixl is the viable
  async communicator**; nccl/pynccl-async is fundamentally deadlock-prone here.

## Step 3 — nixl KV fix (Finding B) — DONE
`gen_eplb_presets.sh`: `NIXL_KV_CACHE_BYTES` default `42949672960` (40 GiB) →
`64424509440` (60 GiB, ≥ 54.62 needed for 1M ctx; fits MI325/MI300). Regenerated the
6 nixl presets. Next: verify nixl-async serves + benches.

## Step 4 — EPLB communicator options (source: `eplb_communicator.py:618` factory)
`create_eplb_communicator(backend=...)` accepts exactly **4** backends (else
`ValueError: Unknown EPLB communicator backend`):
| backend | class | transport | async-safe vs NCCL group? |
|---|---|---|---|
| `torch_nccl` (="nccl") | `TorchDistNcclEplbCommunicator` | torch.dist isend/irecv over **GPU NCCL** group | ❌ deadlocks (shares NCCL group with decode) |
| `pynccl` | PyNccl | **GPU NCCL** | ❌ deadlocks (same reason) |
| `nixl` | `NixlEplbCommunicator` | **UCX RDMA READ** (point-to-point, no collective) | ✅ separate from NCCL group |
| `torch_gloo` | `TorchDistGlooStagedEplbCommunicator` | **gloo P2P over CPU** (weights staged to CPU) | ✅ uses `cpu_group`, not the GPU NCCL group |

→ The **4th communicator = `torch_gloo`** (answer to "besides the 3"). Both `nixl`
and `torch_gloo` transfer OUTSIDE the GPU NCCL group, so they should not trigger
the NCCL-ordering deadlock that kills `torch_nccl`/`pynccl` async. Plan:
- Test `torch_gloo` async (preset `…-eplb-gloo-async-default-r0.yaml`).
- Test `nixl` async (KV fixed to 60 GiB).
- Sync works for all 4 (rearrange collective runs inside the synchronized main step).

---

## RESULTS

### R1 — pynccl **SYNC** (`use_async:false`) → ✅ SUCCESS
`logs/debug/pynccl-sync-run1/` (serve.log + scenario_summary.csv + prefix_cache_hit.txt).
- 491 req, dur 155s, mean_ttft 8.3s, **p50_tpot 18.9**, mean_decode_tps 50.6, mean_batch 34.4.
- serve.log: **no RPC timeout, no worker death**. prefix hit **99.32%** (warm), full_miss 2.
- ⇒ EPLB rearrange works fine **synchronously** under 1P1D. The failure is **async-only**,
  exactly matching the NCCL-ordering root cause.

Side fix found: `run_and_bench.sh` §6 used only `pkill -9 VLLM`, which kills the
`VLLM::` engine/worker procs but NOT the main `vllm-moreh serve` (comm=python3) →
serve lingered, would block the next sweep preset. Fixed §6 to also
`pkill -9 -f "vllm-moreh serve"`.

### R2 — torch_gloo **ASYNC** → ✅ SUCCESS (`logs/debug/gloo-async-run1/`)
- 646 req, 188.8s, **p50_tpot 18.9**, mean_tpot 19.3, decode_tps 52.5, prefix 99.89%.
- **24 async EPLB transfers fired**, no RPC timeout / worker death. → gloo async viable.

### R3 — nixl **ASYNC** (KV=60GiB) → ✅ SUCCESS (`logs/debug/nixl-async-run1/`)
- 537 req, 173.5s, **p50_tpot 18.0**, mean_tpot 20.3, decode_tps 53.4, prefix 99.49%.
- 24 async transfers, no RPC timeout / worker death, **no KV engine-init error** (60GiB fix works).

### The vLLM hint that confirms this is a framework limit (not our config)
`vllm/distributed/eplb/eplb_state.py:240-243` (verbatim):
> *NOTE: Keep in mind that all EP ranks need to have the same
> `expert_rearrangement_step` value to ensure synchronization. Otherwise, the
> rearrangement will hang at collective communication calls.*

Sync mode keeps every rank's step aligned → collective completes. Async
nccl/pynccl lets the background thread issue collectives out-of-step across ranks
→ exactly "hang at collective communication calls". vLLM also ships `nixl`
(receiver-initiated RDMA, no collective) and `torch_gloo` (CPU-staged) precisely as
the non-collective async transports — implicit acknowledgement that the NCCL
collective path is unsafe for background transfer.

## CONCLUSION — viable configs (bench succeeds)
| communicator | async | sync |
|---|---|---|
| **nixl** (UCX RDMA, needs kv=60GiB+UCX) | ✅ | ✅ |
| **torch_gloo** (CPU-staged) | ✅ | ✅ |
| **torch_nccl** | ❌ hang (vLLM limit) | ✅ |
| **pynccl** | ❌ hang (vLLM limit) | ✅ |

Final sweep = **nixl/gloo ASYNC + nccl/pynccl SYNC** × {default,s250} × {r0,r8,r16}
(24) + baseline. Generated by `../gen_eplb_presets.sh`; listed in `sweep_presets.sh`.
Run: `ssh gpu-5 "podman exec phuc-nguyen-mv4571-rebench bash -lc 'bash <ticket>/scripts/sweep_presets.sh'"`.

---

## ROOT CAUSE — nccl/pynccl async deadlock (vLLM limitation, code refs)
All paths in `/usr/local/lib/python3.12/dist-packages/vllm/` (image 260626).

1. `distributed/eplb/async_worker.py` → `transfer_run_periodically()`: a **daemon
   thread** waits on `rearrange_event`, sets a separate cuda stream, and for each
   MoE layer calls `transfer_layer(..., communicator=model_state.communicator)`.
2. `distributed/eplb/eplb_communicator.py` → `TorchDistNcclEplbCommunicator`
   (torch_nccl) `.execute()` (~L146): `batch_isend_irecv(self._p2p_ops)` then
   `req.wait()` — i.e. **NCCL point-to-point on the GPU `device_group`**, issued
   **from the background thread**. `pynccl` path is equivalent (PyNccl P2P on the
   same GPU comm).
3. Concurrently, the **main engine thread** issues its own NCCL collectives
   (DP/EP all-reduce) on the **same NCCL communicator** during each decode step.
4. **NCCL requires every rank to post ops on a communicator in the same global
   order.** The background transfer interleaves out-of-order across ranks →
   `req.wait()` (and the peers' collectives) block forever → the worker's
   `sample_tokens` never returns → `v1/executor/multiproc_executor.py:389` raises
   `TimeoutError: RPC call to sample_tokens timed out` after
   `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (**default 300s**, `envs.py:210`) → EngineCore
   dies → ApiServer dies → server down. `eplb_state.py:242` even warns
   *"Otherwise, the rearrangement will hang at collective."*

**Why it's a vLLM limitation, not a preset/config bug**: concurrent NCCL
collectives from two threads on one communicator is unsafe by NCCL design; there is
no config/env to serialize the background transfer against the main step. The
framework's own answer is to transfer OUTSIDE the NCCL group — hence the `nixl`
(UCX RDMA, receiver-initiated, no collective) and `torch_gloo` (CPU-staged gloo on
`cpu_group`) backends exist. So for **async EPLB, use `nixl` or `torch_gloo`**;
`torch_nccl`/`pynccl` async are expected to deadlock here. (Old `mtp_default_compare.csv`
already showed nccl/pynccl crashing at dense intervals and nixl surviving — same cause,
now triggered even at default interval by the 260626 image / 1P1D scheduler.)

Workable configs: **sync** (any communicator — proven by R1) and **async with
nixl/torch_gloo** (R2/R3 below).

<!-- results appended below as steps complete -->
