# NIXL EPLB communicator on ROCm/MI300 — bug, root cause, and fix

**TL;DR** — vLLM's NIXL EPLB communicator (the default when `enable_eplb: true` is set with no explicit
`communicator`) crashed at init on this ROCm/MI300 build with `NIXL EPLB init failed: buffers`. It was
**NOT** a missing library or a vLLM/rixl bug — it was a **missing UCX environment setting**: the container
ships `UCX_TLS=self,sm,rc_x`, which excludes every ROCm GPU transport, so UCX never loads its ROCm
memory-type detector and misclassifies GPU (VRAM) pointers as host memory, causing NIXL's VRAM buffer
registration to fail. Setting **`UCX_TLS=self,sm,rc_x,rocm_copy,rocm_ipc`** (+ `UCX_MEMTYPE_CACHE=n`)
fixes it — confirmed: `Initialized EPLB communicator: NixlEplbCommunicator.` on all 8 workers + a real
NIXL rearrange.

---

## 1. The bug — what failed and where (the assertion / error chain)

When EPLB is enabled without an explicit communicator, this build resolves the backend to **NIXL**
(`config/parallel.py:907-924`: `enable_eplb and communicator is None and not elastic and is_nixl_available()`
→ `"nixl"`). Serving then dies during engine init on **every** DP worker. The exact chain (from the
`-eplb-freq-nixl` serve.log, run_20260630_195010):

```
[nixl_utils.py:32]  Setting UCX_RCACHE_MAX_UNRELEASED to '1024' ...   # NIXL/rixl IS present
[nixl_utils.py:68]  NIXL is available
[_api.py:361]       Backend UCX was instantiated                       # UCX backend loads
[_api.py:251]       Initialized NIXL agent: eplb-0-8257a099            # agent init OK
E [ucx_utils.cpp:592]  VRAM memory is detected as host by UCX. UCX is likely not configured
                       with CUDA support. VRAM registration cannot proceed.        <-- THE REAL CAUSE
E [nixl_agent.cpp:484] registerMem: registration failed for the specified or all potential backends
Traceback (most recent call last):
  File ".../vllm/distributed/eplb/eplb_communicator.py", line 424, in _init_registered_buffers
    self._nixl_wrapper.register_memory(descs)
  File ".../rixl/_api.py", line 384, in register_memory
    self.agent.registerMem(reg_descs, handle_list)
rixl._bindings.nixlBackendError: NIXL_ERR_BACKEND
  ... wrapped by ...
  File ".../vllm/distributed/eplb/eplb_communicator.py", line 319, in _init_step
    raise RuntimeError(f"NIXL EPLB init failed: {name}") from exc
RuntimeError: NIXL EPLB init failed: buffers
  ... then ...
  File ".../vllm/v1/worker/gpu/eplb_utils.py", line 111, in maybe_register_model
    ...
RuntimeError: Failed to initialize NixlEplbCommunicator (NIXL EPLB init failed: buffers).
```

**Exact assertion/raise sites:**
- Surface error: `distributed/eplb/eplb_communicator.py:319` — `_init_step(name, fn)` wraps any exception
  from a NIXL init sub-step as `RuntimeError("NIXL EPLB init failed: {name}")`. Here `name="buffers"`
  (the first sub-step, `_init_step("buffers", self._init_registered_buffers)`, `eplb_communicator.py:308`).
- The failing call: `distributed/eplb/eplb_communicator.py:424` — `self._nixl_wrapper.register_memory(descs)`
  inside `_init_registered_buffers` (registers the local expert-weight VRAM buffers as NIXL descriptors).
- The library error: `rixl/_api.py:384` `self.agent.registerMem(reg_descs, handle_list)` →
  C++ `nixl_agent.cpp:484 registerMem` → `rixl._bindings.nixlBackendError: NIXL_ERR_BACKEND`.
- The **true** cause line (easy to miss — it's the C++ log just above the traceback):
  `ucx_utils.cpp:592` "**VRAM memory is detected as host by UCX**".

The `"buffers"` in the wrapped message is just the sub-step label; the wrapping (`raise ... from exc`,
`eplb_communicator.py:315-319`) is what hid the real UCX cause behind a generic string — which is why the
first pass (that only grepped for `NIXL EPLB init failed`) looked like an opaque library limit.

## 2. Root cause

NIXL registers GPU tensors so they can be RDMA/copy-transferred between EP ranks during a rearrange. The
descriptor is tagged **VRAM** (`eplb_communicator.py:295 self._nixl_memory_type = "VRAM"`; rixl maps
`"VRAM" → VRAM_SEG`, `rixl/_api.py:238-244`). Registration is delegated to the UCX backend. For UCX to
register device memory it must (a) have ROCm support compiled **and** (b) have a ROCm transport active so
its memory-type *detector* is loaded. On this container:

- **(a) is satisfied** — UCX 1.19.1 is built `--with-rocm` (`ucx_info -b`: `HAVE_ROCM=1`,
  `HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF=1`, `UCX_CONFIGURE_FLAGS ... --with-rocm=/opt/rocm`;
  `/usr/local/ucx/lib/ucx/` contains `libuct_rocm.so`, `libucm_rocm.so`). `ucx_info -d` shows memory
  domains `rocm_cpy` (rocm: access,alloc,reg,cache,**detect**) and `rocm_ipc` (rocm: access,reg,cache).
- **(b) is NOT satisfied** — the container env has `UCX_TLS=self,sm,rc_x` (loopback + shared-mem + RC over
  mlx5). It contains **no** rocm transport (`rocm_copy`/`rocm_ipc`), so the rocm UCM/UCT module — the only
  component that provides the `detect` capability — is never engaged. With `UCX_MEMTYPE_CACHE=try` and no
  active rocm detector, `ucp_mem_query` classifies the ROCm device pointer as `UCS_MEMORY_TYPE_HOST`, and
  the UCX NIXL plugin refuses to register a VRAM_SEG that "looks like host" → `NIXL_ERR_BACKEND`.

Why NIXL even picks UCX and then fails, rather than falling back: `eplb_communicator.py:424` calls
`register_memory(descs)` with no `backends=` list, so NIXL tries every backend that advertises the mem
type. The only NIXL plugins present are UCX and POSIX (`/usr/local/rixl/lib/x86_64-linux-gnu/plugins/`:
`libplugin_UCX.so`, `libplugin_POSIX.so`); UCX advertises `VRAM_SEG` at the plugin level even under the
restricted TLS, so NIXL selects UCX for VRAM and then hits the runtime detection failure. There is no
dedicated ROCm NIXL backend — the VRAM path *must* go through UCX, so UCX's ROCm detection *must* be on.

**⇒ Pure environment misconfiguration.** UCX is ROCm-capable but told (via `UCX_TLS`) to use only
non-ROCm transports, so it can't recognise or register GPU memory.

## 3. The fix

Add the ROCm transports to UCX's transport list (and disable the memtype cache as a belt-and-suspenders so
a stale host-classification can't be reused), **before** UCX/rixl is imported by the worker:

```yaml
# in the preset's env_vars: (serve.sh exports these before `exec vllm serve`, so they override the
# image's baked-in UCX_TLS=self,sm,rc_x)
env_vars:
  UCX_TLS: self,sm,rc_x,rocm_copy,rocm_ipc   # add the two ROCm memory transports -> loads rocm detector
  UCX_MEMTYPE_CACHE: n                        # force live per-pointer memtype detection (no stale cache)
```

Injection point: `auto-script/serve.sh:63-70` does `eval "export KEY=VALUE"` for every `.env_vars` entry,
then `exec vllm serve` (`serve.sh:98`). So preset `env_vars` override the container/image env and are set
before the engine (and thus UCX/rixl) initialises — the required ordering. (Equivalent alternatives:
`UCX_TLS=all`, or unset `UCX_TLS` so UCX auto-selects all compiled transports incl. rocm.)

Preset used for the fix test: `presets/glm5.2/dp8ep8/noMTP-bs64-dg-eplb-freq-nixl-fix.yaml`
(base noMTP + the two UCX env_vars + `enable_eplb: true` + `eplb_config '{"use_async": false,
"communicator": "nixl", "window_size": 100, "step_interval": 100}'`).

## 4. Verification (it works)

Re-serve with the fixed preset (run_20260701_023418, then a clean re-run run_20260701_025415):
```
[eplb_communicator.py:95] Initialized EPLB communicator: NixlEplbCommunicator.   # on ALL 8 DP workers
[eplb_state.py:685] Rearranging experts sync mode (profile)...
[eplb_state.py:789] Rearranged experts  (profile)  in 2.81 s.                    # NIXL rearrange RAN
```
No `VRAM memory is detected as host`, no `registerMem` failure, no `NIXL_ERR_BACKEND`. NIXL EPLB is
functional.

### 4a. Full end-to-end confirmation (run_20260701_034633, noMTP freq @ 10k_c36)
After clearing two secondary init hurdles (below), NIXL ran a COMPLETE serve+profile:
- health = 8/8 workers up; **8 profiler traces** written; analyze_time produced summary_time.json.
- **5 real NIXL rearranges completed** (`Rearranged experts in ... s`, no `(profile)` tag).
- **NIXL imbalance = 1.418** (baseline 1.529 → reduced; ≈ pynccl 1.414 / torch_nccl 1.394) — confirms
  imbalance is backend-independent (NIXL produces the same balanced placement).

### 4b. ⚠️ NIXL sync rearrange is VERY SLOW here (~37 s each)
Each NIXL rearrange took **~37 seconds** (`Rearranged experts in 37.82 s / 37.56 / 36.86 / 36.71 / 36.89`),
vs **<1–6 s** for torch_nccl on the identical freq config (S4: 0.77–6 s). So NIXL *functions* but its
**synchronous** READ-based transfer over UCX (rocm_copy) is ~6–50× slower than nccl/pynccl on this
single-node MI300 box. Sync NIXL is therefore impractical for serving (a 37 s stall per rearrange would
destroy latency). NIXL's design point is **async** (`use_async=True`): the READ transfer overlaps with
compute via the async worker, so the 37 s would be hidden. That is the configuration worth pursuing (§6).

### 4c. Two secondary init hurdles cleared (NOT the NIXL bug)
1. **Memory-profiling assertion** — `gpu_worker.py:452 determine_available_memory`:
   `AssertionError: ... Initial free memory 247.5 GiB, current free memory 389.8 GiB ... other processes
   release GPU memory while vLLM is profiling`. Fired on BOTH nixl runs (not on the non-nixl runs) —
   NIXL's slower/heavier init widens the window for a cross-DP-worker free-memory fluctuation during the
   profiling snapshot. Bypass: set `kv_cache_memory_bytes` in engine_args → `gpu_worker.py:384` **skips
   memory profiling** entirely.
2. **KV-cache too small** — with profiling skipped, `kv_cache_memory_bytes=40 GiB` was < the 53.9 GiB the
   default `max_model_len=1048576` needs → `ValueError: ... larger than the available KV cache`. Fixed by
   `max_model_len: 16384` (the 10k+500 workload fits easily). Final working preset:
   `noMTP-bs64-dg-eplb-freq-nixl-fix-v3.yaml` (UCX_TLS fix + `kv_cache_memory_bytes: 42949672960` +
   `max_model_len: 16384`).
   Neither hurdle is the NIXL registration bug; both are generic init/memory-sizing issues that NIXL's
   extra init cost surfaced.

## 5. How I debugged it (step by step)

1. **First pass gave only the wrapped error.** Grepping `NIXL EPLB init failed` returned the generic
   `buffers` message → looked like an opaque "NIXL unavailable on ROCm" limit. Logged that (wrongly) as
   "library-level, not fixable" in EPLB_PROGRESS.md S5/S9.
2. **User pushed back** ("I think it's still usable, maybe an env var isn't enabled"). Re-opened it.
3. **Pulled the FULL chained traceback** from the nixl serve.log (not just the wrapped line). This surfaced
   the two lines the wrapper hid: `rixl._bindings.nixlBackendError: NIXL_ERR_BACKEND` at
   `register_memory`, and crucially the C++ `ucx_utils.cpp:592 "VRAM memory is detected as host by UCX.
   UCX is likely not configured with CUDA support."` — that sentence reframed it from "NIXL missing" to
   "UCX can't see GPU memory".
4. **Checked UCX capability vs configuration** (read-only, in-container):
   - `ucx_info -b` → UCX built `--with-rocm`, `HAVE_ROCM=1` → capability present.
   - `ucx_info -d` → `rocm_cpy`/`rocm_ipc` memory domains exist, and the `detect` capability lives only on
     the rocm component → detection needs a rocm TL active.
   - `env | grep UCX` → `UCX_TLS=self,sm,rc_x` (no rocm transport) → the smoking gun: capability present
     but disabled by config.
   - `ls /usr/local/ucx/lib/ucx/ | grep rocm` → `libuct_rocm.so`, `libucm_rocm.so` present (modules there).
   - rixl plugins dir → only UCX + POSIX plugins → VRAM must go through UCX (no rocm-native NIXL backend).
5. **Traced the injection point** — confirmed `serve.sh:63-70` exports preset `env_vars` before
   `exec vllm serve`, so overriding `UCX_TLS` via the preset reliably reaches the worker before UCX import.
6. **Built the fixed preset** (`...-eplb-freq-nixl-fix.yaml`) with `UCX_TLS=self,sm,rc_x,rocm_copy,rocm_ipc`
   + `UCX_MEMTYPE_CACHE=n`, served it, and watched serve.log: `UCX_TLS=...rocm_copy,rocm_ipc` was exported,
   and EPLB init reached `Initialized EPLB communicator: NixlEplbCommunicator.` + a real rearrange — fix
   confirmed.
7. **Corrected the earlier wrong conclusion** in EPLB_PROGRESS.md (NIXL is *usable* on this ROCm build with
   the UCX_TLS env fix; it is not a library limitation).

## 6. Consequence for the EPLB conclusions

NIXL is the backend intended for **async** (non-blocking) EPLB rearrange (`use_async=True`). With NIXL now
working, async EPLB becomes available on ROCm — which is exactly what could avoid the sync-rearrange
worker-stall that crashed noMTP under sustained bench (EPLB_PROGRESS.md S7/S8: the sync rearrange blocks an
engine step → RPC timeout). Follow-up worth running: `use_async: true` + `communicator: nixl` (+ the
UCX_TLS env) to see if async NIXL rearrange is both stable AND cheaper than sync pynccl at step=100.
