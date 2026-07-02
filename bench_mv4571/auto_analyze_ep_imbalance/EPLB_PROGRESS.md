# EPLB on GLM5 — does enabling vLLM EPLB reduce TIME imbalance?

Goal: enable vLLM EPLB on GLM5.2, measure whether per-rank MoE-compute **time** imbalance drops vs
baseline (EPLB off). Tune a few EPLB configs (incl. the default = just the enable flag), pick the best
per scenario, then bench 3× (mean/std) to verify the profile-based finding holds.

Scope: 2 scenarios only — **noMTP** and **MTP5**, both at **ISL 10k, conc 36, rate inf, OSL 500**.
Metric: TIME imbalance only (max/min over 8 ranks of per-cluster MoE-compute span), from torch profiler
traces → `analyze_time.py`. (Token imbalance not needed here.)

Paths (HOST): repo `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script` (renamed from
auto-script-gpu-5 earlier). Analysis harness:
`bench_mv4571/auto_analyze_ep_imbalance/auto_analyze_ep_imbalance.sh`. Run via container on gpu-5.

---

## S0 — Environment recon + setup (DONE @ 2026-06-30)

### S0.1 Baseline data check
- Consolidated glm baseline lives in `logs/run_glm5/glm5.2/{noMTP-bs64-dg,MTP5-bs64-dg}/`.
- **No 10k_c36 baseline exists** (noMTP has 8k/100k only; MTP5 has 10k_c8/c22 + 8k/100k, no c36).
  → must run a fresh baseline (EPLB off) at 10k_c36 for both presets so before/after is the SAME scenario.

### S0.2 EPLB config (vLLM `3rdparty/vllm/config/parallel.py:56 EPLBConfig`)
Fields: `window_size`(default 1000), `step_interval`(default 3000), `num_redundant_experts`(default 0),
`log_balancedness`(False). Master flag: `enable_eplb`(False) in ParallelConfig (line 163).
CLI: `--enable-eplb` (store_true) + `--eplb-config '{...}'` (dict). Both are EngineArgs fields
(`arg_utils.py:494-495`).

### S0.3 How a preset reaches `vllm serve` (so I know where to inject EPLB)
- `common/serve.sh` → `bash $SERVE_SH <model> <preset>`; `SERVE_SH = auto-script/serve.sh`.
- `auto-script/serve.sh:73` merges `(.parallelism_args // {}) + (.engine_args // {})` into ONE yaml,
  then `serve.sh:98 exec vllm serve "$MODEL_PATH" --config "$TMP_CONFIG" $@`.
  → vLLM reads the merged yaml via `--config`; any EngineArgs key works regardless of which block.
  → **To enable EPLB: add `enable_eplb: true` (+ optional `eplb_config: '{...}'`) to the preset.**
- Harness phase patching (`auto_analyze_ep_imbalance.sh:128 patch_time_preset`) only flips
  `enforce_eager=false`, `VLLM_MOREH_EP_LOG=0`, injects `profiler_config` — it does NOT touch
  enable_eplb, so EPLB keys placed in the base preset survive into the TIME phase. Good.
- `PHASES=time` env runs ONLY the time phase (skip token) — exactly what we need.

### S0.4 EPLB rearrange-trigger timing (CRITICAL caveat) — `eplb_state.py:431-436`
`expert_rearrangement_step` is initialised to `step_interval - step_interval//4` (= 3/4·interval), so the
**first** rearrange fires after only `step_interval/4` forward steps, then every `step_interval`.
- Profile run at 10k_c36: num_prompts = clamp(conc=36) = 36 prompts, OSL 500 → only ~500–700 forward steps.
- **Default step_interval=3000 → first rearrange at +750 steps → likely NEVER triggers in this short run**
  → default EPLB is expected to show ≈ no change here. (Will confirm from serve log.)
- → tuned configs must use a small `step_interval` (e.g. 100) so rebalancing actually happens in-window.

### S0.5 gpu-5 access + container (node was reset → had to redo)
- ssh failed ("Permission denied") — salloc allocation had expired (squeue empty).
- Re-acquired: `salloc -A ce6 -p 256C8G1H_MI325X_Ubuntu22 --reservation=gpu-5_gpu-6_reservation -w gpu-5
  --exclusive --mem=0 --no-shell` (ran detached, job 29547, node gpu-5). ssh OK after ~12s.
- Container `phuc-nguyen-mv-4571` was in **Created** state with stale bind mounts pointing at the OLD
  `auto-script-gpu-5/...3rdparty/{vllm,vllm_moreh}` (renamed → broken).
- **Fix: removed + recreated** with corrected `auto-script` paths (exact command saved in
  `recreate_container.sh`). Verified inside container:
  - `import vllm` → `/usr/local/lib/python3.12/dist-packages/vllm` v0.23.1.dev0 (bound to patched 3rdparty)
  - `EPLBConfig()` imports: window=1000, step=3000, redundant=0
  - fp8.py EP_COLLECT patch present (2 hits); auto_analyze dir reachable.

### S0.6 Plan / EPLB config matrix (per preset, all at 10k_c36, TIME phase)
1. **baseline** — base preset, EPLB off (the "before").
2. **eplb-default** — `enable_eplb: true` only (window=1000, step=3000, redundant=0) [task-required].
3. **eplb-freq** — `enable_eplb: true`, `eplb_config={window_size:100, step_interval:100}` (rebalances ~5×).
4. **eplb-freq-red16** — freq + `num_redundant_experts:16` (replicate hot experts).
Then: best config per scenario → bench 3× (mean/std) to verify metrics (`bench` not `profile`, since the
imbalance comparison used profiling which alone isn't a fair throughput/latency check).

(Will start with a noMTP pilot: baseline + eplb-default + eplb-freq, to confirm EPLB acts with the AITER
MoE path before launching the full matrix.)

---

## S1 — Phase 1 (noMTP imbalance sweep) launched @ 2026-06-30 18:25

### Setup done
- yq: installed via repo's `ensure_yq` (common/helper.sh, pinned mikefarah v4.50.1 + sha256 OK) inside
  container. Added the same call to `recreate_container.sh` so future node-resets restore it.
- EPLB preset variants created under `presets/glm5.2/dp8ep8/` (copy base + append EPLB keys to engine_args):
  - `noMTP-bs64-dg-eplb-default.yaml`  : `enable_eplb: true` (win1000/step3000/red0)
  - `noMTP-bs64-dg-eplb-freq.yaml`     : `enable_eplb: true` + `eplb_config '{"window_size":100,"step_interval":100}'`
  - `noMTP-bs64-dg-eplb-freq-red16.yaml`: freq + `"num_redundant_experts":16`
  - (same 3 for MTP5-bs64-dg)
- Parse sanity (in container): merged preset → EngineArgs parses; EPLBConfig(window=100, step=100,
  redundant=16, policy='default', use_async=True). enable_expert_parallel=True preserved.
- Scenario: `scenario_eplb_nomtp.yaml` (baseline + freq + freq-red16 + default; order puts freq early
  for fast signal). Launcher: `run_eplb_sweep.sh` (PHASES=time, detached via setsid).

### Launch
- Command (in container, detached): `setsid bash run_eplb_sweep.sh scenario_eplb_nomtp.yaml`  (PID 904)
- Run dir: `logs/run_20260630_182547/glm5.2/<preset>/10k_rinf_c36/time/`
- Log: `claude-logs/artifacts/eplb_scenario_eplb_nomtp_20260630_182547.log`
- Each preset: serve(enforce_eager=false→cudagraph, profiler on) → bench --profile np=36 → wait 8 traces
  → analyze_time.py → summary_time.json. 4 presets, ~6-10min each.
- Status at launch: GPU free, baseline serving. Monitoring for serve errors + (for EPLB presets) whether
  rearrange actually fires (grep serve.log).

---

## S2 — Two bugs found on first sweep, fixed (2026-06-30 18:32)

### Bug A — analyze_time crashes: `ModuleNotFoundError: No module named 'matplotlib'`
- Fresh container (recreated in S0.5) lacks deps the old container had. analyze_time.py imports
  matplotlib/pandas/ijson/numpy (+orjson). Check: matplotlib + orjson MISSING; numpy/pandas/ijson OK.
- Note: baseline still wrote **8 traces** fine — only the post-profile analyze failed. Traces persist
  (harness doesn't delete on analyze failure) → re-analyzable.
- Fix: `pip install matplotlib` (→3.11.0) in container. (orjson only needed if a summary path uses it;
  analyze_time ran after matplotlib was added.) TODO if needed: `pip install orjson`.

### Bug B — EPLB default communicator (NIXL) crashes on ROCm/MI300  ← the important one
- The freq variant DID enable EPLB (preset.yaml has `enable_eplb: true` + eplb_config; serve.log:
  `'enable_eplb': True, 'eplb_config': EPLBConfig(window_size=100, step_interval=100, ...)`, "EPLB rank 0").
- But serve CRASHED at EPLB init:
  `RuntimeError: NIXL EPLB init failed: buffers` → `Failed to initialize NixlEplbCommunicator`.
- Root cause: `EPLBConfig.communicator` default = None. On this moreh build, None → **NIXL** communicator
  (`distributed/eplb/eplb_communicator.py:NixlEplbCommunicator`), which needs NIXL/RIXL staged buffers —
  not functional here. (vLLM stock docstring says None→torch_gloo(async)/torch_nccl(sync), but this build
  picks nixl.) Backends available (`create_eplb_communicator`, line 618): `torch_nccl`, `torch_gloo`,
  `pynccl`, `nixl`.
- **Implication: a *pure* default EPLB run (`enable_eplb: true` only) is NOT runnable on this hardware —
  it crashes.** The minimal working config must override the communicator.
- Fix: set `eplb_config: {"use_async": false, "communicator": "torch_nccl"}` on ALL variants
  (sync rearrange over the device RCCL group; avoids NIXL + async-thread). Regenerated all 6 variants.
  The "eplb-default" variant = enable_eplb + this comm fix + vLLM default window=1000/step=3000.

### Relaunch
- Hard-killed stale harness/vllm (note: `pgrep -f auto_analyze_ep_imbalance` false-matches my own ssh
  path; verified via `ps args` + `rocm-smi --showpidgpus`=GPU FREE before relaunch).
- Preserved consolidated baselines logs/run_glm5 + run_kimi2.6 (rm only `run_2026*`).
- Relaunched `scenario_eplb_nomtp.yaml` (freq ordered FIRST to validate EPLB init fast). New run dir:
  `logs/run_20260630_183918`. Monitoring freq serve.log for "Initialized EPLB communicator" vs NIXL fail.

---

## S3 — Expanded plan: also sweep communicator backends for best performance (user request 2026-06-30)

User: besides EPLB algo configs, also try ALL communicator backends (torch_nccl, pynccl, torch_gloo,
try-to-fix nixl) and find which gives the best PERFORMANCE.

Key distinction:
- **EPLB algo config** (window_size / step_interval / num_redundant_experts) → drives how balanced the
  expert placement becomes → affects **TIME IMBALANCE** (measured by profile → analyze_time).
- **Communicator backend** → only moves expert weights during rearrange → affects **rearrange OVERHEAD**
  → affects end-to-end **PERFORMANCE** (throughput / TTFT / ITL), measured by **bench**. Imbalance itself
  is ~backend-independent (same final placement).

NIXL feasibility (code read, `eplb_communicator.py:241 NixlEplbCommunicator`): fails at
`_init_step("buffers", _init_registered_buffers)` = registering VRAM descriptors with NIXL. NixlWrapper
IS present (else "NIXL/RIXL unavailable"), so it's a VRAM-registration failure — likely NIXL GPU-memory
registration not supported on this ROCm/HIP build (library limitation, not a config knob). Will run it
once to capture the full chained traceback, then either fix if it's a config issue or document as
unsupported.

### Revised 3-phase plan (2 presets: noMTP, MTP5; all @ 10k_c36)
- **Phase 1 — imbalance (profile, torch_nccl)**: {baseline, eplb-default, eplb-freq, eplb-freq-red16}.
  → does EPLB reduce TIME imbalance + which ALGO config is best. [noMTP running: run_20260630_183918]
- **Phase 2 — backend × performance (bench)**: at the best algo config, bench each backend
  {torch_nccl, pynccl, torch_gloo, nixl} + baseline. → best backend for throughput/latency.
- **Phase 3 — verify**: best (algo+backend) per preset → bench 3× (mean/std).

---

## S4 — noMTP imbalance results (Phase 1) + freq-red16 NCCL-timeout (2026-06-30 ~19:21)

Run dir: `logs/run_20260630_183918/glm5.2/<preset>/10k_rinf_c36/time/`. EPLB init OK with torch_nccl
(no NIXL). EPLBConfig in serve.log confirms `use_async=False, communicator='torch_nccl'`.

### TIME imbalance (analyze_time, max/min over 8 ranks of per-cluster MoE-compute span)
| config (noMTP) | all_imb (mean) | headroom% | critical(ms) | balanced(ms) | rearrange events |
|---|---|---|---|---|---|
| baseline (EPLB off)            | **1.529** | 17.32 | 12101 | 10006 | — |
| eplb-freq (win100/step100,red0)| **1.394** | 15.38 | 12511 | 10587 | **5** (real; see S10) |
| eplb-default (win1000/step3000)| **1.502** | 17.00 | 12535 | 10405 | ~0 (as predicted) |
| eplb-freq-red16                | FAILED (NCCL timeout, see below) | | | | |

**Finding (noMTP):** EPLB **freq** reduces per-rank MoE-time imbalance **1.529 → 1.394 (-8.8%)** — and the
serve.log shows **5 real rearrange events** (excluding the startup profile warmup; see S10), so EPLB genuinely rebalanced. EPLB **default** ≈ no change
(1.529→1.502): step_interval=3000 → first rearrange at +750 steps, run is only ~hundreds of steps → it
essentially never rebalances. ⇒ **the vLLM default is unsuitable for short workloads; you must shrink
step_interval for EPLB to act.** (crit/bal rose slightly under EPLB = rearrange overhead; net perf is the
job of the bench phase.)

### freq-red16 (num_redundant_experts=16) FAILED
- serve came up (813KB serve.log) but produced **0 traces**; analyze aborted ("no trace files").
- Root cause (serve.log tail): `[Rank 2] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=4, OpType=COALESCED) ran for 600087 ms before timing out` → `c10::DistBackendError` →
  engine killed. The redundant-expert rearrange issues a COALESCED torch_nccl transfer that deadlocks
  (rank mismatch) → 600s NCCL watchdog timeout → crash.
- → redundant experts + torch_nccl sync rearrange is broken here. Will retry redundant with other
  communicators (pynccl/gloo) in the backend sweep (S5) — per "broken case → re-run, don't ignore".

---

## S5 — Backend comparison (noMTP, freq config) + nixl unfixable (run_20260630_195010, 2026-06-30 ~20:14)

freq (win100/step100, red0) with each communicator. Metrics: imbalance (summary_time) + performance
proxy from the profile-run bench JSON (same profiler overhead across all → relative ranking valid;
absolute is depressed by the profiler). Baseline included for reference.

| backend       | imbalance | crit(ms) | bal(ms) | out_tput | total_tput | mean_ttft(ms) | mean_itl(ms) |
|---------------|-----------|----------|---------|----------|------------|---------------|--------------|
| baseline-off  | 1.529     | 12101    | 10006   | 527.0    | 11067      | 9309          | 49.58        |
| pynccl        | 1.414     | 11982    | 10130   | 452.5    | 9502       | 9772          | 59.94        |
| torch_nccl    | 1.394     | 12511    | 10587   | 403.8    | 8480       | 9483          | 70.03        |
| torch_gloo    | 1.424     | 11999    | 10106   | 106.4    | 2234       | 9975          | 318.84       |
| nixl          | — (FAILED: "NIXL EPLB init failed: buffers") |||||||

Findings:
1. **Imbalance is backend-independent** (1.39–1.42 across nccl/pynccl/gloo) — the communicator only moves
   expert weights; the balanced placement (hence imbalance) is the same. Confirms the design hypothesis.
2. **Backend performance ranking: pynccl > torch_nccl >> torch_gloo.** gloo stages weights through CPU →
   ITL 319ms (≈5× worse), throughput collapses to ~1/5. pynccl beats torch_nccl (ITL 60 vs 70).
   **nixl is unavailable** on this ROCm/MI300 build (VRAM descriptor registration fails — library-level,
   not a config knob; cannot "fix" without NIXL ROCm support).
3. **EPLB appears throughput-negative in this regime (DIRECTIONAL, n=1):** baseline out_tput 527 > pynccl
   452 > nccl 404 — but these are single profiler-active runs (no repeats), so treat as directional only;
   the fair 3× bench (S7/S10) shows the effect is within run-to-run noise for MTP5. The
   rearrange overhead (rebalancing every 100 steps) plausibly outweighs the imbalance gain. NOTE: step=100 is
   aggressive — forced, because the short profile window (~hundreds of steps) won't trigger a larger
   step_interval. A production run (long, infrequent rebalance) would amortize the rearrange cost; that
   regime is exactly what the dedicated 3× bench (Phase 3) should probe.

⇒ Best EPLB config so far (noMTP): **freq + pynccl** (lowest overhead among working backends; imbalance
1.41 vs baseline 1.53).

---

## S6 — MTP5 imbalance results (Phase 1, run_20260630_201534, 2026-06-30 ~20:38)

| config (MTP5) | imbalance | headroom% | crit(ms) | bal(ms) | out_tput | mean_itl | rearrange |
|---|---|---|---|---|---|---|---|
| baseline (off)  | 1.322 | 14.42 | 7833 | 6703 | 716.3 | 120.48 | 0 |
| eplb-freq       | **1.284** | 13.80 | 7343 | 6329 | 656.9 | 152.73 | 4 |
| eplb-default    | 1.318 | 13.92 | 7561 | 6509 | 770.1 | 110.89 | 0 |

Finding (MTP5): EPLB **freq** reduces imbalance **1.322 → 1.284 (-2.9%)** with 4 real rearranges; **default**
≈ baseline (1.318, **0 real rearranges** — the raw grep's "3" were all the startup profile warmup; see S10). Smaller gain than noMTP (-8.8%) because MTP5's baseline is already
more balanced (spec-decode packs ~(1+spec) tokens/step → better routing averaging — matches earlier
MV-4571 findings). Throughput proxy noisy (default 770 > baseline 716 is profile-run noise) — the fair
answer comes from Phase 3 bench.

### Phase 1 summary (both scenarios, 10k_c36, torch_nccl)
| scenario | baseline imb | eplb-freq imb | Δ | eplb-default imb |
|---|---|---|---|---|
| noMTP | 1.529 | 1.394 | **-8.8%** | 1.502 (≈0) |
| MTP5  | 1.322 | 1.284 | **-2.9%** | 1.318 (≈0) |
⇒ EPLB **does** reduce per-rank MoE-time imbalance, but only when step_interval is small enough to
rebalance in-window (freq), and more for the less-balanced scenario (noMTP). Default never helps here.

---

## S7 — Phase 3 verification bench (3× mean/std, NO profiler) — run_eplb_verify_bench (2026-06-30 ~21:14)

env_eplb_bench.yaml: runs=3, 10k@conc36, MODE=bench (no profiler). Best EPLB config = freq-pynccl.
Results dir: `logs/eplb_verify_bench/glm5.2/dp8ep8/<preset>/auto_bench/<ts>/` (run1/2/3.csv + mean/std.csv).

| config | completed (per run) | output_tput (mean) | mean_ttft(ms) | mean_itl(ms) | stable? |
|---|---|---|---|---|---|
| noMTP baseline       | 72/72/72 | **243.8** | 8007 | 131.6 | ✓ |
| noMTP freq-pynccl    | **36 / 0 / 0** | 7.5 (invalid) | — | — | **✗ CRASHED** |
| MTP5 baseline        | 72/72/72 | **611.6** | 6175 | 218.0 | ✓ |
| MTP5 freq-pynccl     | 72/72/72 | **584.1** | 6111 | 224.8 | ✓ |

Findings:
- **MTP5 EPLB freq-pynccl is stable** and costs only **~4.5% output throughput** (611.6→584.1) for the
  imbalance reduction (1.322→1.284). So under a real bench (not the profiler regime), the EPLB overhead
  is modest for MTP5.
- **noMTP EPLB freq-pynccl CRASHED under sustained bench:** run1 completed only 36/72 then the server
  DIED (run2/run3 = 0 completions, dur 0.04s). serve.log:
  `TimeoutError: RPC call to sample_tokens timed out` on EngineCore_DP5 → executor shutdown → ApiServer
  died. ⇒ the synchronous EPLB rearrange (every 100 steps) blocks an engine step long enough to exceed
  the worker RPC timeout; noMTP decodes longer (OSL 500, no spec) → more rebalances → it eventually
  stalls a worker and the engine cascades down. (The shorter PROFILE run, np=36, had survived.)
- ⇒ frequent sync rearrange (step=100) is unstable for sustained noMTP serving. **Broken case → must
  re-run** (and likely raise step_interval to make rearrange rare enough to not stall a step, since the
  async path that would avoid blocking needs NIXL which is unavailable here).

---

## S8 — Re-run broken noMTP case + stability fix (step500) — run_eplb_rerun_bench (2026-06-30 ~21:37)

Per "broken case -> re-run + delete old". Deleted the crashed noMTP-freq-pynccl bench, re-ran it, and
added a less-frequent-rearrange variant (step500: window=500, step_interval=500, pynccl).

| config (noMTP bench, runs=3) | completed | out_tput(mean) | itl(ms) | stable? |
|---|---|---|---|---|
| baseline (off)          | 72/72/72 | 243.8 | 131.6 | ✓ |
| freq-pynccl (step=100)  | 36 / 0 / 0 | 10.6 (invalid) | — | **✗ crash REPRODUCED** |
| step500-pynccl (step=500)| 72/72/72 | **240.6** | 134.3 | ✓ |

- **freq-pynccl crash is systematic** (reproduced: run1=36 then server dies, RPC sample_tokens timeout).
  Frequent sync rearrange (every 100 steps) stalls an engine worker under sustained noMTP serving.
- **step500-pynccl is STABLE and nearly free** (out_tput 240.6 vs baseline 243.8 = **-1.3%**). Rebalancing
  rarely (~every 500 steps) avoids the worker stall.
- ⇒ For stable noMTP serving on this hardware, EPLB must rearrange infrequently (step≈500). Open question
  measured next: does the stable step500 config still REDUCE imbalance? (profile launched.)

---

## S9 — FINAL conclusions (2026-06-30 ~21:48)

step500 (stable config) imbalance measured: **1.462** (1 real rearrange; see S10) — so the stable config DOES still
reduce imbalance. There is a clear tradeoff curve (more frequent rearrange → more balance, less stability).

### Full noMTP matrix (10k_c36), imbalance (profile) + bench (3×, no profiler)
| config | rearrange/step_interval | imbalance | Δ imb | bench completed | bench out_tput | bench stable |
|---|---|---|---|---|---|---|
| baseline (EPLB off)     | —          | 1.529 | —      | 72/72/72 | 243.8 | ✓ |
| eplb-default (step3000) | 0          | 1.502 | -1.8%  | (n/a)    | —     | — |
| eplb-step500 (step500)  | 1          | 1.462 | **-4.4%** | 72/72/72 | 240.6 | ✓ (**-1.3% tput**) |
| eplb-freq (step100)     | 5          | 1.394 | **-8.8%** | 36/0/0  | crash | ✗ (RPC timeout) |
<!-- rearrange = REAL (non-profile) completed rearranges; corrected in S10 (were 2x-inflated grep counts) -->


### MTP5 matrix (10k_c36)
| config | imbalance | Δ imb | bench completed | bench out_tput | stable |
|---|---|---|---|---|---|
| baseline (off)      | 1.322 | —      | 72/72/72 | 611.6 | ✓ |
| eplb-default        | 1.318 | -0.3%  | —        | —     | — |
| eplb-freq (pynccl)  | 1.284 | **-2.9%** | 72/72/72 | 584.1 | ✓ (−4.5% mean, **within run std ≈ noise**; see S10) |

### Backend ranking (noMTP, freq config) — performance
pynccl (out_tp 452) > torch_nccl (404) >> torch_gloo (106, CPU-staged, ITL 5×).
nixl = initially thought unusable ("NIXL EPLB init failed: buffers") — **but this was WRONG; NIXL IS
usable on this ROCm build**. Root cause was a missing UCX env (`UCX_TLS` lacked ROCm transports), fixed
with `UCX_TLS=self,sm,rc_x,rocm_copy,rocm_ipc` + `UCX_MEMTYPE_CACHE=n`. See **EPLB_NIXL_FIX.md** and S11.

### ANSWERS to the goal
1. **Does EPLB reduce GLM5 time imbalance? YES**, when it actually rearranges: noMTP 1.529→1.462 (stable,
   step500) up to 1.394 (step100); MTP5 1.322→1.284. The benefit is larger where baseline is less balanced
   (noMTP > MTP5, since spec-decode already averages routing).
2. **Default config is the wrong choice**: vLLM's default step_interval=3000 never rebalances in these
   workloads (≈ baseline), AND the default async/NIXL communicator crashes on ROCm. EPLB only works here
   with (a) an explicit communicator and (b) a much smaller step_interval.
3. **Best EPLB config (performance + stability):**
   - **noMTP → step500 + pynccl** (imbalance -4.4%, throughput -1.3%, stable). step100 reduces imbalance
     more (-8.8%) but CRASHES sustained serving — not usable.
   - **MTP5 → freq(step100) + pynccl** (imbalance -2.9%, throughput -4.5%, stable).
   - communicator: **pynccl** everywhere (fastest working; gloo too slow; nixl broken).
4. **Net trade-off:** EPLB's imbalance reduction is modest and comes with rearrange overhead + (if too
   frequent) instability. It is only net-positive when rearrange is infrequent enough to be cheap/stable
   yet frequent enough to track load — step≈500 hits that window here. Under the profiler's short window
   EPLB looks throughput-negative; under real bench with infrequent rearrange the cost is ~1-5%.

### Artifacts / where the data lives
- Imbalance (profile): logs/run_20260630_183918 (noMTP baseline/freq/default), run_20260630_195010
  (noMTP freq pynccl/gloo/nixl), run_20260630_201534 (MTP5), run_20260630_213827 (noMTP step500).
- Bench 3× (perf): logs/eplb_verify_bench/glm5.2/dp8ep8/<preset>/auto_bench/<ts>/{run1,2,3,mean,std}.csv
- Presets: presets/glm5.2/dp8ep8/{noMTP,MTP5}-bs64-dg-eplb-*.yaml
- Scripts: run_eplb_sweep.sh, run_eplb_verify_bench.sh, run_eplb_rerun_bench.sh, recreate_container.sh
- Logs: claude-logs/artifacts/eplb_*.log

---

## S10 — Independent adversarial review & corrections (2026-06-30, workflow eplb-deep-dive)

A 7-agent workflow re-verified EPLB_PROGRESS.md against the raw data (summary_time.json, bench CSVs,
serve.logs) and re-read the vLLM source. Verdict: **all hard numbers are correct** — every imbalance
value, headroom/crit/balanced cell, backend throughput/TTFT/ITL, 3× bench means, both crash signatures
(NIXL "init failed: buffers", red16 600s NCCL watchdog), the RPC sample_tokens timeout, and all deltas
(−8.8/−2.9/−4.4/−1.8/−0.3%) match source to rounding. Corrections applied:

### C1 — rearrange counts were ~2× inflated (grep double-counted + included profile warmup)  [FIXED inline]
The reported counts came from `grep -ci rearrang` on serve.log, which counts BOTH the "Rearranging experts
sync mode..." (start) and "Rearranged experts ... in Xs" (done) lines, AND the one startup **profile
warmup** rearrange (`Rearranged experts (profile) in ...`, a dummy rearrange vLLM does during memory
profiling). The REAL load-triggered rearrange count = `grep "Rearranged experts" | grep -v "(profile)"`:

| config | reported (old) | **REAL (corrected)** |
|---|---|---|
| noMTP eplb-freq (step100)   | 12 | **5** |
| noMTP eplb-default (step3000)| ~0 | **0** |
| noMTP eplb-step500 (step500)| 5  | **1** |
| MTP5 eplb-freq (step100)    | 9  | **4** |
| MTP5 eplb-default (step3000)| 3  | **0** |

### C2 — MTP5 eplb-default did ZERO real rearranges, not 3  [FIXED inline]
The "only 3 rearranges" wording self-contradicted the report's own thesis (step_interval=3000 never fires
in a ~hundreds-of-steps run). The 3 grep hits were all the single startup profile warmup. Corrected to 0.
This actually STRENGTHENS the thesis: default rebalanced 0 times in both scenarios. (The imbalance
numbers/interpretation were unaffected — default ≈ baseline stands.)

### C3 — throughput-cost claims: precision/causality softened  [FIXED inline]
- MTP5 "−4.5% throughput cost" (611.6→584.1): the run-to-run std is ~76–96 (≈13–16% CoV; baseline runs
  spanned 518.8/606.0/709.9), so a −4.5% mean gap is **inside 1σ — statistically indistinguishable from
  zero**. Relabelled as "within run std ≈ noise". Honest read: MTP5 EPLB freq-pynccl is stable and its
  throughput cost is **not measurably different from baseline** at n=3.
- S5 "EPLB net-costs throughput": drawn from n=1 profiler-active runs; relabelled DIRECTIONAL. (It was
  already caveated as profiler-regime; now explicit.)

### C4 — minor: NIXL default-communicator characterization  [note]
S2 said stock docstring resolves None→torch_gloo(async)/torch_nccl(sync); the actual build code
(config/parallel.py:907-924) resolves enable_eplb + communicator=None + non-elastic → **nixl** (when
`is_nixl_available()`), which is why default crashed. Core claim (default is unrunnable → NIXL crash)
is correct; the docstring aside was slightly off.

### C5 — note: MTP5+pynccl imbalance not directly measured  [note]
The MTP5 −2.9% imbalance was measured with communicator=**torch_nccl** (run_201534). Attributing it to the
MTP5+pynccl recommendation is justified by the measured backend-independence of imbalance (S5: nccl 1.394
≈ pynccl 1.414 ≈ gloo 1.424), but the exact MTP5+pynccl imbalance was not directly profiled.

Net: conclusions unchanged; the corrections make the rearrange-count evidence sharper (default = 0 real)
and the throughput-cost claims properly hedged for the n=3 variance.

---

## S11 — NIXL is FIXABLE (corrects S2/S5/S9 "unusable") (2026-07-01)

Earlier I concluded NIXL was a library-level limitation ("cannot be fixed"). **That was wrong.** Full
detail in **EPLB_NIXL_FIX.md**; summary:
- Real cause (hidden behind the wrapped `NIXL EPLB init failed: buffers`): `ucx_utils.cpp:592
  "VRAM memory is detected as host by UCX"` → `registerMem` → `NIXL_ERR_BACKEND`. UCX is built --with-rocm
  and has rocm_cpy/rocm_ipc domains, but the container's `UCX_TLS=self,sm,rc_x` excludes all ROCm
  transports → UCX's ROCm memory detector never loads → VRAM misdetected as host → registration refused.
- Fix (preset env_vars, exported by serve.sh:63-70 before vllm serve):
  `UCX_TLS=self,sm,rc_x,rocm_copy,rocm_ipc` + `UCX_MEMTYPE_CACHE=n`.
- Verified: `Initialized EPLB communicator: NixlEplbCommunicator.` on all 8 workers + a real NIXL
  rearrange (run_20260701_02*). Preset: `noMTP-bs64-dg-eplb-freq-nixl-fix.yaml`.
- Corrected claims: S2 Bug B, S5 backend ranking, S9 "nixl unusable / cannot be fixed" → NIXL works with
  the UCX_TLS env. The *pure default* still crashes (because default env lacks UCX_TLS rocm transports),
  but that is a fixable env issue, not a hard limit.
- Follow-up value: NIXL enables **async** EPLB rearrange (use_async=true), which may avoid the sync-rearrange
  worker-stall that crashed noMTP under sustained bench (S7/S8).

### S11a — NIXL end-to-end confirmed + it's slow (run_20260701_034633)
Full serve+profile completed with NIXL: health 8/8, 8 traces, **5 real NIXL rearranges**, imbalance
**1.418** (≈ pynccl/nccl → backend-independent, as expected). BUT each NIXL sync rearrange took **~37 s**
(vs <1–6 s for torch_nccl on the same config) — NIXL's sync READ-over-UCX path is ~6–50× slower here, so
sync NIXL is impractical for serving; its value is the **async** path (use_async=true, overlaps transfer
with compute). Two generic init hurdles were cleared to get here (NOT the NIXL bug): memory-profiling
assertion (bypass via `kv_cache_memory_bytes`) and KV sizing (`max_model_len=16384`). Working preset:
`noMTP-bs64-dg-eplb-freq-nixl-fix-v3.yaml`. Full detail: EPLB_NIXL_FIX.md §4a-4c.

---

## S12 — async NIXL EPLB (use_async=true + communicator=nixl) (2026-07-01)

Follow-up enabled by the NIXL fix (S11): NIXL is the intended backend for ASYNC rearrange (transfer
overlaps compute via a background worker), which should avoid the sync-rearrange worker-stall that
crashed noMTP under sustained bench (S7/S8). Preset `noMTP-bs64-dg-eplb-async-nixl.yaml` (UCX_TLS fix +
kv_cache_memory_bytes + max_model_len=16384 + `eplb_config '{"use_async": true, "communicator": "nixl",
"window_size":100, "step_interval":100}'`).

### Profile (run_20260701_050802): async NIXL is STABLE end-to-end
- health 8/8, **8 traces**, summary produced, **0 errors** (no sample_tokens timeout, no EngineDeadError,
  no watchdog) — async rearrange does NOT stall the engine step (logs "Rearranging experts (async mode)").
- imbalance **1.497** (baseline 1.529, −2.1%) — LESS reduction than sync NIXL (1.418, −7.3%). Because
  async rearranges run in the background and complete after a lag; in the short profile window the new
  placement hadn't fully taken effect when the profiler captured (2 async rearranges started, 0 shown
  completed in-window). In sustained serving the placement would catch up.
- ⇒ async trades peak balancing-in-window for stability (non-blocking). Next: bench stability test.

### S12a — async NIXL bench: STABLE where sync crashed (definitive)
noMTP EPLB under sustained bench (3× runs=3, np=72), 10k_c36:
| config | run1/2/3 completed | out_tput (mean) | itl (mean, ms) | stable |
|---|---|---|---|---|
| baseline (off)            | 72/72/72 | 243.8 | 131.6 | ✓ |
| sync freq-pynccl (step100)| 36/0/0   | (crash) | — | ✗ RPC sample_tokens timeout |
| **async NIXL (step100)**  | **72/72/72** | **212.3** | 153.8 | **✓** |

**Conclusion:** async NIXL EPLB is the ONLY rearranging config that's stable under sustained noMTP serving.
The async worker overlaps the (slow, ~37 s) NIXL READ transfer with compute, so no engine step stalls →
no RPC timeout → no crash (where sync step=100 crashed). Cost: ~13% throughput (243.8→212.3) and +17% ITL
(131.6→153.8) from the background-rearrange overhead. So on this ROCm/MI300 box, if you want EPLB actively
rebalancing noMTP during sustained serving, **async NIXL** (enabled by the UCX_TLS fix) is the viable path;
sync rearrange at a load-tracking cadence is not. (Imbalance under async measured 1.497 in the short
profile — less than sync's 1.418 — because async placement lags the short capture window; it would catch
up under sustained load.)

### Overall EPLB recommendation for GLM5 on this hardware (updated)
- **MTP5**: sync EPLB freq + pynccl — stable, imbalance 1.322→1.284, throughput within noise.
- **noMTP**: sync step=100 crashes; sync step=500 is stable+cheap but barely rebalances (1 rearrange);
  **async NIXL step=100 is stable and actively rebalances** at ~13% throughput cost. Choose async-NIXL if
  active balancing matters, else step500 (or leave EPLB off — the imbalance gain is modest either way).

---

## S13 — Compute-bound-candidate bench: baseline vs async-NIXL across scenarios (2026-07-01, IN PROGRESS)

Goal (user): find which glm5 scenarios EPLB net-benefits. From the S9/S12 findings, EPLB benefit ∝ baseline
imbalance (headroom) AND requires compute-bound (crit>comm, else comm dominates). Baseline imbalance
ranking (run_glm5, 20 cases) → candidates:
- noMTP high+compute-bound: 8k_c8/c22/c31/c36/c52 (imb 1.48-1.62, crit>comm) — PRIME candidates.
- noMTP highest: 100k_c8 (1.64, but comm-bound → contrast), 100k_c31 (1.63, compute-bound).
- MTP5: all ≤1.46 (low headroom) — 8k control.

Design (user choices: **async-NIXL** config, **compute-bound candidates** scope):
- EPLB arm = async NIXL (proven stable under sustained bench, S12a; ~13% overhead is the honest cost).
- Baseline vs async-NIXL share the SAME max_model_len + kv_cache_memory_bytes (only enable_eplb/UCX differ)
  → fair. Presets: `{noMTP,MTP5}-bs64-dg-{base,async}-8k.yaml` (mml16384/kv40GiB) + `noMTP-...-{base,async}-100k.yaml` (mml131072/kv80GiB).
- Cases: noMTP 8k[8,22,31,36,52] + 100k[8,31]; MTP5 8k[8,22,31,36,52] (control). bench runs=3, no profiler.
- One serve per (preset,arm,group) benches all its cases×3 → 6 serves. Driver `run_eplb_candidate_bench.sh`,
  env `env_bench_{8k,100k}.yaml`. Results → `logs/eplb_candidate_bench/.../auto_bench/<ts>/{run1,2,3,mean,std}.csv`.
- Metric: net throughput/latency baseline vs async-NIXL per case → where EPLB (with its overhead) still wins.
