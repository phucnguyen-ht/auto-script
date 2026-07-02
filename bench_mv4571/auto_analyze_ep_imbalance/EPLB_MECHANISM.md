# vLLM EPLB (Expert Parallel Load Balancing) — Mechanism Deep-Dive

Line-by-line documentation of how EPLB works in this vendored vLLM tree
(3rdparty/vllm), produced for GLM5.2 EP-imbalance work (MV-4571). Code refs are
repo-relative under 3rdparty/vllm. See EPLB_PROGRESS.md for the empirical results
and EPLB_NIXL_FIX.md for the NIXL/UCX fix.

---


# 1. Concepts & Configuration Semantics

The exact version comes from a generated `_version.py`. I have all the code needed to produce the section.

## EPLB: Expert Parallel Load Balancing — Concepts and Configuration Semantics

> Code references are repo-relative to `3rdparty/vllm`. The two authoritative files are `config/parallel.py` (the config dataclasses + validation) and `engine/arg_utils.py` (how CLI/YAML populate them). The vendored tree pins its version via a generated `_version.py` (`vllm/version.py:5`), so exact string is build-time; the semantics below are read directly from the source in this tree and are accurate for it. Where behavior is version/hardware-specific I flag it explicitly.

### 1. The problem EPLB solves

Under **Expert Parallelism (EP)**, an MoE layer's experts are physically sharded across ranks (GPUs). A token's router picks a small top-k subset of experts per layer; the token is then dispatched (all-to-all) to whichever rank(s) hold those experts, computed there, and combined back. This is enabled by `enable_expert_parallel` (`config/parallel.py:154`: *"Use expert parallelism instead of tensor parallelism for MoE layers."*).

The failure mode: **real routing traffic is not uniform**. Some experts ("hot" experts) receive far more tokens than others. Because experts are pinned to ranks, a rank holding hot experts becomes a straggler — it does more compute and more all-to-all traffic than its peers. Since the MoE layer cannot finish until the slowest rank finishes, per-rank load imbalance directly caps throughput and inflates latency. The more ranks you scale to (wide-EP), the worse a single hot rank hurts.

**EPLB** attacks this by (a) allowing some experts to be **replicated** across ranks (redundant experts) and (b) **periodically re-deciding the physical placement** of experts onto ranks based on observed load, so that hot experts get more replicas / get spread onto under-loaded ranks. This is a dynamic, runtime rebalancing loop layered on top of static EP.

### 2. Logical vs physical experts, redundant experts

- **Logical experts**: the experts the model actually defines (e.g. 256 routed experts in a DeepSeek-style MoE). The router always addresses logical experts.
- **Physical experts**: the concrete per-rank weight slots that hold expert weights. With EPLB, `num_physical_experts = num_logical_experts + num_redundant_experts`. The extra slots are **replicas** of (typically hot) logical experts.
- **Redundant experts** (`num_redundant_experts`, `config/parallel.py:69`): the count of *extra* physical slots beyond the logical count. `0` (the default) means a pure permutation — placement can be rearranged but no expert is duplicated, so relief only comes from moving experts to better ranks. `>0` means hot logical experts can have multiple physical copies on different ranks, letting their token load be split. There is an indirection map from logical→physical experts, and the router's dispatch is remapped through it. This map is what the rearrange step recomputes.

Related but distinct is the **static placement strategy** (`expert_placement_strategy`, `config/parallel.py:167`), which controls the *initial* logical→physical layout:
- `"linear"` (default): experts placed contiguously — 4 experts / 2 ranks → rank0 `[0,1]`, rank1 `[2,3]`.
- `"round_robin"`: interleaved — rank0 `[0,2]`, rank1 `[1,3]`. The docstring notes this "can help improve load balancing for grouped expert models with no redundant experts" — i.e. it's a cheap static mitigation when you are *not* paying for redundant experts.

EPLB is the dynamic layer on top of whichever static strategy you start from.

### 3. The rearrange cycle (how the loop runs)

EPLB runs a background loop tied to model forward steps:
1. **Record load**: every step, each rank accumulates how many tokens hit each (physical) expert. Only the most recent `window_size` steps are kept (a sliding window of load statistics).
2. **Rearrange**: every `step_interval` steps, the recorded load is fed to the balancing **policy**, which computes a new logical→physical placement. New physical layouts imply moving expert weight tensors between ranks — that movement goes over the EPLB **communicator** backend. When `use_async=True` this movement is non-blocking (overlapped with compute); when `False` it is a blocking, synchronous rearrange.
3. **(Optional) log balancedness**: if `log_balancedness` is on, a balancedness metric is computed and logged (every `log_balancedness_interval` rearranges/steps), at the cost of extra communication.

The relationship between the two intervals is important and is called out in the source itself (`config/parallel.py:62-67`): if `step_interval > window_size`, the rearrange only ever sees the last `window_size` steps of data — the window, not the interval, bounds how much history informs a decision.

### 4. The `EPLBConfig` dataclass — field-by-field

Defined at `config/parallel.py:55-105`, decorated `@config` (`config/parallel.py:55`), which registers it as a vLLM config dataclass (the `@config` decorator lives at `config/utils.py:41+`; it makes the class introspectable by the arg-parsing machinery in §6). Every field uses a pydantic `Field(...)` default with bounds, so out-of-range values are rejected at construction, not silently clamped.

#### `window_size: int = Field(default=1000, gt=0)` — `config/parallel.py:59`
- **What it controls**: the sliding-window length (in forward steps) over which per-expert token load is accumulated before being handed to the rearrange policy.
- **Units**: model forward steps.
- **Default**: `1000`. Constraint `gt=0` (must be strictly positive; `0` or negative rejected).
- **Effect / tradeoff**: larger window = smoother, more stable load estimate (less reactive to transient bursts, less placement churn) but slower to adapt to a genuine shift in the traffic pattern. Smaller window = more reactive but noisier, risking placement thrash. As noted above, if `window_size < step_interval` the extra steps between windows are effectively discarded for that decision.

#### `step_interval: int = Field(default=3000, gt=0)` — `config/parallel.py:61`
- **What it controls**: how often (in steps) the rearrange is actually triggered. Its own docstring (`config/parallel.py:62-67`): *"Interval for rearranging experts in expert parallelism."*
- **Units**: model forward steps.
- **Default**: `3000`. Constraint `gt=0`.
- **Effect / tradeoff**: this is the knob that trades **adaptation latency vs rearrange overhead**. A rearrange moves weight tensors across ranks (communication + potential stall if sync), so it is not free. Small interval → placement tracks load quickly but you pay the rearrange cost often; large interval → cheap amortized cost but placement can be stale for a long time. Default `3000` vs window `1000` means: by default the window is *smaller* than the interval, so each rearrange uses the last 1000 steps.

#### `num_redundant_experts: int = Field(default=0, ge=0)` — `config/parallel.py:69`
- **What it controls**: number of *extra* physical expert slots (replicas) beyond the logical expert count. Docstring `config/parallel.py:70`: *"Number of redundant experts to use for expert parallelism."*
- **Units**: count of physical expert slots (global, across the EP group).
- **Default**: `0`. Constraint `ge=0`.
- **Effect / tradeoff**: `0` = pure re-permutation (placement can move but no expert is duplicated → a single super-hot expert can still bottleneck its one rank). `>0` = hot experts can be replicated so their load is split across ranks, giving the policy real headroom to balance — but each redundant slot costs extra GPU memory (an additional copy of that expert's weights) and adds to the weight-movement volume during rearrange. **Guarded**: setting this `!= 0` while EPLB is disabled is a hard error (see §5) — redundant experts are meaningless without the balancer.

#### `log_balancedness: bool = False` — `config/parallel.py:72`
- **What it controls**: whether a per-step "balancedness" metric is computed and logged. Docstring `config/parallel.py:73-76` explicitly: *"This is turned off by default since it will cause communication overhead."*
- **Default**: `False`.
- **Effect / tradeoff**: purely observability. Turning it on lets you *see* how well-balanced the experts currently are (useful for tuning `window_size`/`step_interval`/`num_redundant_experts`), but computing the metric requires a collective, so it adds communication cost on the hot path. Leave off in production; turn on when tuning.

#### `log_balancedness_interval: int = Field(default=1, gt=0)` — `config/parallel.py:77`
- **What it controls**: how often the balancedness metric is logged (only relevant when `log_balancedness=True`).
- **Units**: steps/log-cycles.
- **Default**: `1` (log every time). Constraint `gt=0`.
- **Effect / tradeoff**: increase it to throttle logging overhead/log volume when you still want the metric but not on every single step. It only matters when `log_balancedness` is on — the validator ties them together (§4.1).

#### `use_async: bool = True` — `config/parallel.py:81`
- **What it controls**: non-blocking vs blocking EPLB. Docstring `config/parallel.py:82-84`: *"Whether to use non-blocking EPLB."*
- **Default**: `True`.
- **Effect / tradeoff**: async (`True`) overlaps the expert-weight movement of a rearrange with ongoing compute, hiding most of the rearrange stall — at the cost of more complex communication and a constrained backend choice (NCCL is explicitly avoided for async — see §7). Sync (`False`) does the rearrange as a blocking operation: simpler, can use `torch_nccl`, but the forward pass stalls while weights move. **Constraint**: async is only valid with the `"default"` policy (enforced by the validator, §4.1).

#### `policy: EPLBPolicyOption = "default"` — `config/parallel.py:86`
- **What it controls**: which load-balancing algorithm computes the new placement. Type is `EPLBPolicyOption = Literal["default"]` (`config/parallel.py:37`).
- **Default / allowed**: `"default"` — and in *this* tree that is the **only** allowed value (the `Literal` has a single member). So today this knob is effectively fixed; it exists as an extension point for future policies. If you pass anything else, argparse's `choices` validation (derived from the `Literal`, §6) rejects it.
- **Interaction**: async EPLB requires this to stay `"default"` (§4.1).

#### `communicator: EPLBCommunicatorBackend | None = None` — `config/parallel.py:89`
- **What it controls**: the transport used to move expert weights between ranks during a rearrange. Type `EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]` (`config/parallel.py:39`). Docstring `config/parallel.py:90-97`:
  - `"torch_nccl"` — `torch.distributed` on the device (GPU) process group.
  - `"torch_gloo"` — `torch.distributed` gloo with **CPU staging** (weights staged through host memory).
  - `"nixl"` — NIXL/RIXL with staged send/recv buffers.
  - `"pynccl"` — PyNCCL send/recv.
  - `None` — **auto-select** (resolved in `__post_init__`, see §7).
- **Default**: `None` (auto).
- **Effect / tradeoff**: this is a performance/compatibility knob for the weight-transfer step. NCCL-based transfer is fastest device-to-device but is **incompatible with async EPLB** (multi-stream conflicts / batched isend-irecv hangs — see §7); gloo/CPU-staging is robust but slower; nixl is preferred when available. Most users leave it `None` and let vLLM pick.

#### 4.1 The `EPLBConfig` validator — `config/parallel.py:99-105`
```python
@model_validator(mode="after")
def _validate_eplb_config(self) -> Self:
    if self.use_async and self.policy != "default":
        raise ValueError("Async EPLB is only supported with the default policy.")
    if self.log_balancedness and self.log_balancedness_interval <= 0:
        raise ValueError("log_balancedness_interval must be greater than 0.")
    return self
```
Runs after all fields are set. Two cross-field invariants: (1) async requires the default policy; (2) if balancedness logging is on, its interval must be positive (a belt-and-suspenders check on top of the field's `gt=0`, guarding the case where the field bound is bypassed). Violations raise at config-construction time — before any GPU work.

### 5. Where EPLB plugs into `ParallelConfig`

`EPLBConfig` is not standalone; it hangs off `ParallelConfig`:

- `enable_eplb: bool = False` — `config/parallel.py:163-164`: *"Enable expert parallelism load balancing for MoE layers."* This is the master on/off switch.
- `eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)` — `config/parallel.py:165`: the nested config. `default_factory` means each `ParallelConfig` gets its own fresh `EPLBConfig` with the defaults above (avoids a shared-mutable-default bug).

The cross-cutting validation lives in `ParallelConfig._validate_parallel_config` (`@model_validator(mode="after")`, `config/parallel.py:424`). The EPLB-relevant block is `config/parallel.py:459-480`:

```python
if self.enable_eplb:
    if not current_platform.is_cuda_alike():
        raise ValueError("Expert parallelism load balancing is only supported on "
                         "CUDA devices or ROCm devices now.")
    if not self.enable_expert_parallel:
        raise ValueError("enable_expert_parallel must be True to use EPLB.")
    if self.tensor_parallel_size * self.data_parallel_size <= 1:
        raise ValueError("EPLB requires tensor_parallel_size or data_parallel_size "
                         f"to be greater than 1, but got "
                         f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}.")
else:
    if self.eplb_config.num_redundant_experts != 0:
        raise ValueError("num_redundant_experts is set to "
                         f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                         "enabled. Either enable EPLB or unset num_redundant_experts.")
```

Block-by-block:
- `config/parallel.py:460-464` — EPLB is only supported on CUDA-alike devices (NVIDIA CUDA **or** AMD ROCm; `is_cuda_alike()` covers both). On any other backend it hard-errors. This is the ROCm-relevant gate for this tree.
- `config/parallel.py:465-466` — EPLB *requires* EP to be on. Load-balancing expert placement is meaningless if experts aren't expert-parallel-sharded in the first place.
- `config/parallel.py:467-472` — EPLB requires `TP * DP > 1`, i.e. there must be more than one rank to balance across. On a single rank there is nothing to rebalance.
- `config/parallel.py:473-480` (the `else`) — if EPLB is **off**, `num_redundant_experts` must be `0`. This is the guard mentioned in §4: redundant experts are a no-op without the balancer, so a nonzero value is treated as a user mistake and rejected with a clear message.

There is also an **elastic-EP** interaction in `ParallelConfig.__post_init__` (`config/parallel.py:786-788`): `enable_elastic_ep` requires `enable_eplb=True` (elastic scale up/down is built on the EPLB machinery).

### 6. How `enable_eplb` / `eplb_config` flow from CLI / `--config` YAML into `ParallelConfig`

The path is: **CLI args / YAML → `EngineArgs` dataclass fields → argparse arguments → `create_engine_config` → `ParallelConfig(...)`**.

**(a) `EngineArgs` mirrors the config fields.** In `engine/arg_utils.py`:
```python
eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")   # arg_utils.py:494
enable_eplb: bool = ParallelConfig.enable_eplb                        # arg_utils.py:495
```
`enable_eplb` copies the plain default (`False`). `eplb_config` uses `get_field(ParallelConfig, "eplb_config")` (`config/utils.py:83-112`) rather than a direct assignment. **Why**: `eplb_config` in `ParallelConfig` is a pydantic `Field(default_factory=EPLBConfig)`; `get_field` extracts that `default_factory` and rebuilds a proper dataclass `field(...)` with the factory intact (`config/utils.py:99-106` unwraps the `FieldInfo`). This preserves the "fresh instance per EngineArgs" semantics and avoids a shared mutable default across engine-args instances.

**(b) argparse kwargs are auto-generated from the field types.** `parallel_kwargs = get_kwargs(ParallelConfig)` (`arg_utils.py:941`) → `_compute_kwargs` (`arg_utils.py:286`). For each field, `_compute_kwargs` inspects the type hints (`arg_utils.py:292`) and picks an argparse `type`/`action`:
- `enable_eplb` is a `bool`, so it hits `arg_utils.py:343-345` → `action = argparse.BooleanOptionalAction`. That auto-creates **both** `--enable-eplb` and `--no-enable-eplb`.
- `eplb_config` is a dataclass, detected at `arg_utils.py:295-296` (`is_dataclass`), so it takes the `dataclass_cls is not None` branch (`arg_utils.py:326-337`). Its argparse `type` becomes the nested `parse_dataclass` closure (`arg_utils.py:328-335`):
  ```python
  def parse_dataclass(val, cls=dataclass_cls):
      val = _expand_json_human_readable_numbers(val)
      return TypeAdapter(cls).validate_json(val)
  ```
  So the CLI value for `--eplb-config` is parsed as **JSON** and validated straight into an `EPLBConfig` via pydantic's `TypeAdapter`. `_expand_json_human_readable_numbers` (`arg_utils.py:263-282`) first expands suffixes like `1k`/`3k` inside the JSON so you can write e.g. `{"window_size": 1k, "step_interval": 3k}`. Any pydantic `ValidationError` is re-raised as an `argparse.ArgumentTypeError` (`arg_utils.py:332-333`), so a bad field surfaces as a normal CLI error. The help text also gets the JSON tip (`arg_utils.py:337`): *"Should either be a valid JSON string or JSON keys passed individually."*

**(c) The arguments are registered.** `arg_utils.py:1093-1094`:
```python
parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
```
`enable_expert_parallel` / `-ep` is registered just above at `arg_utils.py:1061-1065`, and `--expert-placement-strategy` at `arg_utils.py:1095-1098`.

**(d) YAML `--config` and dict form are normalized.** A `--config foo.yaml` value or programmatic `EngineArgs(eplb_config={...})` arrives as a plain `dict`. `EngineArgs.__post_init__` coerces it (`arg_utils.py:729-730`):
```python
if isinstance(self.eplb_config, dict):
    self.eplb_config = EPLBConfig(**self.eplb_config)
```
So whether the config came in as a JSON string on the CLI (→ `parse_dataclass`) or as a nested mapping from YAML/kwargs (→ this coercion), it ends up as a real `EPLBConfig` instance. Note this path (`EPLBConfig(**dict)`) runs the pydantic field bounds + the `_validate_eplb_config` validator (§4.1), so invalid values are caught here too.

**(e) Handoff into `ParallelConfig`.** In `create_engine_config`, the `ParallelConfig(...)` constructor is fed the mirrored fields (`arg_utils.py:1981-1992`):
```python
enable_expert_parallel=self.enable_expert_parallel,   # arg_utils.py:1981
...
enable_eplb=self.enable_eplb,                          # arg_utils.py:1990
eplb_config=self.eplb_config,                          # arg_utils.py:1991
expert_placement_strategy=self.expert_placement_strategy,  # arg_utils.py:1992
```
At this point `ParallelConfig`'s own validators run: `_validate_parallel_config` (§5, the platform/EP/world-size/redundant-expert gates) and `__post_init__` (§7, communicator auto-select + elastic-EP gate). Only after all these pass does the config reach the workers.

**Precedence note**: because YAML is loaded into the same `EngineArgs` fields and explicit CLI flags override them via argparse, the effective order is defaults (`EPLBConfig` field defaults) → `--config` YAML → explicit CLI flags. A partial `--eplb-config '{"num_redundant_experts": 16}'` only overrides that one key; the rest keep their defaults (pydantic fills unspecified fields).

### 7. Communicator auto-selection (`communicator=None`) — `config/parallel.py:907-924`

When EPLB is enabled and `communicator` was left `None`, `ParallelConfig.__post_init__` resolves it:
```python
if self.enable_eplb and self.eplb_config.communicator is None:
    if self.enable_elastic_ep:
        self.eplb_config.communicator = "pynccl"          # parallel.py:912
    else:
        from vllm.distributed.nixl_utils import is_nixl_available
        if is_nixl_available():
            self.eplb_config.communicator = "nixl"        # parallel.py:922
        else:
            self.eplb_config.communicator = "torch_gloo"  # parallel.py:924
```
Logic, block-by-block:
- **Elastic EP → `"pynccl"`** (`parallel.py:908-912`): elastic EP requires stateless process groups, and `torch.distributed.batch_isend_irecv` doesn't support stateless mode, so PyNCCL is forced.
- **Otherwise prefer `"nixl"` if available, else `"torch_gloo"`** (`parallel.py:913-924`). The comment (`parallel.py:914-917`) is important: `torch_nccl` is **deliberately avoided** for auto-selection because "NCCL is fundamentally incompatible with async EPLB due to multi-stream conflicts, and batched isend/irecv hangs under high load" (references pytorch/pytorch#174288). So the auto path never picks NCCL. Since `use_async` defaults to `True`, the default runtime communicator is nixl-or-gloo, not NCCL.

The docstring's summary — *"None: Auto-select backend ('torch_gloo' for async, 'torch_nccl' for sync)"* (`config/parallel.py:96`) — is a **simplified/stale description**: the actual code prefers `nixl` over `torch_gloo` when NIXL is available and special-cases elastic EP to `pynccl`. Treat the code (`parallel.py:907-924`) as authoritative over that one docstring line. If you need NCCL-based transfer you must set `communicator="torch_nccl"` explicitly (and only makes sense with `use_async=False`).

### 8. Minimal usage recap

- Turn it on: `--enable-expert-parallel` (required) + `--enable-eplb`, with `TP*DP > 1`, on CUDA/ROCm.
- Tune it: `--eplb-config '{"num_redundant_experts": 16, "window_size": 1k, "step_interval": 3k}'` (JSON; human-readable suffixes allowed).
- Observe it: add `"log_balancedness": true` (accept the communication overhead), optionally `"log_balancedness_interval": N` to throttle.
- Leave `policy` (`"default"` is the only option here), `use_async` (default `True`), and `communicator` (`None`/auto) alone unless you have a specific transport requirement.

---


# 2. EPLB State Machine (eplb_state.py)

## EPLB State Machine (`distributed/eplb/eplb_state.py`, `distributed/eplb/async_worker.py`)

### Purpose

EPLB continuously rebalances MoE experts across EP (expert-parallel) ranks so no single GPU is overloaded by "hot" experts. The state machine has three responsibilities:

1. **Record** per-forward-pass expert load (token counts) into a sliding window.
2. **Decide** when enough steps have accumulated to trigger a rearrangement.
3. **Execute** the rearrangement — compute a new physical→logical expert assignment and physically move expert weights across ranks — either synchronously (blocking the forward loop) or asynchronously (on a background thread).

The critical distributed-systems constraint threaded through the whole design: **every EP rank must trigger rearrangement on the exact same step**, because rearrangement performs collective communication (`all_reduce`, all-to-all weight shuffles). If ranks disagree on when to rearrange, the collectives deadlock. This is why the trigger is driven by a *deterministic step counter* (`expert_rearrangement_step`) that is guaranteed identical across ranks — not by any per-rank measured quantity.

---

### Key data structures

Two dataclasses plus the `EplbState` controller.

**`EplbModelState`** (`eplb_state.py:90-207`) — per-model tensors (there can be more than one model, e.g. main + speculative-decode drafter). The core fields:

- `physical_to_logical_map` (`eplb_state.py:94`): shape `(num_moe_layers, num_physical_experts)`. For each physical slot on each layer, which logical expert lives there. This is the *source of truth* for the current arrangement.
- `logical_to_physical_map` (`eplb_state.py:110`): shape `(num_moe_layers, num_logical_experts, num_redundant_experts + 1)`. The inverse map — the physical slots holding each logical expert, `-1`-padded. Sparse because a logical expert may be replicated 1..N times. This is what the router reads at inference to pick a physical replica.
- `logical_replica_count` (`eplb_state.py:134`): shape `(num_moe_layers, num_logical_experts)`. Count of non-`-1` entries per logical expert — i.e. how many physical copies each logical expert has.
- `expert_load_pass` (`eplb_state.py:150`): shape `(num_moe_layers, num_physical_experts)`, `int32`. Accumulates the token count each physical expert processes *during the current forward pass*. Written by the MoE layers via a view (see `EplbLayerState.expert_load_view`, `eplb_state.py:952`).
- `expert_load_window` (`eplb_state.py:157`): shape `(window_size, num_moe_layers, num_physical_experts)`, `int32`. A circular buffer of the last `window_size` recorded passes. Note the comment at `eplb_state.py:163-169`: load is recorded for **all** physical experts (not just local ones) so statistics are consistent across dispatch backends (naive all-to-all vs DeepEP); with naive all-to-all the numbers are scaled by `dp_size`.
- `rebalanced` (`eplb_state.py:177`): **async-only** flag. Set `True` by the main thread once new maps are computed → tells the async worker to begin weight transfer. Set back to `False` once all layers are committed. Synchronization between main thread and worker relies on the GIL (`eplb_state.py:184-186`).
- `pending_result` (`eplb_state.py:199`): **async-only**. The async worker publishes one `AsyncEplbLayerResult` here after filling `expert_buffer` for a layer; the main thread consumes it in `_move_to_workspace`. "At most one result is pending at a time" (`eplb_state.py:203`) — this is a hand-off, GIL-synchronized.

**`EplbState`** controller (`eplb_state.py:210`) — the actual state machine. Its `__init__` (`eplb_state.py:215-284`) sets up the counters that drive everything:

- `expert_load_window_step` (`eplb_state.py:223`): current write index into the circular `expert_load_window`. **Per-rank** — the comment at `eplb_state.py:226-228` notes each rank may have its own value (it's local bookkeeping, not a sync point).
- `expert_load_window_size` (`eplb_state.py:230`): constant, from config.
- `expert_rearrangement_step` (`eplb_state.py:235`): steps since last rearrangement — the trigger counter. The docstring at `eplb_state.py:240-243` is the key invariant: *"all EP ranks need to have the same `expert_rearrangement_step` value to ensure synchronization. Otherwise, the rearrangement will hang at collective communication calls."*
- `expert_rearrangement_step_interval` (`eplb_state.py:245`): constant threshold, from config.
- `should_record_tensor` (`eplb_state.py:250`): a single shared scalar bool tensor. Every layer holds a reference to the *same* object, so one `.fill_()` toggles recording for all layers at once (`eplb_state.py:253-256`).
- `is_async` (`eplb_state.py:257`): sync vs async mode.
- `rearrange_event` (`eplb_state.py:261`): a `CpuGpuEvent` used to wake the async worker thread.
- `async_worker` (`eplb_state.py:265`): the background thread handle.
- `num_valid_physical_experts` (`eplb_state.py:273`): number of physical slots actually mapped to logical experts (relevant for elastic EP, where freshly-added ranks may have unmapped slots).

The `__init__` tail (`eplb_state.py:281-284`) captures the CUDA device index for the worker thread.

---

### Build / init: `add_model` (`eplb_state.py:342-472`)

Called once per model to build the initial arrangement and allocate all tensors.

- `eplb_state.py:350` validates the new model's EP config matches any already-registered models (`validate_ep_configuration`, `eplb_state.py:306`).
- `eplb_state.py:351` latches `is_async` from config (`eplb_config.use_async`).
- `eplb_state.py:353-362`: builds the initial physical→logical map via `build_initial_global_physical_to_logical_map` (`eplb_state.py:286-304`). That helper produces `[0,1,...,num_routed-1]` for the base experts, then appends redundant slots as `i % num_routed_experts` — i.e. the first `num_redundant_experts` logical experts get one extra replica each. Simple round-robin seeding; the real balancing happens later.
- `eplb_state.py:366-370`: asserts `num_redundant_experts <= 1023` (`MAX_EXPERT_REDUNDANCY`), sizing the replica dimension. The comment (`eplb_state.py:363-364`) explains the 1024/8 = 128-node ceiling; flagged `TODO(rui): make this configurable`.
- `eplb_state.py:372-386`: allocates `logical_to_physical_map` full of `-1`, then walks every physical slot and fills the inverse map + `logical_replica_count` — the initial forward/inverse consistency is established here.
- `eplb_state.py:388-413`: broadcasts the single-layer maps across all `num_moe_layers` via `unsqueeze(0).expand(...).contiguous()` — all layers start identical.
- `eplb_state.py:415-429`: allocates the zeroed `expert_load_pass` and the `expert_load_window` circular buffer; `expert_load_window_size` is read from `eplb_config.window_size` (`eplb_state.py:420`).

**The 3/4 initialization** (`eplb_state.py:431-436`) — the passage the task specifically calls out:

```python
# Set the initial progress of rearrangement to 3/4
eplb_step_interval = self.parallel_config.eplb_config.step_interval
self.expert_rearrangement_step = max(
    0, eplb_step_interval - eplb_step_interval // 4
)
self.expert_rearrangement_step_interval = eplb_step_interval
```

The counter is *not* initialized to 0. It starts at `step_interval - step_interval//4` — i.e. **3/4 of the way to the threshold**. Effect: the *first* rearrangement fires after only ~`step_interval/4` steps instead of a full interval. The intent is to react to load imbalance early during warm-up rather than waiting a full window. All subsequent intervals are the full `step_interval` (the counter resets to 0 after each rearrangement — `eplb_state.py:603`). Because `step_interval` is a config constant identical on every rank, this initial value is identical on every rank, preserving the cross-rank sync invariant. (Version note: the "3/4" fraction is hard-coded here via `// 4`; older/newer vLLM revisions may differ.)

- `eplb_state.py:438-440`: selects the policy class from `EPLB_POLICIES` by name (default `DefaultEplbPolicy`).
- `eplb_state.py:442-446`: hands `expert_load_pass`, `logical_to_physical_map`, `logical_replica_count` to the model so its MoE layers get views into these tensors.
- `eplb_state.py:447`: `_init_should_record_tensor(model)` — must run *after* `set_eplb_state` because it reads each layer's `eplb_state` (`eplb_state.py:640-642`). It allocates the single shared bool tensor once (`eplb_state.py:650-653`) and points every layer at it (`eplb_state.py:655-656`).
- `eplb_state.py:448-455`: allocates `expert_buffer` (staging area for weights in flight) and builds the `communicator` for weight transfers.
- `eplb_state.py:457-471`: assembles the `EplbModelState` and stores it keyed by `model_config.compute_hash()`.

---

### Per-step: `step()` (`eplb_state.py:474-606`)

Called once per forward pass. Walks the full state machine.

**Profile short-circuit** (`eplb_state.py:500-502`): if `is_profile`, immediately do `self.rearrange(is_profile=True)` and return. A profile step performs a *dummy* rearrangement with maximum communication cost to force allocation of the comm buffer, so real rearrangements later don't OOM (docstring `eplb_state.py:487-490`). No real weights move.

**Dummy step** (`eplb_state.py:504-507`): if `is_dummy`, zero out `expert_load_pass` for every model — load from dummy passes must not count.

**Balancedness logging** (`eplb_state.py:509-556`): every `log_balancedness_interval` steps (and only if `log_stats`), it `_sync_load_pass()` (all-reduces a *clone* of the load pass — `eplb_state.py:872-880`) then computes, per layer, `avg_tokens = mean over ranks`, `max_tokens = max over ranks`, and `balancedness = avg/max` (`eplb_state.py:523-542`). Rank 0 logs it along with steps-until-next-rearrangement (`eplb_state.py:544-556`). Pure telemetry; does not affect the state machine.

**Sliding-window record** (`eplb_state.py:558-571`): if not a dummy step, `should_record = self._should_record_current_step(...)`. When recording, for each model it copies `expert_load_pass` into `expert_load_window[expert_load_window_step]` and then zeroes the pass (`eplb_state.py:562-566`). The window index then advances and wraps (`eplb_state.py:568-571`).

`_should_record_current_step` (`eplb_state.py:608-628`) is the optimization that avoids wasted GPU work: recording is enabled only when `steps_remaining <= window_size` (`eplb_state.py:615-618`) — i.e. we're within one window of the next rearrangement, so these recorded passes will still be in the buffer when the policy reads it. If `log_stats`, it also enables recording near the next log step (`eplb_state.py:623-628`). Any earlier step would just be overwritten in the circular buffer before use (see the explanation in `EplbLayerState.should_record_tensor`, `eplb_state.py:939-943`).

**Advance the trigger counter** (`eplb_state.py:573-577`):

```python
self.expert_rearrangement_step += 1
```

The comment at `eplb_state.py:574-576` is important: this increments **even on dummy steps**, and rearrangement still fires — because all ranks must keep lockstep on collectives. Skipping the increment on some ranks would break sync.

**Async workspace commit** (`eplb_state.py:579-591`): in async mode, for each model, if `rebalanced` is set *and* `_all_ranks_result_ready(...)` returns true, call `_move_to_workspace(...)`. The `rebalanced` check must be *consistent across ranks* (comment `eplb_state.py:583-584`) or the `all_reduce` inside `_all_ranks_result_ready` will hang.

`_all_ranks_result_ready` (`eplb_state.py:824-843`) is the cross-rank barrier for async: each rank sets `has_result = int(pending_result is not None)`, then `all_reduce`s that flag (CPU group if available, else device group) and returns true only if the sum equals the group size — i.e. **every rank has a layer's transfer ready**. This ensures all ranks commit the same layer together, keeping the arrangement globally consistent.

**The trigger** (`eplb_state.py:593-604`):

```python
if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
    if self.is_async and any(rebalanced for ... in model_states):
        self._update_layer_should_record(log_stats=log_stats)
        return
    self.expert_rearrangement_step = 0
    self.rearrange()
```

- When the counter reaches the interval, we want to rearrange.
- **Async guard** (`eplb_state.py:594-602`): if async *and a previous rearrangement is still in flight* (`any(... rebalanced ...)`), we do **not** reset the counter or start a new rearrangement — we just refresh `should_record` (which is always True here since step ≥ interval) and return early. This prevents overlapping async rearrangements; the counter keeps counting past the interval until the outstanding transfer finishes.
- Otherwise (`eplb_state.py:603-604`): reset the counter to 0 and call `rearrange()`. The reset-to-0 (vs the 3/4 init) is why only the *first* period is short.

Finally `_update_layer_should_record(...)` (`eplb_state.py:606`) pushes the recomputed record flag into the shared tensor for all layers (`eplb_state.py:630-635`).

---

### `rearrange()` (`eplb_state.py:658-809`)

Computes and applies a new expert arrangement. Runs on the main thread; behavior forks on `is_async`/`is_profile`.

- `eplb_state.py:674-675`: grabs the EP process group and this rank's rank.
- `eplb_state.py:679-689`: only rank 0 (`is_main_rank`) sets up CUDA timing events (sync/profile path only) and logs the mode.

**Aggregate load into logical space** (`eplb_state.py:691-716`): for each model, it slices the window to valid physical experts (`eplb_state.py:694-696`), then `scatter_add_` accumulates physical-expert load into a `logical_expert_load_window` using `physical_to_logical_map` as the index (`eplb_state.py:704-713`). This collapses replicas of the same logical expert together. It then sums over the window dimension (`eplb_state.py:715`) to get per-rank total logical load.

**All-reduce across ranks** (`eplb_state.py:718`): `_allreduce_list(...)` (`eplb_state.py:845-870`) sums the logical load window across all EP ranks so every rank sees the same *global* load — the policy must run on identical input everywhere to produce an identical arrangement. For multiple models it concatenates, single-`all_reduce`s, and splits back (`eplb_state.py:856-869`).

**Topology sizing** (`eplb_state.py:720-748`): computes `num_replicas`, `num_groups`, `num_nodes`, `num_gpus`. The `rank_mapping` branch (`eplb_state.py:726-737`) handles elastic-EP scale-down (rebalancing onto remaining GPUs before releasing some). If `num_gpus % num_nodes != 0` it falls back to `num_nodes = 1` and disables the hierarchical algorithm (`eplb_state.py:742-748`).

**Sync vs async fork** (`eplb_state.py:751-805`) — per model:

*Sync path* (`if not self.is_async or is_profile`, `eplb_state.py:754-793`):
1. `self.policy.rebalance_experts(...)` (`eplb_state.py:756-763`) runs the balancing algorithm on `global_expert_load_window.cpu()` and returns a new `physical_to_logical_map`.
2. `rearrange_expert_weights_inplace(...)` (`eplb_state.py:766-775`) physically shuffles the expert weights across ranks (all-to-all via the communicator). **This is the blocking collective** — it happens inline in the forward loop.
3. If not profiling, `_commit_eplb_maps(...)` (`eplb_state.py:777-781`) makes the new maps live.
4. Rank 0 records/logs elapsed time (`eplb_state.py:783-793`).

*Async path* (`else`, `eplb_state.py:794-805`): it does **not** compute or move anything here. It only snapshots the global load window (`global_expert_load_window.clone()` at `eplb_state.py:799` — cloned so the worker can read it safely while the main thread continues), stuffs it plus the topology sizes into `eplb_stats` (`eplb_state.py:795-804`), and sets `rebalanced = True` (`eplb_state.py:805`). The actual policy computation and weight transfer are deferred to the background worker.

**Wake the worker** (`eplb_state.py:807-808`): in async, non-profile mode, `self.rearrange_event.record()` signals the async thread that new work is available.

---

### `_commit_eplb_maps` (`eplb_state.py:1117-1151`) and the per-layer variant

Makes a new `physical_to_logical_map` the live arrangement:
- Copies the new physical→logical map into `model_state.physical_to_logical_map` (`eplb_state.py:1127-1138`). The `src.shape[1] != dst.shape[1]` branch handles the rare case where the physical-expert count changed (GPU count changed at runtime) by *replacing* the tensor rather than copying into it.
- Recomputes the inverse map + replica count via `compute_logical_maps(...)` (`eplb_state.py:1140-1141`, defined at `eplb_state.py:994-1070`) and commits them (`_pad_out_tensor` pads the sparse map back to full width with `-1`).

`_commit_eplb_maps_for_layer` (`eplb_state.py:1080-1114`) is the async single-layer version, used because the async worker commits one layer at a time. It asserts the physical-expert count is unchanged (`eplb_state.py:1095-1099`) — async EPLB does not support elastic resize mid-transfer.

---

### Async execution: `_move_to_workspace` + `async_worker.py`

**`_move_to_workspace`** (`eplb_state.py:1154-1179`) — runs on the **main thread**, invoked from `step()` when a layer's transfer is ready:
1. Reads the worker's `pending_result` (`eplb_state.py:1158-1159`).
2. `move_from_buffer(...)` (`eplb_state.py:1160-1166`) copies the staged weights from `expert_buffer` into the live `expert_weights` for that layer.
3. `_commit_eplb_maps_for_layer(...)` (`eplb_state.py:1168-1172`) commits that layer's new maps.
4. If it was the last layer, clears `rebalanced` (`eplb_state.py:1174-1175`) — signalling the whole rearrangement is done and a new one may be scheduled.
5. Resets `pending_result = None` and records `result.consumed_event` (`eplb_state.py:1178-1179`) to unblock the worker so it can proceed to the next layer.

**`async_worker.py`** — the background thread.

- `start_async_worker` (`async_worker.py:25-50`): creates a daemon thread. `thread_target` (`async_worker.py:34-47`) pins the CUDA device (`async_worker.py:36`), creates a dedicated CUDA stream (`async_worker.py:37`) so transfers don't contend with the main compute stream, and runs `transfer_run_periodically`. Started lazily by `EplbState.start_async_loop` (`eplb_state.py:811-822`), which is a no-op unless `is_async` and only spawns one worker.

- `transfer_run_periodically` (`async_worker.py:79-148`) is the worker loop:
  - `state.rearrange_event.wait(stream=cuda_stream)` (`async_worker.py:86`) blocks until the main thread's `rearrange()` records the event.
  - For each model: snapshots `physical_to_logical_map` to CPU on the worker stream (`async_worker.py:98-99`) — synchronized against `rearrange_event` so it sees the pre-rearrange map.
  - `run_rebalance_experts(...)` (`async_worker.py:101-103`, defined `async_worker.py:53-76`): moves the cloned global load window to CPU and runs `policy.rebalance_experts(...)` — **this is where the policy computation happens in async mode** (deferred from `rearrange()`). Returns the new CPU map.
  - Inner loop over layers (`async_worker.py:113-148`), gated on `model_state.rebalanced and layer_idx < num_layers`:
    - `transfer_layer(...)` (`async_worker.py:114-124`) copies that layer's new weights into `expert_buffer`.
    - `cuda_stream.synchronize()` (`async_worker.py:128`) waits for all buffer writes to complete before publishing.
    - Creates a `consumed_event` (`async_worker.py:133`), then publishes `pending_result` (`async_worker.py:135-140`) for the main thread to pick up in `_move_to_workspace`.
    - `consumed_event.wait(stream=cuda_stream)` (`async_worker.py:145`) blocks until the main thread finishes moving the buffer into live weights and records that event — the back-pressure that guarantees `expert_buffer` isn't overwritten before it's consumed.
    - Advances `layer_idx`.

So the async design is a producer/consumer hand-off, one layer per forward pass: the worker produces (compute new map + stage weights into buffer), the main thread's `step()` consumes (move buffer → weights, commit maps) but only once `_all_ranks_result_ready` confirms *every* rank has that layer staged. This keeps the expensive collective weight movement off the critical forward path while preserving the cross-rank lockstep needed for the collectives themselves.

---

### Sync vs async — summary of the two paths

| Aspect | Sync (`is_async=False`) | Async (`is_async=True`) |
|---|---|---|
| Policy compute | Inline in `rearrange()` (`eplb_state.py:756`) | Worker thread, `run_rebalance_experts` (`async_worker.py:66`) |
| Weight move | Inline, blocking (`rearrange_expert_weights_inplace`, `eplb_state.py:766`) | Worker, per-layer `transfer_layer` (`async_worker.py:114`) |
| Commit | `_commit_eplb_maps` (whole model, `eplb_state.py:778`) | `_commit_eplb_maps_for_layer` on main thread in `_move_to_workspace` (`eplb_state.py:1168`) |
| Cross-rank sync | The inline all-to-all collectives themselves | `expert_rearrangement_step` for trigger + `_all_ranks_result_ready` per-layer barrier (`eplb_state.py:824`) |
| Overlap prevention | N/A (blocking) | `rebalanced` guard in `step()` (`eplb_state.py:594-602`) |
| Elastic resize | Supported (`_commit_eplb_maps` shape-change branch) | **Not** supported (assertion `eplb_state.py:1095-1099`) |

In both modes the *trigger* is identical and rank-deterministic: the `expert_rearrangement_step` counter starting at 3/4 of `step_interval` and resetting to 0 thereafter. That single shared counter — not any measured load — is what guarantees all ranks enter the collective rearrangement path on the same step.

---

Source files (absolute paths):
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/async_worker.py`

Version note: line numbers and the hard-coded 3/4 initialization (`// 4`) and `MAX_EXPERT_REDUNDANCY = 1023` are specific to this vendored vLLM copy; upstream vLLM at other revisions may differ.

---


# 3. Rebalance Algorithm (eplb_utils.py)

I have everything needed. Here is the documentation section.

---

# EPLB Rebalance Algorithm

## Important note on file location (version-specific)

The task pointed at `distributed/eplb/eplb_utils.py`, but **in this checkout that file does *not* contain the rebalance algorithm**. `eplb_utils.py` here holds only two helpers: `CpuGpuEvent` (a CUDA-event + `threading.Event` pair that enforces `record()`→`wait()` ordering across the main thread and the async-EPLB thread, `eplb_utils.py:16-61`) and `override_envs_for_eplb` (forces `NCCL_MAX_CTAS=8` to avoid SM-contention deadlocks between NCCL weight-exchange and cooperative MoE kernels, `eplb_utils.py:64-125`). Neither touches placement.

In this vLLM version the rearrangement math has been factored into a **policy package**:

- `distributed/eplb/policy/abstract.py` — the `AbstractEplbPolicy.rebalance_experts` interface.
- `distributed/eplb/policy/default.py` — `DefaultEplbPolicy`, the real algorithm (adapted from [DeepSeek EPLB](https://github.com/deepseek-ai/eplb)).
- `distributed/eplb/policy/__init__.py` — registry `EPLB_POLICIES = {"default": DefaultEplbPolicy}` (`policy/__init__.py:10`), validated against the `EPLBPolicyOption` config enum.

Everything below refers to `distributed/eplb/policy/default.py` unless stated otherwise. (In older vLLM releases this same code lived in `eplb_utils.py` / `rebalance_algo.py`; if you are reading a different checkout the functions may be there instead.)

## Concept and purpose

In an MoE model each token is routed to a small set of *logical* experts. When experts are sharded across GPUs (expert parallelism), a naive `expert_id → GPU` assignment produces **load imbalance**: some experts are "hot" (routed to far more often), so their GPU becomes the bottleneck while others idle. EPLB fixes this by:

1. **Recording** per-expert token load over a window of steps.
2. **Replicating** hot logical experts into multiple *physical* copies (using "redundant" physical slots), so their traffic can be split.
3. **Packing** physical experts onto GPUs so that each GPU's total load is as even as possible, while respecting a two-level (node → GPU) network hierarchy so that expert groups that talk to each other stay on the same node (fast intra-node link, e.g. NVLink).

The output is a `phy2log` map of shape `[layers, num_replicas]`: for each physical expert slot it gives the logical expert id it should serve. Physical slots are laid out contiguously per GPU, so `phy2log` simultaneously encodes *which experts live on which GPU* and *how many replicas each logical expert has*.

Two counts matter throughout:
- `num_log` = number of logical experts (fixed by the model).
- `num_phy` = `num_replicas` = number of physical slots = `slots_per_gpu * num_gpus`. The difference `num_phy − num_log` is the number of **redundant** slots available for replication.

## How the load ("weight") is computed before the algorithm runs

The algorithm is pure and stateless — it consumes a `weight` tensor of shape `[layers, num_logical_experts]`. That tensor is assembled by the caller in `eplb_state.py` before `rebalance_experts` is called:

- Per-step physical-expert load is accumulated into a rolling window `expert_load_window`.
- `logical_expert_load_window.scatter_add_(dim=-1, index=physical_to_logical_map…, src=expert_load_window)` (`eplb_state.py:704-713`) folds physical load back onto logical experts — i.e. it sums the load of all current replicas of a logical expert. This is the crucial step: the algorithm reasons in **logical** load, independent of the *current* replication.
- `global_expert_load_window = logical_expert_load_window.sum(dim=0)` (`eplb_state.py:715`) sums over the window (removing the time dimension), then `_allreduce_list` (`eplb_state.py:718`) sums across all EP ranks so every rank agrees on global logical load.

That aggregated tensor, moved to CPU, is `weight`. The call site is `eplb_state.py:756-763`:

```python
new_physical_to_logical_map = self.policy.rebalance_experts(
    global_expert_load_window.cpu(),
    num_replicas,      # = model.num_physical_experts
    num_groups,        # = model.num_expert_groups
    num_nodes,         # from get_node_count() (or rank_mapping on scale-down)
    num_gpus,          # = ep_group.size()
    eplb_model_state.physical_to_logical_map.cpu(),  # old map, for slot preservation
)
```

## The three primitives

The algorithm is built from three composable functions. I go through each, then the hierarchical driver that chains them.

### 1. `balanced_packing` — even multi-way partitioning by weight

`default.py:22-73`. Purpose (docstring `default.py:26-28`): *"Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs are as balanced as possible."* This is a **cardinality-constrained** greedy partition — every pack ends up with *exactly* the same number of items, and among those valid partitions it greedily minimizes the max pack weight.

- `default.py:38-40`:
  ```python
  num_layers, num_groups = weight.shape
  assert num_groups % num_packs == 0
  groups_per_pack = num_groups // num_packs
  ```
  Items are batched over `num_layers` (each layer is packed independently). The exact-cardinality constraint requires `num_groups` divisible by `num_packs`; `groups_per_pack` is the fixed capacity of every pack.

- `default.py:42-45` — fast path:
  ```python
  if groups_per_pack == 1:
      pack_index = np.tile(np.arange(num_groups …), (num_layers, 1))
      rank_in_pack = np.zeros_like(pack_index …)
      return pack_index, rank_in_pack
  ```
  If each pack holds exactly one item, packing is trivial: item *i* → pack *i*, rank 0. No optimization needed.

- `default.py:48` — `indices = np.argsort(-weight, axis=-1)`: sort items **descending by weight** per layer. This is the classic greedy "largest-first / LPT (longest processing time)" heuristic — placing the biggest items first gives near-optimal balance.

- `default.py:50-54`: allocate outputs `pack_index`/`rank_in_pack` (both `-1`-filled) and running state `pack_weights` (cumulative weight per pack) and `pack_items` (count per pack).

- `default.py:57-71` — the greedy loop, per layer, over items in descending-weight order:
  ```python
  pack = int(np.argmin(weights_row))          # lightest currently-open pack
  pack_index[layer_idx, group] = pack
  rank_in_pack[layer_idx, group] = items_row[pack]
  weights_row[pack] += weight[layer_idx, group]
  items_row[pack] += 1
  if items_row[pack] == groups_per_pack:
      weights_row[pack] = np.inf              # pack full → never chosen again
  ```
  Each item goes into the **lightest non-full pack**. `rank_in_pack` records its position (0…groups_per_pack−1) inside that pack. When a pack reaches capacity its weight is set to `inf` so `argmin` never picks it again — this is how the exact-cardinality constraint is enforced (`default.py:62,69-71` and the comment "*full packs are masked out by inf*").

Returns `(pack_index, rank_in_pack)`: which pack each item landed in, and its slot within that pack.

### 2. `replicate_experts` — placing the redundant replicas

`default.py:75-101`. Purpose (docstring `default.py:79-81`): *"Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized."* This decides **how many physical copies each logical expert gets**, using the spare (redundant) slots.

- `default.py:91-93`:
  ```python
  n, num_log = weight.shape
  num_redundant = num_phy - num_log
  assert num_redundant >= 0
  ```
  `num_redundant` = number of extra physical slots beyond one-per-logical. Must be non-negative (you can't have fewer physical than logical experts).

- `default.py:94-96` — initialization:
  ```python
  phy2log = np.tile(np.arange(num_phy …), (n, 1))   # slots 0..num_log-1 map to logical 0..num_log-1
  logcnt  = np.ones((n, num_log) …)                  # every logical expert starts with 1 replica
  arangen = np.arange(n …)
  ```
  The first `num_log` physical slots are pre-assigned identity (one guaranteed copy per logical expert). `logcnt[l]` tracks the current replica count of logical expert *l*.

- `default.py:97-100` — the redundancy loop, one iteration per extra slot `i` in `[num_log, num_phy)`:
  ```python
  redundant_indices = np.argmax(weight / logcnt, axis=-1)
  phy2log[:, i] = redundant_indices
  logcnt[arangen, redundant_indices] += 1
  ```
  **This is the core replication heuristic.** `weight / logcnt` is the *per-replica* load of each logical expert if its traffic were split evenly across its current replicas. `argmax` picks the logical expert with the highest per-replica load — i.e. the current bottleneck — and gives it one more replica (slot `i`), then increments its count. Adding a replica to expert *l* immediately drops its effective load from `w_l/c` to `w_l/(c+1)`, so the next iteration reconsiders who is now hottest. Greedily assigning each spare slot to the current max-per-replica expert is what "minimizes the maximum replica load."

Returns `(phy2log, logcnt)`: for each of the `num_phy` slots its logical id, and per-logical replica counts.

### 3. `rebalance_experts_hierarchical` — the full node→GPU pipeline

`default.py:103-189`. Chains the two primitives in three stages, respecting the network hierarchy. Setup and divisibility asserts (`default.py:125-132`):

```python
group_size       = num_logical_experts // num_groups   # experts per group
groups_per_node  = num_groups // num_nodes
phy_experts_per_gpu = num_physical_experts // num_gpus  # physical slots per GPU
```

Expert *groups* are the routing unit that should stay together on a node. A local helper `inverse(perm)` (`default.py:134-139`) builds the inverse of a per-row permutation via scatter (`inv[row, perm] = col`); it is used repeatedly to convert "old→new position" maps into "new→old position" maps.

**Step 1 — pack groups to nodes** (`default.py:141-156`):
```python
tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(axis=-1)
group_pack_index, group_rank_in_pack = cls.balanced_packing(tokens_per_group, num_nodes)
log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack)[..., None]
             * group_size) + np.arange(group_size)).reshape(num_layers, num_logical_experts)
mlog2log = inverse(log2mlog)
```
Group load is the sum of its members' loads. `balanced_packing(..., num_nodes)` distributes whole groups evenly across nodes (each node gets `groups_per_node` groups). `log2mlog` then relabels every logical expert into a **node-local ("mlog") ordering**: a group's node index × groups_per_node + its rank-in-node gives its node-local group slot, times `group_size` plus the offset within the group gives the expert's new contiguous index. `mlog2log` is the inverse (node-local → global logical). After this, contiguous blocks of the mlog axis belong to one node.

**Step 2 — build redundant experts within each node** (`default.py:158-165`):
```python
tokens_per_mlog = np.take_along_axis(weight, mlog2log, axis=1).reshape(
    -1, num_logical_experts // num_nodes)
phy2mlog, mlogcnt = cls.replicate_experts(
    tokens_per_mlog, num_physical_experts // num_nodes)
```
Weights are reordered into node-local layout (`take_along_axis` with `mlog2log`) and then **reshaped to fold the node dimension into the batch axis** (`-1, experts_per_node`). So `replicate_experts` runs *per node independently*, distributing that node's share of redundant slots (`num_physical_experts // num_nodes`) among that node's experts. Result: `phy2mlog` (physical slot → node-local logical) and `mlogcnt` (replica counts per node-local logical).

**Step 3 — pack physical experts to GPUs within each node** (`default.py:167-189`):
```python
tokens_per_phy = np.take_along_axis(tokens_per_mlog / mlogcnt, phy2mlog, axis=1)
pack_index, rank_in_pack = cls.balanced_packing(tokens_per_phy, num_gpus // num_nodes)
phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
pphy2phy = inverse(phy2pphy)
```
`tokens_per_mlog / mlogcnt` is each logical expert's per-replica load; `take_along_axis(..., phy2mlog)` broadcasts it onto physical slots, so `tokens_per_phy` is the **effective load of each physical replica** (`default.py:168` comment: *"Effective per-physical load = logical load divided by replica count"*). `balanced_packing(..., num_gpus // num_nodes)` then evenly distributes those physical replicas across the GPUs *within a node*. `phy2pphy` maps a physical slot to its final position (GPU index × slots_per_gpu + slot), and `pphy2phy` inverts it so we can read out slots in final GPU order.

The tail (`default.py:176-188`) translates the packed, node-local physical order back to **global logical ids**:
```python
pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, axis=1)
pphy2mlog = (pphy2mlog.reshape(num_layers, num_nodes, -1)
             + np.arange(0, num_logical_experts, num_logical_experts // num_nodes)[None,:,None]
            ).reshape(num_layers, -1)
pphy2log = np.take_along_axis(mlog2log, pphy2mlog, axis=1)
return pphy2log
```
`pphy2mlog` gathers each final physical slot's node-local logical id. The `+ np.arange(0, …, experts_per_node)` term re-adds the per-node base offset that the Step-2 reshape had stripped (each node's mlog ids were folded to `[0, experts_per_node)`), recovering global mlog ids. Finally `take_along_axis(mlog2log, …)` converts mlog → global logical. The returned `pphy2log` is `[layers, num_replicas]`: the logical expert for each physical slot, laid out GPU-by-GPU.

### `rebalance_experts` — public entry point

`default.py:274-332` (interface in `abstract.py:12-20`). Orchestrates the above:

- `default.py:303-308`: converts `weight` to `float().cpu().numpy()` (the algorithm is NumPy/CPU), and the optional `old_global_expert_indices` likewise.
- `default.py:310-319` — hierarchy fallback:
  ```python
  if num_groups % num_nodes == 0:
      phy2log_np = cls.rebalance_experts_hierarchical(weight_np, num_replicas, num_groups, num_nodes, num_ranks)
  else:
      phy2log_np = cls.rebalance_experts_hierarchical(weight_np, num_replicas, 1, 1, num_ranks)
  ```
  If groups don't divide evenly across nodes, it degenerates to a **flat/global** policy by passing `num_groups=1, num_nodes=1` (one big node, one group) — same code path, no hierarchy. The caller also guards this: `eplb_state.py:742-748` forces `num_nodes = 1` when `num_gpus % num_nodes != 0`.
- `default.py:326-329`: if an old map was supplied, run `preserve_intragpu_slots` (below).
- `default.py:331`: `torch.from_numpy(phy2log_np)` — back to a tensor for the caller.

### `preserve_intragpu_slots` — minimizing weight movement

`default.py:191-272`. Purpose (docstring `default.py:198-203`): reorder the *within-GPU* slot assignment of the new map so that a logical expert that stays on the same GPU keeps its **previous physical slot**, avoiding a needless weight copy. This is a pure cosmetic reshuffle of slots inside each GPU — it does **not** change which experts are on which GPU (that was decided by packing), only their slot order, so it does not affect load balance. It runs only when GPU count and slots-per-GPU are unchanged (`default.py:206-207` guard: `num_phy_experts % num_ranks != 0` → return unchanged).

Per GPU (`default.py:215-270`), over its slot range `[start, end)`:
- **First pass** (`default.py:225-241`): for each old slot, find a new-local slot holding the *same logical id* that hasn't been claimed (`matches = (new_local == old_local[:, slot_idx]) & ~used_new_indices`), and place it at the old slot position (`post_phy2log[..., start + slot_idx] = …`). Marks that source as `used` and the destination as `preserved`. This vectorizes across layers with `argmax(matches)` picking the first match per layer.
- **Second pass** (`default.py:243-270`): incoming experts (not preserved) fill the leftover slots. It builds priority arrays (`np.where(mask, idx_base, sentinel)`), `argsort`s them to get ordered available source/destination positions, takes `min(remaining, fill)` of them per layer, and assigns `post_phy2log[layer, start + dst_pos] = new_local[layer, src_pos]`. This just packs the remaining new experts into the remaining holes in a stable order.

The result is the same GPU→experts assignment as the algorithm produced, but with slot positions chosen to maximize overlap with the old layout — so `rearrange_expert_weights_inplace` (`eplb_state.py:766`) can skip copying weights for experts that didn't actually move.

## End-to-end summary

Given per-logical-expert load `weight[layers, num_log]`:

1. **Groups → nodes**: `balanced_packing` on group loads spreads expert groups evenly across nodes, keeping each group intact (hierarchy locality). Produces a node-local relabeling `mlog2log`.
2. **Replicate within node**: `replicate_experts` hands each node's spare physical slots to its hottest experts (by `weight/logcnt`), minimizing max per-replica load. Produces `phy2mlog`, `mlogcnt`.
3. **Physical → GPUs**: effective per-replica load `weight/mlogcnt` is packed by `balanced_packing` across GPUs within each node, so every GPU carries near-equal load. Indices are mapped back to global logical ids to yield `phy2log[layers, num_replicas]`.
4. **(Optional)** `preserve_intragpu_slots` reorders within-GPU slots to match the old map and avoid weight copies.

The two design ideas that carry the math: **greedy largest-first packing into the lightest non-full bin** (for even partitioning under an exact-cardinality constraint), and **greedily giving each spare replica to the max `load/replica_count` expert** (for load-minimizing replication). All work is per-layer-batched NumPy on CPU; the only stochastic input is the recorded load window.

Relevant file paths (absolute):
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/default.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/abstract.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/policy/__init__.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py` (load aggregation + call site)
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_utils.py` (does NOT contain the algorithm in this version)

---


# 4. Communicator Backends (eplb_communicator.py)

# EPLB Communicators (`distributed/eplb/eplb_communicator.py`)

## Concept and purpose

EPLB (Expert Parallel Load Balancing) periodically rebalances which physical MoE experts live on which rank. When the balancer decides that expert *E* should move from rank *A* to rank *B*, the **expert weight tensors** for *E* must be physically transferred from *A*'s GPU memory to *B*'s. The EPLB *communicator* is the abstraction that performs those transfers.

The file defines one abstract base class (`EplbCommunicator`) and four concrete backends, plus a factory (`create_eplb_communicator`) that picks a backend based on device placement, group type (static vs. elastic), and an explicit backend name:

| Backend | Class | Transport | Where communication happens |
|---|---|---|---|
| `torch_nccl` | `TorchDistNcclEplbCommunicator` | `torch.distributed` isend/irecv (NCCL) | in `execute()` |
| `torch_gloo` | `TorchDistGlooStagedEplbCommunicator` | gloo P2P via CPU staging | in `execute()` |
| `nixl` | `NixlEplbCommunicator` | NIXL/RIXL RDMA READ (zero-copy) | eagerly in `add_recv()`, waited in `execute()` |
| `pynccl` | `PyNcclEplbCommunicator` | PyNCCL ncclSend/ncclRecv | streamed across `add_send`/`add_recv`, closed in `execute()` |

The common calling protocol used by the rebalancer (in `move_to_buffer`-style code) is:

1. Optionally `set_transfer_context(old_indices, layer_idx)` — only NIXL uses it.
2. Enqueue transfers with `add_send(...)` (on the source rank) and `add_recv(...)` (on the destination rank).
3. Call `execute()` — after it returns, all data is guaranteed present in the destination buffers.

Note the send/recv asymmetry across backends: for NCCL/gloo/PyNCCL, both sides enqueue matching ops; for NIXL, only the receiver acts (`add_recv`), because NIXL uses receiver-initiated RDMA READ — `add_send` is a no-op.

---

## Imports and the NIXL availability probe

`3rdparty/vllm/distributed/eplb/eplb_communicator.py:16-20` imports the torch.distributed P2P primitives (`P2POp`, `ProcessGroup`, `batch_isend_irecv`) used by the NCCL/gloo backends. `:22-33` pull in NIXL helpers, the PyNCCL wrapper, group-state helpers (`GroupCoordinator`, `get_pp_group`, `is_local_first_rank`), the `StatelessGroupCoordinator` (elastic EP), and `is_weak_contiguous` (a contiguity check that permits size-1 dims etc.).

`:40-42`:
```python
def has_nixl() -> bool:
    return nixl_utils.NixlWrapper is not None
```
NIXL is optional; `NixlWrapper` is `None` when the package is not installed. The factory calls `has_nixl()` before attempting the NIXL backend.

---

## `EplbCommunicator` (abstract base) — `:45-95`

`:48-64` declare the two abstract enqueue methods. Both take a list of tensors (all the weight tensors for a single expert on a single layer), the peer rank, and an `expert_id`. Note that `expert_id` is documented as **unused** by the NCCL/gloo/PyNCCL backends (see their signatures) — it exists in the interface solely because NIXL needs it to look up the physical source row.

`:66-73` — `execute()` docstring states the key semantic contract: *"Some backends perform communication here; others (e.g. NIXL) issue transfers eagerly in add_recv and only wait here. On return, all data is available in the destination buffers."* This is the invariant every backend must uphold.

`:75-82` — `set_transfer_context(old_indices, layer_idx)` is a concrete **no-op by default** (`# noqa: B027` suppresses the flake8 warning about an empty non-abstract method). Only NIXL overrides it; it provides the per-layer "which physical row holds which expert" context that NIXL needs to compute RDMA source addresses inside `add_recv`.

`:84-88` — `needs_profile_buffer_reservation` defaults to `True`. During vLLM's memory-profiling pass, most backends need a dummy collective to reserve their communication buffers so profiling accounts for that memory. NIXL overrides this to `False` (`:310-312`) because it does zero-copy RDMA and reserves no such buffer.

`:90-91` — `set_stream()` stores a CUDA stream on `self._cuda_stream`; the CUDA backends run their ops inside `torch.cuda.stream(self._cuda_stream)`. NIXL overrides this to a no-op (`:328-329`) since it has no CUDA stream to drive.

`:93-95` — `_log_initialized()` logs the class name once per node (`is_local_first_rank()` gate).

---

## `TorchDistNcclEplbCommunicator` — `:98-152`

The simplest backend: it just accumulates `P2POp` objects and flushes them with `batch_isend_irecv`.

- `:101-109` `__init__` stores `ep_group` (the device/NCCL process group), optional CUDA stream, and an empty `self._p2p_ops` list.
- `:111-125` `add_send` — for each tensor, append a `P2POp(torch.distributed.isend, tensor, dst_rank, self._ep_group)`. Nothing is sent yet; the op is deferred.
- `:127-141` `add_recv` — symmetric; appends `P2POp(torch.distributed.irecv, ...)`.
- `:143-152` `execute` — early-returns if no ops. Otherwise, under the configured CUDA stream, `batch_isend_irecv(self._p2p_ops)` issues them all as one NCCL batch, then `req.wait()` on each returned request blocks until complete. The `finally` clears the op list so the communicator is reusable. Batching all sends+recvs together is what lets NCCL avoid deadlocking on ordered pairwise P2P.

---

## `TorchDistGlooStagedEplbCommunicator` — `:155-238`

gloo has no GPU transport, so this backend **stages through CPU**: copy GPU→CPU, do gloo P2P on CPU tensors, copy CPU→GPU.

- `:158-166` `__init__` stores the `cpu_group` (a gloo process group), optional stream, and `self._ops` as a list of `(op_name, tensor, peer_rank)` triples. Note it records the abstract op, not a `P2POp` — the actual staging happens lazily in `execute`.
- `:168-184` `add_send`/`add_recv` just record `("send", tensor, dst_rank)` / `("recv", tensor, src_rank)`.
- `:186-238` `execute`:
  - `:193-215` `build_ops()` walks the recorded ops. For a **send**, it does `cpu_tensor = tensor.to(device="cpu", non_blocking=True)` (async D2H copy) and appends an isend of that CPU tensor. For a **recv**, it allocates `torch.empty_like(tensor, device="cpu")`, appends an irecv into it, and records `(dst_gpu_tensor, cpu_tensor)` in `recv_staging` so it can be copied back later.
  - `:217-219` runs `build_ops()` inside the CUDA stream (so the D2H copies are enqueued there).
  - `:223-228` — **critical synchronization**: before issuing gloo ops it synchronizes the CUDA stream (or the current stream) so all D2H copies have actually landed in the CPU buffers. Sending a not-yet-populated staging buffer would transmit garbage.
  - `:230-232` `batch_isend_irecv(p2p_ops)` on the gloo group, then wait on each request.
  - `:234-238` — for every received CPU tensor, `dst_tensor.copy_(cpu_tensor, non_blocking=True)` copies it back to GPU under the CUDA stream. (There is no explicit sync after this H2D copy here — correctness relies on the caller ordering subsequent GPU work on the same stream. This is a subtle version-specific detail worth noting.)

---

## `NixlEplbCommunicator` — `:241-571`

This is the most involved backend and the focus of the task. It uses **NIXL/RIXL zero-copy RDMA READ**: every rank pre-registers all of its expert weight memory with NIXL once; at rebalance time, a receiving rank issues an RDMA READ that pulls the needed expert row directly out of a remote rank's registered weight memory into its local receive buffer. There is no explicit send — hence `add_send` is a no-op.

### `__init__` — `:244-308`

- `:250-256` — assert both `all_expert_weights` and `expert_buffer` are non-empty, and that `NixlWrapper` is importable (else `RuntimeError("NIXL/ RIXL is unavailable.")`).
- `:258-260` store the **CPU** process group (used for metadata all-gathers and the post-transfer barrier), plus `world_size`/`rank` derived from it.
- `:262-265`:
  - `self._all_expert_weights` — the full `(num_layers)(num_tensors_per_layer)` weight tensors.
  - `self._expert_buffer` — pre-allocated receive buffers (one per weight tensor of a *single* layer).
  - `self._num_local_experts = all_expert_weights[0][0].shape[0]` — inferred from dim 0 of the first weight tensor (the leading dim indexes local experts).
  - `self._device` — the device of the first tensor.
- `:267-279` — validation loop: every expert weight tensor must be `is_weak_contiguous` (RDMA needs a linear byte layout) and on the same `self._device`; every `expert_buffer` tensor must also be contiguous.
- `:281-286` — transfer state:
  - `self._xfer_entries: list[tuple[int, int, int]]` — in-flight READs as `(local_dlist, remote_dlist, xfer_handle)`; filled by `add_recv`, drained by `execute`.
  - `self._expert_to_src_row: list[dict[int, int]] | None` — per-rank `{expert_id: physical_row}`, set by `set_transfer_context`.
  - `self._layer_idx: int | None` — the layer currently being transferred.
- `:288-294` — build a NIXL agent config (`capture_telemetry=False` if the config helper exists) and construct the wrapper: `self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)`. Each rank is a distinct NIXL agent.
- `:295-302` — `self._nixl_memory_type = "VRAM"` (weights live in GPU VRAM); `self._registered_descs` (registration handles, freed in `__del__`); `self._remote_agents: dict[int, str]` (peer rank → NIXL agent name); and `self._remote_send_meta` — `peer -> (layer, tensor_idx) -> (base_ptr, bytes_per_expert, dev_id)`, the remote address book used to compute RDMA source addresses.
- `:304-308` — the ordered init sequence, each wrapped in `_init_step` for uniform error labeling:
  ```python
  self._cuda_device_id = int(self._device.index or 0)
  self._init_step("buffers", self._init_registered_buffers)
  self._init_step("agents", self._init_remote_agents)
  self._init_step("send meta", self._exchange_remote_send_meta)
  self._log_initialized()
  ```
  Order matters: buffers must be registered before agents exchange metadata, and agents must be known before send-meta is exchanged (send-meta iterates `self._remote_agents`).

### `_init_step` — `:314-319`

A static wrapper that runs `fn(*args, **kwargs)` and, on any exception, re-raises as `RuntimeError(f"NIXL EPLB init failed: {name}")` with `from exc` to preserve the cause. This is why init failures surface as, e.g., `"NIXL EPLB init failed: buffers"`.

### `_make_agent_name` — `:321-326`

Builds a deployment-unique agent name: `f"eplb-{self._rank}{pp_suffix}-{uid}"`, where `pp_suffix` is `-pp{pp_rank}` only when pipeline-parallel size > 1, and `uid` is 8 hex chars from `uuid.uuid4()`. The rank disambiguates peers; the PP suffix disambiguates PP stages; the uuid prevents name collisions across restarts/re-inits.

### `_init_registered_buffers` — `:417-425` (`register_memory` at `:424`)

```python
all_tensors: list[torch.Tensor] = []
for layer_tensors in self._all_expert_weights:
    all_tensors.extend(layer_tensors)
all_tensors.extend(self._expert_buffer)
descs = self._nixl_wrapper.get_reg_descs(all_tensors)
self._nixl_wrapper.register_memory(descs)
self._registered_descs.append(descs)
```
It flattens **all layers' weight tensors plus the receive buffers** into one list, turns them into NIXL registration descriptors (`get_reg_descs`), and registers that memory with NIXL (`register_memory` at `:424`). Registration pins/exposes the memory for RDMA — weights so remote ranks can READ them, buffers so local READs can land in them. The handle is kept in `self._registered_descs` for later deregistration in `__del__`.

### `_init_remote_agents` — `:402-415`

```python
local_metadata = self._nixl_wrapper.get_agent_metadata()
gathered_metadata = [None] * self._world_size
torch.distributed.all_gather_object(gathered_metadata, local_metadata, group=self._cpu_group)
for peer in range(self._world_size):
    if peer == self._rank:
        continue
    peer_metadata = gathered_metadata[peer]
    assert peer_metadata is not None
    self._remote_agents[peer] = self._nixl_wrapper.add_remote_agent(peer_metadata)
```
Each rank serializes its own NIXL agent metadata (connection info) and all-gathers it over the CPU group. For every peer except itself, it registers the peer's agent via `add_remote_agent`, storing the returned agent name in `self._remote_agents[peer]`. After this, the local NIXL agent knows how to reach every other rank's agent.

### `_exchange_remote_send_meta` — `:427-469`

This builds the address book that lets a receiver compute exactly where a given expert's bytes live in a remote rank's registered weight memory.

- `:430-438` — build `local_meta`: for each `(layer_idx, tensor_idx)`, record `(t.data_ptr(), nbytes_per_expert, cuda_device_id)`, where `nbytes_per_expert = t.nbytes // self._num_local_experts`. Because dim 0 of each weight tensor indexes local experts and the tensor is contiguous, expert *r*'s bytes start at `data_ptr() + r * nbytes_per_expert`. This per-expert stride is the key quantity.
- `:443-448` — all-gather `local_meta` over the CPU group into `gathered_meta`.
- `:450-468` — validation for each remote agent's peer meta:
  - `:454-459` — the set of `(layer, tensor)` keys must match exactly across ranks; a mismatch raises `"NIXL EPLB metadata key mismatch with rank {peer}..."`. This guards against ranks with different layer/tensor topology.
  - `:460-468` — for each key, the per-expert stride (`nbytes_per_expert`) must match between local and peer; a mismatch raises `"NIXL EPLB nbytes_per_expert mismatch..."`. This ensures the receiver's row arithmetic is valid against the sender's layout.
  - `:469` — stores the validated peer meta into `self._remote_send_meta[peer]`.

### `set_transfer_context` — `:341-355`

Called once per layer before the layer's `add_recv` calls:
```python
assert not self._xfer_entries, (... "execute() was not called after previous add_recv() calls")
self._layer_idx = layer_idx
n = self._num_local_experts
rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
self._expert_to_src_row = [
    {int(eid): i for i, eid in enumerate(row) if eid != -1}
    for row in rank_experts
]
```
- `:344-348` — asserts no transfers are still pending (i.e., the previous layer was properly `execute()`d).
- `:349-355` — `old_indices` is the *pre-rebalance* physical placement flattened as `(world_size * num_local_experts)`. It is reshaped to `(world_size, num_local_experts)` so `rank_experts[r]` is the list of expert IDs physically resident on rank *r*. For each rank it builds `{expert_id: row_index}`, skipping `-1` (empty slot). This mapping lets `add_recv` translate "I want expert *E* from rank *src*" into "read row *i* of rank *src*'s weights."

### `add_send` — `:331-339`

A no-op. Comment `:337-338`: *"NIXL READ is receiver-initiated. The sender's expert weights are pre-registered and always readable in-place."* The sender does nothing at transfer time.

### `add_recv` — `:357-400` (issues the RDMA READ eagerly)

```python
assert self._expert_to_src_row is not None and self._layer_idx is not None, (...)
src_row = self._expert_to_src_row[src_rank][expert_id]
layer_idx = self._layer_idx
local_descs, remote_descs = [], []
for t_idx, t in enumerate(tensors):
    send_base, send_stride, remote_dev = self._remote_send_meta[src_rank][(layer_idx, t_idx)]
    assert t.nbytes == send_stride, (...)
    local_descs.append((t.data_ptr(), t.nbytes, self._cuda_device_id))
    remote_descs.append((send_base + src_row * send_stride, send_stride, remote_dev))
local_h, remote_h, xfer_h = self._create_peer_xfer(src_rank, local_descs, remote_descs)
self._nixl_wrapper.transfer(xfer_h)
self._xfer_entries.append((local_h, remote_h, xfer_h))
```
- `:366-368` — requires context to be set.
- `:369` — resolve `src_row` = the physical row on `src_rank` holding `expert_id`.
- `:374-394` — for each destination tensor `t` (the receive buffer for one weight tensor):
  - Look up the remote layout `(send_base, send_stride, remote_dev)` from `_remote_send_meta`.
  - `:378-380` — assert the local destination tensor is exactly one expert's worth of bytes (`t.nbytes == send_stride`); i.e., the receive buffer holds a single expert row.
  - `:381-387` — the **local descriptor** is `(dst_data_ptr, nbytes, local_dev)`.
  - `:388-394` — the **remote descriptor** is `(send_base + src_row * send_stride, send_stride, remote_dev)` — the exact byte range of the wanted expert in the remote rank's registered weights. This is the row-address arithmetic enabled by `_exchange_remote_send_meta`.
- `:396-399` — build the prepped transfer via `_create_peer_xfer` and **issue it immediately** with `self._nixl_wrapper.transfer(xfer_h)`. This is the eager behavior noted in the base `execute` docstring — the RDMA READ starts here, overlapping with the rest of the Python enqueue loop.
- `:400` — record the three handles for waiting/cleanup.

### `_create_peer_xfer` — `:487-524`

Builds a single batched READ across multiple descriptors from one peer:
- `:500-506` — turn `local_descs` into a NIXL descriptor list (`get_xfer_descs`) and prep it under the local init agent (`prep_xfer_dlist("NIXL_INIT_AGENT", ...)`).
- `:508-514` — same for `remote_descs`, prepped against the peer's agent name `self._remote_agents[src]`.
- `:516-523` — `make_prepped_xfer("READ", local_handle, indices, remote_handle, indices)` creates the transfer handle; `indices = range(len(local_descs))` pairs each local descriptor with the same-index remote descriptor. Returns `(local_handle, remote_handle, xfer_handle)`.

### `_wait_for_all_transfers` — `:471-485`

Polls until all issued READs finish:
```python
pending = set(handles)
while pending:
    completed = []
    for handle in pending:
        state = self._nixl_wrapper.check_xfer_state(handle)
        if state == "DONE":
            completed.append(handle)
        elif state != "PROC":
            raise RuntimeError(f"NIXL transfer failed with state={state}")
    for handle in completed:
        pending.remove(handle)
    if pending:
        time.sleep(0.0005)
```
It treats `"DONE"` as complete, `"PROC"` as still in progress, and anything else as failure. It sleeps 0.5 ms between polling passes to avoid a busy spin. (This is a CPU-side busy-poll; there is no callback/event mechanism.)

### `execute` — `:526-551`

```python
assert self._layer_idx is not None or not self._xfer_entries, (...)
try:
    self._wait_for_all_transfers([x[2] for x in self._xfer_entries])
    torch.distributed.monitored_barrier(group=self._cpu_group, timeout=timedelta(minutes=5))
finally:
    for local_h, remote_h, xfer_h in self._xfer_entries:
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_xfer_handle(xfer_h)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(local_h)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(remote_h)
    self._xfer_entries.clear()
    self._expert_to_src_row = None
    self._layer_idx = None
```
- `:532` — wait for every in-flight READ (using the third element, the xfer handle) to reach `"DONE"`.
- `:534-540` — a **post-READ `monitored_barrier`** on the CPU group. Comment: *"Correctness fence for zero-copy: prevents overwrite-while-remote-read race."* Because READs pull directly from remote registered weight memory, no rank may start mutating its weights (the next rebalance step) until *all* ranks have finished reading from everyone — the barrier enforces that global ordering. The 5-minute timeout guards against a hung peer.
- `:541-551` — `finally` releases every NIXL transfer handle and both dlist handles (each guarded by `contextlib.suppress(Exception)` so cleanup never masks the real error), then resets `_xfer_entries`, `_expert_to_src_row`, and `_layer_idx` so the communicator is ready for the next layer.

### `__del__` — `:553-571`

Best-effort teardown, everything wrapped in nested `contextlib.suppress(Exception)`:
- `:554-561` — release any straggler transfer/dlist handles left in `_xfer_entries`.
- `:562-566` — `deregister_memory` for each registered desc set, then clear the list.
- `:567-571` — `remove_remote_agent` for each known peer agent, then clear the map.
This mirrors the resources allocated in `_init_registered_buffers` and `_init_remote_agents`.

---

## `PyNcclEplbCommunicator` — `:574-615`

Uses vLLM's own `PyNcclCommunicator` and NCCL group semantics.
- `:577-585` `__init__` stores the `pynccl_comm`, optional stream, and `self._group_started = False`.
- `:587-590` `_ensure_group_started` — lazily calls `self._pynccl_comm.group_start()` on the first send/recv and flips the flag. `group_start`/`group_end` bracket a set of NCCL calls so they are fused into one coalesced group operation.
- `:592-600` `add_send` — ensure group started, then `self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)` per tensor. Sends are issued immediately (inside the open group), not deferred.
- `:602-610` `add_recv` — symmetric with `.recv(...)`.
- `:612-615` `execute` — if a group was started, `group_end()` closes it (which is where NCCL actually completes the coalesced send/recv batch) and resets the flag. If nothing was enqueued, it does nothing.

Unlike the NCCL `P2POp` backend, PyNCCL issues ops eagerly within an open group and completes them at `group_end()`.

---

## `create_eplb_communicator` factory — `:618-734`

Selects and constructs a backend. Signature `:618-623`:
```python
def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str | None,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
) -> EplbCommunicator:
```

**Default backend** — `:644-645`: if `backend is None`, it becomes `"torch_nccl"`.

**Device detection** — `:647-653`:
```python
first_layer = expert_weights[0] if expert_weights else []
tensor_device_type = first_layer[0].device.type if first_layer else "cpu"
torch_group = (
    group_coordinator.cpu_group
    if tensor_device_type == "cpu"
    else group_coordinator.device_group
)
```
It inspects the first weight tensor's device. If weights are on CPU it uses the CPU process group; otherwise the device (GPU/NCCL) group. `torch_group` is only consumed by the `torch_nccl` branch.

**`_create_pynccl` closure** — `:655-689`: validates and builds a `PyNcclEplbCommunicator`:
- `:656-660` — refuses CPU tensors (PyNCCL is CUDA-only).
- `:661-674` — checks every tensor dtype is representable in NCCL (`ncclDataTypeEnum.supports_torch_dtype`); unsupported dtypes raise with the offending list.
- `:676-683` — pulls `pynccl_comm` off the device communicator and requires it to be present, not `disabled`, and `available`.
- `:684-689` — constructs the communicator, wrapping any failure in a descriptive `RuntimeError`.

**Elastic-EP (stateless) promotion rules** — `:691-704`:
```python
is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
if is_stateless:
    if backend not in ("torch_nccl", "pynccl"):
        raise ValueError("Elastic EP requires 'torch_nccl' or 'pynccl' ...")
    if backend == "torch_nccl":
        logger.warning("Stateless elastic EP requires PyNCCL backend. Forcing EPLB communicator to 'pynccl'.")
        backend = "pynccl"
    return _create_pynccl()
```
For elastic EP (`StatelessGroupCoordinator`), only `torch_nccl` and `pynccl` are permitted; `nixl`/`torch_gloo` raise `ValueError`. Crucially, `torch_nccl` is **silently promoted to `pynccl`** (with a warning) — the stateless path only functions over PyNCCL — and the function returns a PyNCCL communicator regardless.

**Static-group branches** — `:706-734`:
- `:706-725` `nixl` — requires `has_nixl()` (else RuntimeError) and CUDA-alike, non-CPU tensors (else RuntimeError). Builds `NixlEplbCommunicator(cpu_group=..., all_expert_weights=expert_weights, expert_buffer=expert_buffer)`, wrapping failures in `RuntimeError(f"Failed to initialize NixlEplbCommunicator ({exc}).")`.
- `:726-729` `torch_gloo` — `TorchDistGlooStagedEplbCommunicator(cpu_group=group_coordinator.cpu_group)`.
- `:730-731` `torch_nccl` — `TorchDistNcclEplbCommunicator(ep_group=torch_group)` (the CPU group if weights are on CPU, else the device group).
- `:732-733` `pynccl` — delegates to `_create_pynccl()`.
- `:734` — any other name raises `ValueError(f"Unknown EPLB communicator backend: {backend}")`.

### Selection summary

- `backend=None` → `torch_nccl`.
- Stateless/elastic EP → forced to `pynccl` (`torch_nccl` promoted with warning; `nixl`/`torch_gloo` rejected).
- CPU tensors → `torch_nccl` uses the CPU group; `torch_gloo` also works; `nixl`/`pynccl` reject CPU.
- `nixl` → requires the NIXL package plus CUDA-alike GPU tensors.

---

## Notes / caveats

- `expert_id` is part of the abstract `add_send`/`add_recv` signature but is only meaningful to NIXL; the other three backends explicitly annotate it `# unused`.
- `set_transfer_context` and the receiver-only `add_recv` flow are NIXL-specific; the other backends ignore context and require matching `add_send`/`add_recv` pairs.
- The gloo backend's final H2D copy (`:236-238`) is enqueued on the CUDA stream without an explicit post-copy synchronize in `execute`; correctness depends on the caller ordering downstream GPU work on the same stream. This is an implementation detail that could differ across versions.
- `_wait_for_all_transfers` is a CPU busy-poll with a 0.5 ms sleep; on very large transfer counts this polling overhead is nonzero but bounded.
- NIXL relies on weight tensors being contiguous with dim 0 = local-expert axis and a uniform per-expert byte stride across ranks; both are enforced at init (`is_weak_contiguous` checks and the `nbytes_per_expert` cross-rank equality check). If a model laid out expert weights differently, this backend would not apply.

---


# 5. MoE Integration & Process Group

I have everything needed to write a precise, line-by-line reference section. Here is the documentation.

---

## EPLB ↔ MoE Integration and the EPLB Process Group

### Concept and purpose

EPLB (Expert Parallel Load Balancing) attacks a specific problem in Mixture-of-Experts inference under Expert Parallelism (EP): with a static, even split of experts across GPUs, real traffic is skewed — a handful of "hot" logical experts get most of the tokens, so the GPUs holding them become the bottleneck while others idle. EPLB fixes this by decoupling **logical experts** (the E experts the model was trained with) from **physical experts** (the E + R replica slots actually laid out across ranks, R = redundant experts). It then (a) *replicates* hot logical experts into extra physical slots, (b) *reassigns* which physical slot lives on which rank, and (c) periodically *rearranges* the actual weight tensors between GPUs to match measured load.

Three distinct pieces of state make this work, and the MoE layer touches all three:

- **`expert_map`** — a per-rank global→local physical index table. It is the *EP* routing mechanism (exists even without EPLB) and tells the kernel which physical experts are resident locally.
- **`logical_to_physical_map` / `logical_replica_count`** — the *EPLB* redirection tables. Routing produces *logical* expert IDs; EPLB rewrites them to *physical* IDs (choosing among replicas) before dispatch.
- **`expert_load_view`** — a counter tensor the forward pass increments per physical expert, feeding the rebalance policy.

Weight movement during rearrangement runs on a **dedicated process group** (`get_eplb_group()`), separate from the EP forward-pass group, so collective weight transfers cannot interleave/deadlock with MoE forward collectives.

Below, `file:line` paths are repo-relative under `3rdparty/vllm`.

---

### 1. The two-level index model: physical vs logical

`distributed/eplb/eplb_state.py:90-148` defines `EplbModelState`, whose docstrings are the authoritative spec:

- `physical_to_logical_map` — shape `(num_moe_layers, num_physical_experts)`. Entry `[layer, p]` = the logical expert that physical slot `p` currently holds. Example given for 6 physical / 4 logical / 3 ranks: `[[0,1,2,3,0,1],[0,2,0,1,0,3]]` — note logical experts 0 and 1 are replicated (appear twice) in layer 0.
- `logical_to_physical_map` — shape `(num_moe_layers, num_logical_experts, num_redundant_experts + 1)`, a **sparse** table padded with `-1`. Entry `[layer, l, :]` lists every physical slot holding logical expert `l`. This is the table the forward path reads to pick a replica.
- `logical_replica_count` — shape `(num_moe_layers, num_logical_experts)`, "exactly the non-`-1` count in the `logical_to_physical_map`."

The **initial** (pre-rearrange) physical→logical layout is built by `FusedMoE.make_expert_params_mapping` at weight-load time — `model_executor/layers/fused_moe/layer.py:1336-1380`:

```python
physical_to_logical_map = (
    EplbState.build_initial_global_physical_to_logical_map(
        num_experts, num_redundant_experts
    )
)
...
f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{base_layer}",
```

- `layer.py:1346` `num_physical_experts = num_experts + num_redundant_experts`.
- `layer.py:1352-1356` the initial map is `[0,1,...,num_experts-1]` followed by redundant slots (see `eplb_state.py:300-304`, which appends `num_routed_experts + i` style filler — the redundant slots start as copies).
- `layer.py:1370` is the load-time contract: the loop iterates `expert_id` over **physical** slots (`layer.py:1374`), but the checkpoint tensor name is keyed by the **logical** id `physical_to_logical_map[expert_id]`. So physical slot 5 that initially mirrors logical expert 1 loads logical-1's checkpoint weights. This is what lets a redundant physical slot be initialized as a genuine replica.

---

### 2. `expert_map` — the EP global→local routing table

This layer is separate from EPLB and exists whenever `ep_size > 1`. `model_executor/layers/fused_moe/expert_map_manager.py:22-113` (`determine_expert_map`) builds it:

- `expert_map_manager.py:63-64` `ep_size == 1` → returns `(global_num_experts, None, None)`: no map needed, all experts local.
- `expert_map_manager.py:67-69` even split; remainder goes to low ranks (`ep_rank < remainder` gets one extra).
- `expert_map_manager.py:72` `expert_map = torch.full((global_num_experts,), -1, ...)` — a full-width table of `-1` (meaning "not on this rank").
- `expert_map_manager.py:75-79` for `"linear"` placement: this rank owns a contiguous block `[start_idx : start_idx+local_num_experts]`, and those entries are filled with `arange(0, local_num_experts)` — i.e. the **local** index. So `expert_map[global_id]` = local physical index if resident, `-1` otherwise. This is exactly the semantics documented at `expert_map_manager.py:297-307`.
- `expert_map_manager.py:80-87` `"round_robin"` variant instead assigns `global_id % ep_size == ep_rank` slots to this rank.
- `expert_map_manager.py:95-111` builds the ROCm-AITER `expert_mask` variant (a 0/1 mask plus a sentinel `-1` slot), used only when AITER fusion is on.

**EPLB-relevant constraint** (`expert_map_manager.py:116-149`, `determine_expert_placement_strategy`): round-robin is silently downgraded to linear when `enable_eplb` is True (`expert_map_manager.py:126-127` requires `num_redundant_experts == 0 and not enable_eplb`). So **with EPLB, placement is always linear.** This is important: EPLB moves experts by editing `physical_to_logical_map` and physically shuffling weights, not by changing the EP block layout.

`ExpertMapManager.expert_map` (`expert_map_manager.py:296-307`) exposes the tensor; `map_global_to_local` (`expert_map_manager.py:336-352`) is the scalar lookup returning `-1`-safe local ids.

---

### 3. How the FusedMoE layer wires all of this at construction

`model_executor/layers/fused_moe/layer.py`, `__init__`:

- `layer.py:182-183`
  ```python
  self.global_num_experts = num_experts + num_redundant_experts
  self.logical_num_experts = num_experts
  ```
  This is the split: `global_num_experts` counts **physical** slots (includes redundant); `logical_num_experts` is the model's real expert count. Every downstream size is derived from these.

- `layer.py:200-212` EPLB gating:
  ```python
  self.eplb_state: EplbLayerState | None = None
  if enable_eplb:
      if self.use_ep and self.global_num_experts % self.ep_size != 0:
          raise ValueError(... "even distribution of experts ...")
      self.eplb_state = EplbLayerState()
  else:
      assert not self.use_ep or num_redundant_experts == 0, (
          "Redundant experts are only supported with EPLB.")
  ```
  `layer.py:202` enforces `global_num_experts % ep_size == 0` — EPLB requires a *clean* even physical distribution (no remainder rank). `layer.py:208` creates the per-layer container `EplbLayerState` (empty until `set_eplb_state` fills it). `layer.py:210-211` is the inverse guard: redundant experts only make sense with EPLB.

- `layer.py:246-259` constructs the `ExpertMapManager` (passing `enable_eplb` so it forces linear placement) and immediately calls `self.update_expert_map_info()`.

- `layer.py:293-312` builds the router and hands it `eplb_state=self.eplb_state` (`layer.py:296`) and `num_logical_experts=self.logical_num_experts` (`layer.py:310`). This is the link that lets routing perform the logical→physical rewrite (Section 6).

- `layer.py:374-383` — quant-method compatibility gate: `if enable_eplb and not self.quant_method.supports_eplb: raise ... "EPLB is not supported {quant_method}"`. Not every quant backend can have its weights rearranged.

#### `update_expert_map_info` — publishing the map as buffers

`layer.py:512-526`:
```python
def update_expert_map_info(self):
    self.local_num_experts = self.expert_map_manager.local_num_experts
    self.expert_placement_strategy = self.expert_map_manager.placement_strategy
    self.register_buffer("_expert_map", self.expert_map_manager.expert_map)
    self.register_buffer("expert_mask", self.expert_map_manager.expert_mask)
    routing_tables = self.expert_map_manager.routing_tables
    if routing_tables is not None:
        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)
```
Registering `_expert_map`/`expert_mask` as **buffers** (not parameters) means they move with `.to(device)` and are visible to the compiled forward graph, but are not treated as trainable/loaded weights. `local_num_experts` (`layer.py:514`) drives the shape of every expert weight tensor and is exactly the width EPLB's `get_expert_weights` will reshape to (Section 5).

The `expert_map` **property** (`layer.py:1330-1334`) is what the kernel path reads:
```python
@property
def expert_map(self) -> torch.Tensor | None:
    return self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
```
On non-AITER (this CUDA/ROCm-fused build path), it returns the plain `_expert_map` global→local table; under AITER fusion it returns the 0/1 `expert_mask` instead — the kernels consume different formats.

`update_expert_map` (`layer.py:543-555`) is the elastic-EP reconfiguration entry: it calls `ExpertMapManager.update(...)` then re-publishes buffers. `ExpertMapManager.update` (`expert_map_manager.py:367-397`) recomputes under a `with device:` block so the new tensors land on the correct GPU.

---

### 4. `set_eplb_state` — attaching the runtime EPLB tables to the layer

`layer.py:1278-1303`:
```python
def set_eplb_state(self, moe_layer_idx, expert_load_view,
                   logical_to_physical_map, logical_replica_count) -> None:
    if self.eplb_state is not None:
        self.eplb_state.set_layer_state(
            moe_layer_idx, expert_load_view,
            logical_to_physical_map, logical_replica_count)
```
This is called **once per MoE layer** from `EplbState.add_model` (`eplb_state.py:442-446`), which passes the **global, all-layer** tensors. `EplbLayerState.set_layer_state` (`eplb_state.py:945-954`) then slices this layer's row and — critically — stores **views**, not copies:
```python
self.expert_load_view = expert_load_view[moe_layer_idx]
self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
self.logical_replica_count = logical_replica_count[moe_layer_idx]
```
Because these are index-views into the tensors owned by `EplbState`, when the rebalancer overwrites the global maps in place (Section 7), **every layer's view updates automatically** — no re-registration needed. Likewise `expert_load_view` is a view the forward pass atomically increments, and `EplbState` reads the same memory.

The shared `should_record_tensor` (`eplb_state.py:933-943`, `EplbLayerState`) is a single scalar-bool tensor referenced by *all* layers; `EplbState` flips it once with `.fill_()` to enable/disable load recording globally. It is False for the first `step_interval - window_size` steps because those samples would be overwritten before the next rearrange anyway (documented at `eplb_state.py:939-942`).

---

### 5. `get_expert_weights` — the memory-safe weight views EPLB rearranges

`layer.py:1201-1276`. EPLB rearrangement must swap the *actual weight memory* between GPUs. This method returns, for one layer, the list of expert weight tensors each reshaped to `(local_num_experts, -1)` so the rebalance kernel can index them by physical slot:

- `layer.py:1270-1276`
  ```python
  return [
      weight.view(self.local_num_experts, -1)
      for name, weight in weights
      if name not in NON_EXPERT_WEIGHTS
      and weight.shape != torch.Size([])
      and not name.startswith(NON_EXPERT_PREFIXES)
  ]
  ```
  `.view(self.local_num_experts, -1)` is a **zero-copy reshape** — the returned tensors alias the layer's parameter storage, so writing into them mutates the live weights. `local_num_experts` here is the count published in `update_expert_map_info`.

- Exclusions matter for correctness:
  - `NON_EXPERT_WEIGHTS` (`layer.py:1248-1252`) — `e_score_correction_bias`, `w13_input_scale`, `w2_input_scale`. The input scales are `.expand()` broadcast views (stride 0) shared across experts, not per-expert, so rearranging them would corrupt memory (`layer.py:1244-1247`).
  - `NON_EXPERT_PREFIXES` (`layer.py:1256-1261`) — shared-expert / gate / transform submodules living under `runner.`; these are replicated, not sharded per expert.

- `_maybe_make_contiguous` (`layer.py:1202-1239`) handles quant scale tensors whose last two dims are transposed. `layer.py:1237-1238` returns a *fresh* `nn.Parameter` wrapping `torch.transpose(p.data, 1, 2)` — the comment at `layer.py:1233-1236` stresses this points at the **same underlying memory**, so the EPLB copy is still moving the real bytes, just through a contiguous view. `layer.py:1263-1268` asserts every non-excluded weight is contiguous before the rearrange runs.

`EplbState.add_model` consumes this at `eplb_state.py:448`: `expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]` — a staging buffer shaped like one layer's expert-weight list.

---

### 6. The forward path: routing tokens after rearrange

This is the heart of "how the layer uses the maps to route tokens." Routing produces **logical** top-k IDs; EPLB rewrites them to **physical** IDs and records load, in one fused Triton kernel.

`model_executor/layers/fused_moe/router/base_router.py`:

- `_apply_eplb_mapping` (`base_router.py:198-213`) is the hook called during routing when `eplb_state is not None`. It forwards the three views + `should_record_tensor` into `eplb_map_to_physical_and_record`.

- `_eplb_map_and_record_i32_kernel` (`base_router.py:18-78`) does per-(token, top-k-slot) work:
  1. `base_router.py:36` loads the routed **logical** `expert_id`.
  2. `base_router.py:41-52` replica selection: reads `logical_replica_count[expert_id]`, clamps to ≥1, and picks a replica via a Knuth multiplicative hash of the token index modulo the replica count (`replica_idx = hashed % replica_count`, `base_router.py:52`). This spreads tokens for a hot logical expert *deterministically and evenly* across its physical replicas without a global counter.
  3. `base_router.py:67-72` `map_index = expert_id * map_slots + replica_idx`, then `physical_id = logical_to_physical_map[map_index]`. This is the logical→physical redirect. `base_router.py:73` writes the physical id back into `topk_ids`.
  4. `base_router.py:75-78` if recording is enabled, `tl.atomic_add(out_ptr + physical_id, 1)` — increments `expert_load_view` **per physical expert**. Per `eplb_state.py:163-165`, load is recorded for *all* physical experts (not just local) so statistics are dispatch-method-agnostic.

After this kernel, `topk_ids` contains **physical** expert IDs. The MoE kernel then applies `expert_map` (Section 2) to convert each physical global ID to a local index (or `-1` to drop it if it lives on another rank) and dispatches. So the ordering is: **router → logical→physical (EPLB) → physical→local (expert_map) → all-to-all dispatch.**

`_validate_eplb_state` (`base_router.py:179-190`) guards that all four tensors are non-None before the first forward, converting a silent mis-wire into a clear error.

(Version note: this build uses a fused Triton kernel; `base_router.py:126-133` shows a non-Triton fallback branch of `eplb_map_to_physical_and_record` for environments without Triton. The mapping semantics are identical.)

---

### 7. The rearrange step and which process group carries the weights

`EplbState.rearrange` (`eplb_state.py:658-780`) recomputes the layout and physically moves weights:

- `eplb_state.py:674-675`
  ```python
  ep_group = get_ep_group().device_group
  ep_rank = ep_group.rank()
  ```
  Load *statistics* (`_allreduce_list`, `eplb_state.py:718`) and the rebalance decision run on the **EP** group.
- `eplb_state.py:691-716` scatters per-physical load back onto logical experts (`scatter_add_` over `physical_to_logical_map`) and sums the sliding window → `global_expert_load_window`.
- `eplb_state.py:718` all-reduces load across ranks; `eplb_state.py:756-763` calls the policy (`rebalance_experts`) to produce `new_physical_to_logical_map`.
- `eplb_state.py:766-772` `rearrange_expert_weights_inplace(...)` performs the actual GPU-to-GPU weight shuffle, and it is handed **two** things: `ep_group` *and* `eplb_model_state.communicator`. The `communicator` (built on the **EPLB** group, next section) is what physically carries the bytes; `ep_group` provides ranks/topology metadata.

This two-group design is documented at the group-creation site, `distributed/parallel_state.py:1878-1881`:
> "Create EPLB group with the same ranks as EP if EPLB is enabled. This is a separate process group to isolate EPLB communications from MoE forward pass collectives and prevent deadlocks."

---

### 8. `get_eplb_group` and the EPLB process group

`distributed/parallel_state.py`:

- `parallel_state.py:1375-1384` the module global and accessor:
  ```python
  _EPLB: GroupCoordinator | None = None
  def get_eplb_group() -> GroupCoordinator:
      assert _EPLB is not None, ("EPLB group is not initialized. ... "
          "Ensure parallel_config.enable_eplb is True.")
      return _EPLB
  ```
  So calling this without EPLB enabled is a hard assertion failure — the group is created lazily and only when needed.

- Creation, `parallel_state.py:1850-1899`, inside `initialize_model_parallel`. The EP group is built first (`parallel_state.py:1850-1876`) from `group_ranks` computed at `parallel_state.py:1854-1864` (a transpose/reshape of the global rank grid so EP spans `dp × pcp × tp`). Then:
  ```python
  global _EPLB
  assert _EPLB is None, "EPLB group is already initialized"
  if config.parallel_config.enable_eplb:
      if enable_elastic_ep:
          _EPLB = _init_stateless_group(group_ranks, "eplb", ...)   # 1886
      else:
          _EPLB = init_model_parallel_group(
              group_ranks, get_world_group().local_rank, backend,
              group_name="eplb")                                    # 1894-1898
  ```
  Key facts:
  - `_EPLB` uses the **exact same `group_ranks` as `_EP`** (`parallel_state.py:1887`, `1895`) — same members, but a *distinct* NCCL/Gloo communicator, which is the whole point (isolation).
  - Non-elastic path (`parallel_state.py:1894`) calls `init_model_parallel_group` (`parallel_state.py:1264-1279`), which builds a full `GroupCoordinator` with `use_device_communicator=True` (default). That means the coordinator gets both a `device_group` and a `cpu_group` and, if `world_size > 1`, a real `device_communicator` (see `GroupCoordinator.__init__`, `parallel_state.py:453-464`).
  - Elastic path (`parallel_state.py:1886`) instead builds a `StatelessGroupCoordinator`.

- Lifecycle plumbing: `prepare_communication_buffer_for_model` (`parallel_state.py:1980-1981`) calls `_EPLB.prepare_communication_buffer_for_model(model)`; `_replace_active_groups` (`parallel_state.py:1320-1328`) and the teardown at `parallel_state.py:2041-2044` destroy `_EPLB` alongside the others.

---

### 9. `communicator=None` resolution on this build

The prompt asks specifically about `communicator=None`. There are two layers to this; on **this build** they resolve as follows.

**(a) Config-time resolution — the real answer for this build.** `config/parallel.py:89` declares:
```python
communicator: EPLBCommunicatorBackend | None = None
```
i.e. the EPLB communicator backend is unset by default. It is resolved in the parallel-config post-init at `config/parallel.py:907-924`:
```python
if self.enable_eplb and self.eplb_config.communicator is None:
    if self.enable_elastic_ep:
        self.eplb_config.communicator = "pynccl"                  # 912
    else:
        # Avoid torch_nccl: NCCL is fundamentally incompatible
        # with async EPLB due to multi-stream conflicts, and
        # batched isend/irecv hangs under high load.
        from vllm.distributed.nixl_utils import is_nixl_available
        if is_nixl_available():
            self.eplb_config.communicator = "nixl"                # 922
        else:
            self.eplb_config.communicator = "torch_gloo"          # 924
```
So on a standard (non-elastic) build **without NIXL**, `communicator=None` is resolved to **`torch_gloo`**, not NCCL. The comment (`parallel.py:914-917`) is explicit that `torch_nccl` is deliberately avoided because NCCL multi-stream conflicts with async EPLB and batched `isend/irecv` hangs under load (referencing pytorch/pytorch#174288). If NIXL is present it prefers `nixl`; elastic EP forces `pynccl`.

**(b) Factory-time fallback.** `create_eplb_communicator` (`distributed/eplb/eplb_communicator.py:618-734`) is where the backend string becomes a concrete communicator. It is called from `EplbState.add_model` (`eplb_state.py:450-455`):
```python
communicator = create_eplb_communicator(
    group_coordinator=get_eplb_group(),          # 451  <-- the EPLB group
    backend=self.parallel_config.eplb_config.communicator,
    expert_weights=model.expert_weights,
    expert_buffer=expert_buffer,
)
```
Inside the factory:
- `eplb_communicator.py:644-645` `if backend is None: backend = "torch_nccl"`. This is a *secondary* fallback — but on this build it is effectively never hit, because config post-init (part a) already turned `None` into `torch_gloo`/`nixl`/`pynccl` before this point.
- `eplb_communicator.py:647-653` picks the torch process group off the **`get_eplb_group()` coordinator**: `cpu_group` if the expert weights are CPU tensors, else `device_group`. This is the concrete tie between "the EPLB process group" and "the object that moves weights."
- Backend dispatch: `"torch_gloo"` → `TorchDistGlooStagedEplbCommunicator(cpu_group=...)` (`eplb_communicator.py:726-729`, gloo P2P with CPU staging); `"torch_nccl"` → `TorchDistNcclEplbCommunicator(ep_group=torch_group)` (`eplb_communicator.py:730-731`); `"pynccl"`/`"nixl"` handled at `eplb_communicator.py:732-733` / `706-725`.
- Stateless (elastic) coordinators are constrained to `torch_nccl`/`pynccl` and silently promote `torch_nccl`→`pynccl` (`eplb_communicator.py:691-704`).

**Bottom line for this build:** with EPLB enabled, non-elastic, and NIXL unavailable, an unset (`None`) communicator becomes **`torch_gloo`**, and that communicator is bound to the **CPU group of `get_eplb_group()`** — a group with the same ranks as EP but an isolated backend. The `EplbModelState.communicator` field (`eplb_state.py:195-198`, set at `eplb_state.py:469`) is what `rearrange` passes into `rearrange_expert_weights_inplace` (Section 7).

---

### 10. `v1/worker/gpu/eplb_utils.py` — `maybe_register_model` (the crash-traceback frame)

`v1/worker/gpu/eplb_utils.py` defines `EPLBController`, the worker-side object that owns the `EplbState` and drives it. In the crash traceback, `maybe_register_model` is the frame where the model first gets attached to EPLB.

Call chain (from `v1/worker/gpu/model_runner.py`):
- `model_runner.py:273` `self.eplb = EPLBController(self.parallel_config, self.device)`.
- `model_runner.py:294` `self.eplb.prepare_load()` — resets state; if EPLB enabled, constructs `EplbState` (`eplb_utils.py:58-62`).
- `model_runner.py:336-340` after model load and communication-buffer prep:
  ```python
  eplb_models_added |= self.eplb.maybe_register_model(
      self.model, self.model_config, load_dummy_weights)
  ```
- `model_runner.py:341` `self.eplb.maybe_start_async_loop(eplb_models_added)`.

`maybe_register_model` itself (`eplb_utils.py:96-113`):
```python
def maybe_register_model(self, model, model_config, load_dummy_weights) -> bool:
    if not self.parallel_config.enable_eplb or load_dummy_weights:
        return False                                              # 102
    model = _unwrap_moe(model)                                    # 105
    if not is_mixture_of_experts(model):
        return False                                              # 106-107
    logger.info_once("EPLB is enabled for model %s.", model_config.model)
    assert self.state is not None                                # 110
    self.state.add_model(model, model_config)                    # 111
    self._has_registered_models = True                           # 112
    return True
```
Block-by-block:
- `eplb_utils.py:102` — early-out unless EPLB is on and this is a real (non-dummy) weight load. Dummy loads skip EPLB because the weights it would rearrange are garbage.
- `eplb_utils.py:105` `_unwrap_moe(model)` (`eplb_utils.py:21-28`) — VLM wrappers (e.g. `KimiK25ForConditionalGeneration`) don't implement `MixtureOfExperts` themselves but hold the MoE LM under `.language_model`; this descends via `get_language_model()`. If the crash was on a multimodal model, this unwrap is the relevant frame.
- `eplb_utils.py:106-107` — if still not an MoE model, bail (dense models have no experts to balance).
- `eplb_utils.py:110-111` — `assert self.state is not None` then `self.state.add_model(model, model_config)`. **This is the frame that triggers the whole registration cascade** documented above: `add_model` builds the initial maps, calls `model.set_eplb_state(...)` on every layer (Section 4), snapshots `expert_weights`, and calls `create_eplb_communicator(group_coordinator=get_eplb_group(), ...)` (Section 9b). So a crash *inside* this call is most likely either (i) `get_eplb_group()`'s assertion (`parallel_state.py:1379`) if the EPLB group was never created — which happens when `enable_eplb` disagreed between config-time group init and here — or (ii) a communicator-construction failure in `create_eplb_communicator` (e.g. a `pynccl`/`nixl`/dtype error at `eplb_communicator.py:657-724`). The `assert self.state is not None` at `eplb_utils.py:110` guards against `prepare_load` not having run.

The sibling `maybe_register_speculator` (`eplb_utils.py:64-94`) does the same for a draft/speculator MoE model but asserts `not enable_elastic_ep` (`eplb_utils.py:83-85`). `setup_from_mapping` (`eplb_utils.py:138-156`) is the elastic-EP path that rebuilds `EplbState` from an externally supplied `expanded_physical_to_logical` map via `EplbState.from_mapping`.

Load-time stepping is driven by the `step_eplb_after` decorator (`eplb_utils.py:31-47`), which calls `self.eplb.step(...)` after a runner method; `EPLBController.step` (`eplb_utils.py:119-136`) no-ops unless `enable_eplb and not suppressed and state is not None and _has_registered_models` — i.e. it will not run until `maybe_register_model` has set `_has_registered_models = True` (`eplb_utils.py:112`).

---

### Files referenced
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/layer.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/expert_map_manager.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/model_executor/layers/fused_moe/router/base_router.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/parallel_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_state.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/distributed/eplb/eplb_communicator.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/config/parallel.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/v1/worker/gpu/eplb_utils.py`
- `/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty/vllm/v1/worker/gpu/model_runner.py`

---
