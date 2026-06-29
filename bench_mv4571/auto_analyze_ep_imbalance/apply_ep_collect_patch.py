#!/usr/bin/env python3
# =============================================================================
# MV-4571 — (re)apply the [EP_COLLECT] instrumentation to the INSTALLED vllm.
#
# The token-imbalance analysis (analyze_tokens.py) reads `[EP_COLLECT] ...` lines
# from serve.log. Those are emitted by a moreh patch in
#   vllm/model_executor/layers/quantization/fp8.py :: Fp8MoEMethod.apply
# which logs, per (rank, MoE layer, step), the per-GLOBAL-expert token-routing
# histogram computed from topk_ids BEFORE the all2all dispatch. That patch lived
# in a dev `3rdparty/vllm` checkout; a freshly-built container ships the base
# package WITHOUT it (so serve.log has zero [EP_COLLECT] lines and analyze_tokens
# aborts with "No [EP_COLLECT] lines found").
#
# This script re-inserts the instrumentation into the installed fp8.py. It is:
#   * idempotent (re-running is a no-op once applied),
#   * gated by env VLLM_MOREH_EP_LOG=1 (zero effect on normal serving / profiling),
#   * requires enforce_eager=true at serve time (under cudagraph the apply() side
#     effect is captured once and not replayed each step).
#
# Run INSIDE the docker container:
#   docker exec phuc-nguyen-mv-4571 python3 \
#     /home/phuc-nguyen/workspace/mv-4571/auto-script/bench_mv4571/auto_analyze_ep_imbalance/apply_ep_collect_patch.py
#
# Output format (matches analyze_tokens.py HEADER_RE):
#   [EP_COLLECT] layer=<layer_name> it=<k> ntok=<local_tokens> E=<global_experts> counts=[c0,...,cE-1]
# (the per-process "Worker_DP<dp>_EP<ep>" prefix is added by vLLM's subprocess logging.)
# =============================================================================
import importlib.util
import sys

MARKER = "_ep_collect_maybe_log"   # presence => already patched

# The instrumentation, inserted at module scope right after `logger = init_logger(__name__)`.
MODULE_BLOCK = '''
# --- [MV-4571 EP_COLLECT] BEGIN (token-imbalance instrumentation) ---------------
_EP_COLLECT_IT: dict = {}


def _ep_collect_maybe_log(layer, topk_ids):
    """Log per-global-expert token-routing histogram (from topk_ids, pre-all2all).
    Gated by VLLM_MOREH_EP_LOG=1; needs enforce_eager. Best-effort, never raises."""
    import os

    if os.environ.get("VLLM_MOREH_EP_LOG", "0") != "1":
        return
    try:
        E = int(getattr(layer, "global_num_experts", 0) or 0)
        if E <= 0 or topk_ids is None:
            return
        name = getattr(layer, "layer_name", None) or "unknown"
        flat = topk_ids.reshape(-1).to(torch.long)
        flat = flat[(flat >= 0) & (flat < E)]
        counts = torch.bincount(flat, minlength=E)[:E].tolist()
        ntok = int(topk_ids.shape[0])
        it = _EP_COLLECT_IT.get(name, 0)
        _EP_COLLECT_IT[name] = it + 1
        logger.info(
            "[EP_COLLECT] layer=%s it=%d ntok=%d E=%d counts=[%s]",
            name, it, ntok, E, ",".join(map(str, counts)),
        )
    except Exception as e:  # never break inference on a logging hiccup
        logger.warning("[EP_COLLECT] logging failed: %s", e)
# --- [MV-4571 EP_COLLECT] END ---------------------------------------------------
'''

MODULE_ANCHOR = "logger = init_logger(__name__)\n"

# The call site, inserted inside Fp8MoEMethod.apply before the moe_kernel dispatch.
CALL_ANCHOR = (
    "        assert not self.is_monolithic\n"
    "        assert self.moe_kernel is not None\n"
    "        return self.moe_kernel.apply(\n"
)
CALL_REPLACEMENT = (
    "        assert not self.is_monolithic\n"
    "        assert self.moe_kernel is not None\n"
    "        _ep_collect_maybe_log(layer, topk_ids)\n"
    "        return self.moe_kernel.apply(\n"
)


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        print("[patch] cannot locate installed vllm", file=sys.stderr)
        return 2
    vroot = spec.submodule_search_locations[0]
    fp8 = f"{vroot}/model_executor/layers/quantization/fp8.py"
    src = open(fp8).read()

    if MARKER in src:
        print(f"[patch] already applied: {fp8}")
        return 0

    if MODULE_ANCHOR not in src:
        print(f"[patch] module anchor not found in {fp8}; aborting", file=sys.stderr)
        return 3
    if CALL_ANCHOR not in src:
        print(f"[patch] call anchor (Fp8MoEMethod.apply) not found in {fp8}; aborting", file=sys.stderr)
        return 4

    src = src.replace(MODULE_ANCHOR, MODULE_ANCHOR + MODULE_BLOCK, 1)
    src = src.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)

    bak = fp8 + ".mv4571.bak"
    try:
        open(bak, "x").write(open(fp8).read())
        print(f"[patch] backup -> {bak}")
    except FileExistsError:
        pass
    open(fp8, "w").write(src)
    print(f"[patch] applied [EP_COLLECT] instrumentation -> {fp8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
