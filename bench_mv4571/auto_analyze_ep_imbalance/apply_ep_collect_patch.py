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

# Per-model MoE method whose apply() receives (layer, ..., topk_ids, ...). Each target =
# (relative file, call_anchor, call_replacement). The same MODULE_BLOCK is inserted after
# MODULE_ANCHOR in every target file. Idempotent per file (MARKER check).
TARGETS = [
    # glm5.2 — AITER Fp8MoEMethod.apply (before the moe_kernel dispatch)
    {
        "model": "glm5.2 (Fp8MoEMethod)",
        "file": "model_executor/layers/quantization/fp8.py",
        "call_anchor": (
            "        assert not self.is_monolithic\n"
            "        assert self.moe_kernel is not None\n"
            "        return self.moe_kernel.apply(\n"
        ),
        "call_replacement": (
            "        assert not self.is_monolithic\n"
            "        assert self.moe_kernel is not None\n"
            "        _ep_collect_maybe_log(layer, topk_ids)\n"
            "        return self.moe_kernel.apply(\n"
        ),
    },
    # kimi2.6 — compressed-tensors CompressedTensorsWNA16MoEMethod.apply (before fused_experts)
    {
        "model": "kimi2.6 (CompressedTensorsWNA16MoEMethod)",
        "file": "model_executor/layers/quantization/compressed_tensors/"
                "compressed_tensors_moe/compressed_tensors_moe_wna16.py",
        "call_anchor": (
            "        from vllm.model_executor.layers.fused_moe import fused_experts\n"
            "\n"
            "        return fused_experts(\n"
        ),
        "call_replacement": (
            "        from vllm.model_executor.layers.fused_moe import fused_experts\n"
            "\n"
            "        _ep_collect_maybe_log(layer, topk_ids)\n"
            "        return fused_experts(\n"
        ),
    },
]


def patch_file(path, call_anchor, call_replacement, label):
    try:
        src = open(path).read()
    except OSError:
        print(f"[patch] SKIP {label}: file not found ({path})")
        return 0
    if MARKER in src:
        print(f"[patch] already applied: {label}")
        return 0
    if MODULE_ANCHOR not in src:
        print(f"[patch] module anchor not found in {label}; aborting", file=sys.stderr)
        return 3
    if call_anchor not in src:
        print(f"[patch] call anchor not found in {label} ({path}); aborting", file=sys.stderr)
        return 4
    src = src.replace(MODULE_ANCHOR, MODULE_ANCHOR + MODULE_BLOCK, 1)
    src = src.replace(call_anchor, call_replacement, 1)
    bak = path + ".mv4571.bak"
    try:
        open(bak, "x").write(open(path).read())
        print(f"[patch] backup -> {bak}")
    except FileExistsError:
        pass
    open(path, "w").write(src)
    print(f"[patch] applied [EP_COLLECT] -> {label}")
    return 0


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        print("[patch] cannot locate installed vllm", file=sys.stderr)
        return 2
    vroot = spec.submodule_search_locations[0]
    rc = 0
    for t in TARGETS:
        r = patch_file(f"{vroot}/{t['file']}", t["call_anchor"], t["call_replacement"], t["model"])
        rc = rc or r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
