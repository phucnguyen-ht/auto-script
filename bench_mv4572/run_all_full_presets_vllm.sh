#!/usr/bin/env bash
# mv-4572 accuracy eval (gsm8k custom script + readable smoke tests) over MULTIPLE
# presets. Each line = one full run_all.sh (readable + gsm8k) on that preset.
# Edit which lines are active (uncomment the presets you want), then:
#   cd auto-script/bench_mv4572 && bash run_all_full_presets_vllm.sh
#
# For ACCURACY the meaningful axis is: no-EPLB vs nixl-EPLB, and num_redundant_experts
# (r0/r8/r16/r32). The rearrange INTERVAL (s100/s250/s500/s1000) does NOT change model
# outputs -- it only changes rearrange frequency (see eplb_src/mxfp4-communicator-support.md
# §8) -- so you don't need to sweep intervals for accuracy.
#
# Runs sequentially; a preset that fails does not stop the rest (no set -e).
# Only nixl works with MXFP4 (gloo/nccl/pynccl records are NOT evaluable -- §1-§2 of the doc).

# --- baseline: NO EPLB (reference accuracy) ---
# PRESET=kimi2.6.mxfp4/dp8ep8/base.yaml                              bash run_all.sh

# PHASE RULE: base.yaml runs READABLE ONLY (RUN_EVAL=0); every EPLB preset runs
# BOTH readable + eval (gsm8k). Keep RUN_EVAL=0 on the base line, none on EPLB lines.

# --- baseline: NO EPLB -> readable ONLY ---
PRESET=kimi2.6.mxfp4/dp8ep8/base.yaml   RUN_EVAL=0 bash run_all.sh   # base = readable only

# --- nixl EPLB, default interval, by redundancy -> readable + eval ---
PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml bash run_all.sh

# PRESET=kimi2.6.mxfp4/dp8ep8/base.yaml   RUN_EVAL=0 bash run_all.sh   # base = readable only

# # --- nixl EPLB, default interval, by redundancy -> readable + eval ---
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml RUN_EVAL=0  bash run_all.sh

# PRESET=kimi2.6.mxfp4/dp8ep8/base.yaml   RUN_EVAL=0 bash run_all.sh   # base = readable only

# # --- nixl EPLB, default interval, by redundancy -> readable + eval ---
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml RUN_EVAL=0  bash run_all.sh
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r8.yaml   bash run_all.sh
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r16.yaml   bash run_all.sh
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r32.yaml   bash run_all.sh
# # PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r32.yaml  bash run_all.sh
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-sync-default-r0.yaml    bash run_all.sh

# --- nixl EPLB at other intervals/redundancy (accuracy should MATCH the above) ---
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-s250-r8.yaml      bash run_all.sh
# PRESET=kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-s500-r16.yaml     bash run_all.sh
