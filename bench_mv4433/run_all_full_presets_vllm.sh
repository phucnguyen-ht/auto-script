export VLLM_ENGINE_READY_TIMEOUT_S=3600

############### GLM5
####### DP8
## Enable mtp, bs=64
# ENFORCE_EAGER_PROFILE=0 RUN_PROFILE=0 PRESET=glm5/dp8ep8/MTP-bs64-moreh.yaml bash run_all.sh
# ENFORCE_EAGER_PROFILE=0 RUN_PROFILE=0 PRESET=glm5/dp8ep8/MTP-bs64-dg.yaml    bash run_all.sh

## Disable mtp, bs=64
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=glm5/dp8ep8/bs64-moreh.yaml bash run_all.sh
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=glm5/dp8ep8/bs64-dg.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=glm5/dp8ep8/MTP-bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/dp8ep8/MTP-bs1-moreh.yaml  bash run_all.sh

## Disable mtp, bs=1
# PRESET=glm5/dp8ep8/bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/dp8ep8/bs1-moreh.yaml  bash run_all.sh

####### TP8
## Enable mtp, bs=64
# ENFORCE_EAGER_PROFILE=0 RUN_PROFILE=0 PRESET=glm5/tp8/MTP-bs64-dg.yaml    bash run_all.sh
# ENFORCE_EAGER_PROFILE=0 RUN_PROFILE=0 PRESET=glm5/tp8/MTP-bs64-moreh.yaml    bash run_all.sh

## Disable mtp, bs=64
# RUN_PROFILE=0 PRESET=glm5/tp8/bs64-dg.yaml bash run_all.sh
# RUN_PROFILE=0 PRESET=glm5/tp8/bs64-moreh.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=glm5/tp8/MTP-bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/tp8/MTP-bs1-moreh.yaml     bash run_all.sh

## Disable mtp, bs=1
# PRESET=glm5/tp8/bs1-dg.yaml  bash run_all.sh
# PRESET=glm5/tp8/bs1-moreh.yaml  bash run_all.sh

############### DEEPSEEK
####### DP8
## Enable mtp, bs=64
# PRESET=deepseek/dp8ep8/MTP-bs64-moreh.yaml bash run_all.sh
# PRESET=deepseek/dp8ep8/MTP-bs64-dg.yaml    bash run_all.sh

# ## Disable mtp, bs=64
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=deepseek/dp8ep8/bs64-moreh.yaml bash run_all.sh
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=deepseek/dp8ep8/bs64-dg.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=deepseek/dp8ep8/MTP-bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/dp8ep8/MTP-bs1-moreh.yaml  bash run_all.sh

## Disable mtp, bs=1
# PRESET=deepseek/dp8ep8/bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/dp8ep8/bs1-moreh.yaml  bash run_all.sh

####### TP8
## Enable mtp, bs=64
# PRESET=deepseek/tp8/MTP-bs64-dg.yaml    bash run_all.sh
# PRESET=deepseek/tp8/MTP-bs64-moreh.yaml    bash run_all.sh

# ## Disable mtp, bs=64
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=deepseek/tp8/bs64-dg.yaml bash run_all.sh
# ENFORCE_EAGER_PROFILE=1 RUN_PROFILE=0 PRESET=deepseek/tp8/bs64-moreh.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=deepseek/tp8/MTP-bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/tp8/MTP-bs1-moreh.yaml     bash run_all.sh

## Disable mtp, bs=1
# PRESET=deepseek/tp8/bs1-dg.yaml  bash run_all.sh
# PRESET=deepseek/tp8/bs1-moreh.yaml  bash run_all.sh

# ## baseline
# RUN_PROFILE=0 RUN_BENCH=0 PRESET=deepseek-v3.2.yaml bash run_all.sh

# # same as glm5, disable mtp, bs=64
RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=0 PRESET=deepseek/tp8/same_as_glm5_deepgemm.yaml  bash run_all.sh
RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=0 PRESET=deepseek/tp8/same_as_glm5_moreh.yaml     bash run_all.sh

RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=0 PRESET=deepseek/dp8ep8/same_as_glm5_deepgemm.yaml  bash run_all.sh
RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=0 PRESET=deepseek/dp8ep8/same_as_glm5_moreh.yaml     bash run_all.sh

# RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=1 PRESET=deepseek-v3.2.yaml     bash run_all.sh

# BACKEND=sglang RUN_PROFILE=0 RUN_BENCH=0 bash run_all.sh

############### SGLANG (DeepSeek-V3.2, no preset)
# BACKEND=sglang has no preset and only supports eval/readable (bench/profile
# are skipped automatically). Logs go under logs_sglang/. SGLang serves on port
# 30000 via serve_sglang_ds3.2.sh.
#
# readable only:
# BACKEND=sglang RUN_PROFILE=0 RUN_BENCH=0 RUN_READABLE=1 bash run_all.sh
# eval + readable:
# BACKEND=sglang RUN_PROFILE=0 RUN_BENCH=0 bash run_all.sh
