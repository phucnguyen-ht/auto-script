export VLLM_ENGINE_READY_TIMEOUT_S=3600

############### GLM5
####### DP8
## Enable mtp, bs=64
ENFORCE_EAGER_PROFILE=0 PRESET=glm5/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
ENFORCE_EAGER_PROFILE=0 PRESET=glm5/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh

## Disable mtp, bs=64
# ENFORCE_EAGER_PROFILE=0 PRESET=glm5/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
# ENFORCE_EAGER_PROFILE=0 PRESET=glm5/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=glm5/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

## Disable mtp, bs=1
# PRESET=glm5/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

####### TP8
## Enable mtp, bs=64
ENFORCE_EAGER_PROFILE=0 PRESET=glm5/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh
ENFORCE_EAGER_PROFILE=0 PRESET=glm5/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml    bash run_all.sh

## Disable mtp, bs=64
# RUN_PROFILE=0 PRESET=glm5/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml bash run_all.sh
# RUN_PROFILE=0 PRESET=glm5/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
 
## Enable mtp, bs=1
# PRESET=glm5/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=glm5/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml     bash run_all.sh

## Disable mtp, bs=1
# PRESET=glm5/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml  bash run_all.sh
# PRESET=glm5/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

############### DEEPSEEK
####### DP8
## Enable mtp, bs=64
# PRESET=deepseek/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
# PRESET=deepseek/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh

## Disable mtp, bs=64
ENFORCE_EAGER_PROFILE=0 PRESET=deepseek/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
ENFORCE_EAGER_PROFILE=0 PRESET=deepseek/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml bash run_all.sh

## Enable mtp, bs=1
# PRESET=deepseek/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

## Disable mtp, bs=1
# PRESET=deepseek/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

####### TP8
## Enable mtp, bs=64
# PRESET=deepseek/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh
# PRESET=deepseek/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml    bash run_all.sh

## Disable mtp, bs=64
RUN_PROFILE=0 PRESET=deepseek/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml bash run_all.sh
RUN_PROFILE=0 PRESET=deepseek/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
 
## Enable mtp, bs=1
# PRESET=deepseek/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=deepseek/tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml     bash run_all.sh

## Disable mtp, bs=1
# PRESET=deepseek/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml  bash run_all.sh
# PRESET=deepseek/tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh