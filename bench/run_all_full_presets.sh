# DP8
# PRESET=dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
# PRESET=dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh

ENFORCE_EAGER_PROFILE=1 PRESET=dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
ENFORCE_EAGER_PROFILE=1 PRESET=dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml bash run_all.sh

# PRESET=dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=dp8ep8/CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

# PRESET=dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=dp8ep8/zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh

# # TP8
# PRESET=tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh
# PRESET=tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-dg.yaml    bash run_all.sh

# PRESET=tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
# PRESET=tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs64-moreh.yaml bash run_all.sh
 
# PRESET=tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh
# PRESET=tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-dg.yaml     bash run_all.sh

# PRESET=tp8/TP8-CG-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh
# PRESET=tp8/TP8-zai-org-glm-5-fp8-amd-mi325x-dp8-moe-tp8-0ic-bs1-moreh.yaml  bash run_all.sh