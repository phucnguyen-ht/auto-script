# export VLLM_ENGINE_READY_TIMEOUT_S=3600

PRESET=kimi2.6/tp8tp8/moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh
PRESET=kimi2.6/dp8ep8/moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh
PRESET=kimi2.6/tp8ep8/moonshotai-kimi-k2.6-amd-mi300x-tp8-moe-ep8.yaml bash run_all.sh

################# ENABLE CUDA GRAPH
# ### DP8EP8
PRESET=kimi2.6/tp8tp8/CG-Full-moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh
PRESET=kimi2.6/dp8ep8/CG-Full-moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh
PRESET=kimi2.6/tp8ep8/CG-Full-moonshotai-kimi-k2.6-amd-mi300x-tp8-moe-ep8.yaml bash run_all.sh