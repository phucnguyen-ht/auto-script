export VLLM_ENGINE_READY_TIMEOUT_S=3600

### DP8EP8
PRESET=kimi2.6/dp8ep8/moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh

### TP8TP8
PRESET=kimi2.6/dp8ep8/moonshotai-kimi-k2.6-amd-mi300x-dp8-moe-ep8.yaml bash run_all.sh

### TP8EP8
PRESET=kimi2.6/tp8ep8/moonshotai-kimi-k2.6-amd-mi300x-tp8-moe-ep8.yaml bash run_all.sh

