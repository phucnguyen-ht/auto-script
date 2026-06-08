

# NUM_PROMPTS_LIST: space- or comma-separated list of NUM_PROMPTS values to benchmark
# NUM_ITERS: number of repetitions per NUM_PROMPTS value
# OUTPUT_DIR: directory for result files (default: ./results)
#
# Examples:
#   bash bench_serving_auto.sh
#   NUM_PROMPTS_LIST="16 32 64" NUM_ITERS=3 bash bench_serving_auto.sh
#   NUM_PROMPTS_LIST="16,32,64" OUTPUT_DIR=/tmp/results bash bench_serving_auto.sh

NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST:-16 18 20 25}"
NUM_ITERS="${NUM_ITERS:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

# Normalize commas to spaces
NUM_PROMPTS_LIST="${NUM_PROMPTS_LIST//,/ }"
read -r -a prompts_arr <<< "$NUM_PROMPTS_LIST"

for np in "${prompts_arr[@]}"; do
    for i in $(seq 1 "$NUM_ITERS"); do
        echo "=========================================================="
        echo "Running bench_serving_glm4p5_65k.sh NUM_PROMPTS=$np iteration $i/$NUM_ITERS"
        OUTPUT_FILE="${OUTPUT_DIR}/dp8ep8_mtp2_model_runner_v2_${np}.jsonl" \
        NUM_PROMPTS="$np" \
        bash "${SCRIPT_DIR}/bench_serving_glm4p5_65k.sh"
    done
done
