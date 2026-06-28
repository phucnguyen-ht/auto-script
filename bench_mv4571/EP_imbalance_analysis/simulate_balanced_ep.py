import torch
from loguru import logger
from tqdm import tqdm
import pandas as pd

from aiter.test_common import perftest
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


# ================================================================================
# Prepare inputs
# ================================================================================
def prepare_expert_maps(global_num_experts, num_gpus, dtype=torch.int32, device="cuda"):
    """Example:
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1], device='cuda:0', dtype=torch.int32)
    """
    expert_maps = []
    num_experts_per_gpu = global_num_experts // num_gpus
    for gpu_id in range(num_gpus):
        expert_map = [-1] * global_num_experts
        expert_map[
            gpu_id * num_experts_per_gpu : \
            gpu_id * (num_experts_per_gpu + 1)
        ] = list(range(0, num_experts_per_gpu))
        expert_maps.append(torch.tensor(expert_map, dtype=dtype, device=device))
    return expert_maps


# hyper parameters for benchmarking
coldi = 2
warmi = 5

# load inputs
logger.info("Loading sample inputs...")
sample_input_path = "/remote/vast0/share-mv/tran/workspace/GPT-OSS/scripts/ep_balance/inputs.pt"
inputs = torch.load(sample_input_path, weights_only=False)
device = "cuda"
num_gpus = 4

# prepare inputs
logger.info("Preparing base inputs...")
w1 = inputs["w1"].to(device)
w2 = inputs["w2"].to(device)
activation = inputs["activation"]
quant_config = inputs["quant_config"]
apply_router_weight_on_input = inputs["apply_router_weight_on_input"]
global_num_experts = inputs["global_num_experts"]
expert_maps = prepare_expert_maps(global_num_experts, num_gpus=num_gpus, dtype=torch.int32, device=device)

# params
hidden_dim = inputs["hidden_states"].shape[-1]
topk = inputs['topk_weights'].shape[-1]
assert topk == inputs['topk_ids'].shape[-1], "topk mismatch"
num_experts_per_gpu = global_num_experts // num_gpus

# ================================================================================
# Load topk_ids and topk_weights from log file
# ================================================================================
# read log file, except lines before "Application startup complete."
log_path = "/remote/vast0/share-mv/tran/workspace/GPT-OSS/log/20251127/dp4-ep/benchmark_logs_full/server_32768_1024_128_384.log"
logger.info(f"Loading topk_ids and topk_weights from log file: {log_path}...")
with open(log_path, "r") as f:
    lines = f.readlines()

start_index = None
for i, line in enumerate(lines):
    if "Application startup complete." in line:
        start_index = i
        break
lines = lines[start_index:]
logger.success(f"Application startup complete at line index: {start_index}")
logger.info(f"Total lines to process: {len(lines)}")

# Load expert count
term = "[TOPK] Expert selection counts: "
expert_lines = [line for line in lines if term in line]
logger.success(f"Found {len(expert_lines)} expert selection logs.")

logger.info("Parsing expert selection counts...")
expert_data = []
for line in tqdm(expert_lines):
    parts = line.split(term)[1].strip().strip("[]").split(", ")
    counts = list(map(int, parts))
    expert_data.append(counts)

expert_dist = [sum(counts) for counts in zip(*expert_data)]
logger.success(f"Expert distribution (total counts per expert): {expert_dist}")

# realtime expert count by GPUs
logger.info("Calculating GPU-wise expert counts...")
gpu_data = []
for data in tqdm(expert_data):
    gpu_counts = [0] * 4
    gpu_counts[0] = sum(data[0:32])
    gpu_counts[1] = sum(data[32:64])
    gpu_counts[2] = sum(data[64:96])
    gpu_counts[3] = sum(data[96:128])
    gpu_data.append(gpu_counts)

gpu_dist = [sum(counts) for counts in zip(*gpu_data)]
logger.success(f"GPU-wise expert distribution (total counts per GPU): {gpu_dist}")

# load real topk_ids from log file
real_topk_ids = []

term = "[TOPK] topk_ids: "
topk_ids_lines = [line for line in lines if term in line]
logger.success(f"Found {len(topk_ids_lines)} topk_ids logs.")

logger.info("Parsing real topk_ids...")
max_num_tokens = 0
for line in tqdm(topk_ids_lines):
    line = line.split(term)[1].strip().strip("[]").split(", ")
    ids = list(map(int, line))
    real_topk_ids.append(ids) # keep ids as list, change to torch tensor later to save memory
    max_num_tokens = max(max_num_tokens, len(ids))
logger.success(f"Parsed {len(real_topk_ids)} topk_ids entries.")
logger.info(f"Maximum number of tokens processed in a single log: {max_num_tokens}")
max_num_tokens = ((max_num_tokens + 32767) // 32768) * 32768 # align to 32768

# generate balanced topk_ids
logger.info("Generating balanced topk_ids based on real expert selection...")
def generate_balance_ids(num_tokens, dtype=torch.int32, device="cuda") -> torch.Tensor:
    topk_ids = torch.randint(0, 32, (num_tokens, 4), dtype=dtype, device=device)
    offsets = torch.tensor([0, 32, 64, 96], dtype=dtype, device=device)
    topk_ids += offsets  # in-place addition
    return topk_ids
balanced_topk_ids = generate_balance_ids(max_num_tokens, dtype=torch.int32, device=device)
logger.success(f"Balanced topk_ids generated. Shape: {balanced_topk_ids.shape}")

# generate fake topk_weights -> this doesn't matter
logger.info("Generating fake topk_weights...")
fake_topk_weights = torch.ones((max_num_tokens, topk), dtype=torch.float32, device=device) / topk
logger.success(f"Fake topk_weights generated. Shape: {fake_topk_weights.shape}")

# prepare data
def prepare_balanced_topk_data(num_tokens: int):
    new_ids = balanced_topk_ids[:num_tokens].clone()
    new_weights = fake_topk_weights[:num_tokens].clone()
    return new_ids, new_weights

def prepare_real_topk_data(index: int):
    new_ids = torch.tensor(real_topk_ids[index]).reshape(-1, topk).to(device)
    num_tokens = new_ids.shape[0]
    new_weights = fake_topk_weights[:num_tokens].clone()
    return new_ids, new_weights

def generate_hidden_states(num_tokens: int):
    return torch.randn((num_tokens, hidden_dim), dtype=torch.bfloat16, device=device)

# ================================================================================
# Benchmark time for fused experts on each cases
# 1. balanced expert parallelism (1 DP only since 4 DP has the same latency)
# 2. original expert parallelism (4 DP) -> longest latency among 4 gpus
# ================================================================================

@perftest(num_warmup=coldi, num_iters=warmi)
def benchmark_fused_experts(
    hidden_states: torch.Tensor, 
    topk_ids: torch.Tensor, 
    topk_weights: torch.Tensor, 
    expert_map: torch.Tensor
):
    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        inplace=True,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

def get_gpu_gap(g_data):
    min_count = max(min(g_data), 1)
    max_count = max(g_data)
    return max_count / min_count

# Run benchmarks
logger.info("Starting benchmarks...")
benchmark_data = []
for index in tqdm(range(len(gpu_data))):
    # for step i
    gpu_gap = get_gpu_gap(gpu_data[index])

    # benchmark real expert parallelism
    logged_topk_ids, logged_topk_weights = prepare_real_topk_data(index)
    num_tokens = logged_topk_ids.shape[0]
    hidden_states = generate_hidden_states(num_tokens)
    assert logged_topk_ids.size() == logged_topk_weights.size(), \
        "Size mismatch between topk_ids and topk_weights: " \
        f"{logged_topk_ids.size()} vs {logged_topk_weights.size()}"

    real_gtimes = []
    for gpu_id in range(num_gpus):
        expert_map = expert_maps[gpu_id]
        _, gtime = benchmark_fused_experts(
            hidden_states=hidden_states,
            topk_ids=logged_topk_ids,
            topk_weights=logged_topk_weights,
            expert_map=expert_map,
        )
        real_gtimes.append(gtime.item())

    # benchmark balanced expert parallelism (1 DP only)
    generated_balanced_topk_ids, generated_balanced_topk_weights = prepare_balanced_topk_data(num_tokens)
    _, balanced_gtime = benchmark_fused_experts(
        hidden_states=hidden_states,
        topk_ids=generated_balanced_topk_ids,
        topk_weights=generated_balanced_topk_weights,
        expert_map=expert_maps[0],  # only need to run on 1 GPU
    )
    balanced_gtime = balanced_gtime.item()
    assert generated_balanced_topk_ids.size() == generated_balanced_topk_weights.size(), \
        "Size mismatch between topk_ids and topk_weights: " \
        f"{generated_balanced_topk_ids.size()} vs {generated_balanced_topk_weights.size()}"

    # store data
    benchmark_data.append({
        "step": index,
        "num_tokens": num_tokens,
        "expert_data": expert_data[index],
        "gpu_data": gpu_data[index],
        "gpu_gap": gpu_gap,
        "real_gtimes": real_gtimes,
        "balanced_gtime": balanced_gtime,
    })
logger.success("Benchmarks completed.")

# save benchmark data as torch file
output_path = "/remote/vast0/share-mv/tran/workspace/GPT-OSS/scripts/ep_balance/benchmark_ep_results.pt"
torch.save(benchmark_data, output_path)
logger.info(f"Benchmark data saved to {output_path}.")