from loguru import logger
import torch
from tqdm import tqdm
import os


# split jobs into 8 splits
n_splits = 8
output_dir = "scripts/ep_balance/jobs"
os.makedirs(output_dir, exist_ok=True)

# ================================================================================
# Load topk_ids and topk_weights from log file
# ================================================================================
# read log file, except lines before "Application startup complete."
log_path = "/remote/vast0/share-mv/tran/workspace/GPT-OSS/log/20251127/dp4-ep/benchmark_logs_full/server_32768_1024_128_384.log"
logger.info(f"Loading topk_ids and topk_weights from log file: {log_path}...")
with open(log_path, "r") as f:
    lines = f.readlines()

# Find start point in log file
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

# Realtime expert count by GPUs
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

# Load real topk_ids from log file
real_topk_ids = []

term = "[TOPK] topk_ids: "
topk_ids_lines = [line for line in lines if term in line]
logger.success(f"Found {len(topk_ids_lines)} topk_ids logs.")

logger.info("Parsing real topk_ids...")
for line in tqdm(topk_ids_lines):
    line = line.split(term)[1].strip().strip("[]").split(", ")
    ids = list(map(int, line))
    real_topk_ids.append(ids) # keep ids as list, change to torch tensor later to save memory
logger.success(f"Parsed {len(real_topk_ids)} topk_ids entries.")


# Split jobs
assert len(real_topk_ids) == len(expert_data) and len(real_topk_ids) == len(gpu_data)
num_jobs = len(real_topk_ids)
jobs_per_split = (num_jobs + n_splits - 1) // n_splits
logger.info(f"Splitting {num_jobs} jobs into {n_splits} splits, each with up to {jobs_per_split} jobs.")

# Save each split to a torch file
for split_idx in range(n_splits):
    start_job = split_idx * jobs_per_split
    end_job = min((split_idx + 1) * jobs_per_split, num_jobs)
    split_topk_ids = real_topk_ids[start_job:end_job]
    split_expert_data = expert_data[start_job:end_job]
    split_gpu_data = gpu_data[start_job:end_job]

    split_file = os.path.join(output_dir, f"job_split_{split_idx}.pt")
    torch.save({
        "topk_ids": split_topk_ids,
        "expert_data": split_expert_data,
        "gpu_data": split_gpu_data,
    }, split_file)
    logger.success(f"Saved job split {split_idx} with jobs {start_job} to {end_job} to {split_file}.")