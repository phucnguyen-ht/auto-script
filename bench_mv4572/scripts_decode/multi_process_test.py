import subprocess
import multiprocessing
import itertools
import requests
import time

# Define the ports for the environments
# ports = [11401]
# ports = [11401, 11403]
# ports = [11401, 11403, 11405, 11407]
# ports = [11401, 11403, 11405, 11407, 11409, 11411, 11413, 11415]
# ports = [11415]
# base_port = 11401
# core_indeces = [0]
# core_indeces = range(0, 4)
# core_indeces = [0, 2, 4, 6]
# core_indeces = [0, 4]
# core_indeces = [0, 1, 2, 3, 5, 6, 7]
# core_indeces = range(0, 8)
# core_indeces = [0, 4]
# ports = [base_port + ci for ci in core_indeces]

# For remote testing, only use port 80 (HTTPS)
ports = [8000]

print(ports)


def check_service_health(port):
    """
    Check if the service on the given port is alive and healthy.
    Returns True if service is healthy, False otherwise.
    """
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[Health Check] Checking port {port} at {current_time}")

        url = f'http://localhost:{port}/health'
        response = requests.get(url, timeout=10)
        is_healthy = response.status_code == 200
        status = "HEALTHY" if is_healthy else f"UNHEALTHY (status: {response.status_code})"

        print(f"[Health Check] Port {port}: {status}")
        return is_healthy
    except requests.exceptions.RequestException as e:
        print(f"[Health Check] Port {port}: UNHEALTHY - {str(e)}")
        return False


def reset_prefix_cache(port):
    """Reset the prefix cache on the given port before each test run."""
    try:
        url = f'http://localhost:{port}/reset_prefix_cache'
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            print(f"[Reset Cache] Port {port}: prefix cache reset successfully")
        else:
            print(f"[Reset Cache] Port {port}: reset returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[Reset Cache] Port {port}: failed to reset prefix cache - {str(e)}")


def wait_for_services_ready(ports, check_interval=60):
    """
    Wait for all services to be ready before starting tests.
    Checks each port every check_interval seconds.
    """
    print(f"\n[Health Check] Starting service health checks...")
    print(f"[Health Check] Monitoring {len(ports)} port(s): {ports}")
    print(f"[Health Check] Check interval: {check_interval} seconds")
    print(f"[Health Check] Waiting for all services to be ready...\n")

    all_ready = False
    attempt = 0

    while not all_ready:
        attempt += 1
        print(f"[Health Check] Attempt #{attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        health_results = {}
        for port in ports:
            health_results[port] = check_service_health(port)

        all_ready = all(health_results.values())

        if all_ready:
            print(f"\n[Health Check] ✓ All services are ready!")
            print(f"[Health Check] Ready ports: {list(health_results.keys())}")
            print(f"[Health Check] Starting tests now...\n")
            break
        else:
            unhealthy_ports = [port for port, healthy in health_results.items() if not healthy]
            print(f"[Health Check] ✗ Waiting for {len(unhealthy_ports)} port(s): {unhealthy_ports}")
            print(f"[Health Check] Next check in {check_interval} seconds...\n")
            time.sleep(check_interval)

    return all_ready

# Define the ranges for parallel_threads and encoding_size
# parallel_threads_range = range(1, 2)
# parallel_threads_range = range(1, 17)
# parallel_threads_range = range(1, 33)
# parallel_threads_range = range(1, 33)
parallel_threads_range = [16, 32, 64, 96, 128, 192, 256]
# Per-dataset concurrency override. A dataset listed here uses its own
# concurrency list instead of the default parallel_threads_range above. The 1M
# dataset only makes sense at low concurrency (very long inputs), so cap it at
# 4 and 8.
dataset_parallel_threads = {
    "1M": [4, 8],
}
# parallel_threads_range = [1, 2, 4, 8, 16, 32, 64, 128, 3, 6, 12, 24, 48, 96, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 9, 11, 13, 15, 18, 22, 26, 30, 36, 44, 52, 60, 72, 88, 104, 120, 17, 19, 21, 23, 25, 27, 29, 31, 34, 38, 42, 46, 50, 54, 58, 62, 68, 76, 84, 92, 100, 108, 116, 124, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127]
# Sweep over LongBench-v2 custom datasets (same data as
# benchmark/bench_kimi26_dp8_moe_ep8.sh) instead of encoding_size. Each entry
# maps to <DATASET_DIR>/longbenchv2-<name>.jsonl.
DATASET_DIR = "/remote/vast0/share-mv/longbenchv2-custom"
datasets_range = ["8k", "10k", "100k", "1M"]

# Approximate input length per dataset (target ISL from the dataset manifest).
# Only used as a numeric label in log file names / [Test] lines.
dataset_target_isl = {"8k": 8192, "10k": 10000, "100k": 100000, "1M": 1000000}

# Define the base command
unit_test_path = "/workspace/vllm-moreh/benchmark_zhipu/unit_test.py"
# base_command = "python3 unit_test.py --time_limit 1840 --ignore_time_start 20 --ignore_time_end 20"
# base_command = "python3 unit_test.py --time_limit 1040 --ignore_time_start 20 --ignore_time_end 20 --cache_warmup"
# Benches no longer warm up individually: warmup is done once per dataset (see
# run_warmup + the dataset-outer loop in __main__), so the per-concurrency
# benches just run the formal test against the already-warmed server.
base_command = f"python3 {unit_test_path} --time_limit 240 --ignore_time_start 20 --ignore_time_end 20"

def run_warmup(port, dataset):
    """(Re-)warm the server for a dataset before a bench.

    unit_test.py's warmup phase processes the whole dataset (9 parallel
    requests per prompt) regardless of parallel_threads, so it is the same work
    for every concurrency level. Run before each bench to restore input
    prefixes that the previous bench's decode may have evicted from the KV
    cache. Cheap because the server stays warm (no prefix-cache reset between
    benches). Emits no bench markers.
    """
    data_path = f"{DATASET_DIR}/longbenchv2-{dataset}.jsonl"
    encoding_size = dataset_target_isl.get(dataset, 0)
    output_path = f"/workspace/vllm-moreh/benchmark_zhipu/results/warmup-longbenchv2-{dataset}_{time.time()}"
    log_file_name = f"warmup_d{dataset}_c_{port % 100}.log"
    # parallel_threads/time_limit are unused by the warmup-only path.
    command = (f"python3 {unit_test_path} --warmup_only --parallel_threads 1 "
               f"--time_limit 1 --data_path {data_path} "
               f"--log_file_name {log_file_name} --port {port} "
               f"--encoding_size {encoding_size} --output_path {output_path}")
    print(f"[Warmup] dataset {dataset} on port {port} (re-warm before bench)")
    subprocess.run(command, shell=True)


def run_bench(port, parallel_threads, dataset):
    """Run one formal-test scenario (no warmup) against the warmed server."""
    data_path = f"{DATASET_DIR}/longbenchv2-{dataset}.jsonl"
    # encoding_size is kept only as a numeric label for unit_test.py's logging.
    encoding_size = dataset_target_isl.get(dataset, 0)

    output_path = f"/workspace/vllm-moreh/benchmark_zhipu/results/longbenchv2-{dataset}_p{parallel_threads}_{time.time()}"
    log_file_name = f"tgi_test_p{parallel_threads}_d{dataset}_c_{port % 100}.log"
    command = f"{base_command} --parallel_threads {parallel_threads} --data_path {data_path} --log_file_name {log_file_name} --port {port} --encoding_size {encoding_size} --output_path {output_path}"

    print(f"[Bench] concurrency {parallel_threads}, dataset {dataset}, port {port}")

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Wait for all services to be ready before starting tests.
    wait_for_services_ready(ports, check_interval=60)

    # Single-port sweep, dataset-outer then concurrency-inner. The prefix cache
    # is reset ONCE per dataset (not per scenario), so it stays warm across the
    # concurrency levels. Before each bench we re-warmup: a prior bench's decode
    # can evict the warmed input prefixes from the KV cache, so we restore them
    # first. This re-warmup is cheap -- the server's kernels/CUDA graphs are
    # already compiled and most prompts are still cached (no reset between
    # benches), so it mostly re-primes evicted prefixes rather than cold-starting.
    port = ports[0]
    for dataset in datasets_range:
        reset_prefix_cache(port)
        # Datasets in dataset_parallel_threads use their own concurrency list;
        # everything else falls back to the default parallel_threads_range.
        threads_range = dataset_parallel_threads.get(dataset, parallel_threads_range)
        for parallel_threads in threads_range:
            run_warmup(port, dataset)
            run_bench(port, parallel_threads, dataset)
