import subprocess
import multiprocessing
import itertools
import math
import requests
import time
import os

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
# parallel_threads_range = [1, 2, 4, 8, 16, 32, 64, 128, 3, 6, 12, 24, 48, 96, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 9, 11, 13, 15, 18, 22, 26, 30, 36, 44, 52, 60, 72, 88, 104, 120, 17, 19, 21, 23, 25, 27, 29, 31, 34, 38, 42, 46, 50, 54, 58, 62, 68, 76, 84, 92, 100, 108, 116, 124, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127]
# Per-dataset concurrency override. A dataset listed here uses its own
# concurrency list instead of the default parallel_threads_range above. The 1M
# dataset only makes sense at low concurrency (very long inputs), so cap it at
# 4 and 8.
dataset_parallel_threads = {
    "1M": [4, 8],
}
# Sweep over LongBench-v2 custom datasets (same data as
# benchmark/bench_kimi26_dp8_moe_ep8.sh) instead of encoding_size. Each entry
# maps to <DATASET_DIR>/longbenchv2-<name>.jsonl.
DATASET_DIR = "/remote/vast0/share-mv/longbenchv2-custom"
datasets_range = ["8k", "10k", "100k", "1M"]
# Wrapper env overrides (sweep passes these; manual runs edit the range lines above).
if os.environ.get("REBENCH_CONC"):
    parallel_threads_range = [int(x) for x in os.environ["REBENCH_CONC"].split(",")]
if os.environ.get("REBENCH_DATASETS"):
    datasets_range = os.environ["REBENCH_DATASETS"].split(",")
print(f"[REBENCH] parallel_threads_range = {parallel_threads_range}", flush=True)
print(f"[REBENCH] datasets_range = {datasets_range}", flush=True)

# Approximate input length per dataset (target ISL from the dataset manifest).
# Only used as a numeric label in log file names / [Test] lines.
dataset_target_isl = {"8k": 8192, "10k": 10000, "100k": 100000, "1M": 1000000}

# Define the base command. unit_test.py sits beside this file; results dir is
# env-overridable (sweep wrapper points it at the run folder), else local ./results.
# Prefill benchmark: no warmup and no prefix-cache reset -- prefix caching is disabled
# at serve time (--no-enable-prefix-caching), so every request recomputes its full
# prompt, which is exactly what we measure.
_HERE = os.path.dirname(os.path.abspath(__file__))
unit_test_path = os.path.join(_HERE, "unit_test.py")
RESULTS_DIR = os.environ.get("REBENCH_RESULTS_DIR") or os.path.join(_HERE, "results")
base_command = f"python3 {unit_test_path}"

# --- Measurement window sizing --------------------------------------------
# A request is only recorded if it BOTH starts and finishes inside the steady
# band [ignore_start, time_limit - ignore_end]. So the usable send window is
# roughly (band_width - TTFT): if the tail TTFT approaches the band width, late
# requests never finish in time and the in-window count collapses (few/no rows,
# noisy prefill_throughput). TTFT grows with (a) input length and (b)
# concurrency (queue wait), so the window must grow on both axes:
#
#   * per-dataset base window -- longer inputs have larger prefill latency.
#     100k/1M need a much larger window than 8k/10k.
#   * a concurrency bump -- at high concurrency TTFT is dominated by queue wait,
#     so widen the window as the client count grows (this is why the high-conc
#     8k/10k cells, not just 100k, were collapsing at the old fixed 240s).
#
# All values are wall-clock seconds and are meant to be tuned per cluster.
# Window sizing is derived, not hand-set: each scenario's time_limit is the wall
# time for one full "first wave" of `concurrency` concurrent prompts to finish
# prefill, plus headroom so the wave lands inside the record window:
#
#     time_limit = ceil(WINDOW_HEADROOM * concurrency * L_dataset / P_dataset)
#
# where L_dataset = mean prompt tokens and P_dataset = cluster prefill throughput
# (tokens/sec summed across DP ranks). Both come from server-side measurement
# (see RUN_AND_BENCH §4); re-measure and update these if the model / preset /
# cluster changes. This scales automatically with both concurrency and input
# length -- long-ISL / high-concurrency cells get a much longer window (100k @
# 256 is ~20-30 min), which is expected and accepted. A floor keeps tiny
# low-concurrency waves from producing a too-short, noisy window.
#
# EXCEPTION -- very long inputs (1M): the conc*L/P model assumes all requests
# share one cluster-wide throughput P, which holds when concurrency >> DP size
# (requests queue through a fixed bottleneck). At 1M we only run conc 4/8 <= DP
# size (8), so each DP rank prefills its own request in parallel and per-request
# TTFT is ~constant across concurrency (~500-530s, bound by chunked prefill at
# max_num_batched_tokens=8192 + the DSA indexer cost that grows with position),
# NOT proportional to conc. The formula therefore under-sizes these windows
# (conc4 -> ~260s, conc8 -> ~520s) and records 0 requests. We floor the 1M window
# instead (see dataset_min_time_limit) so a full wave lands inside it.
dataset_prompt_tokens = {"8k": 8204, "10k": 10012, "100k": 100012, "1M": 1000000}
dataset_prefill_tps = {"8k": 20000, "10k": 20000, "100k": 20000, "1M": 20000}
DEFAULT_PREFILL_TPS = 20000
WINDOW_HEADROOM = 1.25
MIN_TIME_LIMIT = 60          # seconds
# Per-dataset time_limit floor (wall-clock seconds), overriding MIN_TIME_LIMIT.
# For 1M the proportional formula under-sizes the window (see note above), so we
# floor it at ~1.25 * measured per-request TTFT (conc4 ~495s, conc8 ~526s -> 700s)
# so at least one full wave of prefills completes inside the record window.
dataset_min_time_limit = {"1M": 700}
# No warm-up / cool-down: the whole window is measured. Fine for the headline
# prefill_throughput (workers jump to full concurrency at t0, tokens/span is
# self-normalizing); only the TTFT percentiles / mean_prefill_tps read slightly
# optimistic because the initial synchronized wave is included. Set >0 for clean
# latency stats. See RUN_AND_BENCH §2.
IGNORE_FRACTION = 0.0


def window_params(dataset, parallel_threads):
    """(time_limit, ignore_start, ignore_end) for one scenario.

    time_limit is sized so at least the first wave of `parallel_threads`
    concurrent prompts finishes prefill inside the window (times WINDOW_HEADROOM),
    from the measured per-dataset throughput -- scaling automatically with both
    concurrency and input length. A per-dataset floor (dataset_min_time_limit)
    overrides this for datasets where the proportional model does not hold (1M --
    see note above), keeping the window long enough to record a full wave.
    """
    L = dataset_prompt_tokens.get(dataset, dataset_target_isl.get(dataset, 0))
    P = dataset_prefill_tps.get(dataset, DEFAULT_PREFILL_TPS)
    first_wave = (parallel_threads * L / P) if P > 0 else 0.0
    floor = max(MIN_TIME_LIMIT, dataset_min_time_limit.get(dataset, 0))
    time_limit = max(floor, math.ceil(WINDOW_HEADROOM * first_wave))
    ignore = int(round(time_limit * IGNORE_FRACTION))
    return time_limit, ignore, ignore


def run_bench(port, parallel_threads, dataset):
    """Run one prefill scenario (no warmup) against the server."""
    data_path = f"{DATASET_DIR}/longbenchv2-{dataset}.jsonl"
    # encoding_size is kept only as a numeric label for unit_test.py's logging.
    encoding_size = dataset_target_isl.get(dataset, 0)

    time_limit, ignore_start, ignore_end = window_params(dataset, parallel_threads)

    output_path = f"{RESULTS_DIR}/longbenchv2-{dataset}_p{parallel_threads}_{time.time()}"
    log_file_name = f"tgi_test_p{parallel_threads}_d{dataset}_c_{port % 100}.log"
    command = (f"{base_command} --time_limit {time_limit} "
               f"--ignore_time_start {ignore_start} --ignore_time_end {ignore_end} "
               f"--parallel_threads {parallel_threads} --data_path {data_path} "
               f"--log_file_name {log_file_name} --port {port} "
               f"--encoding_size {encoding_size} --output_path {output_path}")

    print(f"[Bench] concurrency {parallel_threads}, dataset {dataset}, port {port} "
          f"(time_limit={time_limit}s, ignore={ignore_start}s/side)")

    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Wait for all services to be ready before starting tests.
    wait_for_services_ready(ports, check_interval=60)

    # Single-port sweep, dataset-outer then concurrency-inner. No warmup and no
    # prefix-cache reset: prefix caching is disabled in the preset, so each bench
    # measures cold prefill throughput directly.
    port = ports[0]
    for dataset in datasets_range:
        # Datasets in dataset_parallel_threads use their own concurrency list;
        # everything else falls back to the default parallel_threads_range.
        threads_range = dataset_parallel_threads.get(dataset, parallel_threads_range)
        for parallel_threads in threads_range:
            run_bench(port, parallel_threads, dataset)
