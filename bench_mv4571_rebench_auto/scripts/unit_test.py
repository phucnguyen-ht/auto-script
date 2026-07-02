import os
import json
import time
import random
import requests
import argparse
import queue
import re
import statistics
import threading
import traceback
import sys
from multiprocessing import Process, Manager
import concurrent.futures
from typing import List, Optional, Union, Dict, Any

ENABLE_PROFILE = True

# Output length (max_tokens) for the formal run. Set in main() based on the
# dataset: 1024 for the 8k dataset (encoding_size == 8192), 500 otherwise,
# matching benchmark/bench_kimi26_dp8_moe_ep8.sh. Forked children inherit it.
OUTPUT_LEN = 500

def start_profile():
    if not ENABLE_PROFILE:
        return
    url = 'http://localhost:8000/start_profile'
    try:
        response = requests.post(url)
        print(f"[start_profile] Status code: {response.status_code}")
    except Exception as e:
        print(f"[start_profile] Error: {e}")

def stop_profile():
    if not ENABLE_PROFILE:
        return
    url = 'http://localhost:8000/stop_profile'
    try:
        response = requests.post(url)
        print(f"[stop_profile] Status code: {response.status_code}")
    except Exception as e:
        print(f"[stop_profile] Error: {e}")
def _percentile(xs, q):
    """np.percentile-equivalent linear interpolation. Returns 0.0 on empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac

def build_chat_input(query, history=None, role="user"):
    def build_single_message(role, metadata, message):
        assert role in ["system", "user", "assistant", "observation"], role
        role_text = f"<|{role}|>" + f"{metadata}\n"
        return role_text + message

    if history is None:
        history = []
    raw_text = "[gMASK]<sop>"
    for item in history:
        content = item["content"]
        if item["role"] == "system" and "tools" in item:
            content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
        raw_text += build_single_message(item["role"], item.get("metadata", ""), content)
    raw_text += build_single_message(role, "", query)
    raw_text += "<|assistant|>"
    return raw_text




# def apply_chat_template(
#     conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"],
#     add_generation_prompt: bool = False,
#     return_dict: bool = False,
#     tokenizer_kwargs: Optional[Dict[str, Any]] = None,
#     add_special_tokens: bool = True,
#     **kwargs,
# ) -> Union[str, List[str], List[List[str]]]:

#     if return_dict:
#         raise ValueError("`return_dict=True` is incompatible when `tokenize=False`.")

#     def build_single_message(role, metadata, message, tokenize=False):
#         assert role in ["system", "user", "assistant", "observation"], role
#         if tokenize:
#             raise ValueError("Tokenization is not supported in this simplified version.")
#         else:
#             return str(f"<|{role}|>{metadata}\n{message}")

#     def handle_single_conversation(conversation):
#         input_message = " " if add_special_tokens else ""
#         for item in conversation:
#             if item.get("tools"):
#                 tools = item["tools"]
#                 content = "你是一个名为 GhatGLM 的人工智能助手。..."
#                 # ... (rest of the tools content setup)
#                 input_message += build_single_message("system", "", content, tokenize=False)
#             if item["content"]:
#                 input_message += build_single_message(
#                     item["role"],
#                     item.get("metadata", ""),
#                     item["content"],
#                     tokenize=False
#                 )
#         if add_generation_prompt:
#             input_message += " "
#         return input_message

#     # Main logic to handle different conversation formats
#     if isinstance(conversation, list) and all(isinstance(i, dict) for i in conversation):
#         result = handle_single_conversation(conversation)
#     elif isinstance(conversation, list) and all(isinstance(i, list) for i in conversation):
#         result = [handle_single_conversation(c) for c in conversation]
#     elif hasattr(conversation, "messages"):
#         result = handle_single_conversation(conversation.messages)
#     else:
#         raise ValueError("Invalid conversation format")

#     return result

def load_data(file_path: str) -> List[str]:
    # LongBench-v2 custom datasets store the prompt under 'prompt'; older
    # encoding datasets use 'text'. Accept whichever is present.
    def _extract(obj):
        for key in ('prompt', 'text'):
            if key in obj:
                return obj[key]
        raise KeyError(f"line has neither 'prompt' nor 'text': keys={list(obj.keys())}")

    with open(file_path, 'r') as file:
        data = [_extract(json.loads(line)) for line in file]
    return data
    # return [build_chat_input('Hello! What''s your name?')]

def make_tgi_payload(prompt):
    return {
        "inputs": prompt,
        "parameters": {
            "best_of": 1,
            "details": True,
            "do_sample": True,
            "frequency_penalty": 0.1,
            "max_new_tokens": 256,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "seed": None,
            "stop": ["</s>", "<|endoftext|>", "<|user|>", "<|observation|>"],
            "temperature": 0.95,
            "top_k": None,
            "top_p": 0.7,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": True
        }
    }

def make_vllm_payload(prompt, is_warmup=False):
    ''' Added by Cambricon to apply chat template for prompt
            1. for GLM-4-9B-Chat, we can follow the README: https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README.md?code=true#L94
            2. we modify the pyload as follows:
                2.1 using 'prompt_' after apply_chat_template instead of original prompt
                2.2 using 'stop_token_ids' instead of 'stop'
    '''
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/data/models/GLM-4-9B-Chat/", trust_remote_code=True)
    # for token_string in ["</s>", "<|endoftext|>", "<|user|>", "<|observation|>"]:
    #     print(token_string, tokenizer.encode(token_string))
    # # prompt_ = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    max_tokens = 8 if is_warmup else OUTPUT_LEN
    return  {
        #  "model": "model",
        "model": "/remote/vast0/share-mv/zai-org/GLM-5.2-FP8/",
        #  "prompt": f"<|user|>{prompt}<|assistant|>",
        # "prompt": prompt,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
         "max_tokens": max_tokens,
         "temperature": 0,
        #   "stop": [ "<|endoftext|>",  "<|user|>",  "<|observation|>",  "<|assistant|>"],
         "stream": True,
         "stream_options": { "include_usage": True },
        # "stop_token_ids": [151329, 151336, 151338],

        #  "stop": ["</s>"],
        #  "n": 1,
        #  "presence_penalty": 0.5,
         # "frequency_penalty": 0.1,
        #  "logprobs": 5,
        #  "echo": True,
        #  "best_of": 1,
        #  "suffix": "",
        #  "user": "user123",
        #  "use_beam_search": False,
        #  "top_k": 50,
        #  "min_p": 0.0,
        #  "repetition_penalty": 1.2,
        #  "length_penalty": 1.0,
        #  "early_stopping": False,
        # "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": True,
         "min_tokens": max_tokens,
        # "skip_special_tokens": False,
        #  "spaces_between_special_tokens": True,
        #  "truncate_prompt_tokens": None,
        "include_stop_str_in_output": True,
        #  "response_format": "json"
    }


def combine_escaped_string(str1, str2):
    return f"b'{str1.strip()[2:-1]}{str2.strip()[2:-1]}'"

def decode_utf8_escaped_string(escaped_str):
    byte_str = bytes(escaped_str.strip()[2:-1], 'utf-8').decode('unicode_escape')
    
    try:
        decoded_str = byte_str.encode('latin1').decode('utf-8')
    except UnicodeDecodeError as e:
        return escaped_str, False
    
    return decoded_str, True

def make_request(data: List[str], log_queue: queue.Queue, official_start_time: float, official_end_time: float, stop_event: threading.Event, log_prefix: str, log_file_name: str, port: int, is_warmup: bool = False):
    # url = f'http://127.0.0.1:{port}/generate_stream'
    # url = f'http://127.0.0.1:{port}/v1/chat/completions'

    # For remote testing with HTTPS
    url = 'http://localhost:8000/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        # 'Authorization': 'Bearer 4tz2eLWWU9CTtyJF1FWDm3DJfPV0tzGmuRuzHwhMqISZAfn0MRyDOoqMCrb2ioxv'
    }
    
    # temp_file = open("/data/workspaces/data/2.txt", "a+")

    if is_warmup:
        # Warmup phase: process all data items in order
        for input_text in data:
            if stop_event.is_set():
                break
            # Initialize reasoning_aggregate for this request
            reasoning_aggregate = ""
            payload = make_vllm_payload(input_text, True)
            test_start_time = time.time()
            response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
            print(f'[Warmup Response status code][{port}] {response.status_code}')

            # Process response without logging
            for line in response.iter_lines():
                if line.startswith(b'data:'):
                    event_data = line[5:]
                    event_data = event_data.decode("utf-8").strip()
                    if event_data == "[DONE]":
                        break
                    try:
                        response_json = json.loads(event_data)
                        if "error_type" in response_json:
                            break
                    except json.JSONDecodeError:
                        continue

            # Small delay between warmup requests to avoid overwhelming the server
            time.sleep(0.1)
    else:
        # Normal testing phase: random selection with time-based logging
        while not stop_event.is_set():
            # Initialize reasoning_aggregate for this request
            reasoning_aggregate = ""
            # input_text = build_chat_input(random.choice(data))
            input_text = random.choice(data)
            # input_text = apply_chat_template([{"role": "user", "content": random.choice(data)}], tokenize=False, add_generation_prompt=True)
            # input_text = random.choice(data)
            payload = make_vllm_payload(input_text)
            # print(payload)
            test_start_time = time.time()
            response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
            print(f'[Response status code][{port}] {response.status_code}')
            first_token_received = False
            total_tokens = 0
            first_token_latency = None

            last_token_received = time.time()
            # [TPOT] Track timestamps of content-bearing chunks only (not every SSE
            # frame), so TPOT is computed between actual emitted tokens.
            first_content_ts = None
            last_content_ts = None
            # [TPOT] Per-token timestamps for the highest-TPOT request analysis.
            token_timestamps = []


            is_error_and_handled = False

            generated_text_aggregate = ""
            incomplete_sequence = ""
            # print(5, response)

            for line in response.iter_lines():
                # print(4, line)

                # temp_file.writelines(line.decode("utf-8").strip() + "\n")

                if line.startswith(b'data:'):
                    event_data = line[5:]
                    # print("[" + event_data + "]")
                    # print("[" + event_data.decode("utf-8") + "]")

                    event_data = event_data.decode("utf-8").strip()     # only vllm
                    if event_data == "[DONE]":
                        break

                    response_json = json.loads(event_data)
                    # response_json = json.loads(event_data.decode("utf-8").strip())
                    # print(response_json)

                    if "error_type" in response_json:
                        if response_json["error_type"] == "generation":
                            if "(out of memory)" in response_json['error']:
                                log_queue.put((log_file_name, f'[Test] {log_prefix} Failed OOM\n'))
                            else:
                                log_queue.put((log_file_name, f'[Test] {log_prefix} Failed unknown_error\n'))
                                log_queue.put((log_file_name, f'[UNKNOWN_ERROR] {json.dumps(response_json, ensure_ascii=False)}\n'))
                            is_error_and_handled = True
                            break

                    test_end_time = time.time()

                    token_latency = time.time() - last_token_received
                    last_token_received = time.time()
                    # print("[Token Latency]", token_latency)

                    if len(response_json.get('choices', [])):
                        # response_text = response_json['choices'][0]['text']
                        choice = response_json['choices'][0]
                        delta = choice.get('delta', {})
                        response_text = delta.get('content')

                        # Handle case where content might be None
                        if response_text is None:
                            response_text = ""

                        # Handle reasoning field - try both 'reasoning' and 'reasoning_content' for compatibility
                        # Different engines may use different field names:
                        # - OpenAI o1/o3 uses 'reasoning'
                        # - Some implementations use 'reasoning_content'
                        reasoning_content = delta.get('reasoning')
                        if reasoning_content is None:
                            reasoning_content = delta.get('reasoning_content')

                        # Aggregate reasoning content
                        if reasoning_content is not None and reasoning_content != "":
                            reasoning_aggregate += reasoning_content

                        # if response_text:
                        # if incomplete_sequence:
                        #     temp_file.writelines(["1", json.dumps({"str":response_text}, ensure_ascii=False), "\n"])
                        #     temp_file.writelines(["3", json.dumps({"str":(combine_escaped_string(incomplete_sequence, response_text))}, ensure_ascii=False), "\n"])
                        #     print(1, json.dumps({"str":response_text}, ensure_ascii=False))
                        #     print(3, json.dumps({"str":(combine_escaped_string(incomplete_sequence, response_text))}, ensure_ascii=False))

                        # response_text, is_decoded = decode_utf8_escaped_string(combine_escaped_string(incomplete_sequence, response_text))
                        is_decoded = True

                        if is_decoded:
                            # If decoding was successful, print and aggregate the text
                            # if incomplete_sequence:
                            #     temp_file.writelines(["2", json.dumps({"str": response_text}, ensure_ascii=False), "\n"])
                            #     print(2, json.dumps({"str": response_text}, ensure_ascii=False))
                            # print(">>>>", generated_text_aggregate)
                            incomplete_sequence = ""
                            generated_text_aggregate += response_text
                        else:
                            # If decoding failed, save the text for the next iteration
                            incomplete_sequence = response_text

                        # Count tokens from both content and reasoning_content
                        if not first_token_received:
                            if (response_text is None or response_text == "") and (reasoning_content is None or reasoning_content == ""):
                                continue
                            first_token_latency = test_end_time - test_start_time
                            # [TPOT] First content-bearing chunk timestamp.
                            first_content_ts = test_end_time
                            last_content_ts = test_end_time
                            first_token_received = True
                            token_timestamps.append(test_end_time)
                            # if test_start_time >= official_start_time and test_end_time <= official_end_time:
                            #     log_queue.put(('test_result.log', f'First token latency: {first_token_latency}\n'))
                        else:
                            if (response_text is not None and response_text != "") or (reasoning_content is not None and reasoning_content != ""):
                                # [TPOT] Update last content-bearing chunk timestamp.
                                last_content_ts = test_end_time
                                total_tokens += 1
                                token_timestamps.append(test_end_time)
                            # No need to check 'details' for OpenAI API

            # Check if we received any valid tokens
            if first_token_latency is None and not is_error_and_handled:
                error_msg = f"[ERROR] Line 340: No valid tokens received. first_token_latency is None.\n"
                error_msg += f"This means all tokens in the stream had empty content (both response_text and reasoning_content were None or empty).\n"
                error_msg += f"Response: {json.dumps(response_json, ensure_ascii=False, indent=2)}\n"
                print(error_msg, file=sys.stderr)
                traceback.print_stack()
                sys.exit(1)

            # print(f"[Time] {test_start_time} {official_start_time} : {test_end_time} {official_end_time}")
            # print(f"[Time Diff] {test_start_time - official_start_time} {test_end_time - official_start_time} {official_end_time - official_start_time}")
            response_json['input_text'] = input_text

            # Combine reasoning content and generated content
            final_content = generated_text_aggregate
            if 'reasoning_aggregate' in locals() and reasoning_aggregate:
                final_content = f"<reasoning>{reasoning_aggregate}</reasoning>{generated_text_aggregate}"

            response_json['generated'] = final_content
            response_json['calculated_tokens'] = total_tokens

            # Get usage information from the last response_json if available
            if 'usage' in response_json and response_json['usage'] is not None:
                response_json['usage_tokens'] = response_json['usage'].get('completion_tokens', 0)
                response_json['token_difference'] = total_tokens - response_json['usage_tokens']
            else:
                response_json['usage_tokens'] = None
                response_json['token_difference'] = None

            # Use actual_tokens for all subsequent checks to handle MTP correctly
            actual_tokens = response_json['usage_tokens'] - 1
            if actual_tokens == 0 and first_token_latency is None:
                log_queue.put((log_file_name, f'[Not Generated] {json.dumps(response_json, ensure_ascii=False)}\n'))

            elif not is_error_and_handled and test_start_time >= official_start_time and test_end_time <= official_end_time:
                # print(actual_tokens, test_end_time - test_start_time, first_token_latency)
                try:
                    # Use actual_tokens from usage field for MTP support
                    # print(1, test_end_time, test_start_time, first_token_latency)
                    generation_time = test_end_time - test_start_time - first_token_latency
                    if generation_time <= 0:
                        tokens_per_second = 0
                    else:
                        tokens_per_second = actual_tokens / generation_time
                except ZeroDivisionError:
                    # print("[ZeroDivisionError]", json.dumps(response_json, ensure_ascii=False) + '\n')
                    # continue
                    tokens_per_second = -1
                    log_queue.put((log_file_name, f'[ZeroDivisionError] actual_tokens={actual_tokens}, first_token_latency={first_token_latency}\n'))

                # [TPOT] Compute per-output-token wall time, MTP-correct via
                # usage.completion_tokens (= actual_tokens + 1). N tokens have
                # N-1 inter-token gaps → divide by actual_tokens. Skip if we
                # only got 1 token (no gaps to measure).
                if (
                    first_content_ts is not None
                    and last_content_ts is not None
                    and actual_tokens >= 1
                    and last_content_ts > first_content_ts
                ):
                    tpot_ms = (last_content_ts - first_content_ts) / actual_tokens * 1000.0
                else:
                    tpot_ms = -1.0


                # [TPOT] Attach per-token timestamps and tpot_ms so the
                # highest-TPOT request can be identified in post-processing.
                response_json['tpot_ms'] = tpot_ms
                response_json['token_timestamps'] = token_timestamps
                log_queue.put(('output.jsonl', json.dumps(response_json, ensure_ascii=False) + '\n'))
                # print(3, actual_tokens, tokens_per_second)
                log_queue.put((log_file_name, f'[Test] {log_prefix} {actual_tokens + 1} {first_token_latency} {tokens_per_second} {total_tokens + 1} {tpot_ms:.4f}\n'))
                # print(response_json)
                # print(f'[Test] {log_prefix} {actual_tokens + 1} {first_token_latency} {tokens_per_second}')
                if actual_tokens == 1:
                    log_queue.put((log_file_name, json.dumps(response_json, ensure_ascii=False) + '\n'))

            # print(1, stop_event.is_set())

def log_writer(log_queue: queue.Queue, stop_event: threading.Event, output_path: str):
    try:
        while not stop_event.is_set() or not log_queue.empty():
            try:
                log_file, log = log_queue.get(timeout=1)
                
                # print("[log]", log_file, log)
                with open(os.path.join(output_path, log_file), 'a+') as file:
                    file.write(log)
                    file.flush()
                
                log_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
                continue
    finally:
        print("[Log writer]", "Done")


        
def main(parallel_threads: int, time_limit: float, ignore_time_start: float, ignore_time_end: float, data_path: str, output_path: str, log_file_name: str, port: int, encoding_size: int, cache_warmup: bool = False, warmup_only: bool = False):
    # 8k dataset (encoding_size == 8192) -> 1024 output tokens, else 500,
    # matching benchmark/bench_kimi26_dp8_moe_ep8.sh.
    global OUTPUT_LEN
    OUTPUT_LEN = 1024 if encoding_size == 8192 else 500
    print(f"[Config] OUTPUT_LEN (max_tokens) = {OUTPUT_LEN} (encoding_size={encoding_size})")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data = load_data(data_path)
    if encoding_size < 100000:
        data = data[:100]
    if encoding_size == 100000:
        data = data[:10]
    if encoding_size == 1000000:
        data = data[:1]

    if cache_warmup or warmup_only:
        print(f"[Cache Warmup] Starting warmup for {len(data)} test cases...")

        # Cache warmup phase
        for pi in range(len(data)):
            with Manager() as manager:
                log_queue = manager.Queue()
                stop_event = manager.Event()

                # For warmup, we set official times far in the future so no results are logged
                dummy_start_time = float('inf')
                dummy_end_time = float('inf')

                log_prefix = f"{16} {encoding_size} [WARMUP]"

                # Start warmup processes
                processes = []
                for i in range(16):  # Start 16 processes for warmup
                    # Each process gets a subset of data for parallel warmup
                    data_subset = [data[pi]]
                    process = Process(target=make_request, args=(data_subset, log_queue, dummy_start_time, dummy_end_time, stop_event, log_prefix, log_file_name, port, True))
                    process.start()
                    processes.append(process)

                log_writer_process = Process(target=log_writer, args=(log_queue, stop_event, output_path))
                log_writer_process.start()

                # Wait for all warmup processes to complete
                for process in processes:
                    process.join()

                stop_event.set()
                log_queue.join()
                log_writer_process.join()

        print("[Cache Warmup] Warmup completed. Starting formal test...")

    # In warmup-only mode we just primed the cache/kernels for this dataset and
    # exit before the formal test -- no bench_start/bench_end markers, so the
    # serve log stays clean. The driver runs this once per dataset, then runs
    # the per-concurrency benches (without warmup) against the warmed server.
    if warmup_only:
        print("[Cache Warmup] warmup_only=True -> skipping formal test.")
        return

    try:
        response = requests.get("http://localhost:8000/v1/bench_start")
    except:
        print("loi roi")
    start_profile()
    # Formal testing phase
    with Manager() as manager:
        log_queue = manager.Queue()
        stop_event = manager.Event()
        start_time = time.time()
        official_start_time = start_time + ignore_time_start
        official_end_time = start_time + time_limit - ignore_time_end

        log_prefix = f"{parallel_threads} {encoding_size}"

        processes = []
        for _ in range(parallel_threads):
            process = Process(target=make_request, args=(data, log_queue, official_start_time, official_end_time, stop_event, log_prefix, log_file_name, port, False))
            process.start()
            processes.append(process)

        log_writer_process = Process(target=log_writer, args=(log_queue, stop_event, output_path))
        log_writer_process.start()

        time.sleep(time_limit)
        stop_event.set()
        log_queue.join()

        for process in processes:
            process.join()
        log_writer_process.join()
    try:
        response = requests.get("http://localhost:8000/v1/bench_end")
    except:
        print("loi roi")
    stop_profile()

    # [TPOT] Aggregate after the log writer has flushed. Read only the per-request
    # [Test] lines that match this run's prefix (warmup lines have a different
    # prefix; failure lines have non-numeric token fields and are skipped).
    log_path = os.path.join(output_path, log_file_name)
    tpot_pattern = re.compile(
        r'^\[Test\] '
        r'(?P<threads>\d+) (?P<enc>\d+) '
        r'(?P<tokens>\d+) (?P<ttft>\S+) (?P<tps>\S+) (?P<chunks>\d+) '
        r'(?P<tpot>-?\d+(?:\.\d+)?)\s*$'
    )
    tpots_ms = []
    ttfts = []
    tpss = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as fh:
            for line in fh:
                m = tpot_pattern.match(line.strip())
                if not m:
                    continue
                if int(m.group('threads')) != parallel_threads or int(m.group('enc')) != encoding_size:
                    continue
                tpot = float(m.group('tpot'))
                if tpot >= 0:
                    tpots_ms.append(tpot)
                try:
                    ttfts.append(float(m.group('ttft')))
                    tpss.append(float(m.group('tps')))
                except ValueError:
                    pass

    summary = {
        "parallel_threads": parallel_threads,
        "encoding_size": encoding_size,
        "requests": len(tpots_ms),
        "mean_ttft_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "p90_ttft_s": round(_percentile(ttfts, 90), 4) if ttfts else None,
        "mean_decode_tps": round(statistics.mean(tpss), 4) if tpss else None,
        "mean_tpot_ms": round(statistics.mean(tpots_ms), 4) if tpots_ms else None,
        "p50_tpot_ms": round(_percentile(tpots_ms, 50), 4) if tpots_ms else None,
        "p90_tpot_ms": round(_percentile(tpots_ms, 90), 4) if tpots_ms else None,
        "p99_tpot_ms": round(_percentile(tpots_ms, 99), 4) if tpots_ms else None,
    }
    summary_line = "[Summary] " + json.dumps(summary)
    print(summary_line, flush=True)
    with open(log_path, 'a+') as fh:
        fh.write(summary_line + "\n")

    # [TPOT] Find requests with the highest and lowest TPOT and save their
    # per-token timestamps to dedicated files for latency analysis.
    def _fmt_ts(raw_ts: float) -> str:
        """Format a time.time() float as MM-DD HH:MM:SS."""
        import datetime
        return datetime.datetime.fromtimestamp(raw_ts).strftime('%m-%d %H:%M:%S')

    def _build_tpot_record(entry: dict, tpot_val: float) -> dict:
        ts_list = entry['token_timestamps']
        t0 = ts_list[0]
        token_ts_records = [
            {
                "token_index": i,
                "timestamp": _fmt_ts(ts),
                "offset_from_first_token_ms": round((ts - t0) * 1000.0, 3),
                "inter_token_gap_ms": round((ts - ts_list[i - 1]) * 1000.0, 3) if i > 0 else 0.0,
            }
            for i, ts in enumerate(ts_list)
        ]
        return {
            "tpot_ms": tpot_val,
            "total_tokens": len(ts_list),
            "generated": entry.get('generated', ''),
            "token_timestamps": token_ts_records,
        }

    output_jsonl_path = os.path.join(output_path, 'output.jsonl')
    if os.path.exists(output_jsonl_path):
        highest_entry, lowest_entry = None, None
        highest_tpot, lowest_tpot = -1.0, float('inf')
        with open(output_jsonl_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tpot = entry.get('tpot_ms')
                if tpot is None or tpot < 0:
                    continue
                if not entry.get('token_timestamps'):
                    continue
                if tpot > highest_tpot:
                    highest_tpot = tpot
                    highest_entry = entry
                if tpot < lowest_tpot:
                    lowest_tpot = tpot
                    lowest_entry = entry

        if highest_entry is not None:
            highest_tpot_path = os.path.join(output_path, 'highest_tpot_token_timestamps.json')
            with open(highest_tpot_path, 'w') as fh:
                json.dump(_build_tpot_record(highest_entry, highest_tpot), fh, ensure_ascii=False, indent=2)
            print(f"[Highest TPOT] {highest_tpot:.4f} ms/token — token timestamps saved to {highest_tpot_path}", flush=True)

        if lowest_entry is not None:
            lowest_tpot_path = os.path.join(output_path, 'lowest_tpot_token_timestamps.json')
            with open(lowest_tpot_path, 'w') as fh:
                json.dump(_build_tpot_record(lowest_entry, lowest_tpot), fh, ensure_ascii=False, indent=2)
            print(f"[Lowest TPOT]  {lowest_tpot:.4f} ms/token — token timestamps saved to {lowest_tpot_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load test script.')
    parser.add_argument('--parallel_threads', type=int, help='Number of parallel threads.', default=3)
    parser.add_argument('--time_limit', type=float, help='Time limit for the load test in seconds.', default=20)
    parser.add_argument('--ignore_time_start', type=float, help='Start time for logging in seconds.', default=0)
    parser.add_argument('--ignore_time_end', type=float, help='End time for logging in seconds.', default=5)
    parser.add_argument('--data_path', type=str, help='Path to the input data file.', default="/")
    parser.add_argument('--output_path', type=str, help='Path to the output directory.', default="/mnt/ceph/yxy/data/data/output/docker_4_with_stop_run2")
    parser.add_argument('--log_file_name', type=str, help='name of log file.', default="test_result.log")
    parser.add_argument('--port', type=int, help='Service port to be tested', default=11401)
    parser.add_argument('--encoding_size', type=int, help='Encoding average size', default=1024)
    parser.add_argument('--cache_warmup', action='store_true', help='Enable cache warmup phase before formal testing')
    parser.add_argument('--warmup_only', action='store_true', help='Only run the warmup phase for the dataset, then exit (no formal test, no markers)')
    args = parser.parse_args()
    main(args.parallel_threads, args.time_limit, args.ignore_time_start, args.ignore_time_end, args.data_path, args.output_path, args.log_file_name, args.port, args.encoding_size, args.cache_warmup, args.warmup_only)
