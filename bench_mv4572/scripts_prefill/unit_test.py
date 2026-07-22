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

# Prefill benchmark: output length is small so each request is dominated by a
# full prefill; TTFT is the prefill latency and the throughput below is a prefill
# measurement. Never overridden per-dataset (unlike the decode bench).
#
# NOTE on OSL and MTP: with this server preset (MTP speculative decoding + PD
# separation + async scheduling) OSL=1 crashes the engine. A request with
# max_tokens=1 finishes in the same step its prefill completes, and the async
# scheduler then looks it up after it has already been freed -> KeyError in
# vllm/v1/core/sched/scheduler.py::_update_after_schedule, killing every
# EngineCore. Using OSL>=2 makes each request take one real decode step before
# finishing, which avoids the race. The prefill metrics are unaffected: TTFT and
# completion_ts are measured from the *first* token, so the extra decode token
# does not change what we report. Override with PREFILL_BENCH_OSL if needed.
OUTPUT_LEN = int(os.environ.get("PREFILL_BENCH_OSL", "2"))

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
        "model": "/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/",
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
            # [Robustness] Reset per request: if the POST fails (non-200) the SSE
            # loop below never assigns response_json; without this reset a stale
            # dict from the previous iteration would be reused with the wrong data.
            response_json = None
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

            # [OSL=1 + MTP] A request can legitimately finish without any
            # content-bearing token we score: GLM-5.2 is a reasoning model, so at
            # max_tokens=1 the single emitted token may be a special marker (e.g.
            # a <think> tag) that the reasoning parser drops from both
            # `content` and `reasoning`, leaving first_token_latency None. This is
            # a rare per-request edge case, NOT a fatal condition -- previously we
            # sys.exit(1) here, which killed this load-generator worker and
            # silently thinned concurrency for the rest of the scenario (skewing
            # prefill_throughput). Warn and fall through: the [Not Generated] path
            # below records it and the while-loop issues the next request.
            if first_token_latency is None and not is_error_and_handled:
                print(f"[WARN] {log_prefix}: request produced no scored token "
                      f"(OSL=1/MTP edge case) -- skipping this request, worker "
                      f"stays alive.", file=sys.stderr)

            # print(f"[Time] {test_start_time} {official_start_time} : {test_end_time} {official_end_time}")
            # print(f"[Time Diff] {test_start_time - official_start_time} {test_end_time - official_start_time} {official_end_time - official_start_time}")
            # [Robustness] A non-200 response (e.g. 404 wrong model, 500) leaves
            # response_json unset -> do NOT crash this worker (that silently zeros
            # the scenario, as the GLM-5.2 model-name bug did). Warn and continue.
            if response_json is None:
                print(f"[WARN] {log_prefix}: request failed (HTTP "
                      f"{response.status_code}) -- skipping, worker stays alive.",
                      file=sys.stderr)
                continue
            response_json['input_text'] = input_text

            # Combine reasoning content and generated content
            final_content = generated_text_aggregate
            if 'reasoning_aggregate' in locals() and reasoning_aggregate:
                final_content = f"<reasoning>{reasoning_aggregate}</reasoning>{generated_text_aggregate}"

            response_json['generated'] = final_content
            response_json['calculated_tokens'] = total_tokens

            # [Prefill] Read the prompt (input) token count from the streamed
            # usage chunk. This is the number of tokens prefilled for this
            # request; TTFT is the wall time to prefill them + emit token #1.
            if 'usage' in response_json and response_json['usage'] is not None:
                response_json['usage_tokens'] = response_json['usage'].get('completion_tokens', 0)
                response_json['prompt_tokens'] = response_json['usage'].get('prompt_tokens', 0)
                response_json['token_difference'] = total_tokens - response_json['usage_tokens']
            else:
                response_json['usage_tokens'] = None
                response_json['prompt_tokens'] = None
                response_json['token_difference'] = None

            prompt_tokens = response_json['prompt_tokens']
            if first_token_latency is None:
                log_queue.put((log_file_name, f'[Not Generated] {json.dumps(response_json, ensure_ascii=False)}\n'))

            elif not is_error_and_handled and test_start_time >= official_start_time and test_end_time <= official_end_time:
                # [Prefill] Per-request prefill throughput = input tokens / TTFT.
                # Under sustained concurrency TTFT includes queue wait, so this
                # is a lower bound; the system-level prefill_throughput in the
                # [Summary] (total input tokens / window span) is the headline
                # number. -1 marks a request we could not compute it for.
                if prompt_tokens and first_token_latency and first_token_latency > 0:
                    prefill_tps = prompt_tokens / first_token_latency
                else:
                    prefill_tps = -1.0

                # [Prefill] first_content_ts is the absolute time this request's
                # first output token arrived (== prefill completion). test_start_time
                # is when the request was sent. We log both so the summary can
                # measure the window as (last completion) - (first send) -- see the
                # window_span note in the aggregation below.
                completion_ts = first_content_ts if first_content_ts is not None else test_end_time
                send_ts = test_start_time

                response_json['prefill_tps'] = prefill_tps
                response_json['send_ts'] = send_ts
                response_json['completion_ts'] = completion_ts
                response_json['token_timestamps'] = token_timestamps
                log_queue.put(('output.jsonl', json.dumps(response_json, ensure_ascii=False) + '\n'))
                # [Test] <threads> <enc> <prompt_tokens> <ttft_s> <prefill_tps> <send_ts> <completion_ts>
                log_queue.put((log_file_name, f'[Test] {log_prefix} {prompt_tokens} {first_token_latency} {prefill_tps:.4f} {send_ts:.6f} {completion_ts:.6f}\n'))

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


        
def main(parallel_threads: int, time_limit: float, ignore_time_start: float, ignore_time_end: float, data_path: str, output_path: str, log_file_name: str, port: int, encoding_size: int):
    # Prefill benchmark: OUTPUT_LEN is small and identical for every dataset.
    # TTFT/completion_ts are measured from the first token, so the metric stays a
    # prefill measurement regardless of the exact OSL (see OUTPUT_LEN note above).
    print(f"[Config] OUTPUT_LEN (max_tokens) = {OUTPUT_LEN} (prefill bench; "
          f"override via PREFILL_BENCH_OSL)")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data = load_data(data_path)

    # No warmup: this is a prefill benchmark with prefix caching disabled in the
    # preset, so every request must recompute its prompt. Warming would only
    # populate a cache we deliberately turned off, so we go straight to the
    # formal measurement window.

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

    # [Prefill] Aggregate after the log writer has flushed. Read only the
    # per-request [Test] lines that match this run (prefix threads+enc); failure
    # lines have non-numeric fields and are skipped by the regex.
    #   [Test] <threads> <enc> <prompt_tokens> <ttft_s> <prefill_tps> <send_ts> <completion_ts>
    log_path = os.path.join(output_path, log_file_name)
    test_pattern = re.compile(
        r'^\[Test\] '
        r'(?P<threads>\d+) (?P<enc>\d+) '
        r'(?P<ptoks>\d+) (?P<ttft>\S+) (?P<ptps>-?\d+(?:\.\d+)?) '
        r'(?P<sts>\d+(?:\.\d+)?) (?P<cts>\d+(?:\.\d+)?)\s*$'
    )
    ttfts = []            # prefill latency (s) per request
    prompt_toks = []      # input tokens per request
    prefill_tpss = []     # per-request prefill throughput (input toks/s)
    send_tss = []         # absolute time each request was sent
    completion_tss = []   # absolute time each request's first token arrived
    if os.path.exists(log_path):
        with open(log_path, 'r') as fh:
            for line in fh:
                m = test_pattern.match(line.strip())
                if not m:
                    continue
                if int(m.group('threads')) != parallel_threads or int(m.group('enc')) != encoding_size:
                    continue
                try:
                    ttfts.append(float(m.group('ttft')))
                    prompt_toks.append(int(m.group('ptoks')))
                    send_tss.append(float(m.group('sts')))
                    completion_tss.append(float(m.group('cts')))
                    ptps = float(m.group('ptps'))
                    if ptps >= 0:
                        prefill_tpss.append(ptps)
                except ValueError:
                    pass

    # [Prefill] System prefill throughput = total input tokens processed in the
    # window / the window span. The span is measured from when the FIRST recorded
    # request was sent to when the LAST recorded request received its first token:
    #     window_span = max(completion_ts) - min(send_ts)
    # Every recorded request's whole prefill (from its send to its first token)
    # lies inside this span, so total_prompt_tokens / window_span is a faithful
    # system input-token rate. (Using max-min of completion_ts alone would drop
    # the first request's own prefill time from the denominator and inflate the
    # number.) The per-request mean_prefill_tps is the mean of input_tokens/TTFT
    # and reads lower under queue wait.
    total_prompt_tokens = sum(prompt_toks)
    window_span_s = ((max(completion_tss) - min(send_tss))
                     if (completion_tss and send_tss) else 0.0)
    prefill_throughput = (round(total_prompt_tokens / window_span_s, 4)
                          if window_span_s > 0 else None)

    summary = {
        "parallel_threads": parallel_threads,
        "encoding_size": encoding_size,
        "requests": len(ttfts),
        "mean_ttft_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "p90_ttft_s": round(_percentile(ttfts, 90), 4) if ttfts else None,
        "p99_ttft_s": round(_percentile(ttfts, 99), 4) if ttfts else None,
        "mean_prompt_tokens": round(statistics.mean(prompt_toks), 2) if prompt_toks else None,
        "total_prompt_tokens": total_prompt_tokens,
        "window_span_s": round(window_span_s, 4) if window_span_s else None,
        "mean_prefill_tps": round(statistics.mean(prefill_tpss), 4) if prefill_tpss else None,
        "prefill_throughput": prefill_throughput,
    }
    summary_line = "[Summary] " + json.dumps(summary)
    print(summary_line, flush=True)
    with open(log_path, 'a+') as fh:
        fh.write(summary_line + "\n")


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
    args = parser.parse_args()
    main(args.parallel_threads, args.time_limit, args.ignore_time_start, args.ignore_time_end, args.data_path, args.output_path, args.log_file_name, args.port, args.encoding_size)
