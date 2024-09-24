import json
import os
import sys
import time
import logging
import traceback
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

HTTP_TIMEOUT = 6 * 60 * 60

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    api_key: Optional[str]
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    thread_id: Optional[int] = None
    request_id: int = 0
    num_requests: int = 1


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""
    thread_id: Optional[int] = None
    request_id: int = 0

# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

# curl -X POST http://10.198.31.25:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/mnt/llm2/llm_perf/hf_models/llama-7b-hf", "prompt": "Once upon a time", "temperature": 0.0, "best_of": 1, "max_tokens": 10, "min_tokens": 10, "stream": true, "ignore_eos": true}'
def request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/completions"
    ), "OpenAI Completions API URL must end with 'v1/completions'."

    assert not request_func_input.use_beam_search
    payload = {
        "model": request_func_input.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.0,
        "best_of": request_func_input.best_of,
        "max_tokens": request_func_input.output_len,
        "min_tokens": request_func_input.output_len,
        "stream": True,
        "ignore_eos": True,
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    
    # curl -X POST 10.198.31.25:8000/v1/completions -H "Authorization: Bearer YOUR_API_KEY" -d '{"model": "/mnt/llm2/llm_perf/hf_models/llama-7b-hf", "prompt": "Once upon a time", "temperature": 0.0, "best_of": 1, "max_tokens": 100, "min_tokens": 100, "stream": true, "ignore_eos": true}'

    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id, 
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(url=api_url, json=payload, 
                           headers=headers, stream=True,
                           timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    
                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                    else:
                        data = json.loads(chunk)
                        
                        if data["choices"][0]["text"]:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                                
                            # Decoding phase
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # do not want to include as inter-token-latency
                            elif data.get("usage", None) is None:
                                output.itl.append(timestamp -
                                                    most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += data["choices"][0]["text"]
                            output_len += 1

                output.generated_text = generated_text
                output.output_len = output_len
                output.success = True
                output.latency = latency
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.text: {response.text}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output

def request_ppl_completions_old(request_func_input: RequestFuncInput) -> RequestFuncOutput:
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc
    
    api_url = request_func_input.api_url
    channel = grpc.insecure_channel(api_url)
    stub = llm_pb2_grpc.LLMServiceStub(channel)
    
    thread_id = request_func_input.thread_id
    request_id = request_func_input.request_id
    num_requests = request_func_input.num_requests
    
    request = llm_pb2.Request(
        id=thread_id * num_requests + request_id,
        prompt=request_func_input.prompt,
        temperature=0.0,
        stopping_parameters=llm_pb2.StoppingCriteriaParameters(
            max_new_tokens=request_func_input.output_len,
            ignore_eos_token=True
        )
    )
    batched_request = llm_pb2.BatchedRequest(req=[request])
    
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id, 
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )
    
    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    
    try:
        response_stream = stub.Generation(batched_request)
        for response in response_stream:
            for rsp in response.rsp:
                if rsp.status == llm_pb2.Status.FAILED:
                    logging.warning(f"Request {request.id} failed")
                    output.success = False
                    output.error = "Response Status: FAILED"
                    break
                
                else:
                    if rsp.generated:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)
                        
                        most_recent_timestamp = timestamp
                        generated_text += rsp.generated
                        output_len += 1
                        
                    if rsp.status == llm_pb2.Status.FINISHED:
                        logging.info(f"Request {request.id} finished")
                        latency = time.perf_counter() - st
                        output.success = True
                        break
        
        output.generated_text = generated_text
        output.output_len = output_len
        output.latency = latency
                       
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        
    return output

def request_ppl_completions(request_func_input: RequestFuncInput) -> RequestFuncOutput:
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc
    
    api_url = request_func_input.api_url
    channel = grpc.insecure_channel(api_url)
    stub = llm_pb2_grpc.LLMServiceStub(channel)
    
    thread_id = request_func_input.thread_id
    request_id = request_func_input.request_id
    num_requests = request_func_input.num_requests
    
    choose_NextToken_parameters=llm_pb2.NextTokenChooserParameters(
        temperature=0.8,
        top_k=40,
        top_p=0.7,
        do_sample=True,
        repetition_penalty=1.05
    )
    
    request = llm_pb2.Request(
        id=thread_id * num_requests + request_id,
        prompt=request_func_input.prompt,
        choosing_parameters=choose_NextToken_parameters,
        stopping_parameters=llm_pb2.StoppingCriteriaParameters(
            max_new_tokens=request_func_input.output_len,
            ignore_eos_token=True
        )
    )
    batched_request = llm_pb2.BatchedRequest(req=[request])
    
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id, 
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )
    
    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    
    try:
        response_stream = stub.Generation(batched_request)
        for response in response_stream:
            for rsp in response.rsp:
                if rsp.status == llm_pb2.Status.FAILED:
                    logging.warning(f"Request {request.id} failed")
                    output.success = False
                    output.error = "Response Status: FAILED"
                    break
                
                else:
                    if rsp.generated:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)
                        
                        most_recent_timestamp = timestamp
                        generated_text += rsp.generated
                        output_len += 1
                        
                    if rsp.status == llm_pb2.Status.FINISHED:
                        logging.info(f"Request {request.id} finished")
                        latency = time.perf_counter() - st
                        output.success = True
                        break
        
        output.generated_text = generated_text
        output.output_len = output_len
        output.latency = latency
                       
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        
    return output

# curl -X POST 127.0.0.1:8000/v2/models/ensemble/generate_stream -d \
#'{"text_input": "What is ML?", "max_tokens": 500, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": -1, "stream": true}'
def request_trtllm_generate_stream(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    assert not request_func_input.use_beam_search
    assert request_func_input.best_of == 1
    
    payload = {
        "text_input": request_func_input.prompt,
        "temperature": 0.0,
        "max_tokens": request_func_input.output_len,
        "bad_words": "",
        "stop_words": "",
        "pad_id": 2,
        "end_id": -1,
        "stream": True,
    }
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(url=api_url, json=payload, 
                           stream=True, timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    
                    data = json.loads(chunk)
                    generated_text += data["text_output"]
                    output_len += 1
                    timestamp = time.perf_counter()
                    # First token
                    if ttft == 0.0:
                        ttft = time.perf_counter() - st
                        output.ttft = ttft
                        
                    # Decoding phase
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)

                    most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.output_len = output_len
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


# curl http://127.0.0.1:8080/generate_stream -X POST -d '{"inputs":"What is AI?","parameters":{"max_new_tokens":17, "frequency_penalty":1}}' -H 'Content-Type: application/json'
def request_lightllm_generate_stream(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    assert not request_func_input.use_beam_search
    assert request_func_input.best_of == 1
    
    payload = {
        "inputs": request_func_input.prompt,
        "parameters": {
            "ignore_eos": True,
            "max_new_tokens": request_func_input.output_len,
            "min_new_tokens": request_func_input.output_len,
            "temperature": 0.0,
            "frequency_penalty": 1,
        }
    }
    headers = {
        "Content-Type": "application/json",
    }
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    # data:{"token": {"id": 29897, "text": ")", "logprob": -0.06751319020986557, "special": false, "count_output_tokens": 17, "prompt_tokens": 6}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}
    try:
        with requests.post(url=api_url, json=payload, headers=headers,
                           stream=True, timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                    
                    data = json.loads(chunk)
                    generated_text += data["token"]["text"]
                    timestamp = time.perf_counter()
                    # First token
                    if ttft == 0.0:
                        ttft = time.perf_counter() - st
                        output.ttft = ttft
                        
                    # Decoding phase
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)

                    most_recent_timestamp = timestamp
                    
                    if data["finished"]:
                        output_len = data["token"]["count_output_tokens"]

                output.generated_text = generated_text
                output.output_len = output_len
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


# curl --location 'https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719/text-generation/generate_stream' \
# --header 'Content-Type: application/json' \
# --header 'Authorization: Bearer eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw' \
# --data '{
#       "inputs": "Who are you?",
#       "parameters": {
#           "do_sample": true,
#           "max_new_tokens": 100,
#           "repetition_penalty": 1.1,
#           "temperature": 0.8,
#           "top_k": 50,
#           "top_p": 0.7
#       }
#   }'
def request_amsv2_generate_stream(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    api_key = request_func_input.api_key
    if api_key is None:
        api_key = 'eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw'
    
    payload = {
        "inputs": request_func_input.prompt,
        "parameters": {
            "do_sample": True,
            "max_new_tokens": request_func_input.output_len,
            "repetition_penalty": 1.1,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.7,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
        
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    # data:{"token": {"id": 29897, "text": ")", "logprob": -0.06751319020986557, "special": false, "count_output_tokens": 17, "prompt_tokens": 6}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}
    try:
        with requests.post(url=api_url, json=payload, headers=headers,
                           stream=True, timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                    
                    data = json.loads(chunk)
                    generated_text += data["generated_text"]
                    timestamp = time.perf_counter()
                    # First token
                    if ttft == 0.0:
                        ttft = time.perf_counter() - st
                        output.ttft = ttft
                        
                    # Decoding phase
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)

                    most_recent_timestamp = timestamp
                    
                    if data.get("details") is not None:
                        output_len = data["details"]["generated_tokens"]

                output.generated_text = generated_text
                output.output_len = output_len
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output

# curl http://localhost:30000/generate -H "Content-Type: application/json" -d '{"text": "Once upon a time,", "sampling_params": {"max_new_tokens": 16, "temperature": 0}, "stream": true}'
def request_sglang_generate(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "/generate"
    ), "SGLang Generate API URL must end with '/generate'."

    assert not request_func_input.use_beam_search
    payload = {
        "text": request_func_input.prompt,
        "sampling_params": {
            "max_new_tokens": request_func_input.output_len,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
    }
    
    output = RequestFuncOutput(
        thread_id=request_func_input.thread_id, 
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(url=api_url, json=payload, 
                           headers=headers, stream=True,
                           timeout=HTTP_TIMEOUT) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    
                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                    else:
                        data = json.loads(chunk)
                        
                        if data["meta_info"]["completion_tokens"] >= 1:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                                
                            # Decoding phase
                            elif data["meta_info"]["completion_tokens"] >= 2:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text = data["text"]
                            output_len += 1

                output.generated_text = generated_text
                output.output_len = output_len
                output.success = True
                output.latency = latency
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.text: {response.text}"

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         trust_remote_code=trust_remote_code)



REQUEST_FUNCS = {
    "vllm": request_openai_completions,
    "ppl": request_ppl_completions,
    "trtllm": request_trtllm_generate_stream,
    "lightllm": request_lightllm_generate_stream,
    "amsv2": request_amsv2_generate_stream,
    "sglang": request_sglang_generate,
}

if __name__ == '__main__':
    import os
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""
    request_func_input = RequestFuncInput(
        prompt="The future of ai is",
        api_url="127.0.0.1:23333",
        # api_url="http://10.198.31.25:8000/v1/completions",
        # api_url='https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719/text-generation/generate_stream',
        # api_key='eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw',
        prompt_len=150,
        output_len=300,
        # model="/mnt/llm2/llm_perf/hf_models/llama-7b-hf",
        thread_id=0,
        request_id=0,
        num_requests=1024,
    )
    
    # output = request_openai_completions(request_func_input)
    # output = request_amsv2_generate_stream(request_func_input)
    output = request_ppl_completions(request_func_input)
    output = request_sglang_generate(request_func_input)
    print(f"output.success: {output.success}")
    print(f"output.generated_text: {output.generated_text}")
    print(f"output.prompt_len: {output.prompt_len}")
    print(f"output.output_len: {output.output_len}")
    print(f"output.latency: {output.latency}")
    print(f"output.ttft: {output.ttft}")
    print(f"output.error: {output.error}")
    
    
