import os
import sys
import time
import json
import logging
import requests
import traceback

from utils import RequestFuncInput, RequestFuncOutput

HTTP_TIMEOUT = 6 * 60 * 60


# curl -X POST http://10.198.31.25:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/mnt/llm2/llm_perf/hf_models/llama-7b-hf", "prompt": "Once upon a time", "temperature": 0.0, "best_of": 1, "max_tokens": 10, "min_tokens": 10, "stream": true, "ignore_eos": true}'
def request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    logger = logging.getLogger()
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
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

    logger.debug(f"request_openai_completions: payload={payload}, headers={headers}")

    # curl -X POST 10.198.31.25:8000/v1/completions -H "Authorization: Bearer YOUR_API_KEY" -d '{"model": "/mnt/llm2/llm_perf/hf_models/llama-7b-hf", "prompt": "Once upon a time", "temperature": 0.0, "best_of": 1, "max_tokens": 100, "min_tokens": 100, "stream": true, "ignore_eos": true}'

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(
            url=api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=HTTP_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")

                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                        logger.debug(
                            f"request_openai_completions: [DONE] latency={latency}s"
                        )
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
                                output.itl.append(timestamp - most_recent_timestamp)

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
                logger.warning(
                    f"client {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                )

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


def request_ppl_completions_old(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    logger = logging.getLogger()
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc

    api_url = request_func_input.api_url
    channel = grpc.insecure_channel(api_url)
    stub = llm_pb2_grpc.LLMServiceStub(channel)

    client_id = request_func_input.client_id
    request_id = request_func_input.request_id
    num_requests = request_func_input.num_requests

    request = llm_pb2.Request(
        id=client_id * num_requests + request_id,
        prompt=request_func_input.prompt,
        temperature=0.0,
        stopping_parameters=llm_pb2.StoppingCriteriaParameters(
            max_new_tokens=request_func_input.output_len, ignore_eos_token=True
        ),
    )
    batched_request = llm_pb2.BatchedRequest(req=[request])

    logger.debug(
        f"request_ppl_completions: id {client_id * num_requests + request_id} prompt {request.prompt}"
    )

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
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
                    logger.warning(
                        f"Request {request.id} (thread {client_id} request {request_id}) failed"
                    )
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
                        latency = time.perf_counter() - st
                        output.success = True
                        logger.debug(
                            f"request_ppl_completions: [DONE] latency={latency}s"
                        )
                        break

        output.generated_text = generated_text
        output.output_len = output_len
        output.latency = latency

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


def request_ppl_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc

    logger = logging.getLogger()
    api_url = request_func_input.api_url
    channel = grpc.insecure_channel(api_url)
    stub = llm_pb2_grpc.LLMServiceStub(channel)

    client_id = request_func_input.client_id
    request_id = request_func_input.request_id
    num_requests = request_func_input.num_requests

    choose_NextToken_parameters = llm_pb2.NextTokenChooserParameters(
        temperature=0.0, top_k=1, top_p=0.0, do_sample=False, repetition_penalty=1.0
    )

    request = llm_pb2.Request(
        id=client_id * num_requests + request_id,
        prompt=request_func_input.prompt,
        choosing_parameters=choose_NextToken_parameters,
        stopping_parameters=llm_pb2.StoppingCriteriaParameters(
            max_new_tokens=request_func_input.output_len, ignore_eos_token=True
        ),
    )
    batched_request = llm_pb2.BatchedRequest(req=[request])

    logger.debug(
        f"request_ppl_completions: id {client_id * num_requests + request_id} prompt {request.prompt}"
    )

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
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
                    logger.warning(
                        f"Request {request.id} (client {client_id} request {request_id}) failed"
                    )
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
                        latency = time.perf_counter() - st
                        output.success = True
                        logger.debug(
                            f"request_ppl_completions: [DONE] latency={latency}s"
                        )
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
    logger = logging.getLogger()
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
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    logger.debug(f"request_trtllm_generate_stream: payload={payload}")

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(
            url=api_url, json=payload, stream=True, timeout=HTTP_TIMEOUT
        ) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")

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
                logger.debug(
                    f"request_trtllm_generate_stream: [DONE] latency={output.latency}s"
                )
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"
                logger.warning(
                    f"thread {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                )

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


# curl http://127.0.0.1:8080/generate_stream -X POST -d '{"inputs":"What is AI?","parameters":{"max_new_tokens":17, "frequency_penalty":1}}' -H 'Content-Type: application/json'
def request_lightllm_generate_stream(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    logger = logging.getLogger()
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
        },
    }
    headers = {
        "Content-Type": "application/json",
    }

    logger.debug(
        f"request_lightllm_generate_stream: payload={payload}, headers={headers}"
    )

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    # data:{"token": {"id": 29897, "text": ")", "logprob": -0.06751319020986557, "special": false, "count_output_tokens": 17, "prompt_tokens": 6}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}
    try:
        with requests.post(
            url=api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=HTTP_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data:")

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
                logger.debug(
                    f"request_lightllm_generate_stream: [DONE] latency={output.latency}"
                )
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"
                logger.warning(
                    f"thread {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                )

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
    logger = logging.getLogger()
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    api_key = request_func_input.api_key
    if api_key is None:
        api_key = "eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw"

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

    logger.debug(f"request_amsv2_generate_stream: payload={payload}, headers={headers}")

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    # data:{"token": {"id": 29897, "text": ")", "logprob": -0.06751319020986557, "special": false, "count_output_tokens": 17, "prompt_tokens": 6}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}
    try:
        with requests.post(
            url=api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=HTTP_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data:")

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
                logger.debug(
                    f"request_amsv2_generate_stream: [DONE] latency={output.latency}"
                )
            else:
                output.success = False
                output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"
                logger.warning(
                    f"thread {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                )

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


# curl http://localhost:30000/generate -H "Content-Type: application/json" -d '{"text": "Once upon a time,", "sampling_params": {"max_new_tokens": 16, "temperature": 0}, "stream": true}'
def request_sglang_generate(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    logger = logging.getLogger()
    logger = logging.getLogger()
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

    logger.debug(f"request_sglang_generate: payload={payload}, headers={headers}")

    output = RequestFuncOutput(
        client_id=request_func_input.client_id,
        request_id=request_func_input.request_id,
        prompt_len=request_func_input.prompt_len,
    )

    generated_text = ""
    output_len = 0
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        with requests.post(
            url=api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=HTTP_TIMEOUT,
        ) as response:
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")

                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                        logger.debug(
                            f"request_sglang_generate: [DONE] latency={latency}s"
                        )
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
                logger.warning(
                    f"thread {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                )

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    return output


REQUEST_FUNCS = {
    "vllm": request_openai_completions,
    "ppl": request_ppl_completions,
    "trtllm": request_trtllm_generate_stream,
    "lightllm": request_lightllm_generate_stream,
    "amsv2": request_amsv2_generate_stream,
    "sglang": request_sglang_generate,
}

if __name__ == "__main__":
    import os

    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""

    request_func_input = RequestFuncInput(
        prompt="The future of ai is",
        api_url="127.0.0.1:33332",
        # api_url="http://10.198.31.25:8000/v1/completions",
        # api_url='https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719/text-generation/generate_stream',
        # api_key='eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw',
        prompt_len=150,
        output_len=300,
        # model="/mnt/llm2/llm_perf/hf_models/llama-7b-hf",
        client_id=0,
        request_id=0,
        num_requests=1024,
    )

    # output = request_openai_completions(request_func_input)
    # output = request_amsv2_generate_stream(request_func_input)
    output = request_ppl_completions(request_func_input)
    # output = request_sglang_generate(request_func_input)
    print(f"output.success: {output.success}")
    print(f"output.generated_text: {output.generated_text}")
    print(f"output.prompt_len: {output.prompt_len}")
    print(f"output.output_len: {output.output_len}")
    print(f"output.latency: {output.latency}")
    print(f"output.ttft: {output.ttft}")
    print(f"output.error: {output.error}")
