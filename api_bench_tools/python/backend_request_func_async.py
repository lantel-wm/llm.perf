import os
import sys
import json
import time
import aiohttp
import asyncio
import logging
import traceback

from typing import List, Optional, Union
from dataclasses import dataclass, field
from utils import RequestFuncInput, RequestFuncOutput


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


# curl -X POST http://10.198.31.25:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/mnt/llm2/llm_perf/hf_models/llama-7b-hf", "prompt": "Once upon a time", "temperature": 0.0, "best_of": 1, "max_tokens": 10, "min_tokens": 10, "stream": true, "ignore_eos": true}'
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    logger = logging.getLogger()
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/completions"
    ), "OpenAI Completions API URL must end with 'v1/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
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

        logger.debug(
            f"async_request_openai_completions: payload={payload}, headers={headers}"
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
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                            logger.debug(
                                f"async_request_openai_completions: [DONE] latency={latency}s"
                            )
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
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
                    output.error = f"HTTP Status Code: {response.status_code}\nresponse.reason: {response.reason}"
                    logger.warning(
                        f"client {request_func_input.client_id} request {request_func_input.request_id} failed: {output.error}"
                    )
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = str.removeprefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


async def async_request_ppl_completions_old(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc

    api_url = request_func_input.api_url
    channel = grpc.aio.insecure_channel(api_url)
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
        async for response in stub.Generation(batched_request):
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
        output.latency = time.perf_counter() - st

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    await channel.close()
    return output


async def async_request_ppl_completions(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    import grpc
    from ppl_server_utils import llm_pb2, llm_pb2_grpc

    logger = logging.getLogger()
    api_url = request_func_input.api_url
    channel = grpc.aio.insecure_channel(api_url)
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
        f"async_request_ppl_completions: id {client_id * num_requests + request_id} prompt {request.prompt}"
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
        async for response in stub.Generation(batched_request):
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
                            f"async_request_ppl_completions: [DONE] latency={latency}s"
                        )
                        break
        output.generated_text = generated_text
        output.output_len = output_len
        output.latency = latency
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    await channel.close()
    return output


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "ppl": async_request_ppl_completions,
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

    request_func = async_request_ppl_completions
    output = asyncio.run(request_func(request_func_input))
    print(f"output.success: {output.success}")
    print(f"output.generated_text: {output.generated_text}")
    print(f"output.prompt_len: {output.prompt_len}")
    print(f"output.output_len: {output.output_len}")
    print(f"output.latency: {output.latency}")
    print(f"output.ttft: {output.ttft}")
    print(f"output.error: {output.error}")
