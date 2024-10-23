import os
import time
import json
import random
import hashlib
import asyncio
import logging
import argparse
import warnings

import numpy as np
import multiprocessing as mp

from typing import AsyncGenerator, List, Optional, Tuple

from utils import (
    Request,
    RequestFuncInput,
    RequestFuncOutput,
    IterMpQueue,
    get_tokenizer,
    api_url_standardize,
)
from benchmark_metrics import calculate_metrics, dump_metrics
from backend_request_func_async import ASYNC_REQUEST_FUNCS
from dataset_sampler import DATASET_SAMPLER


async def get_request(
    input_requests: IterMpQueue,
    request_rate: float,
) -> AsyncGenerator[Request, None]:
    input_requests = iter(input_requests)
    for request_id, request in enumerate(input_requests):
        yield request_id, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[Request],
    best_of: int,
    use_beam_search: bool,
    client_id: int,
    request_rate: float,
    num_requests: int,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger = logging.getLogger()
    logger.info(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request_id, request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = (
            request.prompt,
            request.input_len,
            request.output_len,
        )
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            client_id=client_id,
            request_id=request_id,
            num_requests=num_requests,
        )
        logger.info(
            f"Request {request_id:4d}, prompt_len={prompt_len:4d}, output_len={output_len:4d}, md5={hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]}"
        )
        tasks.append(
            asyncio.create_task(request_func(request_func_input=request_func_input))
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, _ = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
    )

    dump_metrics(metrics)


def main(args: argparse.Namespace):
    # unset http proxy
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""

    logger = logging.getLogger()
    logger.info(args)

    random.seed(0)
    np.random.seed(0)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    api_url = api_url_standardize(args.api_url, args.endpoint, args.backend)
    api_key = (
        args.api_key if args.api_key is not None else os.environ.get("AMSV2_API_KEY")
    )
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    # sample requests
    dataset_sample = DATASET_SAMPLER[args.dataset]
    sampled_requests: List[Request] = dataset_sample(
        dataset_path=args.dataset_path,
        num_requests=args.num_requests,
        num_turns=args.num_turns,
        tokenizer=tokenizer,
        system_prompt_path=args.system_prompt_path,
    )
    input_requests_queue = IterMpQueue(sampled_requests)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            client_id=0,
            request_rate=args.request_rate,
            num_requests=args.num_requests,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Path to the system prompt file. None for no system prompt.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file. Log file is shared by python and shell scripts, None for no python log.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level.",
    )

    args, _ = parser.parse_known_args()
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d %(message)s",
        level=args.log_level,
        filename=args.log_file,
    )
    main(args)
