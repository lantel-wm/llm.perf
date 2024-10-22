import os
import time
import json
import queue
import random
import hashlib
import logging
import warnings
import argparse
import threading

import numpy as np
import multiprocessing as mp

from datetime import datetime
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import AsyncGenerator, List, Optional, Tuple

from dataset_sample import DATASET_SAMPLE, Request
from backend_request_func import (
    REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
    get_tokenizer,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    successful_rate: float
    total_input: int
    total_output: int
    mean_input_tokens: float
    mean_output_tokens: float
    max_input_tokens: int
    max_output_tokens: int
    request_throughput: float
    in_out_throughput: float
    output_throughput: float

    min_ttft_ms: float
    max_ttft_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float

    min_tpot_ms: float
    max_tpot_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float

    min_e2e_ms: float
    max_e2e_ms: float
    mean_e2e_ms: float
    median_e2e_ms: float
    std_e2e_ms: float
    p90_e2e_ms: float
    p99_e2e_ms: float

    min_itl_ms: float
    max_itl_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float


class IterQueue(queue.Queue):
    def __init__(self, request_list: List[Request]):
        super(IterQueue, self).__init__()
        self.queue = queue.deque(request_list)

    def __iter__(self):
        while True:
            try:
                yield self.get_nowait()
            except queue.Empty:
                break


class IterMpQueue:
    def __init__(self, request_list: List[Request]):
        # Create an internal instance of mp.Queue
        self.mp_queue = mp.Queue()
        # Pre-fill the multiprocessing queue if necessary
        for request in request_list:
            self.mp_queue.put(request)

    def __iter__(self):
        while True:
            try:
                yield self.mp_queue.get_nowait()
            except queue.Empty:
                break

    def empty(self) -> bool:
        return self.mp_queue.empty()

    def drain(self):
        while not self.mp_queue.empty():
            self.mp_queue.get()


class BenchThread(threading.Thread):
    """
    Thread class for benchmarking.
    Each thread simulates a client sending requests to the server.
    """

    def __init__(
        self,
        client_id: int,
        ramp_up_time: float,
        stop_time: float,
        backend: str,
        api_url: str,
        api_key: str,
        model_id: str,
        tokenizer: str,
        best_of: int,
        use_beam_search: bool,
        num_requests: int,
        input_requests: List[Request] | IterQueue,
        out_queue: queue.Queue,
    ):
        super(BenchThread, self).__init__(daemon=True)
        self.client_id = client_id
        self.ramp_up_time = ramp_up_time
        self.stop_time = stop_time
        self.backend = backend
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.num_requests = num_requests
        self.input_requests = input_requests
        self.logger = logging.getLogger(f"thread_{client_id}")
        self.out_queue = out_queue

    def run(self):
        time.sleep(self.ramp_up_time)
        outputs = benchmark(
            backend=self.backend,
            api_url=self.api_url,
            model_id=self.model_id,
            input_requests=self.input_requests,
            stop_time=self.stop_time - self.ramp_up_time,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            api_key=self.api_key,
            client_id=self.client_id,
            num_requests=self.num_requests,
            logger=self.logger,
        )
        self.out_queue.put(outputs)


class BenchProcess(mp.Process):
    """
    Process class for benchmarking.
    Each process simulates a client sending requests to the server.
    """

    def __init__(
        self,
        client_id: int,
        stop_time: float,
        ramp_up_time: float,
        backend: str,
        api_url: str,
        api_key: str,
        model_id: str,
        tokenizer: str,
        best_of: int,
        use_beam_search: bool,
        num_requests: int,
        input_requests: List[Request] | IterMpQueue,
        out_queue: mp.Queue,
    ):
        super(BenchProcess, self).__init__(daemon=True)
        self.client_id = client_id
        self.stop_time = stop_time
        self.ramp_up_time = ramp_up_time
        self.backend = backend
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.input_requests = input_requests
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.num_requests = num_requests
        self.logger = logging.getLogger(f"process_{client_id}")
        self.out_queue = out_queue

    def run(self):
        time.sleep(self.ramp_up_time)
        outputs = benchmark(
            backend=self.backend,
            api_url=self.api_url,
            api_key=self.api_key,
            model_id=self.model_id,
            input_requests=self.input_requests,
            stop_time=self.stop_time - self.ramp_up_time,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            client_id=self.client_id,
            num_requests=self.num_requests,
            logger=self.logger,
        )
        self.out_queue.put(outputs)


def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[Request],
    stop_time: float,
    best_of: int,
    use_beam_search: bool,
    api_key: Optional[str] = None,
    client_id: int = -1,
    num_requests: int = -1,
    logger: Optional[logging.Logger] = None,
) -> List[RequestFuncOutput]:
    """Benchmark main function, called from each client.

    Args:
        backend (str): Backend inference framework. Could be "vllm", "lightllm", "sglang" or "ppl". Defaults to "vllm".
        api_url (str): api url of the inference server.
        model_id (str): model id of the inference server, used by vllm.
        input_requests (List[Tuple[str, int, int]]): List of (prompt, prompt_len, output_len) tuples.
        best_of (int): Sampling param.
        use_beam_search (bool): Sampling param.
        api_key (Optional[str], optional): api key. Defaults to None.
        client_id (int, optional): thread id. Defaults to -1.
        num_requests (int, optional): number of requests to send. Defaults to -1.
        logger (Optional[Logger], optional): logger. Defaults to None.

    Raises:
        ValueError: Unknown backend.

    Returns:
        List[RequestFuncOutput]: List of outputs.
    """
    if backend in REQUEST_FUNCS:
        request_func = REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info(
        f"Starting benchmark for backend: {backend}, model_id: {model_id}, client_id: {client_id}, num_requests: {num_requests}, stop_time: {stop_time}"
    )

    benchmark_start_time = time.perf_counter()
    outputs = []
    for request_id, request in enumerate(input_requests):
        if stop_time > 0 and time.perf_counter() - benchmark_start_time >= stop_time:
            logger.info(f"Stop time reached, stopping benchmark for client {client_id}")
            break

        prompt, prompt_len, output_len = (
            request.prompt,
            request.input_len,
            request.output_len,
        )
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            api_key=api_key,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            client_id=client_id,
            request_id=request_id,
            num_requests=num_requests,
        )
        logger.info(
            f"Request {request_id:4d} of client {client_id:4d}: prompt_len={prompt_len:4d}, output_len={output_len:4d}, md5={hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]}"
        )
        outputs.append(request_func(request_func_input=request_func_input))

    return outputs


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> Tuple[BenchmarkMetrics, List[int]]:
    """Calculate benchmark metrics.

    Args:
        outputs (List[RequestFuncOutput]): List of outputs, each element is a RequestFuncOutput
        dur_s (float): Duration of the benchmark in seconds
        tokenizer (PreTrainedTokenizerBase): transformers tokenizer

    Returns:
        Tuple[BenchmarkMetrics, List[int]]: BenchmarkMetrics and List of actual output lengths
    """
    actual_output_lens = []
    total_input_tokens = 0
    max_input_tokens = 0
    max_output_tokens = 0
    completed = 0

    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2es: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            input_len = outputs[i].prompt_len
            output_len = outputs[i].output_len
            # client_id = outputs[i].client_id
            # request_id = outputs[i].request_id

            total_input_tokens += input_len
            actual_output_lens.append(output_len)
            max_input_tokens = max(max_input_tokens, input_len)
            max_output_tokens = max(max_output_tokens, output_len)

            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))

            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2es.append(outputs[i].latency)
            completed += 1

        else:
            actual_output_lens.append(0)

    total_output_tokens = sum(actual_output_lens)

    metrics = BenchmarkMetrics(
        completed=completed,
        successful_rate=completed / len(outputs),
        total_input=total_input_tokens,
        total_output=total_output_tokens,
        mean_input_tokens=total_input_tokens / completed,
        mean_output_tokens=total_output_tokens / completed,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        request_throughput=completed / dur_s,
        in_out_throughput=(total_input_tokens + total_output_tokens) / dur_s,
        output_throughput=total_output_tokens / dur_s,
        # ttfts is empty if streaming is not supported by backend
        min_ttft_ms=np.min(ttfts or 0) * 1000,
        max_ttft_ms=np.max(ttfts or 0) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        min_tpot_ms=np.min(tpots or 0) * 1000,
        max_tpot_ms=np.max(tpots or 0) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        min_e2e_ms=np.min(e2es or 0) * 1000,
        max_e2e_ms=np.max(e2es or 0) * 1000,
        mean_e2e_ms=np.mean(e2es or 0) * 1000,
        median_e2e_ms=np.median(e2es or 0) * 1000,
        std_e2e_ms=np.std(e2es or 0) * 1000,
        p90_e2e_ms=np.percentile(e2es or 0, 90) * 1000,
        p99_e2e_ms=np.percentile(e2es or 0, 99) * 1000,
        min_itl_ms=np.min(itls or 0) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


def dump_metrics_and_results(
    metrics: BenchmarkMetrics,
):
    """Dumps metrics and results to stdout in CSV format.

    Args:
        metrics (BenchmarkMetrics): Metrics to dump.
    """
    print(
        "CSV header output:\
completed,success_rate,qps,total_inlen,total_outlen,avg_inlen,avg_outlen,max_inlen,max_outlen,o_tps,io_tps,\
min_ttft,max_ttft,mean_ttft,median_ttft,std_ttft,p90_ttft,p99_ttft,\
min_tpot,max_tpot,mean_tpot,median_tpot,std_tpot,p90_tpot,p99_tpot,\
min_e2e,max_e2e,mean_e2e,median_e2e,std_e2e,p90_e2e,p99_e2e,\
min_itl,max_itl,mean_itl,median_itl,std_itl,p90_itl,p99_itl"
    )

    csv_line = ""
    csv_line += f"{metrics.completed},"
    csv_line += f"{metrics.successful_rate:.3f},"
    csv_line += f"{metrics.request_throughput:.3f},"
    csv_line += f"{metrics.total_input},"
    csv_line += f"{metrics.total_output},"
    csv_line += f"{metrics.mean_input_tokens:.3f},"
    csv_line += f"{metrics.mean_output_tokens:.3f},"
    csv_line += f"{metrics.max_input_tokens},"
    csv_line += f"{metrics.max_output_tokens},"
    csv_line += f"{metrics.output_throughput:.3f},"
    csv_line += f"{metrics.in_out_throughput:.3f},"

    csv_line += f"{metrics.min_ttft_ms:.3f},"
    csv_line += f"{metrics.max_ttft_ms:.3f},"
    csv_line += f"{metrics.mean_ttft_ms:.3f},"
    csv_line += f"{metrics.median_ttft_ms:.3f},"
    csv_line += f"{metrics.std_ttft_ms:.3f},"
    csv_line += f"{metrics.p90_ttft_ms:.3f},"
    csv_line += f"{metrics.p99_ttft_ms:.3f},"

    csv_line += f"{metrics.min_tpot_ms:.3f},"
    csv_line += f"{metrics.max_tpot_ms:.3f},"
    csv_line += f"{metrics.mean_tpot_ms:.3f},"
    csv_line += f"{metrics.median_tpot_ms:.3f},"
    csv_line += f"{metrics.std_tpot_ms:.3f},"
    csv_line += f"{metrics.p90_tpot_ms:.3f},"
    csv_line += f"{metrics.p99_tpot_ms:.3f},"

    csv_line += f"{metrics.min_e2e_ms:.3f},"
    csv_line += f"{metrics.max_e2e_ms:.3f},"
    csv_line += f"{metrics.mean_e2e_ms:.3f},"
    csv_line += f"{metrics.median_e2e_ms:.3f},"
    csv_line += f"{metrics.std_e2e_ms:.3f},"
    csv_line += f"{metrics.p90_e2e_ms:.3f},"
    csv_line += f"{metrics.p99_e2e_ms:.3f},"

    csv_line += f"{metrics.min_itl_ms:.3f},"
    csv_line += f"{metrics.max_itl_ms:.3f},"
    csv_line += f"{metrics.mean_itl_ms:.3f},"
    csv_line += f"{metrics.median_itl_ms:.3f},"
    csv_line += f"{metrics.std_itl_ms:.3f},"
    csv_line += f"{metrics.p90_itl_ms:.3f},"
    csv_line += f"{metrics.p99_itl_ms:.3f}"

    print(f"CSV format output:{csv_line}")


def api_url_standardize(base_url: str, endpoint: str, backend: str) -> str:
    """standardize api url

    Args:
        api_url (str): api url

    Returns:
        str: standardized api url
    """
    match backend:
        case "vllm" | "openai":
            api_url = f"{base_url}{endpoint}"
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
        case "ppl":
            api_url = base_url
        case "trt":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/v2/models/ensemble/generate_stream"):
                api_url = f"{api_url}/v2/models/ensemble/generate_stream"
        case "amsv2":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/text-generation/generate_stream"):
                api_url = f"{api_url}/text-generation/generate_stream"
        case "sglang":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/generate"):
                api_url = f"{api_url}/generate"
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"using {backend} backend with api url: {api_url}")
    return api_url


def get_input_requests(
    sampled_requests: List[Request],
    args: argparse.Namespace,
) -> List[List[Request]] | IterQueue | IterMpQueue:
    """get input requests according to execute mode and allow_repetitive_requests

    Args:
        sampled_requests (List[Request]): sampled requests
        args (argparse.Namespace): arguments

    Raises:
        ValueError: Unknown execute mode

    Returns:
        List[List[Request]] | IterQueue[Request] | IterMpQueue[Request]: input requests
    """
    if args.allow_repetitive_requests:
        # repetitive sampling
        input_requests = []
        for client_id in range(args.num_clients):
            input_requests_i = (
                sampled_requests[client_id:] + sampled_requests[:client_id]
            )
            if client_id % 2 == 1:
                input_requests_i = input_requests_i[::-1]
            input_requests.append(input_requests_i)
    else:
        # non-repetitive sampling
        if args.execute_mode == "Thread":
            input_requests = IterQueue(sampled_requests)
        elif args.execute_mode == "Process":
            input_requests = IterMpQueue(sampled_requests)
        else:
            raise ValueError(f"Unknown execute mode: {args.execute_mode}")

    return input_requests


def main(args: argparse.Namespace):
    # unset proxy
    os.environ["http_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["https_proxy"] = ""

    logger.info(args)
    assert args.num_requests > 0, "Number of threads must be greater than 0."

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    api_url = api_url_standardize(args.base_url, args.endpoint, args.backend)
    api_key = (
        args.api_key if args.api_key is not None else os.environ.get("AMSV2_API_KEY")
    )
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    # sample requests
    dataset_sample = DATASET_SAMPLE[args.dataset]
    sampled_requests: List[Request] = dataset_sample(
        dataset_path=args.dataset_path,
        num_requests=args.num_requests,
        num_turns=args.num_turns,
        tokenizer=tokenizer,
        system_prompt_path=args.system_prompt_path,
    )

    input_requests = get_input_requests(sampled_requests, args)
    # start benchmark
    benchmark_start_time = time.perf_counter()
    clients = []
    out_queue = queue.Queue() if args.execute_mode == "Thread" else mp.Queue()
    logger.info(f"type(out_queue): {type(out_queue)}")
    for client_id in range(args.num_clients):
        if args.allow_repetitive_requests:
            # input_requests: List[List[Request]], input_requests_i: List[Request]
            input_requests_i = input_requests[client_id]
        else:
            # input_requests: IterQueue[Request] or IterMpQueue[Request]
            input_requests_i = input_requests

        if args.execute_mode == "Thread":
            client = BenchThread(
                client_id=client_id,
                ramp_up_time=client_id * args.ramp_up_time / args.num_clients,
                stop_time=args.stop_time,
                backend=backend,
                api_url=api_url,
                api_key=api_key,
                model_id=model_id,
                tokenizer=tokenizer,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                num_requests=args.num_requests,
                input_requests=input_requests_i,
                out_queue=out_queue,
            )
        elif args.execute_mode == "Process":
            client = BenchProcess(
                client_id=client_id,
                ramp_up_time=client_id * args.ramp_up_time / args.num_clients,
                stop_time=args.stop_time,
                backend=backend,
                api_url=api_url,
                api_key=api_key,
                model_id=model_id,
                tokenizer=tokenizer,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                num_requests=args.num_requests,
                input_requests=input_requests_i,
                out_queue=out_queue,
            )

        client.start()
        clients.append(client)
        logger.info(
            f"client {client_id} launched with ramp up time {client_id * args.ramp_up_time / args.num_clients}"
        )

    logger.info(f"out_queue.empty(): {out_queue.empty()}")
    if not args.allow_repetitive_requests:
        logger.info(f"input_requests.empty(): {input_requests.empty()}")

    all_outputs = []
    for client in clients:
        all_outputs += out_queue.get()

    if not args.allow_repetitive_requests and not input_requests.empty():
        input_requests.drain()

    for client in clients:
        client.join()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    logger.info(f"benchmark duration: {benchmark_duration:.2f}s")
    logger.info(f"output len: {len(all_outputs)}")
    metrics, _ = calculate_metrics(outputs=all_outputs, dur_s=benchmark_duration)
    logger.info(f"metrics calculated")
    dump_metrics_and_results(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        required=True,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key if not using http basic auth.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sharegpt",
        choices=list(DATASET_SAMPLE.keys()),
        help="Dataset type.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--allow-repetitive-requests",
        action="store_true",
        default=False,
        help="Allow repetitive requests sent to the server, see README.md for more details.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",
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
        "--num-clients",
        type=int,
        default=1,
        help="Number of clients to use for the benchmark.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=1,
        help="Number of chat turns to use for the benchmark. A prompt and a completion are considered as one turn.",
    )
    parser.add_argument(
        "--ramp-up-time",
        type=float,
        default=1,
        help="Ramp up time in seconds for each thread.",
    )
    parser.add_argument(
        "--stop-time",
        type=float,
        default=0,
        help="Stop time in seconds for each thread.",
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
    parser.add_argument(
        "--execute-mode",
        type=str,
        default="Thread",
        choices=["Thread", "Process"],
        help="Excute clients in multi-thread or multi-process.",
    )

    args, _ = parser.parse_known_args()
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d %(message)s",
        level=args.log_level,
        filename=args.log_file,
    )
    logger = logging.getLogger()
    main(args)
