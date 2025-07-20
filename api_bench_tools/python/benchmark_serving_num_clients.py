import os
import time
import queue
import hashlib
import logging
import argparse
import threading

import multiprocessing as mp

from typing import List, Optional

from dataset_sampler import DATASET_SAMPLER
from backend_request_func import REQUEST_FUNCS
from benchmark_metrics import dump_metrics, calculate_metrics
from utils import (
    Request,
    RequestFuncInput,
    RequestFuncOutput,
    IterQueue,
    IterMpQueue,
    get_tokenizer,
    api_url_standardize,
)


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
        input_queue: List[Request] | IterQueue,
        output_queue: queue.Queue,
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
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        time.sleep(self.ramp_up_time)
        outputs = benchmark(
            backend=self.backend,
            api_url=self.api_url,
            model_id=self.model_id,
            input_queue=self.input_queue,
            stop_time=self.stop_time - self.ramp_up_time,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            api_key=self.api_key,
            client_id=self.client_id,
            num_requests=self.num_requests,
        )
        self.output_queue.put(outputs)


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
        input_queue: List[Request] | IterMpQueue,
        output_queue: mp.Queue,
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
        self.input_queue = input_queue
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.num_requests = num_requests
        self.output_queue = output_queue

    def run(self):
        time.sleep(self.ramp_up_time)
        outputs = benchmark(
            backend=self.backend,
            api_url=self.api_url,
            api_key=self.api_key,
            model_id=self.model_id,
            input_queue=self.input_queue,
            stop_time=self.stop_time - self.ramp_up_time,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            client_id=self.client_id,
            num_requests=self.num_requests,
        )
        self.output_queue.put(outputs)


def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_queue: List[Request] | IterQueue | IterMpQueue,
    stop_time: float,
    best_of: int,
    use_beam_search: bool,
    api_key: Optional[str] = None,
    client_id: int = -1,
    num_requests: int = -1,
) -> List[RequestFuncOutput]:
    """Benchmark main function, called from each client.

    Args:
        backend (str): Backend inference framework. Could be "vllm", "lightllm", "sglang" or "ppl". Defaults to "vllm".
        api_url (str): api url of the inference server.
        model_id (str): model id of the inference server, used by vllm.
        input_requests (List[Request] | IterQueue | IterMpQueue): List of requests to send.
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

    logger = logging.getLogger()
    logger.info(
        f"Starting benchmark for backend: {backend}, model_id: {model_id}, client_id: {client_id}, num_requests: {num_requests}, stop_time: {stop_time}"
    )

    logger.info(f"id(input_queue): {id(input_queue)}")

    benchmark_start_time = time.perf_counter()
    outputs = []
    for request_id, request in enumerate(input_queue):
        if stop_time > 0 and time.perf_counter() - benchmark_start_time >= stop_time:
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

    logger.info(f"Stop time reached, stopping benchmark for client {client_id}")

    return outputs


def get_input_requests_queue(
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

    logger = logging.getLogger()
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
    dataset_sampler = DATASET_SAMPLER[args.dataset]
    sampled_requests: List[Request] = dataset_sampler(
        dataset_path=args.dataset_path,
        num_requests=args.num_requests,
        num_turns=args.num_turns,
        tokenizer=tokenizer,
        system_prompt_path=args.system_prompt_path,
    )
    input_queue = get_input_requests_queue(sampled_requests, args)

    # start benchmark
    benchmark_start_time = time.perf_counter()
    clients = []
    output_queue = queue.Queue() if args.execute_mode == "Thread" else mp.Queue()
    logger.info(f"type(input_requests_queue): {type(input_queue)}")
    logger.info(f"type(out_queue): {type(output_queue)}")
    for client_id in range(args.num_clients):
        if args.allow_repetitive_requests:
            # input_requests: List[List[Request]], input_requests_i: List[Request]
            input_queue_i = input_queue[client_id]
        else:
            # input_requests: IterQueue[Request] or IterMpQueue[Request]
            input_queue_i = input_queue

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
                input_queue=input_queue_i,
                output_queue=output_queue,
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
                input_queue=input_queue_i,
                output_queue=output_queue,
            )

        client.start()
        clients.append(client)
        logger.info(
            f"client {client_id} launched with ramp up time {client_id * args.ramp_up_time / args.num_clients}"
        )

    all_outputs = []
    for client in clients:
        all_outputs += output_queue.get()
        if args.execute_mode == "Thread":
            client.join()

    if args.execute_mode == "Process":
        for client in clients:
            client.join()

        if not args.allow_repetitive_requests:
            input_queue.clear()
            logger.info(f"out_queue.empty(): {output_queue.empty()}")
            logger.info(f"input_queue.empty(): {input_queue.empty()}")

    benchmark_duration = time.perf_counter() - benchmark_start_time
    logger.info(f"benchmark duration: {benchmark_duration:.2f}s")
    logger.info(f"output len: {len(all_outputs)}")
    metrics, _ = calculate_metrics(outputs=all_outputs, dur_s=benchmark_duration)
    logger.info(f"metrics calculated")
    dump_metrics(metrics)


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
        choices=list(DATASET_SAMPLER.keys()),
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
    main(args)
