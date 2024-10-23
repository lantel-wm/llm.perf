import numpy as np

from dataclasses import dataclass
from typing import List, Tuple

from utils import RequestFuncOutput


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

    ttfts: List[float] = []
    tpots: List[float] = []
    itls: List[float] = []
    e2es: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            input_len = outputs[i].prompt_len
            output_len = outputs[i].output_len

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


def dump_metrics(metrics: BenchmarkMetrics):
    """Dumps metrics stdout in CSV format which will be parsed by the shell scripts.

    Args:
        metrics (BenchmarkMetrics): Metrics to dump.
    """
    print(
        "CSV header output:\
completed,success_rate,qps,total_inlen,total_outlen,\
avg_inlen,avg_outlen,max_inlen,max_outlen,o_tps,io_tps,\
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
