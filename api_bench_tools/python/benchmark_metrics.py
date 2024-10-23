import numpy as np
from dataclasses import dataclass
from typing import List

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
    

def get_metrics(
    completed: int,
    succeeded: int,
    total_input_tokens: int,
    total_output_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    dur_s: float,
    ttfts: List[float],
    tpots: List[float],
    e2es: List[float],
    itls: List[float],
) -> BenchmarkMetrics:
    BenchmarkMetrics(
        completed=completed,
        successful_rate=completed / succeeded,
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