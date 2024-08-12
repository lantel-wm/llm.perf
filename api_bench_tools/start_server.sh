#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")


if [ -z "$BACKEND" ]; then
    echo "BACKEND environment variable not set"
    exit 1
fi

MODEL_SIZE=$1

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=7
fi

TP_SIZE=$2

if [ -z "$TP_SIZE" ]; then
    TP_SIZE=1
fi

MODE=$3

if [ -z "$MODE" ]; then
    MODE="fp16"
fi

if [ "$BACKEND" == "vllm" ]; then
    bash "$PERF_BASE_PATH/../vllm/api_bench/benchmark_server_templ_cuda.sh" "$MODEL_SIZE" "$TP_SIZE" "$MODE"
elif [ "$BACKEND" == "ppl" ]; then
    bash "$PERF_BASE_PATH/../ppl_llm/api_bench/benchmark_server_templ_cuda.sh" "$MODEL_SIZE" "$TP_SIZE" "$MODE"
elif [ "$BACKEND" == "lightllm" ]; then
    bash "$PERF_BASE_PATH/../lightllm/api_bench/benchmark_server_templ_cuda.sh" "$MODEL_SIZE" "$TP_SIZE" "$MODE"
else
    echo "Unsupported backend: $BACKEND"
    exit 1
fi
  


