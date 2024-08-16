#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

# 测试参数，根据测试用例随时进行更改
# export BACKEND="vllm"
# export BACKEND="ppl"
export BACKEND="lightllm"
export MODEL_SIZE=70
export TP_SIZE=4
export MODE="fp16"

# 脚本路径、数据集路径、模型路径、服务地址等
export BENCHMARK_LLM="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/gitlab/llm.perf/api_bench_tools/python/benchmark_serving_num_clients.py"
export DATASET_PATH="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/gitlab/llm.perf/api_bench_tools/datasets/samples_1024.json"
export OPMX_MODEL_PATH="/mnt/llm/LLaMA/test/opmx_models"
export HF_MODEL_PATH="/mnt/llm2/llm_perf/hf_models"
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export PPL_SERVER_URL="127.0.0.1:23333"
export LIGHTLLM_SERVER_URL="http://127.0.0.1:8080"



