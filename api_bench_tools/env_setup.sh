#!/bin/bash

# 测试参数，根据测试用例随时进行更改
# export BACKEND="vllm"
# export BACKEND="ppl"
# export BACKEND="lightllm"
# export BACKEND="amsv2"
export BACKEND="sglang"
export MODEL_TAG="llama2-7b_tp1_fp16"
export ENABLE_SYSTEM_PROMPT=1
export DATASET="sharegpt"
# export DATASET="xiaomi"


# benchmark_serving_num_clients.py 脚本路径
export BENCHMARK_LLM="/mnt/llm/workspace/zhaozhiyu/work/llm.perf/api_bench_tools/python/benchmark_serving_num_clients.py"
# 数据集路径
export SHAREGPT_DATASET_PATH="/mnt/llm/workspace/zhaozhiyu/work/llm.perf/api_bench_tools/datasets/samples_1024.json"
export XIAOMI_DATASET_PATH="/mnt/llm/workspace/zhaozhiyu/work/llm.perf/api_bench_tools/datasets/xiaomi_data1_medium.jsonl"
# 系统提示词路径
export SYSTEM_PROMPT_PATH="/mnt/llm/workspace/zhaozhiyu/work/llm.perf/api_bench_tools/datasets/system_prompt_sample.txt"
# benchmark tokenizer路径
export BENCHMARK_TOKENIZER_PATH="/mnt/llm2/llm_perf/hf_models/llama-7b-hf"
# OPMX模型路径，用于ppl_llm_server
export OPMX_MODEL_PATH="/mnt/llm2/llm_perf/ppl_llm/opmx_models"
# Huggingface模型路径，用于vllm_server和lightllm_server
export HF_MODEL_PATH="/mnt/llm2/llm_perf/hf_models"
# URLS
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export PPL_SERVER_URL="127.0.0.1:23333"
export LIGHTLLM_SERVER_URL="http://127.0.0.1:8080"
export SGLANG_SERVER_URL="http://127.0.0.1:30000"
export AMSV2_SERVER_URL="https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719"
# AMSV2 API Key
export AMSV2_API_KEY="eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw"



