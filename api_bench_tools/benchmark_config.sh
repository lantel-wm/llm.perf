#-------------------------------------------------------------------------------------------------------
# 测试参数，根据测试用例随时进行更改
#-------------------------------------------------------------------------------------------------------
# 测试的推理后端，支持vllm、ppl、lightllm、amsv2、sglang
# export BACKEND="vllm"
export BACKEND="ppl"
# export BACKEND="lightllm"
# export BACKEND="amsv2"
# export BACKEND="sglang" 
# 模型标签，仅起标识作用，请自行保证模型标签与实际测试的模型一致
export MODEL_TAG="llama2-7b_tp1_fp16" 
# system prompt开关，1表示使用system prompt，0表示不使用system prompt（system prompt文件路径在配置参数中指定）
export ENABLE_SYSTEM_PROMPT=1
# 测试使用的数据集，支持sharegpt和xiaomi (数据集路径在配置参数中指定)
export DATASET="sharegpt"
# export DATASET="xiaomi"
# 测试并发数
export NUM_CLIENTS=(1 5 10 20 30 40 50 100 200 300)
# 停止时间，单位为秒
export STOP_TIME=300


#-------------------------------------------------------------------------------------------------------
# 配置参数，设定后一般无需更改
#-------------------------------------------------------------------------------------------------------
# benchmark_serving_num_clients.py 脚本路径
export BENCHMARK_LLM="./python/benchmark_serving_num_clients.py"
# 数据集路径
export SHAREGPT_DATASET_PATH="./datasets/samples_1024.json"
export XIAOMI_DATASET_PATH="./datasets/xiaomi_data1_medium.jsonl"
# system_prompt路径
export SYSTEM_PROMPT_PATH="./datasets/system_prompt_sample.txt"
# benchmark tokenizer路径
export BENCHMARK_TOKENIZER_PATH="/mnt/llm2/llm_perf/hf_models/llama-7b-hf"
# OPMX模型路径，用于ppl_llm_server
export OPMX_MODEL_PATH="/mnt/llm2/llm_perf/ppl_llm/opmx_models"
# Huggingface模型路径
export HF_MODEL_PATH="/mnt/llm2/llm_perf/hf_models"
# server urls
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export PPL_SERVER_URL="127.0.0.1:23334"
export LIGHTLLM_SERVER_URL="http://127.0.0.1:8080"
export SGLANG_SERVER_URL="http://127.0.0.1:30000"
export AMSV2_SERVER_URL="https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719"
# AMSV2 API Key
export AMSV2_API_KEY="eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw"



