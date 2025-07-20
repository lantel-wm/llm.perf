# vllm

## vllm server 环境配置

```shell
pip install vllm==0.4.2
# 或者
pip install vllm==0.5.1
```

## vllm server 启动

```shell
# 需要修改--model和-tp
python -m vllm.entrypoints.openai.api_server --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --swap-space 16 --disable-log-requests --enforce-eager --host 127.0.0.1 --port 8000 -tp 1
```