# sglang

## sglang server 环境配置

```shell
pip install --upgrade pip
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

## sglang server 启动

```shell
# 需要修改--model-dir、--tp
python -m sglang.launch_server --model-path /mnt/llm2/llm_perf/hf_models/llama-7b-hf --port 30000 --tp 1
```