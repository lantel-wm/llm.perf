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

# lightllm

## lightllm server 环境配置

```shell
git clone https://github.com/ModelTC/lightllm.git && cd lightllm
pip install -r requirements.txt
pip install triton==2.0.0
python setup.py install
pip install triton==2.1.0
```

requirements.txt中torch==2.0.0，triton==2.1.0，但torch依赖triton==2.0.0。所以需要先安装triton==2.0.0使`python setup.py install`顺利执行，安装成功后再换回triton==2.1.0。

报错：FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'

解决方法：export PATH=$PATH:/sbin

## lightllm server 启动

```shell
# 需要修改--model-dir、--tp
python -m lightllm.server.api_server --model_dir /mnt/llm2/llm_perf/hf_models/llama-7b-hf --host 127.0.0.1 --port 8080 --tp 1 --max_total_token_num 150000 --tokenizer_mode fast
```

# ppl

## ppl server 环境配置

自行编译ppl.serving：https://gitlab.sz.sensetime.com/HPC/llm/ppl.llm.serving

## ppl server 启动

```shell
# ppl ./ppl_llm_server在ppl.llm.serving/ppl-build目录
# 需要修改--model-dir、--model-param-path、--tokenizer-path、--tensor-parallel-size
./ppl_llm_server --model-dir /mnt/llm/llm_perf/opmx_models/llama_65b_8gpu --model-param-path /mnt/llm/llm_perf/opmx_models/llama_65b_8gpu/params.json --tokenizer-path /mnt/llm/llm_perf/hf_models/llama-65b-hf/tokenizer.model --tensor-parallel-size 8 --top-p 0.0 --top-k 1 --max-tokens-scale 0.94 --max-input-tokens-per-request 4096 --max-output-tokens-per-request 4096 --max-total-tokens-per-request 8192 --max-running-batch 1024 --max-tokens-per-step 8192 --host 127.0.0.1 --port 23333
```

# sglang

## sglang server 环境配置

```shell
pip install --upgrade pip
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

## sglang server 启动

```shell
# 需要修改--model-dir、--tp，使用--disable-radix-cache来关闭prefix cache
python -m sglang.launch_server --disable-radix-cache --model-path /mnt/llm2/llm_perf/hf_models/llama-7b-hf --port 30000 --tp 1
```