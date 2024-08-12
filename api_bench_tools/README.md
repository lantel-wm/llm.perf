# api-benchmark

## 前置准备

下载数据集：

```shell
$ cd api_bench/datasets
$ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
$ wget https://raw.githubusercontent.com/openppl-public/ppl.llm.serving/master/tools/samples_1024.json
```

在`env_setup.sh`中设置环境变量：
```shell
#!/bin/bash

SCRIPT=$(realpath -s "$0")
PERF_BASE_PATH=$(dirname "$SCRIPT")

# 测试参数，根据测试用例随时进行更改
# export BACKEND="vllm"
# export BACKEND="ppl"
export BACKEND="lightllm"
export MODEL_SIZE=7
export TP_SIZE=1
export MODE="fp16"

# 脚本路径、数据集路径、模型路径、服务地址等
export BENCHMARK_LLM="$PERF_BASE_PATH/python/benchmark_serving_num_clients.py"
export DATASET_PATH="$PERF_BASE_PATH/datasets/samples_1024.json"
export OPMX_MODEL_PATH="/mnt/llm/LLaMA/test/opmx_models"
export HF_MODEL_PATH="/mnt/llm2/llm_perf/hf_models"
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export PPL_SERVER_URL="127.0.0.1:23333"
export LIGHTLLM_SERVER_URL="127.0.0.1:8080"
```
然后执行`env_setup.sh`：
```shell
$ source env_setup.sh
```

## 测试启动

在`env_setup.sh`中设置BACKEND、MODEL_SIZE、TP_SIZE、MODE等参数，然后执行

```shell
$ source env_setup.sh
```

启动server

```shell
$ bash ./start_server.sh
SERVER STARTED 27250
```

启动测试脚本

```shell
$ bash benchmark_all_cuda.sh
```

测试完成后，关闭server

```shell
$ kill -9 27250
```

结果保存在`result/`中，历史结果会自动归档在`result/$date`目录中。

需要修改推理后端、数据集、服务器url等参数时,修改env_setup.sh的相应内容，然后重新执行`source env_setup.sh`。

## 使用Python测试单个case

### vLLM

启动 server：
```shell
$ python -m vllm.entrypoints.openai.api_server --model PATH_TO_MODEL --swap-space 16 --disable-log-requests --enforce-eager --host YOUR_IP_ADDRESS  --port 8000
```

启动 client，开始benchmark：
```shell
$ python python/benchmark_serving_num_clients.py --host YOUR_IP_ADDRESS --port 8000 --model PATH_TO_MODEL --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompt 200 --thread-num 48 --disable-tqdm --thread-stop-time 300
```

### PPL

启动 server：（需要先编译ppl.llm.serving）
```shell
$ /path/to/ppl.llm.serving/ppl.llm.serving/ppl-build/ppl_llm_server \
    --model-dir /mnt/llm/LLaMA/test/opmx_models/7B_db1_fq1_fk1_ff1_ac1_qc1_cm0_cl3_1gpu \
    --model-param-path /mnt/llm/LLaMA/test/opmx_models/7B_db1_fq1_fk1_ff1_ac1_qc1_cm0_cl3_1gpu/params.json \
    --tokenizer-path /mnt/llm/LLaMA/tokenizer.model \
    --tensor-parallel-size 1 \
    --top-p 0.0 \
    --top-k 1 \
    --max-tokens-scale 0.94 \
    --max-input-tokens-per-request 4096 \
    --max-output-tokens-per-request 4096 \
    --max-total-tokens-per-request 8192 \
    --max-running-batch 1024 \
    --max-tokens-per-step 8192 \
    --host 127.0.0.1 \
    --port 23333
```

启动 client，开始benchmark：
```shell
python python/benchmark_serving.py --host YOUR_IP_ADDRESS --port 23333 --model PATH_TO_MODEL --dataset-name samples_1024 --dataset-path ./samples_1024.json --num-requests 200 --num-threads 48 --disable-tqdm --thread-stop-time 300
```

## 参数设置

`--backend`：测试使用的后端，目前支持vllm和ppl。

`--num-threads`：并发数，为1则执行动态并发，大于1则控制并发数固定。

`--num-requests`：每个并发的测试的请求数量。总请求数量需要乘以并发数。

`--thread-stop-time`：每个并发的测试的持续时间，单位秒。

`--ramp-up-time`：所有线程的起转时间，用来模拟实际情况中request交错到来的情况，单位秒。

