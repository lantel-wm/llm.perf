# api_bench_tools

## 脚本说明

`api_bench_tools`包含一套测试LLM动态推理性能的工具，能够模拟真实用户场景下的LLM推理性能。

- `api_bench_tools/python/benchmark_serving_num_clients.py`模拟了固定用户数量的场景。

    设置并发数为n，每个并发线程模拟一个用户，每个用户有一个固定的`thread_id`。每个线程阻塞式地向server发送请求，即直到server返回结果之前一直等待，server返回结果后再发送下一个请求。

    用户发送的内容从数据集中选取，为了更好地模拟真实情况，每个用户发送的内容序列各不相同；同时为了测试的可复现性，不同次测试中，`thread_id`相同的用户发送的内容序列相同。

    真实场景中，用户请求大概率是交错到达的。如果server在服务n个用户，n个用户中应该有若干请求在进行prefill，若干请求在进行decode，还可能有若干请求在排队，不太可能出现所有用户请求处于同一状态的情况。
    
    然而在测试启动时，若直接同时启动n个线程，将导致n个用户请求同时处于prefill阶段，这与实际情况不符，会导致测得的首字延迟大大增加。因此`benchmark_serving_num_clients.py`设置了`ramp_up_time`参数，即缓起时间。在测试开始时，n个线程不是同时启动，而是在`ramp_up_time`秒内均匀地逐个启动，模拟真实场景中用户请求交错到来的情况。默认的`ramp_up_time`为 n*0.1 秒。

- `api_bench_tools/python/benchmark_serving_request_rate.py`模拟了固定请求速率的场景。

    设置请求速率为r，每隔t秒发送一次请求，其中t服从参数为r的指数分布。

    这种测试方式异步地发送请求，因此无法控制请求并发数。

**注意：** 以上脚本只模拟用户行为，不模拟server行为。并发数=n仅代表当前有n个用户在发送请求，不保证当前在server内这n个请求在同一个batch内，也不保证这n个请求都在运行。server内部的状态，包括server的tokenizer对测试脚本都是黑盒状态。测试脚本只负责模拟用户行为，根据server的返回生成token的时间间隔计算性能指标。

## 性能指标

- `completed`：完成的请求数量。

- `success_rate`：请求成功率，即完成的请求数量/总请求数量。

- `qps`：requests per second，每秒请求数量，即完成的请求数量/测试总时间。

- `total_inlen`：测试过程中输入token的总数。

- `total_outlen`：测试过程中输出token的总数。

- `avg_inlen`：所有请求平均输入token数。

- `avg_outlen`：所有请求平均输出token数。

- `max_inlen`：所有请求的最大输入token数。

- `max_outlen`：所有请求的最大输出token数。

- `o_tps`：Output tokens per second，每秒输出token数，输出吞吐，=`total_outlen`/测试总时间。

- `io_tps`：Input Output tokens per second，每秒输入输出token数，总吞吐，=(`total_inlen` + `total_outlen`)/测试总时间。

- `TTFT`： Time to fist token，首字延迟，从请求发送到接收到第一个token的时间间隔。每个线程的每个请求都有一个TTFT值，指标计算了所有请求的TTFT的max，min，mean，median，std，p90，p99。

- `TPOT`：Time per output token，每个输出token的延迟。每个线程的每个请求都有一个TPOT值，计算公式为：输出token数 / (latency - TTFT)，指标计算了所有请求的TPOT的max，min，mean，median，std，p90，p99。

- `E2E_TIME`：End to end time（Lantency），从请求发送到接收到最后一个token的时间间隔。每个线程的每个请求都有一个E2E_TIME值，指标计算了所有请求的E2E_TIME的max，min，mean，median，std，p90，p99。

- `ITL`：Inter token latency，每个token之间的时间间隔。每个输出token都有一个ITL值，指标计算了所有ITL的max，min，mean，median，std，p90，p99。ITL和TPOT的区别在于ITL的统计粒度为token，而TPOT的统计粒度为请求。

    **注意**：

    统计输入token数需要用到tokenizer将prompt文本转为tokens，由于server对测试脚本为黑盒状态，测试脚本无法得知server使用的tokenizer。因此测试脚本使用固定的tokenizer，以保证不同模型、不同推理后端所测的输入token数一致。由于测试使用的tokenizer可能不是正确的tokenizer，因此测得的`total_inlen`，`avg_inlen`，`mean_inlen`，`max_inlen`，`io_tps`，可能不准确，仅供测试结果之间进行对比，其绝对值不具有参考价值。

    输出token数不存在这个问题，输出token数的统计是通过记录server返回的token数量直接计算，不涉及tokenizer，即`total_outlen`，`avg_outlen`，`mean_outlen`，`max_outlen`，`o_tps`是准确的。



## 前置准备

配置python环境：
```shell
conda creante -n perf python=3.11
conda activate perf
pip install -r requirements.txt
```

下载数据集：

```shell
$ cd api_bench/datasets
$ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
$ wget https://raw.githubusercontent.com/openppl-public/ppl.llm.serving/master/tools/samples_1024.json
```

在`env_setup.sh`中设置环境变量：
```sh
#!/bin/bash

# 测试参数，根据测试用例随时进行更改
# export BACKEND="vllm"
export BACKEND="ppl"
# export BACKEND="lightllm"
# export BACKEND="amsv2"
export MODEL_SIZE=7
export TP_SIZE=1
export MODE="fp16"
export ENABLE_SYSTEM_PROMPT=0


# benchmark_serving_num_clients.py 脚本路径
export BENCHMARK_LLM="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/gitlab/llm.perf/api_bench_tools/python/benchmark_serving_num_clients.py"
# 数据集路径
export DATASET_PATH="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/gitlab/llm.perf/api_bench_tools/datasets/samples_1024.json"
# 系统提示词路径
export SYSTEM_PROMPT_PATH="/mnt/nvme0n1/workspace/zhaozhiyu/work/llm-bench/gitlab/llm.perf/api_bench_tools/datasets/system_prompt_sample.txt"
# benchmark tokenizer路径
export BENCHMARK_TOKENIZER_PATH="/mnt/llm2/llm_perf/hf_models/llama-7b-hf"
# OPMX模型路径，用于ppl_llm_server
export OPMX_MODEL_PATH="/mnt/llm/LLaMA/test/opmx_models"
# Huggingface模型路径，用于vllm_server和lightllm_server
export HF_MODEL_PATH="/mnt/llm2/llm_perf/hf_models"
# URLS
export VLLM_SERVER_URL="http://127.0.0.1:8000"
export PPL_SERVER_URL="127.0.0.1:23333"
export LIGHTLLM_SERVER_URL="http://127.0.0.1:8080"
export AMSV2_SERVER_URL="https://devsft.studio.sensecoreapi.cn/gpu8-sensechat590-20240719"
# AMSV2 API Key
export AMSV2_API_KEY="eyJhbGciOiJFUzI1NiIsImtpZCI6ImNiMTY1YTA1LWY1ZTctNDkzYS1hNjMwLTcyOTM3YmE1YTM0ZiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwMzcxNjgzMTEsImlhdCI6MTcyMTYzNTUxMSwiaXNzIjoiaHR0cHM6Ly9pYW0taW50ZXJuYWwuc2Vuc2Vjb3JlYXBpLmNuLyIsImp0aSI6IjQ1YmYzMWE4LTdmZjItNDM5OC04NmMwLTQwMDg5ZjU0M2M3NiIsInJlc291cmNlX2lkIjoiZDkxN2JkYmQtNDViMC0xMWVmLTkwMjktM2U2NDkxYjJlNmY1Iiwic3ViIjoiNjMwZmI3MTI2MWViNjgxMjAwMjNmZTY1YWNjNWFiNDgiLCJ1cmkiOiJkZXZzZnQuc3R1ZGlvLnNlbnNlY29yZWFwaS5jbi9ncHU4LXNlbnNlY2hhdDU5MC0yMDI0MDcxOSJ9.4Dt712ONtlKHVcCTv9AVpCBLTo0osDXHHqIzzDsIsLTU1rGKqsjBQaW4xPKM-pIGbVoSb1KzyO1T4gTFSU6Xgw"
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

若使用`bash ./start_server.sh`启动不成功，可以手动启动相应的server。命令如下：

```shell
# vLLM
# 需要修改--model和-tp
python -m vllm.entrypoints.openai.api_server --model /mnt/llm2/llm_perf/hf_models/llama-7b-hf --swap-space 16 --disable-log-requests --enforce-eager --host 127.0.0.1 --port 8000 -tp 1

# ppl ./ppl_llm_server在ppl.llm.serving/ppl-build目录
# 需要修改--model-dir、--model-param-path、--tokenizer-path、--tensor-parallel-size
./ppl_llm_server --model-dir /mnt/llm/llm_perf/opmx_models/llama_65b_8gpu --model-param-path /mnt/llm/llm_perf/opmx_models/llama_65b_8gpu/params.json --tokenizer-path /mnt/llm/llm_perf/hf_models/llama-65b-hf/tokenizer.model --tensor-parallel-size 8 --top-p 0.0 --top-k 1 --max-tokens-scale 0.94 --max-input-tokens-per-request 4096 --max-output-tokens-per-request 4096 --max-total-tokens-per-request 8192 --max-running-batch 1024 --max-tokens-per-step 8192 --host 127.0.0.1 --port 23333

# lightllm
# 需要修改--model-dir、--tp
python -m lightllm.server.api_server --model_dir /mnt/llm2/llm_perf/hf_models/llama-7b-hf --host 127.0.0.1 --port 8080 --tp 1 --max_total_token_num 150000 --tokenizer_mode fast
```

## 使用Python测试单个case

`benchmark_all_cuda.sh`调用了`python/benchmark_serving_num_clients.py`，如果想测试单个测试用例，或者想debug，可以直接运行`benchmark_serving_num_clients.py`，步骤如下：

1.手动启动对应的server

2.启动 client，开始benchmark：
```shell
$ python python/benchmark_serving_num_clients.py --base-url YOUR_SERVER_URL --backend vllm --model PATH_TO_HF_MODEL --dataset-path datasets/samples_1024.json --num-requests 1024 --num-turns 1 --num-threads 100 --ramp-up-time 10 --thread-stop-time 300 
```

## 添加新的测试后端

`api_bench_tools`目前支持vllm，ppl，lightllm三种后端。若要添加新的后端，需要在`python/backend_request_func.py`中添加新的请求函数，实现client与server的通信。

目前的请求函数的通信协议有两种：

- http：vllm，lightllm，可以使用curl命令向server发送请求
- grpc：ppl，无法使用curl命令向server发送请求

对于http协议的server，可以参照`python/backend_request_func.py`中的`request_openai_completions`函数，实现自己的请求函数。

对于grpc协议的server，可以参照`python/backend_request_func.py`中的`request_ppl_completions`函数，实现自己的请求函数。

在`python/backend_request_func.py`的`if __name__ == "__main__"`中有请求函数的调试代码，可以修改参数，然后运行`python backend_request_func.py`，测试请求函数是否正确。

为了避免随机性，需要保证server返回固定的长度。常用的方法是将`ignore_eos`设为`True`，将`max_new_tokens`设为想要输出的长度。如果server没有类似的参数，需要自己想办法控制输出的长度。

**注意：** 推理服务需要支持Stream模式，即逐token返回生成结果，否则无法测试动态性能。

