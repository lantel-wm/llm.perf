# api_bench_tools

## 前置准备

1.配置测试脚本的python环境：
```shell
$ conda create -n perf python=3.11
$ conda activate perf
$ pip install -r requirements.txt
```

2.配置server环境：[README_SERVER.md](README_SERVER.md)

3.修改`benchmark_config.sh`

## 测试启动

1.在`benchmark_config.sh`中设置以下参数：
- `BACKEND`
- `MODEL_TAG`
- `ENABLE_SYSTEM_PROMPT`
- `DATASET`

2.启动`BACKEND`对应的server：[README_SERVER.md](README_SERVER.md)

3.启动测试脚本

```shell
$ bash benchmark_all_cuda.sh
```

4.测试完成后，关闭server

## 测试结果

- 测试结果保存在`result/`目录中，历史结果会自动归档在`result/$date`目录中。

- 如果需要修改推理后端、数据集、服务器url等参数，请修改`benchmark_config.sh`的相应内容即可。

## 注意事项

- `benchmark_config.sh`中的`MODEL_TAG`仅起标识作用，与server的启动设置无关。请保证`MODEL_TAG`与server的实际设置相对应。


## 使用Python测试单个case

`benchmark_all_cuda.sh`调用了`python/benchmark_serving_num_clients.py`，如果想测试单个测试用例或者debug，可以运行`benchmark_serving_num_clients.py`，步骤如下：

1.启动对应的server

2.启动 client，开始benchmark：
```shell
$ python python/benchmark_serving_num_clients.py \
--base-url YOUR_SERVER_URL \
--backend vllm \
--model PATH_TO_HF_MODEL \
--dataset-path datasets/samples_1024.json \
--num-requests 1024 \
--num-turns 1 \ 
--num-threads 100 \ 
--ramp-up-time 10 \ 
--thread-stop-time 300
```

## 添加新的测试后端

`api_bench_tools`目前支持的推理后端有ppl，vllm，lightllm，sglang。若要添加新的后端，需要在`python/backend_request_func.py`中添加新的请求函数，实现client与server的通信。

目前的请求函数的通信协议有两种：

- http：vllm，lightllm，sglang使用http server，可以使用curl命令向server发送请求
- grpc：ppl使用grpc server，无法使用curl命令向server发送请求

对于http协议的server，可以参照`python/backend_request_func.py`中的`request_openai_completions`函数，实现自己的请求函数。

对于grpc协议的server，可以参照`python/backend_request_func.py`中的`request_ppl_completions`函数，实现自己的请求函数。

在`python/backend_request_func.py`的`if __name__ == "__main__"`中有请求函数的调试代码，可以修改参数，然后运行`python backend_request_func.py`，测试请求函数是否正确。

为了避免随机性，需要保证server返回固定的长度。常用的方法是将`ignore_eos`设为`True`，将`max_new_tokens`设为想要输出的长度。如果server没有类似的参数，需要自己想办法控制输出的长度。

**注意：** 推理服务需要支持Stream模式，即逐token返回生成结果，否则无法测试动态性能。

## 详细说明

`api_bench_tools`包含一套测试LLM动态推理性能的工具，能够模拟真实用户场景下的LLM推理性能。

- `api_bench_tools/python/benchmark_serving_num_clients.py`模拟了固定用户数量的场景。

    设置并发数为n，每个并发线程模拟一个用户，每个用户有一个固定的`thread_id`。每个线程阻塞式地向server发送请求，即直到server返回结果之前一直等待，server返回结果后再发送下一个请求。

    用户发送的内容从数据集中选取，为了更好地模拟真实情况，每个用户发送的内容序列各不相同；同时为了测试的可复现性，不同次测试中，`thread_id`相同的用户发送的内容序列相同。

    真实场景中，用户请求大概率是交错到达的。如果server在服务n个用户，n个用户中应该有若干请求在进行prefill，若干请求在进行decode，还可能有若干请求在排队，不太可能出现所有用户请求处于同一状态的情况。
    
    然而在测试启动时，若直接同时启动n个线程，将导致n个用户请求同时处于prefill阶段，这与实际情况不符，会导致测得的首字延迟大大增加。因此`benchmark_serving_num_clients.py`设置了`ramp_up_time`参数，即缓起时间。在测试开始时，n个线程不是同时启动，而是在`ramp_up_time`秒内均匀地逐个启动，模拟真实场景中用户请求交错到来的情况。默认的`ramp_up_time`为 n*0.1 秒。

    多线程的模式可能会因为python的GIL遇到性能问题，可以采用多进程的方式来运行client，通过`excute_mode`参数控制，默认为`Thread`表示使用线程，设置为`Process`即可使用多进程方式运行

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

- `TPOT`：Time per output token，每个输出token的延迟。每个线程的每个请求都有一个TPOT值，计算公式为：(latency - TTFT) / 输出token数，指标计算了所有请求的TPOT的max，min，mean，median，std，p90，p99。

- `E2E_TIME`：End to end time（Lantency），从请求发送到接收到最后一个token的时间间隔。每个线程的每个请求都有一个E2E_TIME值，指标计算了所有请求的E2E_TIME的max，min，mean，median，std，p90，p99。

- `ITL`：Inter token latency，两个相邻输出token之间的时间间隔。每个输出token都有一个ITL值，指标计算了所有ITL的max，min，mean，median，std，p90，p99。ITL和TPOT的区别在于ITL的统计粒度为token，且第一个输出token的ITL即为首字延迟；而TPOT的统计粒度为请求。

**注意**：

统计输入token数需要用到tokenizer将prompt文本转为tokens，由于server对测试脚本为黑盒状态，测试脚本无法得知server使用的tokenizer。因此测试脚本使用固定的tokenizer，以保证不同模型、不同推理后端所测的输入token数一致。由于测试使用的tokenizer可能不是正确的tokenizer，因此测得的`total_inlen`，`avg_inlen`，`mean_inlen`，`max_inlen`，`io_tps`，可能不准确，仅供测试结果之间进行对比，其绝对值不具有参考价值。

输出token数不存在这个问题，输出token数的统计是通过记录server返回的token数量直接计算，不涉及tokenizer，即`total_outlen`，`avg_outlen`，`mean_outlen`，`max_outlen`，`o_tps`是准确的。
