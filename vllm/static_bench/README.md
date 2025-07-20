# static-benchmark

## 测试启动

在运行脚本之前，先把BENCKMARK_LLM环境变量设为`benchmark_latency.py`的路径，`benchmark_latency.py`是vllm静态测试脚本，路径为`llm.perf/vllm/static_bench/python/benchmark_lantency.py`


启动`benchmark_all_cuda.sh`即可开始测试：

```shell
bash benchmark_all_cuda.sh
```
启动`benmark_one_cuda_fp16.sh`可以进行单个case的测试并查看详情log:

```shell
bash benmark_one_cuda_fp16.sh
```

## 调整测试用例

`benchmark_all_cuda.sh`中可以调整测试的batch size，model size，tp size，in len，out len，mode等参数。

## 结果保存

测试结果保存在benchmark_all_cuda.sh，如果确认没问题，可以重命名成benchmark_result_{显卡型号}_{yyyymmdd}.csv，并移动到result文件夹，
例如benchmark_result_a800_20240501.csv

