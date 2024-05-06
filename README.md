# LLM Perf

用于放置不同后端的推理性能测试脚本以及结果

## 测试类型

### static_bench

静态推理性能测试，测试引擎的原生性能

### queue_bench

以数据集为输入的静态推理性能测试，以模拟推理引擎在不同场景下的静态性能。也考察推理引擎的动态特性，如remove padding，early finish

### api_bench

线上服务的api性能测试，测试推理引擎+serving接口的整体性能。考察调度以及continuous batching，kv cache管理等服务特性

测试时需要先启动api_server，再启动benchmark client进行请求发送和性能测试，此测试可能会受网路波动影响

## 原始模型

模型文件在hf_models文件夹下，具体目录/mnt/llm2/llm_perf/hf_models，不能上传到git

## PPL.LLM

模型文件在opmx_models文件夹下，具体目录/mnt/llm2/llm_perf/ppl_llm/opmx_models，不能上传到git
