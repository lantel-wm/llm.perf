# LLM Perf

用于放置不同后端的推理性能测试脚本以及结果

## 测试类型

### static_bench

静态推理性能测试，测试引擎的原生性能

仅ppl支持静态性能测试，脚本位于`ppl_llm/static_bench`


### api_bench

线上服务的api性能测试，测试推理引擎+serving接口的整体性能。考察调度以及continuous batching，kv cache管理等服务特性

测试时需要先启动api_server，再启动benchmark client进行请求发送和性能测试，此测试可能会受网路波动影响

脚本位于`api_bench_tools`

详细文档见: [api_bench_tools/README.md](api_bench_tools/README.md)


