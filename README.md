# llm.perf

本仓库提供了一套完整的LLM推理性能测试框架，支持多种主流推理引擎的性能评估与对比分析。

## 🎯 项目概述

LLM Perf旨在为不同大语言模型推理后端提供标准化的性能测试方案，包含：
- **静态推理性能测试**：评估引擎原生计算性能
- **API服务性能测试**：评估完整服务栈性能（含调度、缓存、批处理等）

## 📊 测试类型

### 1. 静态推理测试 (Static Bench)

**测试目标**：评估推理引擎的**原生计算性能**，排除网络和服务层开销

**适用引擎**：
- ✅ PPL.LLM（完整支持）
- ✅ vLLM（支持）
- ✅ LightLLM（支持）

**测试内容**：
- 单批次推理延迟
- 吞吐量基准
- 内存使用效率
- 不同精度（FP16/INT8/INT4）性能对比

**测试位置**：`ppl_llm/static_bench/`

### 2. API服务测试 (API Bench)

**测试目标**：评估**完整服务栈性能**，包括：
- 推理引擎性能
- 服务接口开销
- 连续批处理（Continuous Batching）
- KV缓存管理
- 请求调度策略

**测试流程**：
1. 启动目标推理服务（api_server）
2. 运行benchmark client发送请求
3. 收集性能指标

**注意事项**：测试结果可能受网络环境影响，建议在局域网或本机环境执行

**测试位置**：`api_bench_tools/`

## 🗂️ 目录结构

```
llm.perf/
├── api_bench_tools/          # API服务测试工具
│   ├── python/              # Python测试脚本
│   ├── datasets/            # 测试数据集
│   └── plot/                # 结果可视化
├── ppl_llm/                 # PPL.LLM相关测试
│   ├── static_bench/        # 静态测试
│   ├── api_bench/          # API测试
│   └── queue_bench/        # 队列测试
├── vllm/                    # vLLM相关测试
├── sglang/                  # SGLang相关测试
└── lightllm/               # LightLLM相关测试
```

## 🚀 快速开始

### 静态测试示例
```bash
# PPL.LLM静态测试
cd ppl_llm/static_bench
./benchmark_all_cuda.sh

# vLLM静态测试
cd vllm/static_bench
./benchmark_all_cuda.sh
```

### API测试示例
```bash
# 启动API服务（以PPL.LLM为例）
cd ppl_llm/api_bench
./start_ppl_server.sh

# 运行性能测试
cd api_bench_tools
python python/benchmark_serving_num_clients.py
```

## 📈 结果分析

所有测试结果将保存在对应目录的`result/`文件夹中，包含：
- **CSV格式**：详细性能指标
- **可视化图表**：性能趋势分析
- **对比报告**：不同配置下的性能差异

## 📖 详细文档

- [API测试工具文档](api_bench_tools/README.md)
- [PPL.LLM测试指南](ppl_llm/README.md)
- [vLLM测试指南](vllm/README.md)
- [SGLang测试指南](sglang/README.md)