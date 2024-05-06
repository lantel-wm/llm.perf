# 调整测试算例

调整benchmark_all_cuda.sh中的batch size，model size，还有tp size，dataset，mode等参数即可，脚本的详细内容不赘述，请读懂

# 测试启动

在运行脚本之前，一定要设置BENCHMARK_LLM环境变量
BENCHMARK_LLM指向编译完成的pplnn中的tools/benchmark_llama

启动benchmark_all_cuda.sh即可开始测试

启动benchmark_one_cuda_fp16.sh可以进行单个case的测试并查看详情log

# 结果保存

benchmark_all_cuda.sh跑好的结果，如果确认没问题，可以重命名成benchmark_result_{显卡型号}_{yyyymmdd}.csv，并移动到result文件夹，
例如benchmark_result_a800_20240501.csv

# 不同显卡的处理

为不同显卡准备了不同的benchmark_templ_cuda，需要使用时将其复制成benchmark_templ_cuda.sh即可
备注：
- benchmark_templ_cuda_ampere.sh是为A系列显卡准备的
- benchmark_templ_cuda_non_ampere.sh是为更新的显卡准备的，主要增加了NCCL_PROTO=^Simple来防止崩溃，在A系显卡上使用可能会导致多卡通信变慢
