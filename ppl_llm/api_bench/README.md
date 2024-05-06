这部分尚未完成，目前看来，还需要云瑞/国宇整理好serving的开源和闭源代码才可以继续推进。

另外，该部分由于需要启动两个executeable，即client和server，目前看来无法全自动完成，
必须先由人工启动对应模型的server才能起client脚本测试。

需要的文件：
- benchmark_server_templ_cuda.sh：启动server的模板，接受model_size, tp_size，mode（mode指fp16 w8a8 w4a16）启动对应的模型
- benchmark_client_templ_cuda.sh：启动client的模板，接受request_rate，dataset，启动客服端进行请求发送和计时，输出里一定有一行包含输入的各个指标数据的csv格式输出
- benchmark_server_cuda_fp16.sh：启动fp16的server，接受model_size, tp_size，指定mode
- benchmark_all_cuda.sh：遍历request_rate列表和dataset列表，运行benchmark_client_templ_cuda.sh，将输出汇总到benchmark_all_cuda_result.csv
