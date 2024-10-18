此文件夹包含了3个数据集

1. overall数据集
    文件
        overall_input_token_ids.txt
        overall_output_length.txt

2. P99 prefill数据集
    文件
        P99_prefill_input_token_ids.txt
        P99_prefill_output_length.txt

3. P99 avg数据集
    文件
        P99_avg_input_token_ids.txt
        P99_avg_output_length.txt


文件含义：
input_token_ids.txt文件：总共有M行，每行包含一个输入token序列。每行有K个由逗号分隔的token编号T（0<T<16384）
output_length.txt文件：总共有M行，每行包含一个数字L，表示input_token_ids.txt中对应行的输入token序列所需要产生的输出长度。



P99数据集采集方法介绍
1. P99 prefill数据集
将overall数据集中的数据按输入长度由长到短排序取得数据集A，取出数据集的前2%数据作为子集B，再将子集B的前50%去除得到最终输出
2. P99 avg数据集
将overall数据集中的数据按（输入长度+输出长度）由长到短排序取得数据集A，取出数据集的前2%数据作为子集B，再将子集B的前50%去除得到最终输出
