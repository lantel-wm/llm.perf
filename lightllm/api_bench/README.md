# lightllm

## lightllm server 环境配置

```shell
git clone https://github.com/ModelTC/lightllm.git && cd lightllm
pip install -r requirements.txt
pip install triton==2.0.0
python setup.py install
pip install triton==2.1.0
```

requirements.txt中torch==2.0.0，triton==2.1.0，但torch依赖triton==2.0.0。所以需要先安装triton==2.0.0使`python setup.py install`顺利执行，安装成功后再换回triton==2.1.0。

报错：FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'

解决方法：export PATH=$PATH:/sbin

## lightllm server 启动

```shell
# 需要修改--model-dir、--tp
python -m lightllm.server.api_server --model_dir /mnt/llm2/llm_perf/hf_models/llama-7b-hf --host 127.0.0.1 --port 8080 --tp 1 --max_total_token_num 150000 --tokenizer_mode fast
```

