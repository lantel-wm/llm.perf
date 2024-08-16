# lightllm/api_bench

启动ppl_llm_server，用法见api_bench_tools/README.md

## lightllm install

```shell
git clone https://github.com/ModelTC/lightllm.git && cd lightllm
pip install -r requirements.txt
pip install triton==2.0.0
python setup.py install
pip install triton==2.1.0
```

requirements.txt中torch==2.0.0，triton==2.1.0，但torch依赖triton==2.0.0。所以需要先安装triton==2.0.0使`python setup.py install`顺利执行，安装成功后再换回triton==2.1.0。

## server error

报错：FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'

解决方法：export PATH=$PATH:/sbin

