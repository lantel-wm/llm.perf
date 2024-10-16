import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, List

# 获取默认颜色
default_colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
# default_colors = ['#616c8c', '#568c87', '#b2d59b', '#f2de79', '#d95f18']

# 定义BenchResult类
class BenchResult:
    def __init__(
        self, 
        result_path: str,
        model_name: str,
        device: str,
        backend: str,
        tp_size: int,
        ep_size: Optional[int],
        dataset: str,
        use_system_prompt: bool,
    ): 
        self.result_path = result_path
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.dataset = dataset
        self.use_system_prompt = use_system_prompt
        
        # 生成tag
        self.tag = f"{model_name}_{dataset}{'-system-prompt' if use_system_prompt else ''}_tp{tp_size}"
        if ep_size is not None:
            self.tag += f"_ep{ep_size}"
        self.tag += f"_{device}_{backend}"
        
        # 读取csv文件
        self.df = pd.read_csv(self.result_path)
    
    # 重载 [] 运算符
    def __getitem__(self, key: str):
        if key not in self.df.columns:
            raise KeyError(f"Key {key} not found in dataframe")
        return np.array(self.df[key])
    
 
# 绘制单条柱状图
def bar1(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    fig, ax = plt.subplots(figsize=(6, 5))
    res = result_list[0]
    x = np.arange(len(res['num_clients']))
    xticks = [str(int(x)) for x in res['num_clients']]
    y = res[metric]
    
    # 绘制柱状图
    ax.bar(x, y, width=0.5, color=default_colors[0])
    ax.set_xticks(x, xticks)
    # 添加数值标签
    for a, b in zip(x, y):
        ax.text(a, b, f'{b:.2f}', ha='center', va='bottom', color=default_colors[0], fontsize=8)

    ax.set_xlabel('number of clients')
    ax.set_ylabel(metric)
    ax.set_title(f"{res.tag}")
    
    # 保存图片
    image_save_path = os.path.join(image_save_dir, f"{res.tag}_{metric}.png")
    fig.savefig(image_save_path, bbox_inches='tight', dpi=300)
    print(f'Figure saved to {image_save_path}')


# 绘制单条折线图
def line1(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    fig, ax = plt.subplots(figsize=(6, 5))
    res = result_list[0]
    x = np.arange(len(res['num_clients']))
    xticks = [str(int(x)) for x in res['num_clients']]
    y = res[metric]
    
    # 绘制折线图
    ax.grid()
    ax.plot(x, y, color=default_colors[0])
    ax.set_xticks(x, xticks)
    # 添加数值标签
    for a, b in zip(x, y):
        ax.text(a, b, f'{b:.2f}', ha='center', va='bottom', color=default_colors[0], fontsize=8)

    ax.set_xlabel('number of clients')
    ax.set_ylabel(metric)
    ax.set_title(f"{res.tag}")
    
    # 保存图片
    image_save_path = os.path.join(image_save_dir, f"{res.tag}_{metric}.png")
    fig.savefig(image_save_path, bbox_inches='tight', dpi=300)
    print(f'Figure saved to {image_save_path}')


def find_different_attribute(
    result_list: List[BenchResult],
    compare_attrs: List[str] = ['device', 'backend', 'tp_size', 'ep_size', 'dataset', 'use_system_prompt'],
):
    """ 查找不同的属性

    Args:
        result_list (List[BenchResult]): benchmark result列表
        compare_attrs (List[str], optional): 需要对比的属性. Defaults to ['device', 'backend', 'tp_size', 'ep_size', 'dataset', 'use_system_prompt'].

    Raises:
        Exception: 如果所有benchmark配置都相同, 则抛出异常
        Exception: 如果benchmark配置不同之处大于1, 则抛出异常

    Returns:
        Tuple[str, List[Any]]: 不同属性名和对应的值列表
    """
    dicts = [vars(res) for res in result_list]
    keys = dicts[0].keys()
    
    diffs = {}
    for k in keys:
        if k not in compare_attrs:
            continue
        values = [dict[k] for dict in dicts]
        if not all(v == values[0] for v in values):
            diffs[k] = values
            
    if len(diffs) == 0:
        raise Exception("All benchmark configurations are the same.")
    elif len(diffs) >= 2:
        raise Exception("More than one benchmark configuration is different.")
    else:
        return list(diffs.keys())[0], diffs[list(diffs.keys())[0]]
    
    
def strip_different_attribute(
    result_list: List[BenchResult],
    compare_attrs: List[str] = ['model_name', 'dataset', 'use_system_prompt', 'tp_size', 'ep_size', 'device', 'backend'],
):
    """ 去掉不同的属性, 生成tag

    Args:
        result_list (List[BenchResult]): benchmark result列表
        compare_attrs (List[str], optional): 需要对比的属性. Defaults to ['model_name', 'dataset', 'use_system_prompt', 'tp_size', 'ep_size', 'device', 'backend'].

    Returns:
        str: tag
    """
    diff_attr, _ = find_different_attribute(result_list)
    result_dict = vars(result_list[0])
    
    tag = ""
    for attr in compare_attrs:
        if attr == diff_attr:
            continue
        value = result_dict[attr]
        match attr:
            case 'model_name' | 'device' | 'dataset' | 'backend':
                tag += f"{value}_"
            case 'tp_size':
                tag += f"tp{value}_"
            case 'ep_size':
                tag += f"ep{value}_" if value is not None else ""
            case 'use_system_prompt':
                tag += "sysprompt_" if value else ""
           
    return tag[:-1]


# 绘制双条柱状图
def bar2(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    fig, ax = plt.subplots(figsize=(6, 5))
    res0, res1 = result_list
    x = np.arange(len(res0['num_clients']))
    xticks = [str(int(x)) for x in res0['num_clients']]
    y0 = res0[metric]
    y1 = res1[metric]
    width = 0.3
    
    # 查询不同的属性，作为图例
    _, labels = find_different_attribute(result_list)
    ax.bar(x - width / 2, y0, width, color=default_colors[0], label=labels[0])
    ax.bar(x + width / 2, y1, width, color=default_colors[1], label=labels[1])
    ax.legend(loc='upper left')
    ax.set_xticks(x, xticks)
    for a, b in zip(x, y0):
        ax.text(a - width / 2, b, f'{b:.2f}', ha='center', va='bottom', color=default_colors[0], fontsize=8)
    for a, b in zip(x, y1):
        ax.text(a + width / 2, b, f'{b:.2f}', ha='center', va='bottom', color=default_colors[1], fontsize=8)

    ax.set_xlabel('number of clients')
    ax.set_ylabel(metric)
    
    # 查询相同的属性，作为标题
    title = strip_different_attribute(result_list)
    ax.set_title(title)
    
    # 保存图片
    image_save_path = os.path.join(image_save_dir, f"{title}_{metric}.png")
    fig.savefig(image_save_path, bbox_inches='tight', dpi=300)
    print(f'Figure saved to {image_save_path}')

# 绘制双条折线图
def line2(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    fig, ax = plt.subplots(figsize=(6, 5))
    res0, res1 = result_list
    x = np.arange(len(res0['num_clients']))
    xticks = [str(int(x)) for x in res0['num_clients']]
    y0 = res0[metric]
    y1 = res1[metric]
    
    # 查询不同的属性，作为图例
    _, labels = find_different_attribute(result_list)
    ax.plot(x, y0, color=default_colors[0], label=labels[0])
    ax.plot(x, y1, color=default_colors[1], label=labels[1])
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xticks(x, xticks)
    for a, b in zip(x, y0):
        ax.text(a, b, f'{b:.2f}', ha='center', va='bottom', color=default_colors[0], fontsize=8)
    for a, b in zip(x, y1):
        ax.text(a, b, f'{b:.2f}', ha='center', va='top', color=default_colors[1], fontsize=8)

    ax.set_xlabel('number of clients')
    ax.set_ylabel(metric)
    
    # 查询相同的属性，作为标题
    title = strip_different_attribute(result_list)
    ax.set_title(title)
    
    # 保存图片
    image_save_path = os.path.join(image_save_dir, f"{title}_{metric}.png")
    fig.savefig(image_save_path, bbox_inches='tight', dpi=300)
    print(f'Figure saved to {image_save_path}')

# 绘制三条柱状图
def bar3(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    raise NotImplementedError

# 绘制三条折线图
def line3(
    result_list: List[BenchResult],
    metric: str,
    image_save_dir: str = './image',
):
    raise NotImplementedError
    
# 绘制单条结果
def plot_single(
    result_list: List[BenchResult],
    metric_list: List[str],
    image_save_dir: str = './image',
):
    for metric in metric_list:
        if metric.endswith('ttft') or metric.endswith('tpot') or metric.endswith('e2e') or metric.endswith('itl'):
            line1(result_list, metric, image_save_dir)
        else:
            bar1(result_list, metric, image_save_dir)


# 绘制双条结果
def plot_double(
    result_list: List[BenchResult],
    metric_list: List[str],
    image_save_dir: str = './image',
):
    for metric in metric_list:
        if metric.endswith('ttft') or metric.endswith('tpot') or metric.endswith('e2e') or metric.endswith('itl'):
            line2(result_list, metric, image_save_dir)
        else:
            bar2(result_list, metric, image_save_dir)


# 绘制三条结果
def plot_triple(
    result_list: List[BenchResult],
    metric_list: List[str],
    image_save_dir: str = './image',
):
    for metric in metric_list:
        if metric.endswith('ttft') or metric.endswith('tpot') or metric.endswith('e2e') or metric.endswith('itl'):
            line3(result_list, metric, image_save_dir)
        else:
            bar3(result_list, metric, image_save_dir)

    
# 绘制结果
def plot(
    result_list: List[BenchResult],
    metric_list: List[str],
    image_save_dir: str = './image',
):
    match len(result_list):
        case 1:
            plot_single(result_list, metric_list, image_save_dir)
        case 2:
            plot_double(result_list, metric_list, image_save_dir)
        case 3:
            plot_triple(result_list, metric_list, image_save_dir)
        case _:
            raise ValueError("Only support 1, 2 or 3 result")
    
    
if __name__ == '__main__':
    # 需要保证所有 bench result 的 model_name, device, backend, tp_size, ep_size, dataset, use_system_prompt 中有且仅有一项不同
    res0 = BenchResult(
        result_path='../../sglang/api_bench/result/benchmark_result_num_client_h200_20240928/benchmark_sglang_sharegpt-system-prompt_llama2-70b-tp4-fp16_result.csv',
        model_name='llama2-70b',
        device='H200',
        backend='sglang',
        tp_size=4,
        ep_size=None,
        dataset='sharegpt',
        use_system_prompt=True,
    )
    
    res1 = BenchResult(
        result_path='../../sglang/api_bench/result/benchmark_result_num_client_h800_20240929/benchmark_sglang_sharegpt-system-prompt_llama2-70b-tp4-fp16_result.csv',
        model_name='llama2-70b',
        device='H800',
        backend='sglang',
        tp_size=4,
        ep_size=None,
        dataset='sharegpt',
        use_system_prompt=True,
    )

    # 可用的metric: 
    # completed, success_rate, qps, 
    # total_inlen, total_outlen, avg_inlen, avg_outlen, max_inlen, max_outlen, 
    # o_tps, io_tps, 
    # min_ttft, max_ttft, mean_ttft, median_ttft, std_ttft, p90_ttft, p99_ttft, 
    # min_tpot, max_tpot, mean_tpot, median_tpot, std_tpot, p90_tpot, p99_tpot, 
    # min_e2e, max_e2e, mean_e2e, median_e2e, std_e2e, p90_e2e, p99_e2e, 
    # min_itl, max_itl, mean_itl, median_itl, std_itl, p90_itl, p99_itl
    
    plot(
        result_list=[res0, res1], 
        metric_list=['p99_ttft', 'mean_ttft', 'p99_tpot', 'mean_tpot', 'qps', 'o_tps'],
    )