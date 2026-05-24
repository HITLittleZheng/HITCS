# lab5\utils\utils.py
from typing import Callable, List, Any, Optional
import time
import matplotlib.pyplot as plt
import random

def generate_test_data(sizes: List[int]) -> List[List[int]]:
    """
    生成指定数量的测试数据
    :param sizes: 数据规模大小列表
    :return: 生成的数据集列表
    """
    datasets = []
    for size in sizes:
        data = [random.randint(1, 100000) for _ in range(size)]
        datasets.append(data)
    return datasets

def measure_algorithm_time(algorithm: Callable[[List[Any]], Any], data: List[Any]) -> float:
    """
    测量算法运行时间
    :param algorithm: 要测试的算法函数
    :param data: 输入数据
    :return: 运行时间（毫秒）
    """
    start_time = time.perf_counter()
    try:
        algorithm(data)
    except Exception as e:
        print(f"Algorithm execution failed: {e}")
        return float('inf')
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000

def plot_performance_curve(algorithm: Callable, data_sets: List[List[Any]], 
                         show: bool = True, save_path: Optional[str] = None) -> None:
    """
    绘制算法性能曲线
    :param algorithm: 要测试的算法
    :param data_sets: 不同规模的数据集
    :param show: 是否显示图像
    :param save_path: 保存路径（可选）
    """
    if not data_sets:
        print("No data sets provided")
        return
        
    sizes = [len(data) for data in data_sets]
    times = [measure_algorithm_time(algorithm, data) for data in data_sets]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, '-o', label='Running Time')
    plt.xlabel("Data Size")
    plt.ylabel("Running Time (ms)")
    plt.title(f"Performance Curve for {algorithm.__name__}")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
