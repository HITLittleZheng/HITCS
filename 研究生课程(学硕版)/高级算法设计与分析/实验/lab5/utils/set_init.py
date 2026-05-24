# lab5\utils\set_init.py
import random
from typing import List, Tuple

def createData(sizes: List[int], sample_method: str = 'uniform') -> List[List[int]]:
    """
    生成指定数量的测试数据集
    :param sizes: 每个数据集的大小列表
    :param sample_method: 采样方法 ('uniform' 或 'normal')
    :return: 生成的数据集列表，每个数据集是一个整数列表
    """
    datasets = []
    for size in sizes:
        if sample_method == 'uniform':
            data = [random.randint(1, 100000) for _ in range(size)]
        elif sample_method == 'normal':
            # 改进的正态分布生成
            data = []
            while len(data) < size:
                val = int(random.gauss(50000, 20000))  # 均值50000，标准差20000
                if 1 <= val <= 100000:  # 确保值在合理范围内
                    data.append(val)
        else:
            raise ValueError(f"未知的采样方法: {sample_method}")
            
        random.shuffle(data)
        datasets.append(data)
        
    return datasets
