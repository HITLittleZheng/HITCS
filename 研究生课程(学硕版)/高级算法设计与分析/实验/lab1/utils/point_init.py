# lab1/utils/point_init.py

import random


def createData(sizes=None, sample_method="uniform"):
    """
    生成指定数量的二维随机点数据集

     "uniform"：均匀分布
        "gauss"：高斯分布

    """

    # 避免使用可变默认参数
    if sizes is None:
        sizes = []

    datasets = []

    for size in sizes:

        points = []

        # 高斯分布的数据集使用同一个中心点
        if sample_method == "gauss":
            x_avg = random.randint(20, 80)
            y_avg = random.randint(20, 80)

        for _ in range(size):

            # 均匀分布
            if sample_method == "uniform":

                x = random.uniform(0, 100)
                y = random.uniform(0, 100)

            # 高斯分布
            elif sample_method == "gauss":

                # 不断重新采样，直到点落入合法区域
                while True:

                    # 生成高斯分布随机点
                    x = random.gauss(x_avg, 6)
                    y = random.gauss(y_avg, 6)

                    # 点合法则退出循环
                    if 0 <= x <= 100 and 0 <= y <= 100:
                        break

            else:
                raise ValueError(
                    "sample_method 只能是 'uniform' 或 'gauss'"
                )

            # 保存点
            points.append((x, y))

        datasets.append(points)

    return datasets


if __name__ == '__main__':

    sizes = [1000, 2000, 3000]

    datasets = createData(
        sizes,
        sample_method="gauss"
    )

    print(datasets)