# main.py
import argparse
import time
import random

from ConvexHull import (
    EnumerateConvexHull,
    GrahamScanConvexHull,
    DivideConvexHull
)
from utils import createData, draw_convex_hull, draw_performance_curve


# 设置随机种子，保证每次生成的数据一致，便于复现实验结果
random.seed(10043)


def get_args():
    """
    获取命令行参数

    :return: 命令行参数对象
    """

    parser = argparse.ArgumentParser(description="convex hull algorithm")

    # 选择凸包算法
    parser.add_argument(
        "--name",
        type=str,
        default="divide_conquer",
        choices=["enumerate", "grahamScan", "divide_conquer"],
        help="算法选择：enumerate、grahamScan 或 divide_conquer"
    )

    # 手动指定多个数据规模，例如：1000,2000,3000
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,2000,3000",
        help="数据规模，多个规模之间用逗号分隔，例如：1000,2000,3000"
    )

    # 最大数据规模
    # 如果 max_size 不为 -1，则根据 max_size 和 group_size 自动生成 sizes
    parser.add_argument(
        "--max_size",
        type=int,
        default=-1,
        help="最大数据规模；若不为 -1，则自动生成测试规模"
    )

    # 将 0 到 max_size 分成多少组
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="数据规模分组数量"
    )

    # 是否显示凸包图像
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="是否显示凸包图像"
    )

    # 随机点生成方式
    parser.add_argument(
        "--sample_method",
        type=str,
        default="gauss",
        choices=["uniform", "gauss"],
        help="采样方法：uniform 或 gauss"
    )
    # 是否绘制凸包图
    parser.add_argument(
        '--draw_hull',
        action='store_true',
        default=False,
        help='是否绘制凸包图'
)

    return parser.parse_args()


def parse_sizes(args):
    """
    根据命令行参数解析数据规模

    :param args: 命令行参数对象
    :return: 数据规模列表
    """

    # 如果指定了 max_size，则自动生成数据规模
    if args.max_size != -1:
        if args.group_size <= 0:
            raise ValueError("group_size 必须大于 0")

        step = args.max_size // args.group_size

        if step <= 0:
            raise ValueError("max_size 必须大于或等于 group_size")

        # 例如 max_size=3000, group_size=3
        # 得到 [1000, 2000, 3000]
        return list(range(0, args.max_size + 1, step))[1:]

    # 否则使用 --sizes 手动指定的数据规模
    return list(map(int, args.sizes.split(",")))


def get_algorithm(algo_name):
    """
    根据算法名称返回对应的凸包算法函数

    :param algo_name: 算法名称
    :return: 对应的算法函数
    """

    if algo_name == "enumerate":
        return EnumerateConvexHull
    elif algo_name == "grahamScan":
        return GrahamScanConvexHull
    elif algo_name == "divide_conquer":
        return DivideConvexHull
    else:
        raise ValueError("algorithm not implemented {}".format(algo_name))


def run_algorithm(algorithm, dataset):
    """
    运行指定凸包算法，并统计运行时间

    :param algorithm: 凸包算法函数
    :param dataset: 输入点集
    :return: 算法运行时间和凸包结果
    """

    # 复制数据，避免算法内部排序时修改原始数据
    data = dataset[:]

    start_time = time.time()
    convex_hull = algorithm(data)
    end_time = time.time()

    # 返回运行时间，单位为毫秒
    return (end_time - start_time) * 1000, convex_hull


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()

    # 算法名称
    algo_name = args.name

    # 获取测试数据规模
    data_sizes = parse_sizes(args)

    # 根据数据规模生成随机点集
    datasets = createData(data_sizes, sample_method=args.sample_method)

    # 打印实验设置
    print("algorithm: {}".format(algo_name))
    print("sizes: {}".format(data_sizes))
    print("sample method: {}".format(args.sample_method))

    # 根据算法名称选择算法函数
    algorithm = get_algorithm(algo_name)

    # 保存每组数据的运行时间和实际数据规模
    run_times = []
    real_sizes = []

    for dataset in datasets:
        # 运行算法并统计时间
        run_time, convex_hull = run_algorithm(algorithm, dataset)

        # 保存运行时间和数据规模
        run_times.append(run_time)
        real_sizes.append(len(dataset))

        # 绘制当前数据集的凸包图像
        # draw_convex_hull(
        #     dataset,
        #     convex_hull,
        #     title="Convex Hull_{}".format(len(dataset)),
        #     save_file_path="./convex_hull_{}.png".format(len(dataset)),
        #     show=args.show
        # )
        if args.draw_hull:
            draw_convex_hull(
                dataset,
                convex_hull,
                title="Convex Hull_{}".format(len(dataset)),
                save_file_path="./convex_hull_{}.png".format(len(dataset)),
                show=args.show
            )

    # 绘制性能曲线
    draw_performance_curve(real_sizes, run_times, algo_name)