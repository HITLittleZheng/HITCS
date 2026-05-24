# lab1/utils/draw.py
import matplotlib.pyplot as plt
import argparse
try:
    
    from .point_init import createData
except ImportError:
    
    from point_init import createData


def draw_points_with_order(points, title="Points", save_file_path="./points.png", show=False):
    """
    绘制点集，并标出每个点的序号

    """

    plt.figure(figsize=(8, 8))

    # 提取所有点的 x 坐标和 y 坐标
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    # 绘制散点图
    plt.scatter(points_x, points_y, c="blue", marker="o")

    # 给每个点添加编号
    for i, (xi, yi) in enumerate(zip(points_x, points_y)):
        plt.text(xi + 0.05, yi + 0.05, f"{i}", fontsize=10)

    plt.title(title)
    plt.savefig(save_file_path)

    if show:
        plt.show()

    plt.close()


def draw_points_with_order_with_inner_point(
    points,
    inner_point,
    title="Points",
    save_file_path="./points.png",
    show=False
):
    """
    绘制点集，并额外标出一个内部点


    """

    plt.figure(figsize=(8, 8))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    # 绘制普通点
    plt.scatter(points_x, points_y, c="blue", marker="o")

    # 绘制内部点
    plt.scatter(inner_point[0], inner_point[1], c="green", marker="x")

    # 给普通点添加编号
    for i, (xi, yi) in enumerate(zip(points_x, points_y)):
        plt.text(xi + 0.05, yi + 0.05, f"{i}", fontsize=10)

    plt.title(title)
    plt.savefig(save_file_path)

    if show:
        plt.show()

    plt.close()


def draw_init_points(points, title="Points", save_file_path="./points.png", show=False):
    """
    绘制初始点集，不显示点的序号


    """

    plt.figure(figsize=(8, 8))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    # 绘制散点图
    plt.scatter(points_x, points_y, c="blue", marker="o")

    plt.title(title)
    plt.savefig(save_file_path)

    if show:
        plt.show()

    plt.close()


def draw_convex_hull_with_inner_point(
    points,
    convex_hull,
    inner_point,
    title="Convex Hull",
    save_file_path="./convex_hull.png",
    show=False
):
    """
    绘制点集、凸包以及一个内部点

    
    """

    plt.figure(figsize=(8, 8))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    # 绘制原始点
    plt.scatter(points_x, points_y, c="blue", marker="o")

    # 绘制凸包
    if convex_hull:
        closed_hull = convex_hull + [convex_hull[0]]
        convex_hull_x = [point[0] for point in closed_hull]
        convex_hull_y = [point[1] for point in closed_hull]
        plt.plot(convex_hull_x, convex_hull_y, c="red", marker="o")

    # 绘制内部点
    plt.scatter(inner_point[0], inner_point[1], c="green", marker="x")

    plt.title(title)
    plt.savefig(save_file_path)

    if show:
        plt.show()

    plt.close()


def draw_convex_hull(points, convex_hull, title="Convex Hull", save_file_path="./convex_hull.png", show=False):
    """
    绘制原始点集和凸包


    """

    plt.figure(figsize=(8, 8))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    # 绘制原始点
    plt.scatter(points_x, points_y, c="blue", marker="o")

    # 绘制凸包，自动连接首尾点形成闭合多边形
    if convex_hull:
        closed_hull = convex_hull + [convex_hull[0]]
        convex_hull_x = [point[0] for point in closed_hull]
        convex_hull_y = [point[1] for point in closed_hull]
        plt.plot(convex_hull_x, convex_hull_y, c="red", marker="o")

    plt.title(title)
    plt.savefig(save_file_path)

    if show:
        plt.show()

    plt.close()


def draw_performance_curve(sizes, times, algorithm_name, show=False):
    """
    绘制算法性能曲线


    """

    plt.figure(figsize=(8, 8))

    # 绘制运行时间随数据规模变化的折线图
    plt.plot(sizes, times, "-o")

    plt.xlabel("Data Size")
    plt.ylabel("Running Time (ms)")
    plt.title("Performance Curve for {}".format(algorithm_name))

    # 保存性能曲线图片
    plt.savefig("Performance_Curve_for_{}.png".format(algorithm_name))

    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="绘制点集和凸包的工具脚本")
    
    # 1. 添加 sizes 参数
    # nargs='+' 表示至少传入1个或多个参数，将它们组合成一个列表
    # type=int 表示传入的参数会被转换为整数
    parser.add_argument(
        '--sizes', 
        nargs='+', 
        type=int, 
        default=[1000, 2000, 3000], 
        help="数据规模列表，例如：1000 2000 3000 (默认：1000 2000 3000)"
    )
    
    # 2. 添加 sample_method 参数
    # default='uniform' 保证了你不传该参数时，默认值为 'uniform'
    parser.add_argument(
        '--sample_method', 
        type=str, 
        default='uniform', 
        choices=['uniform', 'gauss'],  # 可选：限制只能输入某些合法的字符串
        help="采样方法，例如：uniform, gaussian (默认：uniform)"
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 从解析结果中提取变量
    sizes = args.sizes
    sample_method = args.sample_method
    

    # 生成随机点集（将 sample_method 传递给 createData）
    datasets = createData(sizes=sizes, sample_method=sample_method)

    # 绘制每个数据集的初始点分布图
    for dataset in datasets:
        draw_init_points(
            dataset,
            title="Points_Init_{}_{}".format(len(dataset), sample_method), # 标题加上采样方式更清晰
            save_file_path="./points_init_{}_{}.png".format(len(dataset), sample_method),
            show=True
        )