# lab1/ConvexHull/enumerate.py
from math import atan2


def cross_product(A, B, P):
    """
    计算向量 AB 和 AP 的叉积


    > 0：P 在  AB 的左侧
    < 0：P 在  AB 的右侧
    = 0：P 与 A、B 共线
    """
    return (B[0] - A[0]) * (P[1] - A[1]) - (P[0] - A[0]) * (B[1] - A[1])


def distance(p, q):
    
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def in_triangle(A, B, C, P):
    """

    如果 P 和三角形的第三个顶点分别位于每条边的同一侧，
    则 P 在三角形内部。

    """

    if cross_product(A, B, C) == 0:
        return False

    return (
        cross_product(A, B, P) * cross_product(A, B, C) > 0 and
        cross_product(A, C, P) * cross_product(A, C, B) > 0 and
        cross_product(B, C, P) * cross_product(B, C, A) > 0
    )


def in_any_triangle(P, points):
    """
    
    如果 P 在任意一个三角形内部，则 P 一定不是凸包顶点。
    复杂度为 O(n^3)
    """
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                A, B, C = points[i], points[j], points[k]

                # P 本身不能作为三角形的顶点
                if P != A and P != B and P != C:
                    if in_triangle(A, B, C, P):
                        return True

    return False


def EnumerateConvexHull(points):
    """
    
    如果一个点位于其他三个点构成的某个三角形内部，
    则该点一定不是凸包顶点。

    
    return: 按极角排序后的凸包顶点列表
    """

    n = len(points)

    # 点数小于 3 时，所有点都属于凸包
    if n < 3:
        return points[:]

    points_copy = points[:]

    # flag[i] 表示第 i 个点是否可能是凸包顶点
    flag = [True] * n

    # 枚举每一个点，判断它是否在某个三角形内部，复杂度为 O(n^4)
    for i, p in enumerate(points_copy):
        if in_any_triangle(p, points_copy):
            flag[i] = False

    # 保留没有被三角形包含的点，即凸包候选点
    hull_points = [p for i, p in enumerate(points_copy) if flag[i]]

    # 如果凸包点数量小于 3，直接返回
    if len(hull_points) < 3:
        return hull_points

    # 先找到最左下角的点，作为极角排序的基准点
    hull_points.sort(key=lambda p: (p[1], p[0]))
    base = hull_points[0]

    # 对其他点按照相对于 base 的极角排序
    # 如果极角相同，则按照距离 base 的远近排序
    hull_points[1:] = sorted(
        hull_points[1:],
        key=lambda p: (
            atan2(p[1] - base[1], p[0] - base[0]),
            distance(p, base)
        )
    )

    return hull_points


if __name__ == '__main__':
    # 测试点集
    points = [[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]]

    # 计算凸包
    outputs = EnumerateConvexHull(points)

    # 输出凸包结果
    print(outputs)

    # 期望结果
    target = [[1, 1], [2, 0], [3, 3], [2, 4], [4, 2]]

    # 判断结果是否正确
    print(sorted(target) == sorted(outputs))