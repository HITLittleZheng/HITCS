# lab1/ConvexHull/grahamScan.py

from math import atan2


def cross_product(A, B, C):
    """

    return:
        >0 左转
        <0 右转
        =0 共线
    """
    return (B[0] - A[0]) * (C[1] - A[1]) - \
           (B[1] - A[1]) * (C[0] - A[0])


def distance(p, q):
   
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def GrahamScanConvexHull(points):
    """
    Graham Scan 求凸包
    """

    # 去重 时间复杂度 O(n)
    points = list(map(list, set(map(tuple, points))))

    n = len(points)

    if n <= 2:
        return points

    # 找最左下点 时间复杂度 O(n log n)
    points.sort(key=lambda p: (p[1], p[0]))

    base = points[0]

    # 极角排序 时间复杂度 O(n log n)
    points[1:] = sorted(
        points[1:],
        key=lambda p: (
            atan2(p[1] - base[1], p[0] - base[0]),
            distance(base, p)
        )
    )

    #  找到共线部分的分界线
    r = n - 1

    while r > 0 and cross_product(base, points[-1], points[r]) == 0:
        r -= 1
        
    # 反转共线区间

    l, h = r + 1, n - 1

    while l < h:
        points[l], points[h] = points[h], points[l]
        l += 1
        h -= 1

    # 栈扫描
    stack = [points[0], points[1]]

    for i in range(2, n):

        while len(stack) >= 2 and \
                cross_product(stack[-2], stack[-1], points[i]) <= 0:
            stack.pop()

        stack.append(points[i])

    return stack