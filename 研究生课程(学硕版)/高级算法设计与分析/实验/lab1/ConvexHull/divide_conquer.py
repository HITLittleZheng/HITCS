# lab1/ConvexHull/divide_conquer.py

from math import atan2

from .grahamScan import GrahamScanConvexHull


def angle(p0, p1):
    """
    极角
    """
    return atan2(
        p1[1] - p0[1],
        p1[0] - p0[0]
    )


def get_inner_point(points, hull):
    """
    找一个不在凸包上的内点
    """
    hull_set = set(map(tuple, hull))
    for p in points:
        if tuple(p) not in hull_set:
            return p
    return None


def circular_slice_ccw(hull, start, end):
    """
    逆时针截取
    """
    n = len(hull)
    result = []
    i = start
    while True:
        result.append(hull[i])
        if i == end:
            break
        i = (i + 1) % n
    return result


def circular_slice_cw(hull, start, end):
    """
    顺时针截取
    """
    n = len(hull)
    result = []
    i = start
    while True:
        result.append(hull[i])
        if i == end:
            break
        i = (i - 1 + n) % n
    return result


def merge(left_hull, right_hull, inner_point):
    """
    合并左右凸包 
    """
    # 如果没有内点，说明左凸包的点都在其边界上，直接合并所有点
    if inner_point is None:
        merged = left_hull + right_hull
        return GrahamScanConvexHull(merged)

    # 在 CH(Q_R) 中找与 p 的极角最大和最小顶点 u 和 v;
    min_angle = float("inf")
    max_angle = float("-inf")

    u = 0
    v = 0

    for i in range(len(right_hull)):
        a = angle(inner_point, right_hull[i])

        if a < min_angle:
            min_angle = a
            v = i

        if a > max_angle:
            max_angle = a
            u = i

    
    # 按逆时针排列的 CH(Q_L) 的所有顶点,
    seq1 = left_hull[:]

    # 按逆时针排列的 CH(Q_R) 从 u 到 v 的顶点,
    seq2 = circular_slice_ccw(right_hull, u, v)

    #  按顺时针排列的 CH(Q_R) 从 u 到 v 的顶点;
    seq3 = circular_slice_cw(right_hull, u, v)

    # 4. 合并上述三个序列;
    merged = seq1 + seq2 + seq3

    # 5. 在合并的序列上应用 Graham-Scan.
    return GrahamScanConvexHull(merged)



def DivideConvexHull(points):

    points = list(map(list, set(map(tuple, points))))

    n = len(points)

    if n <= 5:
        return GrahamScanConvexHull(points)

    sorted_points = sorted(points) #时间复杂度 O(n log n)

    mid = n // 2

    left_points = sorted_points[:mid]
    right_points = sorted_points[mid:]

    left_hull = DivideConvexHull(left_points)
    right_hull = DivideConvexHull(right_points)

    inner_point = get_inner_point(left_points, left_hull)

    merged_points = merge(
        left_hull,
        right_hull,
        inner_point
    )

    return merged_points


if __name__ == '__main__':
    points = [
        [1, 1],
        [2, 2],
        [2, 0],
        [2, 4],
        [3, 3],
        [4, 2]
    ]

    outputs = DivideConvexHull(points)
    print(outputs)

    target = [
        [1, 1],
        [2, 0],
        [4, 2],
        [2, 4]
    ]

    print(sorted(outputs) == sorted(target))
