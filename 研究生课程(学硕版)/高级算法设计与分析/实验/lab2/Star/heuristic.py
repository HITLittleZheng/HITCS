# lab2\Star\heuristic.py
import math

def h(p1, p2, type="h1"):
    if type == "h1":
        return h1(p1, p2)
    elif type == "h2":
        return h2(p1, p2)
    elif type == "h3":
        return h3(p1, p2)
    elif type == "h4":
        return h4(p1, p2)
   
    else:
        raise ValueError("Invalid heuristic type")


# Octile Distance (八方向标准距离) 
# 8方向移动
# 所有地形横纵代价=1，对角代价=sqrt(2)

def h1(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    diag = min(dx, dy)
    straight = max(dx, dy) - diag

    return math.sqrt(2) * diag + straight


# 曼哈顿距离
# 4方向移动
# 在8方向移动中使用会高估代价，可能导致A*找不到最优路径！
def h2(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return abs(x1 - x2) + abs(y1 - y2)


# 欧式距离
# 任意角度移动时的理论最短距离
# 在8方向网格中处于“可采纳(不超估)”的状态，但通常比h1扩展的节点多
def h3(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# 切比雪夫距离
# 适用于: 8方向移动 且 横纵/对角移动代价完全相同(比如都是1)的情况
def h4(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return max(abs(x1 - x2), abs(y1 - y2))



