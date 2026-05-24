
# lab2\Star\astar.py
import pygame
import heapq
import math
import time

try:
    from ..utils.draw import EXTRA_COST, reconstruct_path, bye
except ImportError:
    from utils.draw import EXTRA_COST, reconstruct_path, bye

try:
    from .heuristic import h
except ImportError:
    from heuristic import h


def astar(draw, grid, start, end, type="h1"):
 

    # open_set 是优先队列，存放待搜索节点
    
    # count 用于避免 f_score 相同时 heapq 比较 Spot 对象时报错
    open_set = []
    count = 0
    heapq.heappush(open_set, (0, count, start))

    # came_from 记录路径：came_from[当前节点] = 前一个节点
    came_from = {}

    # g_score 表示从起点到某节点的实际代价
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    # f_score = g_score + h_score
    # 表示从起点经过当前节点到终点的估计总代价
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos(), type)

    # 用集合快速判断某节点是否已经在 open_set 中
    open_set_hash = {start}

    pause = 0

    while open_set:
        # 处理 pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                bye()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause ^= 1
                elif event.key == pygame.K_q:
                    bye()

        # 暂停时不继续搜索
        if pause:
            continue

        # 取出当前 f_score 最小的节点
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)


        # 如果到达终点，回溯路径并返回成功
        if current == end:
            print(f"Total Cost: {g_score[current]:.2f}")
            reconstruct_path(came_from, end, draw)
            return True

        # 遍历当前节点的所有邻居
        for neighbor in current.neighbors:
            
            # 计算从起点经过 current 到 neighbor 的 g_score
            temp_g_score = g_score[current] + current.get_move_cost(neighbor)

            # 如果找到一条到 neighbor 更短的路径，则更新记录
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(
                    neighbor.get_pos(),
                    end.get_pos(),
                    type
                )

                # 如果 neighbor 不在 open_set 中，则加入搜索队列
                if neighbor not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

    
        time.sleep(0.005)

        # 起点不标记为 closed
        if current != start:
            current.make_closed()

        # 刷新画面
        draw()

    # open_set 为空仍未到达终点，说明无路径
    return False
