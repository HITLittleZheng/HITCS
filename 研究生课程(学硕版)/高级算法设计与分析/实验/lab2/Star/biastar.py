# lab2\Star\biastar.py
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


def biastar(draw, grid, start, end, type="h1"):
    """
    双向 A* 寻路算法 (尾节点地形代价丢失)
    """
    idx = 0
    
    open_set = [[], []]
    count = 0
    heapq.heappush(open_set[0], (0, count, start))
    count += 1
    heapq.heappush(open_set[1], (0, count, end))
    
    came_from = [{}, {}]
    
    g_score = [
        {spot: float("inf") for row in grid for spot in row},
        {spot: float("inf") for row in grid for spot in row}
    ]
    
    # 正向起步价依然是 0，反向起步价直接设为终点的地形代价！
    g_score[0][start] = 0
    g_score[1][end] = EXTRA_COST[end.cost_idx]
    
    f_score = [
        {spot: float("inf") for row in grid for spot in row},
        {spot: float("inf") for row in grid for spot in row}
    ]
    #反向的 f_score 也要加上这个起步价
    f_score[0][start] = h(start.get_pos(), end.get_pos(), type)
    f_score[1][end] = EXTRA_COST[end.cost_idx] + h(end.get_pos(), start.get_pos(), type)
    
    open_set_hash = [{start}, {end}]
    target_nodes = [end, start]

    pause = 0

    while len(open_set[0]) > 0 and len(open_set[1]) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                bye()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause ^= 1
                elif event.key == pygame.K_q:
                    bye()
        
        if pause: 
            continue

        current_f, current_cnt, current = heapq.heappop(open_set[idx])
        open_set_hash[idx].remove(current)

        # 相遇条件,另一个方向的搜索已经访问过当前节点
        if g_score[idx ^ 1][current] != float("inf"):
            
            
            # 正向 g_score 不含起点代价，反向 g_score 含了终点代价。
            # 拼接时，相遇点 current 的代价被包含了两次，必须减去一次！
            optimal_cost = g_score[0][current] + g_score[1][current] - EXTRA_COST[current.cost_idx]
            
            print(f"Path Found! Optimal Cost: {optimal_cost:.2f} (Meet at: {current.row},{current.col})")
            
            reconstruct_path(came_from[0], current, draw, back=False)
            reconstruct_path(came_from[1], current, draw, back=True)
            draw()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[idx][current] + current.get_move_cost(neighbor)

            if temp_g_score < g_score[idx][neighbor]:
                came_from[idx][neighbor] = current
                g_score[idx][neighbor] = temp_g_score
                f_score[idx][neighbor] = temp_g_score + h(neighbor.get_pos(), target_nodes[idx].get_pos(), type)

                if neighbor not in open_set_hash[idx]:
                    count += 1
                    heapq.heappush(open_set[idx], (f_score[idx][neighbor], count, neighbor))
                    open_set_hash[idx].add(neighbor)
                    neighbor.make_open()

        time.sleep(0.005)

        if current != start and current != end:
            current.make_closed()
            
        draw()

        if len(open_set[idx]) > len(open_set[idx ^ 1]):
            idx ^= 1

    return False
