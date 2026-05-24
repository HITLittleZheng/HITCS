# lab2\Star\biastar_2.py
from utils import *
import heapq
import math
import time  # 移到外面
try:
    from .heuristic import h
except ImportError:
    from heuristic import h
def biastar(draw, grid, start, end):
    idx = 0
    open_set = [[], []]
    
    # 【修复】必须加入 count，防止比较 Spot 对象报错
    count = 0
    heapq.heappush(open_set[0], (0, count, start))
    count += 1
    heapq.heappush(open_set[1], (0, count, end))
    
    came_from = [{}, {}]
    g_score = [{spot: float("inf") for row in grid for spot in row},
               {spot: float("inf") for row in grid for spot in row}]
    g_score[0][start] = g_score[1][end] = 0
    
    f_score = [{spot: float("inf") for row in grid for spot in row},
               {spot: float("inf") for row in grid for spot in row}]
    f_score[0][start] = h(start.get_pos(), end.get_pos())
    f_score[1][end] = h(end.get_pos(), start.get_pos())
    
    open_set_hash = [{start}, {end}]
    ends = [end, start]
    
    best_total_cost = float("inf")
    meeting_point = None
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
        if pause: continue
        
        current_f, current_cnt, current = heapq.heappop(open_set[idx])
        open_set_hash[idx].remove(current)
        
        # 【修复逻辑】如果当前节点被另一侧遍历过（即在另一侧的 came_from 中）
        if current in came_from[idx ^ 1]:
            # 【关键修复】真实总代价必须是 g_score 相加，绝对不能是 f_score 相加！
            current_total_cost = g_score[0][current] + g_score[1][current]
            if current_total_cost < best_total_cost:
                best_total_cost = current_total_cost
                meeting_point = current
        
        # 【高性能修复】提前终止条件
        # 因为堆顶的 f 值就是当前队列最小的 f 值，所以直接 peek 堆顶即可，O(1)复杂度！
        if meeting_point is not None and len(open_set[0]) > 0 and len(open_set[1]) > 0:
            min_f_0 = open_set[0][0][0] # 正向队列最小 f
            min_f_1 = open_set[1][0][0] # 反向队列最小 f
            
            # 如果 当前找到的最优真实代价 < min(f正向, f反向)
            # 说明无论怎么搜，都不可能找到比这个更好的路径了，直接终止！
            if best_total_cost <= min(min_f_0, min_f_1):
                print(f"Optimal termination! Cost: {best_total_cost:.2f}")
                reconstruct_path(came_from[0], meeting_point, draw)
                reconstruct_path(came_from[1], meeting_point, draw, back=True)
                return True

        for neighbor in current.neighbors:
            # 【修复】正确计算加权8方向移动代价
            temp_g_score = g_score[idx][current] + current.get_move_cost(neighbor)

            if temp_g_score < g_score[idx][neighbor]:
                came_from[idx][neighbor] = current
                g_score[idx][neighbor] = temp_g_score
                f_score[idx][neighbor] = temp_g_score + h(neighbor.get_pos(), ends[idx].get_pos())
                
                if neighbor not in open_set_hash[idx]:
                    count += 1
                    heapq.heappush(open_set[idx], (f_score[idx][neighbor], count, neighbor))
                    open_set_hash[idx].add(neighbor)
                    neighbor.make_open()

        time.sleep(0.005)
        if current != start and current != end:
            current.make_closed()
            
        idx ^= 1 # 严格交替
        draw()

    # 如果循环结束是因为某一边空了，但之前找到过相遇点（虽然没触发提前终止），依然返回成功
    if meeting_point is not None:
        print(f"Exhausted search. Cost: {best_total_cost:.2f}")
        reconstruct_path(came_from[0], meeting_point, draw)
        reconstruct_path(came_from[1], meeting_point, draw, back=True)
        return True

    return False
