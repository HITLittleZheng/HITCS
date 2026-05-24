
# lab3\setcover\lpSetCover.py
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, PULP_CBC_CMD
import random
import math

def lpSetCover(data, rounding_mode="random"):
  
    universe, subsets = data
    n_subsets = len(subsets)
   
    # 定义连续变量 x_i ∈ [0, 1]，代表子集 i 被选中的程度
    subset_vars = LpVariable.dicts('S', range(n_subsets), lowBound=0, upBound=1, cat='Continuous')

    problem = LpProblem('Set_Cover_LP', LpMinimize)
    # 目标函数：最小化选中的子集数量 
    problem += lpSum([subset_vars[i] for i in range(n_subsets)])

    # 构建倒排索引：记录每个元素属于哪些子集的索引
    # O(1) 时间查找元素所属子集，加速约束构建
    # 频率计算，统计每个元素被多少子集包含
    element_to_indices = {e: [] for e in universe}
    for idx, s in enumerate(subsets):
        for e in s:
            if e in element_to_indices:  
                element_to_indices[e].append(idx)

    # 极速构建约束：对于全集中的每个元素，包含它的所有子集变量之和 >= 1
    for e, indices in element_to_indices.items():
        if indices:  # 确保该元素至少在某个子集中
            problem += lpSum([subset_vars[i] for i in indices]) >= 1

    # 求解 LP 问题
    problem.solve(PULP_CBC_CMD(msg=False))

    # 将求解结果提取到普通列表中，断开与 PuLP 底层对象的耦合
    lp_values = [subset_vars[i].value() for i in range(n_subsets)]


    # 将分数解转化为确定的 0/1 解
    selected_subsets = []

    if rounding_mode == "random":
        # 简单随机舍入
        # 原理：以概率 x_i 选中子集 i
        # 单次运行无法保证覆盖所有元素
        for i in range(n_subsets):
            if random.uniform(0, 1) < lp_values[i]:
                selected_subsets.append(subsets[i])

    elif rounding_mode == "frequency":
        # 基于最大频率的确定性舍入 
        # 原理：根据集合覆盖的对偶拟合理论，定义最大频率 f 为：
        # 全集中，被最多子集包含的那个元素，它被 f 个子集包含。
        # 只要设定阈值 1/f，将所有 x_i >= 1/f 的子集选中，就能 100% 保证覆盖所有元素。
       
        max_freq = max((len(indices) for indices in element_to_indices.values()), default=1)
        threshold = 1.0 / max_freq
        
        for i in range(n_subsets):
            if lp_values[i] >= threshold:
                selected_subsets.append(subsets[i])

    elif rounding_mode == "random_improved":
        # 放大随机舍入 ---
        # 原理：重复执行多次随机舍入取并集。根据切尔诺夫界，遗漏概率呈指数级下降。
        # 计算重复次数：O(log(|U|))
        if len(universe) <= 1:
            random_times = 1
        else:
            c = 1 + math.log(4) / math.log(len(universe))
            random_times = math.ceil(c * math.log(len(universe)))
            
        
        for _ in range(random_times):
            for i in range(n_subsets):
                if random.uniform(0, 1) < lp_values[i]:
                    selected_subsets.append(subsets[i])
                    
    return selected_subsets

if __name__ == '__main__':
    universe = {1, 2, 3, 4, 5}
    subsets = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]
    data = universe, subsets
    
    print("Random 舍入:", lpSetCover(data, "random"))
    print("Frequency 舍入:", lpSetCover(data, "frequency"))
    print("Improved 舍入:", lpSetCover(data, "random_improved"))
