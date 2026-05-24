import random

def createData(sizes=[], sample_method="uniform_two_dimension", max_size=20):
    datasets = []
    for size in sizes:
        data = generate_instance(size, size, sample_method=sample_method, max_size=max_size)
        datasets.append(data)
    return datasets

def generate_instance(num_points, num_sets, sample_method="uniform_two_dimension", max_size=20):
    points = []
    
    if sample_method == "gauss":
        point_set = set()
        while len(point_set) < num_points:
            x_avg = random.randint(20, 80)
            y_avg = random.randint(20, 80)
            x = max(0.0, min(100.0, random.gauss(x_avg, 10)))
            y = max(0.0, min(100.0, random.gauss(y_avg, 10)))
            point_set.add((x, y))
        points = list(point_set)
        
    elif sample_method == "uniform_two_dimension":
        point_set = set()
        while len(point_set) < num_points:
            x = random.uniform(0, 100)  
            y = random.uniform(0, 100)
            point_set.add((x, y))
        points = list(point_set)
        
    elif sample_method == "range_one_dimension":
        points = list(range(num_points))
        
    else:
        raise ValueError(f"不支持的采样方法: {sample_method}")

    universe = set(points)
    assert len(universe) == num_points, "生成的点存在重复，无法达到 num_points 要求"
    
    # 根据要求生成可行解集合

    feasible_sets = []
    
    # 当 universe 大小小于 20 时，选择所有点作为 S0
    sample_size = min(20, len(universe))
     # S0: 随机选 X 中的 20 个点放入其中
    S0 = frozenset(random.sample(list(universe), sample_size))
    feasible_sets.append(S0)
    
    covered_so_far = set(S0)
    remaining_points = universe - covered_so_far
    
    while len(remaining_points) >= sample_size and len(feasible_sets) < num_sets:
         # 产生一个 1-20 的随机数 n 代表 S_i 集合大小
        n = random.randint(1, 20)
        # 随机一个 1-n 的随机数 x 代表需要重 X-S0 随机选 x 个点放入 S1
        x = random.randint(1, n)
        # 从剩余点中选 x 个点
        from_remaining = set(random.sample(list(remaining_points), min(x, len(remaining_points))))
        # 从已覆盖点中选 n-x 个点
        from_covered = set(random.sample(list(covered_so_far), min(n - x, len(covered_so_far))))
        # 合并得到新的集合
        Si = frozenset(from_remaining | from_covered)
        feasible_sets.append(Si)
        # 更新已覆盖点和剩余点
        covered_so_far |= Si
        remaining_points = universe - covered_so_far
    # 当剩余点小于 20 时，直接作为最后一个集合
    if remaining_points and len(feasible_sets) < num_sets:
        feasible_sets.append(frozenset(remaining_points))
    # 计算已生成的集合数量
    y = len(feasible_sets)
    # 生成其余集合，确保总数达到 num_sets
    remaining_sets = []
    target_remaining = num_sets - y
    
    attempts = 0
    max_attempts = target_remaining * 20
    
    while len(remaining_sets) < target_remaining and attempts < max_attempts:
        attempts += 1
        # 随机生成集合大小，不超过 max_size 且至少为 1
        s_size = random.randint(1, max_size)
        s = frozenset(random.sample(list(universe), min(s_size, len(universe))))
        # 确保集合不重复
        if s not in feasible_sets and s not in remaining_sets:
            remaining_sets.append(s)

    subsets = feasible_sets + remaining_sets
    return universe, subsets

if __name__ == '__main__':
    print("测试大规模 2D...")
    data = generate_instance(1000, 1000)
    print(f"Universe: {len(data[0])}, Subsets: {len(data[1])}")
    
    print("\n测试小规模 2D (原代码会崩)...")
    data_small = generate_instance(10, 15)
    print(f"Universe: {len(data_small[0])}, Subsets: {len(data_small[1])}")
    
    print("\n测试高斯分布 (原代码进不去)...")
    data_gauss = generate_instance(100, 100, sample_method="gauss")
    print(f"Universe: {len(data_gauss[0])}, Subsets: {len(data_gauss[1])}")
