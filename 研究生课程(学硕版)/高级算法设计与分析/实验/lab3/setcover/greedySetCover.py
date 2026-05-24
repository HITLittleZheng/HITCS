# lab3\setcover\greedySetCover.py
def greedySetCover(data,rounding_mode="random"):
 
    universe, subsets = data

    # 存储所有被覆盖的元素
    covered = set()

    # 存储选中的子集
    selected_subsets = []

    # 如果还有元素没有被覆盖
    while covered != universe:
        # 过滤掉已经被完全覆盖的子集
        valid_subsets = [s for s in subsets if not s.issubset(covered)]
        
        # 如果没有有效子集了，说明剩下的元素无论如何都无法被覆盖，强行跳出
        if not valid_subsets:
            # print(f"警告：无法完全覆盖，剩余未覆盖元素: {universe - covered}")
            break

        # 选择未被覆盖元素最多的子集
        best_subset = max(valid_subsets, key=lambda s: len(s - covered))
        
        # 二次防御：如果最优子集带来的新元素是 0
        if len(best_subset - covered) == 0:
            break
        
        # 将这个子集加入到被选中的子集列表中
        selected_subsets.append(best_subset)

        # 将这个子集覆盖的元素加入到已被覆盖的元素列表中
        covered |= best_subset

    return selected_subsets

if __name__ == '__main__':
    # 测试用例 1：正常情况
    universe1 = {1, 2, 3, 4, 5}
    subsets1 = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]
    data1 = universe1, subsets1
    print("正常情况:", greedySetCover(data1)) 
    # 预期输出: [{1, 2, 3}, {4, 5}]
    
    # 测试用例 2：无解情况（原代码会死循环，现在会安全返回）
    universe2 = {1, 2, 3, 4}
    subsets2 = [{1, 2}, {2, 3}] # 缺少 4
    data2 = universe2, subsets2
    print("无解情况:", greedySetCover(data2)) 
    # 预期输出: [{1, 2}, {2, 3}] (并附带 break 跳出)
