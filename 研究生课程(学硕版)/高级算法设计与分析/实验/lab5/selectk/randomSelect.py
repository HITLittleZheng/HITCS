# lab5\selectk\randomSelect.py
import random
from typing import List, Tuple


def rank(arr: List[int], x: int) -> Tuple[int, int]:
    """
    计算 x 在原数组中的排名区间
    时间复杂度: O(N)
    
    返回：
    less_count: 严格小于 x 的元素个数
    leq_count: 小于等于 x 的元素个数 (less + equal)

    """
    less_count = 0
    leq_count = 0

    for el in arr:
        if el < x:
            less_count += 1
        if el <= x:
            leq_count += 1

    return less_count, leq_count


def randomSelect(S: List[int], k: int, sample_rate: float = 3 / 4, test_once: bool = False) -> Tuple[int, int]:
    """
    
     O(N^a * log(N^a))
    

 
    """
    n = len(S)

    if n == 0:
        raise ValueError("Input array is empty")
    if k < 0 or k >= n:
        raise IndexError(f"k={k} is out of range for array of size {n}")

    retry_times = 0

    while True:
        # 随机采样 
        # 采样规模 m = n^a
        m = int(n ** sample_rate)
        m = max(1, min(m, n))

        # 排序耗时 O(m log m)
        R = random.sample(S, k=m)
        R.sort()

        # 估算目标在样本中的位置 
        # 线性映射
        x = int(k * m / n)
        x = max(0, min(x, m - 1)) 

        #  扩大搜索窗口 
        gap = int(n ** 0.5)

        l = max(x - gap, 0)
        h = min(x + gap, m - 1)

        L = R[l]
        H = R[h]

        #  计算真实边界 
        # 耗时 O(N)
        L_less, L_leq = rank(S, L)
        H_less, H_leq = rank(S, H)

        #  圈定候选集 组成候选集 P 耗时 O(N)
        P = [y for y in S if L <= y <= H]

        retry_times += 1

        
        max_P_size = 4 * (n ** sample_rate) + 1

        # 判断本次采样是否成功
        # L_less <= k < H_leq，即目标元素 k 确实被夹在 L 和 H 之间
        # len(P) <= max_P_size，即候选集足够小，没有因为重复元素过多导致窗口爆炸
        if L_less <= k < H_leq and len(P) <= max_P_size:
            
            P.sort()
            
            # 计算目标在 P 中的相对位置
           
            index_in_P = k - L_less

            if 0 <= index_in_P < len(P):
                return P[index_in_P], retry_times

        # ================= 7. 实验统计控制 =================
        # 如果只是测试一次运行的成功率，失败直接返回 -1
        if test_once:
            return -1, retry_times
        
        # 否则，while True 会继续循环，重新采样，直到成功为止





def test_randomSelect():
   
    print("Running tests for randomSelect...")
    
    # 测试用例1：基础无重复数组
    nums1 = list(range(1, 101)) # 1-100
    random.shuffle(nums1)
    for k in [0, 25, 50, 99]: # 测最小、中间、最大
        result, retries = randomSelect(nums1.copy(), k)
        expected = sorted(nums1)[k]
        assert result == expected, f"Test 1 failed at k={k}: expected {expected}, got {result}"
    print("✅ Test 1 (Basic array) passed.")

    # 测试用例2：海量重复元素数组 
    nums2 = [5] * 80 + [1, 2, 3, 4, 6, 7, 8, 9, 10] * 2
    random.shuffle(nums2)
    for k in [0, 50, len(nums2)-1]: 
        result, retries = randomSelect(nums2.copy(), k)
        expected = sorted(nums2)[k]
        assert result == expected, f"Test 2 failed at k={k}: expected {expected}, got {result}"
        print(f"   -> Duplicates test at k={k} succeeded with {retries} retries.")
    print("✅ Test 2 (Heavy duplicates) passed.")

    # 测试用例3：小数组 (验证 m = max(1, min(m, n)) 的边界保护)
    nums3 = [3, 1, 4]
    for k in range(len(nums3)):
        result, retries = randomSelect(nums3.copy(), k)
        expected = sorted(nums3)[k]
        assert result == expected, f"Test 3 failed at k={k}: expected {expected}, got {result}"
    print("✅ Test 3 (Small array) passed.")

   

    print("\n🎉 All tests passed!")


if __name__ == '__main__':
    
    
    test_randomSelect()
    
    # 日常使用演示
    nums = list(range(1, 10001))
    random.shuffle(nums)
    k = 4321
    result, retries = randomSelect(nums, k)
    print(f"\nDemo: The {k+1}th smallest element is {result}, found in {retries} retry(es).")
    print(f"Verification: {sorted(nums)[k]}")