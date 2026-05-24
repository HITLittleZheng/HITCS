# lab5\selectk\quickSelect.py
import random
from typing import List

def quickselect(arr: List[int], k: int) -> int:
    """
    快速选择算法，查找第k小元素（0-based）
    
    Args:
        arr: 输入数组
        k: 要查找的元素索引（0-based）
        
    Returns:
        第k小的元素
    """
    # 边界检查
    if not arr:
        raise ValueError("Input array is empty")
    if k < 0 or k >= len(arr):
        raise IndexError(f"k={k} is out of range for array of size {len(arr)}")
    
    # 基准情况：数组很小，直接排序返回
    if len(arr) <= 5:
        return sorted(arr)[k]
    
    # 随机选择pivot
    randomIdx = random.randint(0, len(arr) - 1)
    pivot = arr[randomIdx]
    
    # 分区
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    pivot_index_min = len(left)
    pivot_index_max = len(left) + len(mid) - 1
    
    # 根据k的位置递归查找
    if k < pivot_index_min:
        return quickselect(left, k)
    elif k > pivot_index_max:
        return quickselect(right, k - pivot_index_max - 1)
    else:
        return pivot

def test_quickselect():
    """测试函数"""
    # 测试用例1：简单数组
    nums1 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    k1 = 3
    result1 = quickselect(nums1.copy(), k1)
    expected1 = sorted(nums1)[k1]
    assert result1 == expected1, f"Test 1 failed: expected {expected1}, got {result1}"
    
    # 测试用例2：重复元素
    nums2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    k2 = 7
    result2 = quickselect(nums2.copy(), k2)
    expected2 = sorted(nums2)[k2]
    assert result2 == expected2, f"Test 2 failed: expected {expected2}, got {result2}"
    
    # 测试用例3：大数组
    nums3 = list(range(1, 101)) + [24, 25, 33, 32, 24, 24, 24, 24, 24]
    random.shuffle(nums3)
    k3 = 26
    result3 = quickselect(nums3.copy(), k3)
    expected3 = sorted(nums3)[k3]
    assert result3 == expected3, f"Test 3 failed: expected {expected3}, got {result3}"
    
    # 测试用例4：边界情况
    nums4 = [5, 4, 3, 2, 1]
    k4 = 0
    result4 = quickselect(nums4.copy(), k4)
    expected4 = sorted(nums4)[k4]
    assert result4 == expected4, f"Test 4 failed: expected {expected4}, got {result4}"
    
    print("All tests passed!")

if __name__ == '__main__':
    # 运行测试
    test_quickselect()
    
    # 示例使用
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    result = quickselect(nums, k)
    print(f"The {k+1}th smallest element is: {result}")
    print(f"Verification: {sorted(nums)[k]}")
