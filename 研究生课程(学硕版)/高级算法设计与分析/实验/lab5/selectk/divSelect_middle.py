# lab5\selectk\divSelect_middle.py
import random
from typing import List, Tuple

def insert_sort(nums: List[int]) -> List[int]:
    """
     N <= 5，所以实际为 O(1)
    """
    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1
        
        while j >= 0 and key < nums[j]:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key
    return nums

def partition(nums: List[int], pivot: int) -> Tuple[int, int, List[int], List[int]]:
    """
    三路分区函数（
    
    """
    left = []
    middle = []
    right = []
    
    for num in nums:
        if num < pivot:
            left.append(num)
        elif num == pivot:
            middle.append(num)
        else:
            right.append(num)
    
    
    pivot_index_min = len(left)
    pivot_index_max = len(left) + len(middle) - 1
    
    return pivot_index_min, pivot_index_max, left, right

def divSelect_middle(nums: List[int], k: int) -> int:
    """

    T(N) = T(N/5) + T(7N/10) + O(N)
 
    """
    
    c = 5
    
    # 边界检查
    if k < 0 or k >= len(nums):
        raise IndexError(f"k={k} is out of range for array of size {len(nums)}")
    
    
    if len(nums) <= c:
        return insert_sort(nums)[k]
    
    #分组并找到每组的中位数，O(N)
    medians = []
    for i in range(0, len(nums), c):
        group = nums[i:i+c]
        group = insert_sort(group) # 5个元素，排序O(1)
        medians.append(group[len(group)//2]) 
    
    # 递归找中位数的中位数作为pivot，T(N/5)
    pivot_value = divSelect_middle(medians, len(medians)//2)
    
    # 根据pivot进行三路分区，O(N)
    pivot_index_min, pivot_index_max, left, right = partition(nums, pivot_value)
    
    # 第4步：根据k的位置判断去左子区还是右子区递归，T(7N/10)
    if k < pivot_index_min:
        # k 在左子数组中
        return divSelect_middle(left, k)
    elif k > pivot_index_max:
        # k 在右子数组中，更新 k 的索引
        return divSelect_middle(right, k - pivot_index_max - 1)
    else:
        # k 刚好落在主元的范围内
        return pivot_value

def test_divSelect_middle():
    
    nums1 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    k1 = 3
    result1 = divSelect_middle(nums1.copy(), k1)
    expected1 = sorted(nums1)[k1]
    assert result1 == expected1, f"Test 1 failed: expected {expected1}, got {result1}"
    
    
    nums2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    k2 = 7
    result2 = divSelect_middle(nums2.copy(), k2)
    expected2 = sorted(nums2)[k2]
    assert result2 == expected2, f"Test 2 failed: expected {expected2}, got {result2}"
    
    
    nums3 = list(range(1, 101)) + [24, 25, 33, 32, 24, 24, 24, 24, 24]
    random.shuffle(nums3)
    k3 = 26
    result3 = divSelect_middle(nums3.copy(), k3)
    expected3 = sorted(nums3)[k3]
    assert result3 == expected3, f"Test 3 failed: expected {expected3}, got {result3}"
    
    print("All tests passed!")

if __name__ == '__main__':
    # 运行测试
    test_divSelect_middle()
    
    # 示例使用
    nums = list(range(1, 101)) + [24, 25, 33, 32, 24, 24, 24, 24, 24]
    random.shuffle(nums)
    k = 26
    pivot = divSelect_middle(nums, k)
    sorted_nums = sorted(nums)
    print(f"Original array (shuffled): {nums}")
    print(f"Sorted array: {sorted_nums}")
    print(f"The {k+1}th smallest element is: {pivot}")
    print(f"Verification: {sorted_nums[k]}")
