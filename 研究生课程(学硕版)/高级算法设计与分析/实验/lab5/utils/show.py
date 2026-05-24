# lab5\utils\show.py
from typing import List, Tuple, Set, Any

def print_set_cover(dataset: Tuple[Set[Any], List[Set[Any]]], 
                  selected_subsets: List[Set[Any]], 
                  algorithm_name: str) -> None:
    """
    打印集合覆盖算法的结果
    
    Args:
        dataset: 包含全集和子集的元组 (universe, subsets)
        selected_subsets: 算法选中的子集列表
        algorithm_name: 算法名称
    """
    universe, subsets = dataset
    
    # 处理空选中子集的情况
    if not selected_subsets:
        print(f"{'='*10}{algorithm_name} algorithm result{'='*10}")
        print(f"len(universe): {len(universe)}")
        print(f"len(subsets): {len(subsets)}")
        print(f"len(selected_subsets): 0")
        print(f"union_subsets is empty, universe not covered")
        print(f"{'='*(20+len(algorithm_name)+2+len('algorithmresult'))}")
        return
    
    union_subsets = set().union(*selected_subsets)
    
    print(f"{'='*10}{algorithm_name} algorithm result{'='*10}")
    print(f"len(universe): {len(universe)}")
    print(f"len(subsets): {len(subsets)}")
    print(f"len(selected_subsets): {len(selected_subsets)}")
    print(f"len(union_subsets): {len(union_subsets)} vs len(universe): {len(universe)}")
    
    # 检查是否完全覆盖
    is_covered = union_subsets == universe
    print(f"Is fully covered: {is_covered}")
    
    print(f"{'='*(20+len(algorithm_name)+2+len('algorithmresult'))}")


