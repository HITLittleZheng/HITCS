# lab3\utils\show.py

def print_set_cover(dataset, selected_subsets, algorithm_name):
    universe, subsets = dataset
    union_subsets = set().union(*selected_subsets)
    
    total_len = 36 + len(algorithm_name)
    
    print(f"{'='*10}{algorithm_name} algorithm result{'='*10}")
    print(f"len(universe): {len(universe)}")
    print(f"len(subsets): {len(subsets)}")
    print(f"len(selected_subsets): {len(selected_subsets)}")
    print(f"len(union_subsets): {len(union_subsets)} vs len(universe): {len(universe)}")
    
    print("=" * total_len)

