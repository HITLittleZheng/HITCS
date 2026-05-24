# lab3\utils\draw.py
import matplotlib.pyplot as plt

try:
    from .set_init import createData
except:
    from set_init import createData

def draw_init_points(dataset, title="Points", save_file_path="./points.png", show=False):
    universe, subsets = dataset
    plt.figure(figsize=(8, 8))
    
   
    first_elem = list(universe)[0]
    if isinstance(first_elem, (tuple, list)):
        points_x = [point[0] for point in universe]
        points_y = [point[1] for point in universe]
    else:
        
        points_x = list(universe)
        points_y = [0] * len(universe)
        
    plt.scatter(points_x, points_y, c='blue', marker='o')
    plt.title(title)
    plt.savefig(save_file_path, bbox_inches='tight') # 加上 bbox_inches 防止保存时坐标轴标签被截断
    if show:
        plt.show()
    plt.close()

def draw_points_with_set_cover(dataset, selected_subsets, title="Cover Sets", save_file_path="./cover_sets.png", show=False):
    universe, subsets = dataset
    plt.figure(figsize=(8, 8))
    
    # 提取实际被覆盖的元素
    union_subsets = set().union(*selected_subsets) if selected_subsets else set()
    uncovered_points = universe - union_subsets # 找出未被覆盖的点
    
    
    first_elem = list(universe)[0]
    if isinstance(first_elem, (tuple, list)):
        uncov_x = [p[0] for p in uncovered_points]
        uncov_y = [p[1] for p in uncovered_points]
        cov_x = [p[0] for p in union_subsets]
        cov_y = [p[1] for p in union_subsets]
    else:
        uncov_x = list(uncovered_points)
        uncov_y = [0] * len(uncovered_points)
        cov_x = list(union_subsets)
        cov_y = [0] * len(union_subsets)

    
    if uncov_x:
        plt.scatter(uncov_x, uncov_y, c='gray', marker='o', label='Uncovered')
    if cov_x:
        plt.scatter(cov_x, cov_y, c='red', marker='o', label='Covered')
    
    plt.title(title)
    plt.legend() # 显示图例
    plt.savefig(save_file_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def draw_performance_curve(sizes, lens, algorithm_name, show=False):
    plt.figure(figsize=(8, 6)) # 性能曲线通常不需要正方形，8x6更美观
    plt.plot(sizes, lens, '-o')
    plt.xlabel("Data Size")
    plt.ylabel("Number of Selected Sets")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.grid(True, linestyle='--', alpha=0.6) # 加上网格线，看数据更清晰
    plt.savefig("Performance_Subsets_Curve_for_{}.png".format(algorithm_name), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print("Performance Subsets Curve Image Saved to {}".format("Performance_Subsets_Curve_for_{}.png".format(algorithm_name)))

def draw_times_curve(sizes, times, algorithm_name, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, '-o')
    plt.xlabel("Data Size")
    plt.ylabel("Running Time (ms)")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.grid(True, linestyle='--', alpha=0.6) # 加上网格线
    plt.savefig("Performance_Time_Curve_for_{}.png".format(algorithm_name), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print("Performance Time Curve Image Saved to {}".format("Performance_Time_Curve_for_{}.png".format(algorithm_name)))

if __name__ == '__main__':
    sizes = [1000, 2000, 3000]
    datasets = createData(sizes)
    for i in range(len(datasets)):
        draw_init_points(datasets[i], title="Points_Init_{}".format(len(datasets[i][0])), save_file_path="./points_init_{}.png".format(len(datasets[i][0])), show=True)
