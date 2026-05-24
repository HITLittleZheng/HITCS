# lab5\utils\draw.py
import time
import matplotlib.pyplot as plt
try:
    from .set_init import createData
except ImportError:
    from set_init import createData






def draw_init_points(dataset,title="Points",save_file_path="./points.png",show=False):
    universe, subsets = dataset
    plt.figure(figsize=(8, 8))
    points_x = [point[0] for point in universe]
    points_y = [point[1] for point in universe]
    plt.scatter(points_x, points_y, c='blue', marker='o')
    plt.title(title)
    plt.savefig(save_file_path)
    if show:
        plt.show()
    plt.close()

def draw_points_with_set_cover(dataset,selected_subsets,title="Cover Sets",save_file_path="./cover_sets.png",show=False):
    universe, subsets = dataset
    plt.figure(figsize=(8, 8))
    points_x = [point[0] for point in universe]
    points_y = [point[1] for point in universe]
    union_subsets = set().union(*selected_subsets)
    covered_points_x = [point[0] for point in union_subsets]
    covered_points_y = [point[1] for point in union_subsets]
    plt.scatter(points_x, points_y, c='blue', marker='o')
    plt.scatter(covered_points_x, covered_points_y, c='red', marker='o')
    plt.title(title)
    plt.savefig(save_file_path)
    if show:
        plt.show()
    plt.close()
def draw_performance_curve(sizes,lens,algorithm_name,show=False):
    plt.figure(figsize=(8, 8))
    plt.plot(sizes, lens, '-o')
    plt.xlabel("Data Size")
    plt.ylabel("Number of Selected Sets")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.savefig("Performance_Subsets_Curve_for_{}.png".format(algorithm_name))
    if show:
        plt.show()
    plt.close()
    print("Performance Subsets Curve Image Saved to {}".format("Performance_Subsets_Curve_for_{}.png".format(algorithm_name)))
def draw_times_curve(sizes,times,algorithm_name,sample_method,show=False):
    plt.figure(figsize=(8, 8))
    plt.plot(sizes, times, '-o')
    plt.xlabel("Data Size")
    plt.ylabel("Running Time (ms)")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.savefig("Performance_Time_Curve_for_{}_{}.png".format(algorithm_name,sample_method))
    if show:
        plt.show()
    plt.close()
    print("Performance Time Curve Image Saved to {}".format("Performance_Time_Curve_for_{}_{}.png".format(algorithm_name,sample_method)))
def draw_retry_times_curve(radios,retry_times,algorithm_name,show=False):
    plt.figure(figsize=(8, 8))
    plt.plot(radios, retry_times, '-o')
    plt.xlabel("Radio")
    plt.ylabel("Retry Times")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.savefig("Performance_Retry_Times_Curve_for_{}.png".format(algorithm_name))
    if show:
        plt.show()
    plt.close()
    print("Performance Retry Times Curve Image Saved to {}".format("Performance_Retry_Times_Curve_for_{}.png".format(algorithm_name)))

def draw_acc_curve(radios,accs,algorithm_name,show=False):
    plt.figure(figsize=(8, 8))
    plt.plot(radios, accs, '-o',label="Accuracy")
    plt.xlabel("Radio")
    plt.ylabel("Accuracy")
    plt.title("Performance Curve for {}".format(algorithm_name))
    plt.savefig("Performance_Accuracy_Curve_for_{}.png".format(algorithm_name))
    if show:
        plt.show()
    plt.close()
    print("Performance Accuracy Curve Image Saved to {}".format("Performance_Accuracy_Curve_for_{}.png".format(algorithm_name)))



if __name__ == '__main__':
    sizes = [1000, 2000, 3000]
    datasets = createData(sizes)
    for i in range(len(datasets)):
        draw_init_points(datasets[i],title="Points_Init_{}".format(len(datasets[i])),save_file_path="./points_init_{}.png".format(len(datasets[i])),show=True)
        import random
        # convex_hull = random.sample(datasets[i],len(datasets[i])//100)
        # draw_convex_hull(datasets[i],convex_hull,title="Convex Hull_{}".format(len(datasets[i])),save_file_path="./convex_hull_{}.png".format(len(datasets[i])))

