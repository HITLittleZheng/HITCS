from setcover import greedySetCover,lpSetCover
from utils import createData, draw_performance_curve,draw_init_points,draw_points_with_set_cover,print_set_cover,draw_times_curve   
import argparse
import random
import time
import os
def get_args():
    '''
    获取命令行参数
    '''
    parser = argparse.ArgumentParser(description='convex hull algorithm')
    # 算法名
    parser.add_argument('--name',
                    type=str,
                    default='lp',
                    help='算法选择')
    # 点的数量
    parser.add_argument('--sizes',
                    type=str,
                    default='1000,2000,5000',
                    help='数据大小')
    # 点的数量
    parser.add_argument('--max_size',
                    type=int,
                    default=-1,
                    help='数据大小')
    # 分组个数
    parser.add_argument('--group_size',
                    type=int,
                    default=4,
                    help='数据大小')
    
    # 是否显示凸包
    parser.add_argument('--show',
                    action='store_true',
                    default=False,
                    help='是否显示凸包')
    # 是否显示凸包
    parser.add_argument('--sample_method',
                    type=str,
                    default="range_one_dimension",
                    help='采样方法')
    parser.add_argument('--max_size_of_set',
                    type=int,
                    default=500,
                    help='最大集合大小')
    parser.add_argument('--rounding_mode',
                    type=str,
                    default="random_improved",
                    help='松弛变量取值方式')
    parser.add_argument('--result_path',
                    type=str,
                    default="./result_3",
                    help='结果保存路径')
    args = parser.parse_args()
    return args
def run_algorithm(algorithm, dataset,rounding_mode):
    start_time = time.time()
    selected_subsets = algorithm(dataset,rounding_mode=rounding_mode)
    end_time = time.time()
    return (end_time - start_time) * 1000,selected_subsets
    
if __name__ == '__main__':
    args = get_args()
    algo_name = args.name
    sizes = list(map(int, args.sizes.split(','))) 
    if args.max_size!=-1:
        sizes = list(range(0, args.max_size+1, args.max_size//args.group_size))[1:]
        print(sizes)
    datasets = createData(sizes,sample_method=args.sample_method,max_size=args.max_size_of_set)
    if algo_name == 'greedy':
        algorithm = greedySetCover
    elif algo_name == 'lp':
        algorithm = lpSetCover
    else:
        raise ValueError("algorithm not implemented {}".format(algo_name))
    lens = []
    times = []
    sizes = []
    for dataset in datasets:
        '''
        运行算法
        '''
        # draw_init_points(dataset,title="Points",save_file_path="./points.png",show=args.show)
        run_time,selected_subsets = run_algorithm(algorithm,dataset,args.rounding_mode)
        lens.append(len(selected_subsets))
        times.append(run_time)
        sizes.append(len(dataset[0]))
        print_set_cover(dataset,selected_subsets,algo_name)
        # draw_points_with_set_cover(dataset,selected_subsets,title="Cover Sets",save_file_path="./cover_sets.png",show=args.show)
    draw_performance_curve(sizes,lens,algo_name+"_"+args.rounding_mode)
    draw_times_curve(sizes,times,algo_name+"_"+args.rounding_mode)
    args.result_path = "./result_{}".format(len(sizes))
    os.makedirs(args.result_path,exist_ok=True)
    with open(args.result_path+"/{}_result.json".format(algo_name+"_"+args.rounding_mode),"w") as f:
        result_json = [{'size':sizes[i],'lens':lens[i],'times':times[i]} for i in range(len(sizes))]
        import json
        json.dump(result_json,f)
# python main.py --algo greedySetCover --sizes 100,200,300,400,500,1000,2000

# python main.py --algo lpSetCover --sizes 100,200,300,400,500,1000,2000



