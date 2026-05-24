from utils import createData,draw_times_curve,draw_retry_times_curve
import argparse
from selectk import randomSelect,divSelect_middle,quickselect
import time
import os
import random
def get_args():
    '''
    获取命令行参数
    '''
    parser = argparse.ArgumentParser(description='convex hull algorithm')
    # 算法名
    parser.add_argument('--name',
                    type=str,
                    default='random',
                    help='算法选择')
    # 点的数量
    parser.add_argument('--sizes',
                    type=str,
                    default='1000,2000,5000,10000,20000,50000,100000',
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
                    default="normal",
                    help='采样方法')
    parser.add_argument('--max_size_of_set',
                    type=int,
                    default=500,
                    help='最大集合大小')
    parser.add_argument('--result_path',
                    type=str,
                    default="./result_3",
                    help='结果保存路径')
    parser.add_argument('--sample_rate',
                    type=float,
                    default=0.75,  # 默认 3/4
                    help='采样比例 (e.g., 0.125 for 1/8, 0.25 for 1/4)')
    args = parser.parse_args()
    return args
# def run_algorithm(algorithm, dataset,k):
#     start_time = time.time()
#     retry_times = 0
#     if algorithm == randomSelect:
#         k_value,retry_times = algorithm(dataset,k,test_once=False)
#     else:
#         k_value = algorithm(dataset,k)
#     end_time = time.time()
    
#     return (end_time - start_time) * 1000,k_value,retry_times
def run_algorithm(algorithm, dataset, k, sample_rate=None):
    start_time = time.time()
    retry_times = 0
    if algorithm == randomSelect:
        # 将 sample_rate 传递给 randomSelect 函数
        k_value, retry_times = algorithm(dataset, k, sample_rate=sample_rate, test_once=False)
    else:
        k_value = algorithm(dataset, k)
    end_time = time.time()
    
    return (end_time - start_time) * 1000, k_value, retry_times
    
if __name__ == '__main__':
    args = get_args()
    algo_name = args.name
    sizes = list(map(int, args.sizes.split(','))) 
    if args.max_size!=-1:
        sizes = list(range(0, args.max_size+1, args.max_size//args.group_size))[1:]
        print(sizes)
    datasets = createData(sizes,sample_method=args.sample_method)
    if algo_name == 'div':
        algorithm = divSelect_middle
    elif algo_name == 'random':
        algorithm = randomSelect
    elif algo_name == 'quick':
        algorithm = quickselect
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
        k = random.randint(0,len(dataset)-1)
        run_time_sum = 0
        # 重复1000次，取平均值
        from tqdm import tqdm
        bar = tqdm(range(1000))
        retry_times_sum = 0
        for i in range(1000):
            # run_time,k_value,retry_times = run_algorithm(algorithm,dataset,k)
            run_time, k_value, retry_times = run_algorithm(algorithm, dataset, k, sample_rate=args.sample_rate)
            run_time_sum += run_time
            retry_times_sum += retry_times
            bar.update(1)
        bar.close()
        times.append(run_time_sum/1000)
        sizes.append(len(dataset))
        print("retry_times_sum:{}".format(retry_times_sum/1000))
        print("k:{},k_value:{} != sorted(dataset)[k]:{} while k-1 value:{} and k+1 value:{}".format(k,k_value,sorted(dataset)[k],sorted(dataset)[k-1],sorted(dataset)[k+1]))
        assert k_value == sorted(dataset)[k],"k:{},k_value:{} != sorted(dataset)[k]:{} while k-1 value:{} and k+1 value:{}".format(k,k_value,sorted(dataset)[k],sorted(dataset)[k-1],sorted(dataset)[k+1])
        # print_set_cover(dataset,selected_subsets,algo_name)
        # draw_points_with_set_cover(dataset,selected_subsets,title="Cover Sets",save_file_path="./cover_sets.png",show=args.show)
    draw_times_curve(sizes,times,algo_name,sample_method=args.sample_method)
    args.result_path = "./result_{}".format(len(sizes))
    os.makedirs(args.result_path,exist_ok=True)
    with open(args.result_path+"/{}_result.json".format(algo_name),"w") as f:
        result_json = [{'size':sizes[i],'times':times[i]} for i in range(len(sizes))]
        import json
        json.dump(result_json,f)







