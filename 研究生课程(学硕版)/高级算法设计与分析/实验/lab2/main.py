import argparse
from utils import init_map, excute
from Star import astar, biastar
import os

def get_parser():
    parser = argparse.ArgumentParser(description='A star search algorithm')
    
    
    parser.add_argument('--algo',
                        type=str,
                        default='biastar',
                        choices=['astar', 'biastar'], # 加上 choices 防止手误打错
                        help='算法选择')
    
    
    parser.add_argument('--data_path',
                        type=str,
                        default='data/map2.txt',
                        help='地图数据路径')
    
    parser.add_argument('--reverse',
                        action='store_true',
                        help='是否反向搜索 (交换起点和终点)')
    
    parser.add_argument('--type',
                        type=str,
                        default='h1',
                        choices=['h1', 'h2', 'h3', 'h4', 'h5'], # 限制可选的启发式函数
                        help='启发式函数类型')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    algo_name = args.algo 
    reverse = args.reverse
    
    
    if not os.path.exists(args.data_path):
        print(f"错误: 找不到地图文件 '{args.data_path}'")
        print("请通过 --data_path 参数指定正确的地图文件路径")
        exit(1)

    WIN, ROWS, COLS, GAP, grid, start, end = init_map(args.data_path, algo_name)
    
    
    if algo_name == 'astar':
        algorithm = astar
    elif algo_name == 'biastar':
        algorithm = biastar
    else:
        raise ValueError("algorithm not implemented {}".format(algo_name))
        
    if reverse:
        print("Running reverse search...")
        excute(algorithm, WIN, ROWS, COLS, GAP, grid, end, start, args.type)
    else:
        print("Running normal search...")
        excute(algorithm, WIN, ROWS, COLS, GAP, grid, start, end, args.type)

# 统一修改后的运行示例：
# python main.py --algo astar --data_path data/map2.txt --type h1
# python main.py --algo biastar --data_path data/map2.txt --type h5
# python main.py --algo astar --reverse
