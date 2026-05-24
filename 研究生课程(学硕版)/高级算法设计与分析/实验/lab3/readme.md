## 实验三：近似算法

### 贪心近似算法不同 size 的解决方案
python main.py --rounding_mode frequency --name greedy 
python main.py --rounding_mode frequency --name greedy --max_size 3000 --group_size 10

### 线性规划近似算法不同 size 的解决方案
python main.py --rounding_mode frequency --name lp
python main.py --rounding_mode frequency --name lp --max_size 3000 --group_size 10


### 随机舍入和改善随机舍入的对比
python main.py --rounding_mode random --name lp 
python main.py --rounding_mode random_improved --name lp 







