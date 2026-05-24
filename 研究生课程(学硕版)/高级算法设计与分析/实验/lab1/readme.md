## 实验一：分治算法

### 实验运行指令
conda activate robot

### 生成数据集合
python draw.py 

python draw.py --sample_method gauss


### 枚举法凸包可视化

python main.py --name enumerate --sizes 50,100,200 --draw_hull

### 枚举法性能曲线
python main.py --name enumerate --max_size 200 --group_size 10

### Graham-Scan 凸包可视化

python main.py --name grahamScan --sizes 1000,2000,3000 --draw_hull

### Graham-Scan 性能曲线
python main.py --name grahamScan --max_size 100000 --group_size 10

### 分治法凸包可视化

python main.py --name divide_conquer --sizes 1000,2000,3000 --draw_hull

### 分治法性能曲线
python main.py --name divide_conquer --max_size 100000 --group_size 5




















