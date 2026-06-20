import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def loss_function(params, points):
    """
    损失函数：计算每个点与三条直线的竖直方向距离，取最小绝对差的平方和。
    参数：
        params: 列表 [a, c, c1, c2]，对应直线方程 y = -a * x - c 等（b=1 简化形式）。
        points: (n, 2) 数组，每行为 (x, y) 坐标。
    返回：
        损失值（标量）。
    """
    a, c, c1, c2 = params
    x = points[:, 0]
    y = points[:, 1]
    # 计算三条直线的竖直差
    d1 = y + a * x + c   # 直线1: y = -a*x - c
    d2 = y + a * x + c1  # 直线2: y = -a*x - c1
    d3 = y + a * x + c2  # 直线3: y = -a*x - c2
    # 绝对值并取每个点的最小值
    abs_d1 = np.abs(d1)
    abs_d2 = np.abs(d2)
    abs_d3 = np.abs(d3)
    min_abs = np.minimum(np.minimum(abs_d1, abs_d2), abs_d3)
    # 平方和
    loss = np.sum(min_abs ** 2)
    return loss

def generate_sample_data(seed=42):
    """
    生成示例数据点，从三条平行线附近添加高斯噪声。
    返回：
        points: (n, 2) 数组，数据点。
        a_true, c_true, c1_true, c2_true: 真实参数。
    """
    np.random.seed(seed)
    a_true = -2.0
    c_true = 1.0
    c1_true = 3.0
    c2_true = 5.0
    
    points_list = []
    # 为每条直线生成20个点
    for c_val in [c_true, c1_true, c2_true]:
        x_vals = np.random.uniform(-10, 10, 100)
        y_vals = -a_true * x_vals - c_val + np.random.normal(0, 0.5, len(x_vals))
        points_list.append(np.column_stack((x_vals, y_vals)))
    points = np.vstack(points_list)  # 合并所有点
    return points, a_true, c_true, c1_true, c2_true

def plot_results(points, params, title="三条平行直线拟合结果"):
    """
    绘制数据点和拟合的直线
    参数：
        points: (n, 2) 数组，数据点
        params: 列表 [a, c, c1, c2]，拟合参数
        title: 图形标题
    """
    a_fit, c_fit, c1_fit, c2_fit = params
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6, label='数据点', s=30)
    
    # 生成x范围
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    x_range = np.linspace(x_min - 1, x_max + 1, 100)
    
    # 计算三条拟合直线
    y1_fit = -a_fit * x_range - c_fit
    y2_fit = -a_fit * x_range - c1_fit
    y3_fit = -a_fit * x_range - c2_fit
    
    # 绘制拟合直线
    plt.plot(x_range, y1_fit, 'r-', linewidth=2, label=f'直线1: y = {-a_fit:.3f}x - {c_fit:.3f}')
    plt.plot(x_range, y2_fit, 'g-', linewidth=2, label=f'直线2: y = {-a_fit:.3f}x - {c1_fit:.3f}')
    plt.plot(x_range, y3_fit, 'm-', linewidth=2, label=f'直线3: y = {-a_fit:.3f}x - {c2_fit:.3f}')
    
    # 添加图形元素
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.axis('equal')
    
    # 添加信息框
    info_text = f'拟合参数:\n斜率 a = {a_fit:.4f}\n截距 c = {c_fit:.4f}\n截距 c1 = {c1_fit:.4f}\n截距 c2 = {c2_fit:.4f}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成示例数据
    points, a_true, c_true, c1_true, c2_true = generate_sample_data()
    print("真实参数：")
    print(f"a = {a_true:.4f}")
    print(f"c = {c_true:.4f}")
    print(f"c1 = {c1_true:.4f}")
    print(f"c2 = {c2_true:.4f}")
    print(f"数据点数量: {len(points)}")
    
    # 初始猜测 [a, c, c1, c2]
    initial_guess = [1.0, 0.0, 2.0, 4.0]
    
    # 使用 Powell 方法优化（不依赖梯度）
    result = minimize(loss_function, initial_guess, args=(points,), method='Powell')
    
    # 输出拟合结果
    a_fit, c_fit, c1_fit, c2_fit = result.x
    print("\n拟合参数：")
    print(f"a = {a_fit:.4f}")
    print(f"c = {c_fit:.4f}")
    print(f"c1 = {c1_fit:.4f}")
    print(f"c2 = {c2_fit:.4f}")
    print(f"损失值: {result.fun:.4f}")
    
    # 输出直线方程（形式: ax + y + c = 0，其中 b=1）
    print("\n拟合直线方程 (形式: a*x + y + c = 0):")
    print(f"直线1: {a_fit:.4f} * x + y + {c_fit:.4f} = 0")
    print(f"直线2: {a_fit:.4f} * x + y + {c1_fit:.4f} = 0")
    print(f"直线3: {a_fit:.4f} * x + y + {c2_fit:.4f} = 0")

 # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置常用中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plot_results(points, [a_fit, c_fit, c1_fit, c2_fit], "三条平行直线拟合结果")