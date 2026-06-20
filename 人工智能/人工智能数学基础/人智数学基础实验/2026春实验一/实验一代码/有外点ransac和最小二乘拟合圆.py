import numpy as np
import matplotlib.pyplot as plt
import random

# 设置随机种子
np.random.seed(42)
random.seed(42)
# 真实圆参数
a_true, b_true, r_true = 3.0, 2.0, 5.0
# 生成圆上点
num_points = 100
theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
x_noisy = a_true + r_true*np.cos(theta) + np.random.normal(0, 0.01, num_points)
y_noisy = b_true + r_true*np.sin(theta) + np.random.normal(0, 0.01, num_points)
#外点
num_outliers = 30
num_points+=num_outliers
x_outliers = np.random.uniform(-2, 8, num_outliers)  # 在圆周围随机位置生成外点
y_outliers = np.random.uniform(-3, 7, num_outliers)
# 合并内点和外点
x_noisy = np.concatenate([x_noisy, x_outliers])
y_noisy = np.concatenate([y_noisy, y_outliers])
points = np.column_stack([x_noisy, y_noisy])


# 最小二乘法拟合圆
# 圆的一般方程: x² + y² + Dx + Ey + F = 0
# 其中 D = -2a, E = -2b, F = a² + b² - r²
# 我们需要拟合参数 D, E, F
# 构建线性方程组: 对于每个点 (x_i, y_i)，有 x_i² + y_i² = -D*x_i - E*y_i - F
# 可以写成 Ax = b 的形式，其中 x = [D, E, F]^T
A = np.column_stack([x_noisy, y_noisy, np.ones(num_points)])
b = -(x_noisy**2 + y_noisy**2)

# 使用最小二乘法求解
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
D_fit, E_fit, F_fit = params

# 从D, E, F计算圆心和半径
a_fit1 = -D_fit / 2
b_fit1 = -E_fit / 2
r_fit1 = np.sqrt(a_fit1**2 + b_fit1**2 - F_fit)

# 计算损失函数（点到圆心的距离与半径的差的平方）
distances1 = np.sqrt((x_noisy - a_fit1)**2 + (y_noisy - b_fit1)**2)
loss1 = np.mean((distances1 - r_fit1)**2)


def fit_circle_three_points(p1, p2, p3):
    """三点拟合圆"""
    x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
    
    # 防止共线
    if (y2-y1)*(x3-x2) == (y3-y2)*(x2-x1):
        return None
    
    A, B, C, D = x2-x1, y2-y1, x3-x1, y3-y1
    E = A*(x1+x2) + B*(y1+y2)
    F = C*(x1+x3) + D*(y1+y3)
    G = 2*(A*(y3-y2) - B*(x3-x2))
    
    if abs(G) < 1e-10:
        return None
    
    a = (D*E - B*F)/G
    b = (A*F - C*E)/G
    r = np.sqrt((x1-a)**2 + (y1-b)**2)
    return a, b, r

def distance_to_circle(point, circle):
    """点到圆的距离"""
    x, y, a, b, r = *point, *circle
    return abs(np.sqrt((x-a)**2 + (y-b)**2) - r)

def ransac_fit_circle(points, max_iter=1000, threshold=1.0, min_inlier_ratio=0.8):
    """RANSAC拟合圆"""
    n = len(points)
    best_circle, best_inliers, best_count = None, [], 0
    
    for it in range(max_iter):#max_iter轮
        # 随机采样三点
        idx = random.sample(range(n), 3)
        sample_points = points[idx]  # 保存这三个点
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        # 拟合圆
        circle = fit_circle_three_points(p1, p2, p3)
        if circle is None:
            continue
        
        # 统计内点
        distances = np.array([distance_to_circle(p, circle) for p in points])#求出points数组中每个p到圆的距离distances，distances也是一个数组
        inliers = np.where(distances < threshold)[0]#inliers为distance数组中<1.0的元素的索引元组
        inlier_count = len(inliers)
        # 更新最佳模型
        if inlier_count > best_count:
            best_circle, best_inliers,best_count = circle, inliers,inlier_count
            best_sample_points = sample_points
        
        # 提前终止
        if inlier_count/n >= min_inlier_ratio:
            print(f"提前终止：迭代{it+1}，内点比例{inlier_count/n:.2%}")#达到min_inlier_ratio认为已经足够好，停止迭代
            break
    
    print(f"RANSAC完成：迭代{it+1}，内点{best_count}/{n}({best_count/n:.2%})")
    return best_circle, best_inliers,best_sample_points
# 执行拟合
print("开始RANSAC圆拟合...")
best_circle, inliers,best_sample_points = ransac_fit_circle(points, max_iter=1000, threshold=0.8, min_inlier_ratio=0.8)
# 计算损失
a_fit2, b_fit2, r_fit2 = best_circle
distances2 = np.sqrt((x_noisy-a_fit2)**2 + (y_noisy-b_fit2)**2)
loss2 = np.mean((distances2-r_fit2)**2)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(21, 6))

# 生成圆的绘制点
theta_circle = np.linspace(0, 2*np.pi, 200)

# 左图：最小二乘法拟合
ax1 = axes[0]
# 绘制数据点
ax1.scatter(x_noisy, y_noisy, s=20, alpha=0.7, c='blue', label='points')
# 绘制真实圆
x_true = a_true + r_true*np.cos(theta_circle)
y_true = b_true + r_true*np.sin(theta_circle)
ax1.plot(x_true, y_true, 'k-', linewidth=3, alpha=0.7, label='TRUE')
# 绘制最小二乘法拟合圆
x_fit1 = a_fit1 + r_fit1*np.cos(theta_circle)
y_fit1 = b_fit1 + r_fit1*np.sin(theta_circle)
ax1.plot(x_fit1, y_fit1, 'r-', linewidth=2, label=f'Least Squares\nheart:({a_fit1:.2f},{b_fit1:.2f})\nr:{r_fit1:.2f}')
# 绘制圆心
ax1.scatter(a_true, b_true, s=100, c='black', marker='*', label='heart_true')
ax1.scatter(a_fit1, b_fit1, s=100, c='red', marker='+', linewidths=3, label='heart_fit')

ax1.set_title(f'Least Squares (loss: {loss1:.4f})')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 右图：RANSAC拟合
ax2 = axes[1]
# 绘制所有数据点
ax2.scatter(x_noisy, y_noisy, s=20, alpha=0.7, c='blue', label='points')
# 绘制内点
inlier_points = points[inliers]
ax2.scatter(inlier_points[:, 0], inlier_points[:, 1], s=30, c='green', alpha=0.7, label='inliers')
# 绘制RANSAC用于拟合的三个点
if best_sample_points is not None:
    ax2.scatter(best_sample_points[:, 0], best_sample_points[:, 1], s=150, c='orange', marker='s', edgecolors='red', linewidths=2, label='RANSAC fit point')
# 绘制真实圆
ax2.plot(x_true, y_true, 'k-', linewidth=3, alpha=0.7, label='TRUE')
# 绘制RANSAC拟合圆
x_fit2 = a_fit2 + r_fit2*np.cos(theta_circle)
y_fit2 = b_fit2 + r_fit2*np.sin(theta_circle)
ax2.plot(x_fit2, y_fit2, 'r-', linewidth=2, label=f'RANSAC fit\nheart:({a_fit2:.2f},{b_fit2:.2f})\nr:{r_fit2:.2f}')
# 绘制圆心
ax2.scatter(a_true, b_true, s=100, c='black', marker='*', label='heart_true')
ax2.scatter(a_fit2, b_fit2, s=100, c='red', marker='+', linewidths=3, label='heart_fit')
# 添加内点数量条目
ax2.scatter([], [], s=0, label=f'Inliers: {len(inliers)}/{num_points}')

ax2.set_title(f'RANSAC fit (loss: {loss2:.4f})')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

# 打印对比结果
print("\n=== 对比结果 ===")
print(f"真实圆参数: 圆心({a_true:.2f}, {b_true:.2f}), 半径{r_true:.2f}")
print(f"最小二乘法: 圆心({a_fit1:.2f}, {b_fit1:.2f}), 半径{r_fit1:.2f}, 损失{loss1:.6f}")
print(f"RANSAC: 圆心({a_fit2:.2f}, {b_fit2:.2f}), 半径{r_fit2:.2f}, 损失{loss2:.6f}")