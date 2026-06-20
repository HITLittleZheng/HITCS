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
points = np.column_stack([x_noisy, y_noisy])

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
a_fit, b_fit, r_fit = best_circle
distances = np.sqrt((x_noisy-a_fit)**2 + (y_noisy-b_fit)**2)
loss = np.mean((distances-r_fit)**2)


# 可视化
plt.figure(figsize=(10, 8))

# 1. 绘制真实圆
theta_plot = np.linspace(0, 2*np.pi, 200)
x_true_circle = a_true + r_true * np.cos(theta_plot)
y_true_circle = b_true + r_true * np.sin(theta_plot)
plt.plot(x_true_circle, y_true_circle, 'b-', label='True Circle', linewidth=2, alpha=0.7)

# 2. 绘制噪声点
plt.scatter(x_noisy, y_noisy, c='gray', s=20, alpha=0.6, label='Noisy Points')

# 3. 绘制拟合圆
x_fit_circle = a_fit + r_fit * np.cos(theta_plot)
y_fit_circle = b_fit + r_fit * np.sin(theta_plot)
plt.plot(x_fit_circle, y_fit_circle, 'r-', label='Fitted Circle', linewidth=2, alpha=0.7)

# 4. 绘制用于拟合的三个点
if best_sample_points is not None:
    plt.scatter(best_sample_points[:, 0], best_sample_points[:, 1], 
                c='green', s=120, marker='*', edgecolors='black', 
                linewidths=1.5, label='Sample Points (3)', zorder=5)

# 5. 绘制圆心
plt.scatter(a_true, b_true, c='blue', s=100, marker='o', label='True Center', zorder=4)
plt.scatter(a_fit, b_fit, c='red', s=100, marker='x', label='Fitted Center', zorder=4)

# 6. 添加文本信息
plt.text(0.02, 0.98, f'True: center({a_true:.2f}, {b_true:.2f}), r={r_true:.2f}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
plt.text(0.02, 0.92, f'Fitted: center({a_fit:.2f}, {b_fit:.2f}), r={r_fit:.2f}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
plt.text(0.02, 0.86, f'Loss: {loss:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2))
plt.text(0.02, 0.80, f'Inliers: {len(inliers)}/{num_points}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

# 7. 设置图形属性
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC Circle Fitting Visualization')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

# 8. 添加注释显示采样点坐标
if best_sample_points is not None:
    for i, (x, y) in enumerate(best_sample_points):
        plt.annotate(f'P{i+1}({x:.2f}, {y:.2f})', 
                     xy=(x, y), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8, 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()