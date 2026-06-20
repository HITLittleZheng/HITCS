import numpy as np
import matplotlib.pyplot as plt
# 设置随机种子以便结果可重现
np.random.seed(42)
# 定义真实直线参数 ax + by + c = 0
a_true = 2.0
b_true = -3.0
c_true = 5.0
# 生成满足真实直线的随机点
num_points = 200  # 点数
x = np.random.uniform(-10, 10, num_points)  # 随机生成x值
# 由直线方程计算对应的y值：y = (-a*x - c) / b
y = (-a_true * x - c_true) / b_true
# 添加高斯噪声(噪声服从正态分布)
noise_std = 0.5  # 噪声标准差
x_noisy = x + np.random.normal(0, noise_std, num_points)
y_noisy = y + np.random.normal(0, noise_std, num_points)
# 最小二乘法拟合直线 ax + by + c = 0
# 构建矩阵 M = [x_noisy, y_noisy, 1]
M = np.column_stack([x_noisy, y_noisy, np.ones(num_points)])
# 使用奇异值分解（SVD）求解，最小奇异值对应的右奇异向量即为拟合参数
U, S, Vt = np.linalg.svd(M, full_matrices=False)
params = Vt[-1, :]  # 取最后一行，对应最小奇异值
a_fit, b_fit, c_fit = params
# 输出结果
print("真实直线参数:")
print(f"  a = {a_true:.6f}, b = {b_true:.6f}, c = {c_true:.6f}")
print("\n拟合直线参数 (归一化 a²+b²=1):")
print(f"  a = {a_fit:.6f}, b = {b_fit:.6f}, c = {c_fit:.6f}")
# 计算损失函数
distances = (a_fit * x_noisy + b_fit * y_noisy + c_fit)**2  
mean_distance = np.mean(distances)
print(f"\n损失: {mean_distance:.6f}")



# -------------------------- 可视化部分 --------------------------
plt.figure(figsize=(12, 8))
# 1. 绘制加了噪声的点
plt.scatter(x_noisy, y_noisy, alpha=0.6, c='steelblue', s=20, label=f'noise_point (n={num_points})', zorder=2)
# 2. 绘制真实直线
x_plot = np.linspace(-12, 12, 400)
# 从直线方程 ax + by + c = 0 解出 y
y_true_line = (-a_true * x_plot - c_true) / b_true
plt.plot(x_plot, y_true_line, 'r-', linewidth=3, label='TRUE: $2x - 3y + 5 = 0$', zorder=3)
# 3. 绘制拟合直线
y_fit_line = (-a_fit * x_plot - c_fit) / b_fit
plt.plot(x_plot, y_fit_line, 'g--', linewidth=3, label=f'fit: ${a_fit:.3f}x {b_fit:+.3f}y {c_fit:+.3f} = 0$', zorder=4)
# 4. 添加图例和标签
plt.legend(loc='best', fontsize=12)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('TRUE vs fit ', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.axis('equal')
plt.xlim([-12, 12])
plt.ylim([-12, 12])
# 5. 添加信息文本框
info_text = f'noise_point_number: {num_points}\nLoss: {mean_distance:.4f}'
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
# 6. 添加对比说明
plt.figtext(0.5, 0.01, 
            f"TRUE: y = {(-a_true/b_true):.3f}x + {(-c_true/b_true):.3f}  |  fit: y = {(-a_fit/b_fit):.3f}x + {(-c_fit/b_fit):.3f}",
            ha='center', fontsize=11, style='italic', bbox=dict(facecolor='lightgray', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.show()
