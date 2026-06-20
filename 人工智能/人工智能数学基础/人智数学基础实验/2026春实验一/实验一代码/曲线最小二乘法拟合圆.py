import numpy as np
import matplotlib.pyplot as plt
# 设置随机种子以便结果可重现
np.random.seed(42)
# 定义真实圆的参数
# 圆的一般方程: (x - a)² + (y - b)² = r²
a_true = 3.0    # 圆心x坐标
b_true = 2.0    # 圆心y坐标
r_true = 5.0    # 半径
# 生成圆上的点
num_points = 100
# 在0到2π之间均匀生成角度
theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
# 生成无噪声的圆上点
x_true = a_true + r_true * np.cos(theta)
y_true = b_true + r_true * np.sin(theta)
# 添加高斯噪声
noise_std = 0.3
x_noisy = x_true + np.random.normal(0, noise_std, num_points)
y_noisy = y_true + np.random.normal(0, noise_std, num_points)
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
a_fit = -D_fit / 2
b_fit = -E_fit / 2
r_fit = np.sqrt(a_fit**2 + b_fit**2 - F_fit)

# 计算损失函数（点到圆心的距离与半径的差的平方）
distances = np.sqrt((x_noisy - a_fit)**2 + (y_noisy - b_fit)**2)
loss = np.mean((distances - r_fit)**2)



# 可视化部分
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# 原始数据和拟合结果对比
ax1 = axes

# 绘制带噪声的数据点
ax1.scatter(x_noisy, y_noisy, c='steelblue', s=20, alpha=0.6, label=f'noisy (n={num_points})')

# 绘制真实圆
theta_plot = np.linspace(0, 2*np.pi, 200)
x_true_circle = a_true + r_true * np.cos(theta_plot)
y_true_circle = b_true + r_true * np.sin(theta_plot)
ax1.plot(x_true_circle, y_true_circle, 'r-', linewidth=3, label=f'TRUE: heart_true({a_true}, {b_true}), r_true{r_true}', zorder=3)

# 绘制拟合圆
x_fit_circle = a_fit + r_fit * np.cos(theta_plot)
y_fit_circle = b_fit + r_fit * np.sin(theta_plot)
ax1.plot(x_fit_circle, y_fit_circle, 'g--', linewidth=3, label=f'fit: heart_fit({a_fit:.2f}, {b_fit:.2f}), r_fit{r_fit:.2f}', zorder=4)

# 绘制圆心
ax1.scatter(a_true, b_true, c='red', s=100, marker='o', label='true', zorder=5)
ax1.scatter(a_fit, b_fit, c='green', s=100, marker='s', label='fit', zorder=5)

# 设置图属性
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axis('equal')
ax1.set_xlim([a_true - 1.2*r_true, a_true + 1.2*r_true])
ax1.set_ylim([b_true - 1.2*r_true, b_true + 1.2*r_true])

# 添加信息文本框
info_text1 = f'number: {num_points}\nloss: {loss:.4f}'
ax1.text(0.02, 0.98, info_text1, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# 输出拟合圆的参数方程
print("\n圆方程对比:")
print(f"真实圆: (x - {a_true})² + (y - {b_true})² = {r_true}²")
print(f"拟合圆: (x - {a_fit:.4f})² + (y - {b_fit:.4f})² = {r_fit:.4f}²")
print(f"\n或表示为一般形式:")
print(f"真实圆: x² + y² - {2*a_true}x - {2*b_true}y + {a_true**2 + b_true**2 - r_true**2} = 0")
print(f"拟合圆: x² + y² + {D_fit:.4f}x + {E_fit:.4f}y + {F_fit:.4f} = 0")