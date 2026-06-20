import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge

# 参数设置 
np.random.seed(42)          # 设置种子以确保结果可复现
N = 1000                    # 采样点数
a= 1.0                      # 振幅
w1= 2 * np.pi               # 角频率
noise_std = 0.1             # 高斯噪声标准差
# 生成随机 t (假设 t 在 [0, 10] 均匀分布)
t = np.random.uniform(0, 10, N)
# 计算信号并添加高斯噪声 (均值 0, 标准差 noise_std)
y = a * np.sin(w1 * t) +np.random.normal(0, noise_std, N)

# 定义模型函数
def model(x, a, w1):
    return a * np.sin(w1 * x) 
# 非线性最小二乘拟合
# 注意：必须提供初始猜测 p0，否则极易陷入局部最优或无法收敛
initial_guess = [1.5, 6.5] 
params, covariance = curve_fit(model, t, y, p0=initial_guess)

# 输出结果
print(f"TRUE:y={a}sin({w1:.3f}t)")
print(f"回归：y={params[0]:.3f}sin({params[1]:.3f}t)")


# 构建候选频率字典 (将非线性问题转化为线性稀疏选择问题)
# 注意：频率必须落在网格上，否则会有基匹配误差 (Basis Mismatch)
w_grid = np.linspace(0, 10, 1000)  # 范围需覆盖真实频率，密度影响精度
X = np.sin(np.outer(t, w_grid))   # 特征矩阵 (N_samples, N_features)

# L1 正则化拟合 (Lasso)
# fit_intercept=False 因为正弦波组合均值为 0
# alpha 需根据噪声水平调整，过大导致欠拟合，过小导致过拟合
lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=10000)
lasso.fit(X, y)
# 选择主导频率
idx = np.argmax(np.abs(lasso.coef_))
w_l1=w_grid[idx]
X=np.column_stack([np.sin(w_l1*t)])
a_l1=np.linalg.lstsq(X,y,rcond=None)[0]
a_l1_fit=a_l1[0]
print(f"L1：y={a_l1_fit:.3f}sin({w_l1:.3f}t)")

w_grid = np.linspace(0, 10, 1000)
X = np.sin(np.outer(t, w_grid))
# L2 正则化拟合 (Ridge)
ridge = Ridge(alpha=0.001, fit_intercept=False)  
ridge.fit(X, y)
idx = np.argmax(np.abs(ridge.coef_))
w_l2=w_grid[idx]
X=np.column_stack([np.sin(w_l2*t)])
a_l2=np.linalg.lstsq(X,y,rcond=None)[0]
a_l2_fit=a_l2[0]
print(f"L2：y={a_l2_fit:.3f}sin({w_l2:.3f}t)")

# 可视化
plt.figure(figsize=(12, 8))
# 绘制数据点
plt.scatter(t, y, alpha=0.3, s=10, color='lightgray', label='Noisy Data')
# 创建用于绘制平滑曲线的x轴
t_smooth = np.linspace(0, 10, 1000)
# 真实曲线
y_true = a * np.sin(w1 * t_smooth)
plt.plot(t_smooth, y_true, 'r-', linewidth=2, label=f'True: y={a}sin({w1:.2f}t)')
# 非线性最小二乘回归曲线
y_nls = params[0] * np.sin(params[1] * t_smooth)
plt.plot(t_smooth, y_nls, 'b--', linewidth=2, label=f'NLS: y={params[0]:.2f}sin({params[1]:.2f}t)')

# L1回归曲线
y_l1 = a_l1_fit * np.sin(w_l1 * t_smooth)
plt.plot(t_smooth, y_l1, 'g:', linewidth=2, label=f'L1: y={a_l1_fit:.2f}sin({w_l1:.2f}t)')

# L2回归曲线
y_l2 = a_l2_fit * np.sin(w_l2 * t_smooth)
plt.plot(t_smooth, y_l2, 'm-.', linewidth=2, label=f'L2: y={a_l2_fit:.2f}sin({w_l2:.2f}t)')

# 设置图形属性
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Amplitude (y)', fontsize=12)
plt.title('Comparison of Regression Methods for Sinusoidal Signal', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 为了更清晰地比较拟合结果，创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Detailed Comparison of Fitting Results', fontsize=16, fontweight='bold')

# 真实曲线
axes[0, 0].scatter(t, y, alpha=0.3, s=5, color='lightgray')
axes[0, 0].plot(t_smooth, y_true, 'r-', linewidth=2)
axes[0, 0].set_title(f'True Signal: y={a}sin({w1:.2f}t)')
axes[0, 0].set_xlabel('Time (t)')
axes[0, 0].set_ylabel('Amplitude (y)')
axes[0, 0].grid(True, alpha=0.3)

# 非线性最小二乘回归
axes[0, 1].scatter(t, y, alpha=0.3, s=5, color='lightgray')
axes[0, 1].plot(t_smooth, y_nls, 'b--', linewidth=2)
axes[0, 1].set_title(f'Nonlinear Least Squares: y={params[0]:.3f}sin({params[1]:.3f}t)')
axes[0, 1].set_xlabel('Time (t)')
axes[0, 1].set_ylabel('Amplitude (y)')
axes[0, 1].grid(True, alpha=0.3)

# L1回归
axes[1, 0].scatter(t, y, alpha=0.3, s=5, color='lightgray')
axes[1, 0].plot(t_smooth, y_l1, 'g:', linewidth=2)
axes[1, 0].set_title(f'L1 Regression: y={a_l1_fit:.3f}sin({w_l1:.3f}t)')
axes[1, 0].set_xlabel('Time (t)')
axes[1, 0].set_ylabel('Amplitude (y)')
axes[1, 0].grid(True, alpha=0.3)
# L2回归
axes[1, 1].scatter(t, y, alpha=0.3, s=5, color='lightgray')
axes[1, 1].plot(t_smooth, y_l2, 'm-.', linewidth=2)
axes[1, 1].set_title(f'L2 Regression: y={a_l2_fit:.3f}sin({w_l2:.3f}t)')
axes[1, 1].set_xlabel('Time (t)')
axes[1, 1].set_ylabel('Amplitude (y)')
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()