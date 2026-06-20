import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 参数设置 
np.random.seed(42)          # 设置种子以确保结果可复现
N = 1000                    # 采样点数
a= 1.0                      # 振幅
w1= 2 * np.pi               # 角频率
noise_std = 0.1             # 高斯噪声标准差
# 生成随机 t 
t = np.random.uniform(0, 5, N)
# 计算信号并添加高斯噪声 (均值 0, 标准差 noise_std)
y = a * np.sin(w1 * t) +np.random.normal(0, noise_std, N)

# 转换为PyTorch张量
t_tensor = torch.FloatTensor(t).reshape(-1, 1)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
epochs = 10000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(t_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()#反向传播
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 预测
with torch.no_grad():
    predicted = model(t_tensor).numpy().flatten()
# 生成原始正弦曲线（无噪声）用于可视化对比
t_linspace = np.linspace(0, 5, 1000)  # 用于绘制平滑曲线
y_clean = 1.0 * np.sin(2 * np.pi * t_linspace)  # 原始无噪声信号

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制噪声点（训练数据点）
plt.scatter(t, y, c='gray', s=10, alpha=0.6, label='point', zorder=1)

# 绘制原始曲线（无噪声）
plt.plot(t_linspace, y_clean, 'b-', linewidth=2, alpha=0.8, label='TRUE', zorder=2)

# 绘制MLP拟合曲线
# 注意：我们需要对t_linspace进行预测来得到平滑的拟合曲线
t_linspace_tensor = torch.FloatTensor(t_linspace).reshape(-1, 1)
with torch.no_grad():
    y_fitted = model(t_linspace_tensor).numpy().flatten()

plt.plot(t_linspace, y_fitted, 'r-', linewidth=2.5, label='MLP', zorder=3)

# 设置图形属性
plt.xlabel('t', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('MLP', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right')
plt.xlim(0, 5)
plt.ylim(-1.5, 1.5)

# 显示图形
plt.tight_layout()
plt.show()

# 可选：显示损失函数变化曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, 'b-', alpha=0.7, linewidth=1.5)
plt.fill_between(range(len(losses)), losses, alpha=0.3, color='blue')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss (MSE)', fontsize=12)
plt.title('LOSS', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
