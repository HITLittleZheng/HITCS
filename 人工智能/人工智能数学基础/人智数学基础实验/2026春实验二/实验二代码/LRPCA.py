import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
device = torch.device('cpu')
dtype = torch.float32

# 视频尺寸
height = 30          # 帧高度
width = 30           # 帧宽度
n_frames = 100       # 帧数
n_pixels = height * width  # 每帧像素数

# 小球参数
ball_radius = 3 #小球的半径（以像素为单位）
ball_value = 2 #小球区域的像素值（亮度值）

# LRPCA 参数
rank = 1                     # 背景是静态图像，秩为 1
max_iter =20                 # 迭代层数（可学习参数的层数）
lr = 0.01                    # 学习率
epochs = 100                 # 训练轮数

# 生成模拟视频
def generate_video():
    """生成渐变背景 + 移动小球的视频"""
    # 创建渐变背景（每帧相同，秩为1）
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    background_img = (xx + yy) / 2   # 范围 [0,1]，从左上(0)到右下(1)渐变
    background_img = background_img.astype(np.float32)

    # 低秩矩阵 X_star：每列都是相同的背景图像
    X_star = background_img.flatten()[:, np.newaxis]   # 二维图像展开成一维向量(n_pixels, 1)
    X_star = np.tile(X_star, (1, n_frames))            # 一维向量拼接成二维矩阵(n_pixels, n_frames)

    # 稀疏前景 S_star：小球移动
    S_star = np.zeros((n_pixels, n_frames), dtype=np.float32)

    for t in range(n_frames):
        # 小球沿正弦曲线移动
        center_x = int(ball_radius + (width - 2*ball_radius) * t / (n_frames-1))
        center_y = int(height/2 + height/4 * np.sin(2 * np.pi * t / n_frames))
        # 绘制小球
        for dx in range(-ball_radius, ball_radius+1):
            for dy in range(-ball_radius, ball_radius+1):
                if dx*dx + dy*dy <= ball_radius*ball_radius:
                    x = center_x + dx
                    y = center_y + dy
                    if 0 <= x < width and 0 <= y < height:
                        idx = y * width + x
                        S_star[idx, t] = ball_value

    # 添加少量高斯噪声
    noise_level = 0.03
    noise = np.random.randn(n_pixels, n_frames) * noise_level
    Y = X_star + S_star + noise

    # 转换为张量
    X_star_t = torch.tensor(X_star, dtype=dtype, device=device)
    S_star_t = torch.tensor(S_star, dtype=dtype, device=device)
    Y_t = torch.tensor(Y, dtype=dtype, device=device)
    return Y_t, X_star_t, S_star_t

#LRPCA 网络 
class LRPCA(nn.Module):
    def __init__(self, rank, max_iter):
        super(LRPCA, self).__init__()
        self.rank = rank
        self.max_iter = max_iter
        # 可学习的阈值 zeta 和步长 eta（每个迭代层一个）
        self.zeta = nn.ParameterList([nn.Parameter(torch.tensor(0.1, dtype=dtype)) for _ in range(max_iter+1)])
        self.eta  = nn.ParameterList([nn.Parameter(torch.tensor(0.5, dtype=dtype)) for _ in range(max_iter+1)])

    def soft_threshold(self, M, theta):
        """软阈值算子"""
        return torch.sign(M) * torch.max(torch.abs(M) - theta, torch.tensor(0.0, dtype=dtype))

    def forward(self, Y):
        # 初始化
        S = self.soft_threshold(Y, self.zeta[0])
        # 对 Y - S 做截断 SVD，得到初始 L, R
        # 使用 torch.svd_lowrank (CPU 上可用)
        U, s, V = torch.svd_lowrank(Y - S, q=self.rank, niter=2)
        sqrt_s = torch.sqrt(s)
        L = U * sqrt_s           # (n_pixels, rank)
        R = V * sqrt_s           # (n_frames, rank)

        # 迭代更新
        for t in range(1, self.max_iter+1):
            X = L @ R.T
            S = self.soft_threshold(Y - X, self.zeta[t])
            E = Y - X - S

            # 缩放梯度下降
            RTR_inv = torch.inverse(R.T @ R + 1e-8 * torch.eye(self.rank, device=device))#计算矩阵的逆
            LTR_inv = torch.inverse(L.T @ L + 1e-8 * torch.eye(self.rank, device=device))#计算矩阵的逆

            L = L - self.eta[t] * (E @ R) @ RTR_inv
            R = R - self.eta[t] * (E.T @ L) @ LTR_inv

        # 最终恢复的低秩矩阵
        X_final = L @ R.T
        return X_final, S

# 训练
def train():
    # 生成数据（训练集只用一组数据，也可每轮重新生成，这里固定以便观察）
    Y_train, X_true, S_true = generate_video()
    print(f"视频大小: {Y_train.shape}, 真实背景秩: {torch.linalg.matrix_rank(X_true).item()}")

    model = LRPCA(rank, max_iter)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_pred, S_pred = model(Y_train)
        loss = loss_fn(X_pred, X_true)   # 直接监督背景恢复
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # 测试最终结果
    with torch.no_grad():
        X_final, S_final = model(Y_train)
        final_loss = loss_fn(X_final, X_true).item()
        print(f"\n训练完成，最终loss: {final_loss:.6f}")

    # 可视化
    visualize_results(Y_train, X_true, S_true, X_final, S_final, height, width)

    return model

# 可视化
def visualize_results(Y, X_true, S_true, X_pred, S_pred, h, w):
    """显示第一帧的真实背景、观测、恢复背景等"""
    # 取第一帧（索引0）
    frame_obs = Y[:, 0].cpu().numpy().reshape(h, w)
    frame_bg_true = X_true[:, 0].cpu().numpy().reshape(h, w)
    frame_fg_true = S_true[:, 0].cpu().numpy().reshape(h, w)
    frame_bg_pred = X_pred[:, 0].cpu().numpy().reshape(h, w)
    frame_fg_pred = S_pred[:, 0].cpu().numpy().reshape(h, w)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1); plt.imshow(frame_obs, cmap='gray'); plt.title("Observe")
    plt.subplot(2, 3, 2); plt.imshow(frame_bg_true, cmap='gray'); plt.title("True background")
    plt.subplot(2, 3, 3); plt.imshow(frame_fg_true, cmap='gray'); plt.title("True ball")
    plt.subplot(2, 3, 4); plt.imshow(frame_bg_pred, cmap='gray'); plt.title("recovered background")
    plt.subplot(2, 3, 5); plt.imshow(frame_fg_pred, cmap='gray'); plt.title("recovered ball")
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    train()