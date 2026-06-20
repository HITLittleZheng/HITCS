import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# RPCA 实现 (基于ADMM)
def rpca_admm(M, lam=None, mu=None, max_iter=100, tol=1e-6):
    """
    使用ADMM求解RPCA问题：min ||L||_* + λ||S||_1  s.t. M = L + S
    参数：
        M : 输入矩阵 (m x n)
        lam : 稀疏部分的正则化参数，默认 1/sqrt(max(m,n))
        mu : ADMM惩罚参数，默认 10 * lam
        max_iter : 最大迭代次数
        tol : 收敛容忍度
        verbose : 是否打印迭代信息
    返回：
        L : 低秩矩阵
        S : 稀疏矩阵
    """
    m, n = M.shape
    # 默认参数
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = 10 * lam

    # 初始化变量
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)

    # 软阈值函数
    def soft_threshold(X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    # SVD 阈值函数 (核范数近端)
    def svd_threshold(X, tau):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - tau, 0)
        return U @ np.diag(s_thresh) @ Vt

    for i in range(max_iter):
        # 更新 L: L = SVD阈值化(M - S + Y/mu, 1/mu)
        L = svd_threshold(M - S + Y / mu, 1 / mu)
        # 更新 S: S = 软阈值化(M - L + Y/mu, lam/mu)
        S = soft_threshold(M - L + Y / mu, lam / mu)
        # 更新 Y: Y = Y + mu * (M - L - S)
        Y = Y + mu * (M - L - S)
        # 检查收敛条件 (M-L-S的Frobenius范数)
        residual = np.linalg.norm(M - L - S, 'fro')
        if residual < tol:
            break

    return L, S

# 加载MNIST数据
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X = mnist.data[:100]          # 100张图片，每张28x28=784,X是一个100行784列的矩阵
y = mnist.target[:100].astype(int)

# 转换为浮点数 (RPCA要求浮点运算)
X = X.astype(np.float64)

print("Data shape:", X.shape)   # (100, 784)

# 应用RPCA 
print("Running RPCA (this may take a few minutes)...")
L, S = rpca_admm(X, lam=None, mu=None, max_iter=80, tol=1e-5)

print("Decomposition completed.")
print(f"Low-rank part shape: {L.shape}, Sparse part shape: {S.shape}")

# 可视化
n_samples = 5
indices = np.arange(n_samples)

# 重塑图像 (28x28)
original = X[indices].reshape(-1, 28, 28)
low_rank = L[indices].reshape(-1, 28, 28)
sparse = S[indices].reshape(-1, 28, 28)
reconstructed = low_rank + sparse

# 创建 4 行 × n_samples 列的子图
fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 2, 8))

# 定义每行的标题
row_titles = ['Original', 'Low-rank', 'Sparse', 'Reconstructed']

# 填充图像
for i, row_images in enumerate([original, low_rank, sparse, reconstructed]):
    for j in range(n_samples):
        axes[i, j].imshow(row_images[j], cmap='gray')
        axes[i, j].axis('off')
        # 只在第一列添加行标题
        if j == 0:
            axes[i, j].set_ylabel(row_titles[i], fontsize=12, rotation=0, labelpad=30)
        # 在第一行上方添加列标题（图片编号）
        if i == 0:
            axes[i, j].set_title(f"Sample {j}", fontsize=10)

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 划分训练集和测试集（低秩特征 L）
X_train, X_test, y_train, y_test = train_test_split(L, y, test_size=0.3, random_state=42)

# 训练线性 SVM 分类器
clf = LinearSVC(max_iter=1000, random_state=42, dual='auto')
clf.fit(X_train, y_train)

# 预测并输出准确率
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"RPCA + 低秩特征 + SVM 分类准确率: {acc:.4f}")