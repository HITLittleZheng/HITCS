import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 加载MNIST数据集（只取前100个样本以加快速度）
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X = mnist.data[:100]  # 100张图片，每张28x28=784,X是一个100行784列的矩阵
y = mnist.target[:100].astype(int)

# 数据标准化：中心化 + 缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA（保留95%的方差）
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: {X_pca.shape[1]}")
print(f"累计解释方差比例: {pca.explained_variance_ratio_.sum():.2f}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# 训练逻辑回归分类器
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
# 输出PCA保留的主成分个数
print(f"PCA保留了 {pca.n_components_} 个主成分，解释了 {sum(pca.explained_variance_ratio_):.2%} 的方差")
