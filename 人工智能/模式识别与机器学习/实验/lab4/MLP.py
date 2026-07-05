import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子保证可复现性
np.random.seed(42)

# ===================== 数据生成 =====================
def generate_spiral(n_samples, n_classes=4, noise=0.00001):
    """
    生成多类螺旋线数据，每类从一个不同角度起始的螺旋臂。
    参数:
        n_samples: 总样本数
        n_classes: 类别数 (>=2)
        noise: 高斯噪声标准差
    返回:
        X: shape (n_samples, 2)
        y: shape (n_samples,)
    """
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    for c in range(n_classes):
        # 每类螺旋臂起始角度偏移
        theta_start = c * (2 * np.pi / n_classes)
        # 半径从 0.5 线性增加到 2.5
        t = np.linspace(0, 1, samples_per_class)
        r = 0.5 + 1.0 * t
        theta = theta_start + 1.0 * np.pi * t  
        x1 = r * np.cos(theta) + noise * np.random.randn(samples_per_class)
        x2 = r * np.sin(theta) + noise * np.random.randn(samples_per_class)
        X_list.append(np.stack([x1, x2], axis=1))
        y_list.append(np.full(samples_per_class, c))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    # 随机打乱
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ===================== 基础模型 =====================
def softmax(z):
    """稳定版本的softmax"""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失，y_pred是概率矩阵，y_true是整数标签"""
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(m), y_true])
    return np.mean(log_likelihood)

def one_hot(y, n_classes):
    """将整数标签转换为one-hot"""
    m = y.shape[0]
    oh = np.zeros((m, n_classes))
    oh[np.arange(m), y] = 1
    return oh

class LinearClassifier:
    """线性分类器：Softmax回归（无隐藏层）"""
    def __init__(self, input_dim, n_classes):
        self.W = np.random.randn(input_dim, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))
    
    def forward(self, X):
        self.z = X @ self.W + self.b
        self.probs = softmax(self.z)
        return self.probs
    
    def backward(self, X, y_true_onehot):
        m = X.shape[0]
        # 梯度计算: dL/dz = probs - y_onehot
        dZ = self.probs - y_true_onehot
        dW = (X.T @ dZ) / m
        db = np.mean(dZ, axis=0, keepdims=True)
        return dW, db
    
    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

class MLP:
    """多层感知机：一个隐藏层，可配置神经元数量"""
    def __init__(self, input_dim, hidden_dim, n_classes, activation='relu'):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, n_classes) * 0.01
        self.b2 = np.zeros((1, n_classes))
        self.activation = activation
    
    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Unsupported activation")
    
    def _derivative_activate(self, a):
        if self.activation == 'relu':
            return (a > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - a**2
    
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activate(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs
    
    def backward(self, y_true_onehot):
        m = self.X.shape[0]
        # 输出层梯度
        dZ2 = self.probs - y_true_onehot
        dW2 = (self.a1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)
        # 隐藏层梯度
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._derivative_activate(self.a1)
        dW1 = (self.X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2
    
    def update(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# ===================== 训练流程 =====================
def train_model(model, X_train, y_train, X_val, y_val, lr=0.01, epochs=500, verbose=True):
    """训练模型并记录损失与准确率"""
    n_classes = len(np.unique(y_train))
    y_train_oh = one_hot(y_train, n_classes)
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # 前向与反向传播
        if isinstance(model, LinearClassifier):
            model.forward(X_train)
            dW, db = model.backward(X_train, y_train_oh)
            model.update(dW, db, lr)
            loss = cross_entropy_loss(model.probs, y_train)
        else:  # MLP
            model.forward(X_train)
            dW1, db1, dW2, db2 = model.backward(y_train_oh)
            model.update(dW1, db1, dW2, db2, lr)
            loss = cross_entropy_loss(model.probs, y_train)
        
        train_losses.append(loss)
        
        # 验证集准确率
        y_pred = model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        val_accs.append(acc)
        
        if verbose and (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Acc: {acc:.4f}")
    
    return train_losses, val_accs

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return np.mean(y_pred == y_test)

# ===================== 可视化 =====================
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """绘制决策边界，X为二维特征，y为标签"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.tab10, edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_loss_curve(losses, title):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.grid(True)
    plt.show()

# ===================== 主实验 =====================
if __name__ == "__main__":
    # 生成数据集（非线性螺旋形，4类）
    n_total = 800
    X, y = generate_spiral(n_total, n_classes=4, noise=0.1)
    
    # 分割训练/验证/测试集 (60%,20%,20%)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    
    # ------------------- 实验3.1：线性分类器 vs MLP -------------------
    print("=== 实验3.1: 线性分类器 vs MLP ===")
    # 线性分类器
    linear_model = LinearClassifier(input_dim=2, n_classes=4)
    print("训练线性分类器...")
    losses_lin, accs_lin = train_model(linear_model, X_train, y_train, X_val, y_val, lr=0.05, epochs=500, verbose=False)
    test_acc_lin = evaluate(linear_model, X_test, y_test)
    print(f"线性分类器测试准确率: {test_acc_lin:.4f}")
    plot_decision_boundary(linear_model, X_test, y_test, title="Linear Classifier Decision Boundary")
    
    # MLP (隐藏层20个神经元，ReLU)
    mlp_model = MLP(input_dim=2, hidden_dim=20, n_classes=4, activation='relu')
    print("训练MLP...")
    losses_mlp, accs_mlp = train_model(mlp_model, X_train, y_train, X_val, y_val, lr=0.1, epochs=5000, verbose=False)
    test_acc_mlp = evaluate(mlp_model, X_test, y_test)
    print(f"MLP测试准确率: {test_acc_mlp:.4f}")
    plot_decision_boundary(mlp_model, X_test, y_test, title="MLP (20 hidden, ReLU) Decision Boundary")
    
    # ------------------- 实验3.2：不同数据量对MLP的影响 -------------------
    print("\n=== 实验3.2: 不同数据量对MLP的影响 ===")
    data_sizes = [50, 100, 300]
    test_accs_size = []
    for size in data_sizes:
        X_sub, y_sub = generate_spiral(size, n_classes=4, noise=0.05)
        split = int(0.8*size)  # 简单拆分，不重复验证
        X_tr, X_te = X_sub[:split], X_sub[split:]
        y_tr, y_te = y_sub[:split], y_sub[split:]
        mlp = MLP(2, 20, 4, 'relu')
        train_model(mlp, X_tr, y_tr, X_te, y_te, lr=1.0, epochs=3000, verbose=False)
        acc = evaluate(mlp, X_te, y_te)
        test_accs_size.append(acc)
        print(f"数据量 = {size}, 测试准确率 = {acc:.4f}")
    
    # 绘图对比
    plt.figure()
    plt.plot(data_sizes, test_accs_size, 'o-')
    plt.xlabel("Training Set Size")
    plt.ylabel("Test Accuracy")
    plt.title("MLP Performance vs Data Size")
    plt.grid(True)
    plt.show()
    
    # ------------------- 实验3.3：不同隐藏层大小对MLP的影响 -------------------
    print("\n=== 实验3.3: 不同隐藏层神经元数对MLP的影响 ===")
    hidden_sizes = [5, 20, 30]
    test_accs_hidden = []
    for h in hidden_sizes:
        mlp = MLP(2, h, 4, 'relu')
        train_model(mlp, X_train, y_train, X_val, y_val, lr=1.0, epochs=5000, verbose=False)
        acc = evaluate(mlp, X_test, y_test)
        test_accs_hidden.append(acc)
        print(f"隐藏层大小 = {h}, 测试准确率 = {acc:.4f}")
    plt.figure()
    plt.plot(hidden_sizes, test_accs_hidden, 's-')
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Test Accuracy")
    plt.title("MLP Performance vs Hidden Size")
    plt.grid(True)
    plt.show()
    
    # ------------------- 实验3.4：不同学习率对比 -------------------
    print("\n=== 实验3.4: 不同学习率对MLP的影响 ===")
    lrs = [0.001, 0.01, 0.1,1.0]
    test_accs_lr = []
    for lr in lrs:
        mlp = MLP(2, 20, 4, 'relu')
        train_model(mlp, X_train, y_train, X_val, y_val, lr=lr, epochs=500, verbose=False)
        acc = evaluate(mlp, X_test, y_test)
        test_accs_lr.append(acc)
        print(f"学习率 = {lr}, 测试准确率 = {acc:.4f}")
    plt.figure()
    plt.plot(lrs, test_accs_lr, 'd-')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.title("MLP Performance vs Learning Rate")
    plt.grid(True)
    plt.show()
    
    # 展示MLP训练过程中的损失下降曲线
    plot_loss_curve(losses_mlp, "MLP Loss Curve (Training)")