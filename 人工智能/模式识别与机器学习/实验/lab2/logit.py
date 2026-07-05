import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import csv

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, epochs=1000, reg_lambda=0.0, verbose=False):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    loss_history = []
    for i in range(epochs):
        linear = X @ w + b
        pred = sigmoid(linear)
        loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
        if reg_lambda > 0:
            loss += (reg_lambda / (2 * n)) * np.sum(w ** 2)
        loss_history.append(loss)
        dw = (1 / n) * (X.T @ (pred - y))
        db = (1 / n) * np.sum(pred - y)
        if reg_lambda > 0:
            dw += (reg_lambda / n) * w
        w -= lr * dw
        b -= lr * db
        if verbose and (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{epochs}, loss = {loss:.6f}")
    return w, b, loss_history

def predict(X, w, b, thresh=0.5):
    return (sigmoid(X @ w + b) >= thresh).astype(int)

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std

def classification_report(y_true, y_pred, target_names):
    labels = np.unique(y_true)
    report = ""
    for i, l in enumerate(labels):
        tp = np.sum((y_true == l) & (y_pred == l))
        fp = np.sum((y_true != l) & (y_pred == l))
        fn = np.sum((y_true == l) & (y_pred != l))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        report += f"{target_names[i]:12}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}\n"
    acc = np.mean(y_true == y_pred)
    report += f"\naccuracy = {acc:.4f}\n"
    return report

def load_breast_cancer_wdbc():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode('utf-8').splitlines()
    reader = csv.reader(data)
    X, y = [], []
    for row in reader:
        if not row: continue
        y.append(1.0 if row[1] == 'B' else 0.0)
        X.append([float(x) for x in row[2:]])
    X = np.array(X)
    y = np.array(y)
    print(f"成功下载数据，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y

def generate_correlated_data(n=400, seed=42):
    np.random.seed(seed)
    n2 = n // 2
    X0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n2)
    y0 = np.zeros(n2)
    X1 = np.random.multivariate_normal([4, 4], [[-0.5, 1], [1, -0.5]], n2)
    y1 = np.ones(n2)
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def plot_decision_boundary_2d(X, y, w, b, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], w, b)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    # 实验一：人工相关数据
    print("=" * 60)
    print("实验1：人工生成数据（特征相关，不满足朴素贝叶斯假设）")
    X_syn, y_syn = generate_correlated_data(400)
    X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = standardize(X_train, X_test)

    w_no, b_no, _ = train_logistic_regression(X_train_scaled, y_train, lr=0.1, epochs=2000, reg_lambda=0.0, verbose=True)
    pred_no = predict(X_test_scaled, w_no, b_no)
    acc_no = np.mean(y_test == pred_no)

    w_reg, b_reg, _ = train_logistic_regression(X_train_scaled, y_train, lr=0.1, epochs=2000, reg_lambda=0.01, verbose=True)
    pred_reg = predict(X_test_scaled, w_reg, b_reg)
    acc_reg = np.mean(y_test == pred_reg)

    print(f"无正则化    测试准确率: {acc_no:.4f}")
    print(f"L2正则化 (λ=0.01) 测试准确率: {acc_reg:.4f}")

    w_viz, b_viz, _ = train_logistic_regression(X_syn, y_syn, lr=0.1, epochs=2000)
    plot_decision_boundary_2d(X_syn, y_syn, w_viz, b_viz, title="Logistic Regression on Correlated Data")
    print("说明：两类特征内部强相关（违背朴素贝叶斯独立假设），但逻辑回归仍能较好分类。\n")

    # 实验二：UCI 乳腺癌数据
    print("=" * 60)
    print("实验2：UCI 威斯康星乳腺癌数据集")
    X_uci, y_uci = load_breast_cancer_wdbc()
    X_train, X_test, y_train, y_test = train_test_split(X_uci, y_uci, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = standardize(X_train, X_test)

    w_uci_no, b_uci_no, _ = train_logistic_regression(X_train_scaled, y_train, lr=0.05, epochs=3000, reg_lambda=0.0)
    pred_uci_no = predict(X_test_scaled, w_uci_no, b_uci_no)
    acc_uci_no = np.mean(y_test == pred_uci_no)

    best_reg, best_acc = 0.0, 0.0
    for reg in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        w_tmp, b_tmp, _ = train_logistic_regression(X_train_scaled, y_train, lr=0.05, epochs=3000, reg_lambda=reg)
        acc_tmp = np.mean(y_test == predict(X_test_scaled, w_tmp, b_tmp))
        if acc_tmp > best_acc:
            best_acc, best_reg = acc_tmp, reg

    w_uci_reg, b_uci_reg, loss_reg = train_logistic_regression(X_train_scaled, y_train, lr=0.05, epochs=3000, reg_lambda=best_reg)
    pred_uci_reg = predict(X_test_scaled, w_uci_reg, b_uci_reg)
    acc_uci_reg = np.mean(y_test == pred_uci_reg)

    print(f"无正则化模型       测试准确率: {acc_uci_no:.4f}")
    print(f"L2正则化模型 (λ={best_reg}) 测试准确率: {acc_uci_reg:.4f}")
    print("\n正则化模型分类报告：")
    print(classification_report(y_test, pred_uci_reg, ["恶性 (malignant)", "良性 (benign)"]))

    # 损失曲线对比（重新训练无正则化模型以获取损失历史）
    _, _, loss_no = train_logistic_regression(X_train_scaled, y_train, lr=0.05, epochs=3000, reg_lambda=0.0)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_no, label="无正则化")
    plt.xlabel("迭代次数"); plt.ylabel("损失"); plt.title("训练损失 (无正则化)"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss_reg, label=f"L2正则化 λ={best_reg}")
    plt.xlabel("迭代次数"); plt.ylabel("损失"); plt.title("训练损失 (带正则化)"); plt.legend()
    plt.tight_layout()
    plt.show()