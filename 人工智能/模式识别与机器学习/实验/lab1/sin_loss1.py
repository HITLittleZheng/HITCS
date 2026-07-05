import numpy as np

# 超参数 
seed=42          # 设置种子以确保结果可复现
train_sizes = [10, 100] # 创建不同训练集大小
a= 1.0                      # 振幅
omega= 2 * np.pi               # 角频率
noise_std = 0.1             # 高斯噪声标准差
degrees = [3, 6, 9] # 多项式阶数
lambdas = [0.0, 0.1]# 正则化系数
LR=0.01
N_ITER=20000
epsilon=1e-12 # 避免除零
train_ratio=0.6
valid_ratio=0.2

def buildX(x, degree):
    """
    构造X：第一列为1（截距），后续为 x, x^2, ..., x^degree
    参数:
        x: (N,) 一维数组
        degree: 多项式阶数
    返回:
        X: (N, degree+1) 设计矩阵
    """
    N = len(x)
    X = np.zeros((N, degree + 1))
    for i in range(degree + 1):
        X[:, i] = x ** i
    return X

def data(N, seed, train_ratio, valid_ratio):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, N)
    x = 2 * (x - x.min()) / (x.max() - x.min() + 1e-12) - 1
    y = 1.0 * np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, N)
    n_train = int(N * train_ratio)
    n_valid = int(N * valid_ratio)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
    x_test, y_test = x[n_train+n_valid:], y[n_train+n_valid:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def loss1(y, x, w, lambda_reg,degree):
    """
    if lambda_reg < 0:
        报错
    elif lambda_reg==0:
        计算带无正则化的损失
        损失 = 1/2 * Σ(y_pred-y)^2
    else:
        计算带 L2 正则化的损失
        损失 = 1/2* Σ(y_pred-y)^2 + (lambda/2) * Σ_{j>=1} w_j^2
    """
    X=buildX(x,degree)
    y_pred = X @ w
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative (>= 0)")
    loss1 = 0.5 * np.sum((y_pred-y) ** 2)
    regular= 0.5 * lambda_reg * np.sum(w ** 2)
    return loss1 + regular

def loss1_gradient(X, y, w, lambda_reg):
    """
    计算损失函数关于参数 w 的梯度
    梯度 =  X^T (Xw-y) + lambda * w
    """
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative (>= 0)")
    y_pred = X @ w
    grad = X.T @ (y_pred-y)+lambda_reg * w
    return grad

# 梯度下降优化 
def loss1_gradient_descent(x, y, degree, lambda_reg=0.0, learning_rate=0.1, n_iter=2000):
    """
    使用梯度下降求解多项式回归参数
    参数:
        x: (N,) 一维数组
        y: 目标值 (一维数组)
        degree: 多项式阶数
        lambda_reg: L2 正则化系数
        learning_rate: 学习率
        n_iter: 迭代次数
    返回:
        w: 最优参数 (degree+1,)
        loss_history: 每次迭代的损失值列表
    """
    X=buildX(x,degree)
    w = np.random.randn(degree + 1) * 0.01  # 小随机初始化
    loss_history = []
    
    for i in range(n_iter):
        loss = loss1(y, x, w, lambda_reg,degree)
        loss_history.append(loss)
        
        grad = loss1_gradient(X, y, w, lambda_reg)
        w -= learning_rate * grad
        if np.sqrt(np.sum(grad ** 2))<1e-6:
            num=i+1
            break
        num=i+1
        
    return w, loss_history,num

def loss1_solve(x, y, lambda_reg,degree):
    """
    解析法求w
    参数:
        x: (N,) 一维数组
        y: 目标值 (一维数组)
        lambda_reg: L2 正则化系数
    返回:
        w: 最优参数 (degree+1,)
    """
    X=buildX(x,degree)
    if lambda_reg==0:
        w = np.linalg.pinv(X) @ y #防止病态矩阵
    else:
        I = np.eye(degree+1)
        inv=np.linalg.inv(X.T@X+lambda_reg*I)
        w = inv@X.T @ y
    return w

# 共轭梯度
def CG(x, y, degree, lambda_reg=0.0, n_iter=2000):
    """
    使用梯度下降求解多项式回归参数
    参数:
        x: (N,) 一维数组
        y: 目标值 (一维数组)
        degree: 多项式阶数
        lambda_reg: L2 正则化系数
        n_iter: 迭代次数
    返回:
        w: 最优参数 (degree+1,)
        loss_history: 每次迭代的损失值列表
    """
    X=buildX(x,degree)
    w = np.random.randn(degree + 1) * 0.01  # 小随机初始化
    I = np.eye(degree+1)
    A=X.T@X+lambda_reg*I
    b=X.T@y
    loss_history = []
    
    r_k=b-A@w
    p=r_k
    for i in range(n_iter):
        a=(r_k.T@r_k)/(p.T@A@p)
        w+=a*p
        r_k1=b-A@w
        beta=(r_k1.T@r_k1)/(r_k.T@r_k)
        p=r_k1+beta*p
        r_k=r_k1
        loss = loss1(y, x, w, lambda_reg,degree)
        loss_history.append(loss)
        if np.sqrt(np.sum(r_k** 2))<1e-6:
            num=i+1
            break
        num=i+1

    return w, loss_history,num

#主实验
for N in train_sizes:
    x_train, y_train, x_valid, y_valid, x_test, y_test=data(N, seed, train_ratio, valid_ratio)
    for deg in degrees:
        for lam in lambdas:
            w1, _ ,num= loss1_gradient_descent(x_train, y_train, deg, lam, learning_rate=LR, n_iter=N_ITER)
            train_loss_1=loss1(y_train,x_train,w1,lam,deg)
            valid_loss_1=loss1(y_valid,x_valid,w1,lam,deg)
            test_loss_1=loss1(y_test,x_test,w1,lam,deg)
            print(f"===训练集数量为{N},阶数为{deg},lamdda为{lam}的梯度下降法,所需迭代次数为{num}===")
            print(f"train_loss1={train_loss_1:.6f}|vaild_loss1={valid_loss_1:.6f}|test_loss1={test_loss_1:.6f}")

            w1, _ ,num= CG(x_train, y_train, deg, lam, n_iter=N_ITER)
            train_loss_1=loss1(y_train,x_train,w1,lam,deg)
            valid_loss_1=loss1(y_valid,x_valid,w1,lam,deg)
            test_loss_1=loss1(y_test,x_test,w1,lam,deg)
            print(f"===训练集数量为{N},阶数为{deg},lamdda为{lam}的共轭梯度,所需迭代次数为{num}===")
            print(f"train_loss1={train_loss_1:.6f}|vaild_loss1={valid_loss_1:.6f}|test_loss1={test_loss_1:.6f}")

            w1= loss1_solve(x_train, y_train,lam,deg)
            train_loss_1=loss1(y_train,x_train,w1,lam,deg)
            valid_loss_1=loss1(y_valid,x_valid,w1,lam,deg)
            test_loss_1=loss1(y_test,x_test,w1,lam,deg)
            print(f"===训练集数量为{N},阶数为{deg},lamdda为{lam}的解析法===")
            print(f"train_loss1={train_loss_1:.6f}|vaild_loss1={valid_loss_1:.6f}|test_loss1={test_loss_1:.6f}")

import matplotlib.pyplot as plt
def predict(x, w, degree):
    X = buildX(x, degree)
    return X @ w
# 损失收敛曲线（梯度下降过程中）
# 注意：我们需要在梯度下降函数调用时，把 loss_history 保存下来用于绘图。
# 为了不产生过多弹窗，单独运行一次代表性（ N=100, deg=6, lam=0）的梯度下降并绘制收敛曲线
N = 100
deg = 6
lam = 0.0
x_train, y_train, x_valid, y_valid, x_test, y_test=data(N, seed, train_ratio, valid_ratio)

w_gd_rep, loss_hist,_ = loss1_gradient_descent(x_train, y_train, deg, lam,learning_rate=LR, n_iter=N_ITER)
plt.figure(figsize=(8,5))
plt.plot(loss_hist, linewidth=1.5)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Gradient Descent Convergence (N={N}, degree={deg}, λ={lam})')
plt.grid(True, alpha=0.3)
plt.show()

# 拟合曲线对比（梯度下降 vs 解析法，并显示真实曲线及散点）
highlight_cases = [
    (10, 3, 0.0),(10, 9, 0.0),(100, 9, 0.0)
]

for (N, deg, lam) in highlight_cases:
    x_train, y_train, x_valid, y_valid, x_test, y_test=data(N, seed, train_ratio, valid_ratio)
    
    # 梯度下降求 w
    w_gd, _ ,_ = loss1_gradient_descent(x_train, y_train, deg, lam,
                                     learning_rate=0.01, n_iter=20000)
    # 解析法求 w
    w_ana = loss1_solve(x_train, y_train, lam, deg)
    
    # 生成密集预测曲线
    x_dense = np.linspace(-1, 1, 300)
    y_true_dense = 1.0 * np.sin(2 * np.pi * x_dense)
    y_gd_dense = predict(x_dense, w_gd, deg)
    y_ana_dense = predict(x_dense, w_ana, deg)
    
    plt.figure(figsize=(8,5))
    plt.plot(x_dense, y_true_dense, 'k-', label='True sin(2πx)', linewidth=2)
    plt.plot(x_dense, y_gd_dense, 'b--', label='Gradient Descent', alpha=0.8)
    plt.plot(x_dense, y_ana_dense, 'r:', label='Analytical', alpha=0.8)
    plt.scatter(x_train, y_train, c='green', marker='o', s=25, label='Train', alpha=0.6)
    plt.scatter(x_valid, y_valid, c='orange', marker='s', s=25, label='Valid', alpha=0.6)
    plt.scatter(x_test, y_test, c='purple', marker='^', s=25, label='Test', alpha=0.6)
    plt.xlabel('x (scaled to [-1,1])')
    plt.ylabel('y')
    plt.title(f'N={N}, degree={deg}, λ={lam}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
# 正则化作用对比子图（抑制过拟合）
np.random.seed(42)
N = 300
deg=100
x_train, y_train, x_valid, y_valid, x_test, y_test=data(N, seed, train_ratio, valid_ratio)

def rmse(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
ln_lambdas = np.linspace(-20, -2, 30)  
lambdas = np.exp(ln_lambdas) 

# 存储 RMSE
train_rmse = []
valid_rmse = []
test_rmse = []
for lam in lambdas:
    # 解析法求解参数
    w= loss1_solve(x_train, y_train,lam,deg)
    
    # 预测
    y_train_pred = buildX(x_train, deg) @ w
    y_valid_pred = buildX(x_valid, deg) @ w
    y_test_pred  = buildX(x_test,  deg) @ w
    
    # 计算 RMSE（注意这里直接用真实 y，不用正则化损失）
    train_rmse.append(rmse(y_train, y_train_pred))
    valid_rmse.append(rmse(y_valid, y_valid_pred))
    test_rmse.append(rmse(y_test, y_test_pred))
# 绘图
plt.figure(figsize=(10, 6))
plt.plot(ln_lambdas, train_rmse, 'b-o', label='Train RMSE', linewidth=2)
plt.plot(ln_lambdas, valid_rmse, 'g-s', label='Validation RMSE', linewidth=2)
plt.plot(ln_lambdas, test_rmse,  'r-^', label='Test RMSE', linewidth=2)
plt.xlabel('ln λ ')
plt.ylabel('RMSE')
plt.title(f'RMSE vs λ (degree={deg}, N_train={len(x_train)})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# 打印最佳 λ（验证集 RMSE 最小）
best_idx = np.argmin(valid_rmse)
best_ln_lambda = ln_lambdas[best_idx]
print(f"Best lnλ based on validation RMSE: {best_ln_lambda:.6f}")
print(f"Corresponding test RMSE: {test_rmse[best_idx]:.6f}")

# 正则化
highlight_cases = [
    (300,100,0.0),(300,100,np.exp(-18.137931))
]

for (N, deg, lam) in highlight_cases:
    x_train, y_train, x_valid, y_valid, x_test, y_test=data(N, seed, train_ratio, valid_ratio)
    
    # 解析法求 w
    w_ana = loss1_solve(x_train, y_train, lam, deg)
    
    # 生成密集预测曲线
    x_dense = np.linspace(-1, 1, 300)
    y_true_dense = 1.0 * np.sin(2 * np.pi * x_dense)
    y_ana_dense = predict(x_dense, w_ana, deg)
    
    plt.figure(figsize=(8,5))
    plt.plot(x_dense, y_true_dense, 'k-', label='True sin(2πx)', linewidth=2)
    plt.plot(x_dense, y_ana_dense, 'r:', label='Analytical', alpha=0.8)
    plt.scatter(x_train, y_train, c='green', marker='o', s=25, label='Train', alpha=0.6)
    plt.scatter(x_valid, y_valid, c='orange', marker='s', s=25, label='Valid', alpha=0.6)
    plt.scatter(x_test, y_test, c='purple', marker='^', s=25, label='Test', alpha=0.6)
    plt.xlabel('x (scaled to [-1,1])')
    plt.ylabel('y')
    plt.title(f'N={N}, degree={deg}, λ={lam}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()