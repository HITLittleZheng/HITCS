# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 定义正弦函数
def sin_func(x):
    return np.sin(2 * np.pi * x)

# 定义多项式函数
def poly_func(w, x):
    n = len(w)
    y = 0
    for i in range(n):
        y += w[i] * (x ** i)
    return y

# 定义损失函数（均方误差）
def loss_func(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred) ** 2)

# 定义带惩罚项的损失函数（L2范数）
def loss_func_reg(y_true, y_pred, w, lam):
    return loss_func(y_true, y_pred) + 0.5 * lam * np.sum(w ** 2)

# 定义梯度下降法
def gradient_descent(x, y, w, lr, epochs, lam=0):
    n = len(w)
    m = len(x)
    loss_list = [] # 存储损失函数值
    for i in range(epochs):
        # 计算预测值
        y_pred = poly_func(w, x)
        # 计算损失值
        loss = loss_func_reg(y, y_pred, w, lam)
        loss_list.append(loss)
        # 计算梯度
        grad = np.zeros(n)
        for j in range(n):
            grad[j] = np.mean((y_pred - y) * (x ** j)) + lam * w[j]
        # 更新权重
        w = w - lr * grad
    return w, loss_list

# 生成数据，加入噪声
np.random.seed(2023) # 设置随机种子
sample_size = 50 # 样本数量
x = np.linspace(0, 1, sample_size) # 在[0,1]区间内均匀采样
y = sin_func(x) + np.random.normal(0, 0.1, sample_size) # 加入正态分布噪声

# 用不同阶数多项式函数拟合曲线（建议正弦函数曲线）
poly_orders = [1, 3, 5, 7,9,11,15,19,23] # 多项式阶数列表
w_init = np.random.normal(0, 1, max(poly_orders) + 1) # 初始化权重
lr = 0.1 # 学习率
epochs = 10000 # 迭代次数

# 不加惩罚项的情况
w_no_reg_list = [] # 存储不同阶数的权重
loss_no_reg_list = [] # 存储不同阶数的损失值
for order in poly_orders:
    w_no_reg, loss_no_reg = gradient_descent(x, y, w_init[:order+1], lr, epochs)
    w_no_reg_list.append(w_no_reg)
    loss_no_reg_list.append(loss_no_reg)

# 加惩罚项的情况
lam = 0.01 # 惩罚系数
w_reg_list = [] # 存储不同阶数的权重
loss_reg_list = [] # 存储不同阶数的损失值
for order in poly_orders:
    w_reg, loss_reg = gradient_descent(x, y, w_init[:order+1], lr, epochs, lam)
    w_reg_list.append(w_reg)
    loss_reg_list.append(loss_reg)

# 绘制拟合曲线和损失曲线
x_test = np.linspace(0, 1, 100) # 测试数据
y_test = sin_func(x_test) # 真实值

for i in range(len(poly_orders)):
    order = poly_orders[i]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_no_reg_list[i], x_test), label='Fitted function (no reg)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} without regularization'.format(order))
      
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_reg_list[i], x_test), label='Fitted function (reg)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} with regularization'.format(order))
    
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 8))
plt.plot(poly_orders, [loss_no_reg[-1] for loss_no_reg in loss_no_reg_list], label='Loss (no reg)')
plt.plot(poly_orders, [loss_reg[-1] for loss_reg in loss_reg_list], label='Loss (reg)')

plt.xlabel('Polynomial order')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curve with different polynomial order')
plt.show()

# 用不同阶数多项式函数拟合曲线（建议正弦函数曲线）
poly_order = 7 # 多项式阶数
w_init = np.random.normal(0, 1, poly_order + 1) # 初始化权重
lr = 0.1 # 学习率
epochs = 10000 # 迭代次数

# 不加惩罚项的情况
w_no_reg_dict = {} # 存储不同数据量的权重
loss_no_reg_dict = {} # 存储不同数据量的损失值

# 加惩罚项的情况
lam = 0.001 # 惩罚系数
w_reg_dict = {} # 存储不同数据量的权重
loss_reg_dict = {} # 存储不同数据量的损失值

# 不同数据量列表
sample_sizes = [10,  30,  50, 70, 100]

# 对每个数据量进行拟合和绘图
for sample_size in sample_sizes:  
    # 生成数据，加入噪声
    np.random.seed(2023) # 设置随机种子
    x = np.linspace(0, 1, sample_size) # 在[0,1]区间内均匀采样
    y = sin_func(x) + np.random.normal(0, 0.1, sample_size) # 加入正态分布噪声  
    # 不加惩罚项的情况
    w_no_reg, loss_no_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs)
    w_no_reg_dict[sample_size] = w_no_reg
    loss_no_reg_dict[sample_size] = loss_no_reg  
    # 加惩罚项的情况
    w_reg, loss_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs, lam)
    w_reg_dict[sample_size] = w_reg
    loss_reg_dict[sample_size] = loss_reg  
    # 绘制拟合曲线
    x_test = np.linspace(0, 1, 100) # 测试数据
    y_test = sin_func(x_test) # 真实值   
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_no_reg, x_test), label='Fitted function (no reg)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} without regularization (sample size = {})'.format(poly_order, sample_size))
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_reg, x_test), label='Fitted function (reg)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} with regularization (sample size = {})'.format(poly_order, sample_size))
    plt.tight_layout()
    plt.show()
# 绘制损失值随数据量的变化
plt.figure(figsize=(8, 8))
plt.plot(sample_sizes, [loss_no_reg_dict[s][-1] for s in sample_sizes], label='Loss (no reg)')
plt.plot(sample_sizes, [loss_reg_dict[s][-1] for s in sample_sizes], label='Loss (reg)')
plt.xlabel('Sample size')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curve with different sample sizes')
plt.show()


# 不同学习率列表（新增）
learning_rates = [0.01, 0.05 ,0.1,0.5,1,1.1]

# 对每个学习率进行拟合和绘图（修改）
for lr in learning_rates: # 新增
    
    # 不加惩罚项的情况
    w_no_reg, loss_no_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs)
    w_no_reg_dict[lr] = w_no_reg # 修改
    loss_no_reg_dict[lr] = loss_no_reg # 修改
    
    # 加惩罚项的情况
    w_reg, loss_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs, lam)
    w_reg_dict[lr] = w_reg # 修改
    loss_reg_dict[lr] = loss_reg # 修改
    
    # 绘制拟合曲线
    x_test = np.linspace(0, 1, 100) # 测试数据
    y_test = sin_func(x_test) # 真实值
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_no_reg, x_test), label='Fitted function (no reg)', color='red', linestyle='--')
    plt.plot(x_test, poly_func(w_reg, x_test), label='Fitted function (reg)', color='blue', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} with different regularization methods (sample size = {}, learning rate = {})'.format(poly_order, sample_size, lr)) # 修改
    plt.show()

# 绘制损失值随学习率的变化（修改）
plt.figure(figsize=(8, 8))
plt.plot(learning_rates, [loss_no_reg_dict[lr][-1] for lr in learning_rates], label='Loss (no reg)') # 修改
plt.plot(learning_rates, [loss_reg_dict[lr][-1] for lr in learning_rates], label='Loss (reg)') # 修改
plt.xlabel('Learning rate') # 修改
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curve with different learning rates') # 修改
plt.show()

# 不同迭代次数列表（新增）
epoch_nums = [1000, 5000, 10000,20000]

# 固定学习率为0.05（新增）
lr = 0.1

# 对每个迭代次数进行拟合和绘图（修改）
for epochs in epoch_nums: # 新增
    
    # 不加惩罚项的情况
    w_no_reg, loss_no_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs)
    w_no_reg_dict[epochs] = w_no_reg # 修改
    loss_no_reg_dict[epochs] = loss_no_reg # 修改
    
    # 加惩罚项的情况
    w_reg, loss_reg = gradient_descent(x, y, w_init[:poly_order+1], lr, epochs, lam)
    w_reg_dict[epochs] = w_reg # 修改
    loss_reg_dict[epochs] = loss_reg # 修改
    
    # 绘制拟合曲线
    x_test = np.linspace(0, 1, 100) # 测试数据
    y_test = sin_func(x_test) # 真实值
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, label='Training data')
    plt.plot(x_test, y_test, label='True function')
    plt.plot(x_test, poly_func(w_no_reg, x_test), label='Fitted function (no reg)', color='red', linestyle='--')
    plt.plot(x_test, poly_func(w_reg, x_test), label='Fitted function (reg)', color='blue', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial fitting of order {} with different regularization methods (sample size = {}, learning rate = {}, epoch num = {})'.format(poly_order, sample_size, lr, epochs)) # 修改
    plt.show()

# 绘制损失值随迭代次数的变化（修改）
plt.figure(figsize=(8, 8))
plt.plot(epoch_nums, [loss_no_reg_dict[e][-1] for e in epoch_nums], label='Loss (no reg)') # 修改
plt.plot(epoch_nums, [loss_reg_dict[e][-1] for e in epoch_nums], label='Loss (reg)') # 修改
plt.xlabel('Epoch num') # 修改
plt.ylabel('Loss')
plt.legend()
plt.title('Loss curve with different epoch nums') # 修改
plt.show()

