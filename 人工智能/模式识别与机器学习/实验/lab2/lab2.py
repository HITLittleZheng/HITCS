import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D # 导入mplot3d模块

# 生成两个类别的数据，用高斯分布
np.random.seed(0)
n = 1000 # 样本数

# 满足朴素贝叶斯假设的数据
x1_nb = np.random.multivariate_normal([-1,-1], [[0.5,0],[0,0.5]], n) # 类别1
x2_nb = np.random.multivariate_normal([2,2], [[0.5,0],[0,0.5]], n) # 类别2
x_nb = np.vstack((x1_nb,x2_nb)) # 合并数据
y_nb = np.array([0]*n + [1]*n) # 标签



# 不满足朴素贝叶斯假设的数据
x1_nnb = np.random.multivariate_normal([-1,-1], [[1,0.5],[0.5,1]], n) # 类别1
x2_nnb = np.random.multivariate_normal([2,2], [[1,-0.5],[-0.5,1]], n) # 类别2
x_nnb = np.vstack((x1_nnb,x2_nnb)) # 合并数据
y_nnb = np.array([0]*n + [1]*n) # 标签



# 定义逻辑回归模型（无正则化）
def logistic_regression_no_reg(x, y, lr=0.01, max_iter=100):
    """
    x: 特征矩阵，shape为(n,m)，n为样本数，m为特征数
    y: 标签向量，shape为(n,)
    lr: 学习率，默认为0.01
    max_iter: 最大迭代次数，默认为100
    """
    n, m = x.shape # 获取样本数和特征数
    x = np.c_[np.ones(n), x] # 在特征矩阵前加一列全1，表示截距项
    w = np.zeros(m+1) # 初始化参数向量为全0，shape为(m+1,)
    for i in range(max_iter): # 迭代更新参数
        z = x.dot(w) # 计算线性部分，shape为(n,)
        p = 1 / (1 + np.exp(-z)) # 计算逻辑函数值，shape为(n,)
        g = x.T.dot(p - y) / n # 计算梯度（无正则化项），shape为(m+1,)
        w = w - lr * g # 更新参数，shape为(m+1,)
    return w # 返回参数向量

# 定义逻辑回归模型（加入L1正则化）
def logistic_regression_l1_reg(x, y, lr=0.01, max_iter=100, lam=0.01):
    """
    x: 特征矩阵，shape为(n,m)，n为样本数，m为特征数
    y: 标签向量，shape为(n,)
    lr: 学习率，默认为0.01
    max_iter: 最大迭代次数，默认为100
    lam: 正则化系数，默认为0.01
    """
    n, m = x.shape # 获取样本数和特征数
    x = np.c_[np.ones(n), x] # 在特征矩阵前加一列全1，表示截距项
    w = np.zeros(m+1) # 初始化参数向量为全0，shape为(m+1,)
    for i in range(max_iter): # 迭代更新参数
        z = x.dot(w) # 计算线性部分，shape为(n,)
        p = 1 / (1 + np.exp(-z)) # 计算逻辑函数值，shape为(n,)
        g = x.T.dot(p - y) / n + lam * np.sign(w) # 计算梯度（加入L1正则化项），shape为(m+1,)
        w = w - lr * g # 更新参数，shape为(m+1,)
    return w # 返回参数向量

def calculate_accuracy(x, y, w):
    n = len(y)
    x = np.c_[np.ones(n), x]
    y_pred = (x.dot(w) >= 0).astype(int)  # Predict 1 if the linear combination is >= 0, else 0
    accuracy = (y_pred == y).mean()
    return accuracy

# 调用逻辑回归模型（无正则化），得到参数估计
w_nb = logistic_regression_no_reg(x_nb, y_nb)
print("The parameter estimates for data satisfying the naive Bayes hypothesis are:", w_nb)
w_nnb = logistic_regression_no_reg(x_nnb, y_nnb)
print("The parameter estimates for data that do not satisfy the naive Bayes hypothesis are:", w_nnb)

# 绘制数据点和决策边界
plt.subplot(121)
plt.scatter(x1_nb[:,0], x1_nb[:,1], c='r', label='category1')
plt.scatter(x2_nb[:,0], x2_nb[:,1], c='b', label='category2')
xx = np.linspace(-3, 6, 100)
yy_nb = -(w_nb[0] + w_nb[1] * xx) / w_nb[2]
plt.plot(xx, yy_nb, c='g', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data that satisfy the naive Bayes hypothesis')
plt.legend()

plt.subplot(122)
plt.scatter(x1_nnb[:,0], x1_nnb[:,1], c='r', label='category1')
plt.scatter(x2_nnb[:,0], x2_nnb[:,1], c='b', label='category2')
xx = np.linspace(-3, 6, 100)
yy_nnb = -(w_nnb[0] + w_nnb[1] * xx) / w_nnb[2]
plt.plot(xx, yy_nnb, c='g', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data that does not satisfy the naive Bayes hypothesis')
plt.legend()

plt.show()

# 调用逻辑回归模型（无正则化），得到参数估计
w_nb_l1 = logistic_regression_l1_reg(x_nb, y_nb)
print("满足朴素贝叶斯假设的数据的参数估计为：", w_nb_l1)
w_nnb_l1= logistic_regression_l1_reg(x_nnb, y_nnb)
print("不满足朴素贝叶斯假设的数据的参数估计为：", w_nnb_l1)

# 绘制数据点和决策边界
plt.subplot(121)
plt.scatter(x1_nb[:,0], x1_nb[:,1], c='r', label='category1')
plt.scatter(x2_nb[:,0], x2_nb[:,1], c='b', label='category2')
xx = np.linspace(-3, 6, 100)
yy_nb = -(w_nb_l1[0] + w_nb_l1[1] * xx) / w_nb_l1[2]
plt.plot(xx, yy_nb, c='g', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data that satisfy the naive Bayes hypothesis')
plt.legend()

plt.subplot(122)
plt.scatter(x1_nnb[:,0], x1_nnb[:,1], c='r', label='category1')
plt.scatter(x2_nnb[:,0], x2_nnb[:,1], c='b', label='category2')
xx = np.linspace(-3, 6, 100)
yy_nnb = -(w_nnb_l1[0] + w_nnb_l1[1] * xx) / w_nnb_l1[2]
plt.plot(xx, yy_nnb, c='g', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data that does not satisfy the naive Bayes hypothesis')
plt.legend()

plt.show()
accuracy_nb = calculate_accuracy(x_nb, y_nb, w_nb)
accuracy_nnb = calculate_accuracy(x_nnb, y_nnb, w_nnb)

print("Accuracy for data satisfying Naive Bayes assumption:", accuracy_nb)
print("Accuracy for data not satisfying Naive Bayes assumption:", accuracy_nnb)

# ... (existing code for plotting) ...

# Calculate accuracy for both datasets with L1 regularization
accuracy_nb_l1 = calculate_accuracy(x_nb, y_nb, w_nb_l1)
accuracy_nnb_l1 = calculate_accuracy(x_nnb, y_nnb, w_nnb_l1)

print("Accuracy for data satisfying Naive Bayes assumption with L1 regularization:", accuracy_nb_l1)
print("Accuracy for data not satisfying Naive Bayes assumption with L1 regularization:", accuracy_nnb_l1)

# 读取Skin数据集
data = np.loadtxt('skin.csv', delimiter=',', encoding='utf-8-sig')

x = data[:, :3] # 特征矩阵，shape为(245057, 3)
y = data[:, -1] # 标签向量，shape为(245057,)

# 将x和y分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 调用逻辑回归模型（无正则化），得到参数估计
w_no_reg = logistic_regression_no_reg(x_train, y_train)
print("The unregularized parameter is estimated as：", w_no_reg)

# 在测试集上计算准确率和混淆矩阵
y_pred_no_reg = np.where(np.c_[np.ones(len(x_test)), x_test].dot(w_no_reg) > 0, 1, 0)
acc_no_reg = accuracy_score(y_test, y_pred_no_reg)
cm_no_reg = confusion_matrix(y_test, y_pred_no_reg)
print("无正则化的准确率为：", acc_no_reg)
print("无正则化的混淆矩阵为：\n", cm_no_reg)

# 绘制数据点和决策边界
fig = plt.figure() # 创建一个图形对象
ax = fig.add_subplot(121, projection='3d') # 创建一个三维子图
# 根据不同的类别，用不同的颜色绘制数据点
ax.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], x_test[y_test == 1, 2], c='r', label='category1')
ax.scatter(x_test[y_test == 2, 0], x_test[y_test == 2, 1], x_test[y_test == 2, 2], c='g', label='category2')
# 绘制决策边界，即w.dot(x) = 0的平面
xx, yy = np.meshgrid(np.linspace(0, 255, 10), np.linspace(0, 255, 10)) # 创建网格点
zz = -(w_no_reg[0] + w_no_reg[1] * xx + w_no_reg[2] * yy) / w_no_reg[3] # 计算z坐标
ax.plot_surface(xx, yy, zz, alpha=0.5) # 绘制平面
ax.set_xlabel('B') # 设置x轴标签
ax.set_ylabel('G') # 设置y轴标签
ax.set_zlabel('R') # 设置z轴标签
ax.set_title('Unregularized decision boundary') # 设置标题
ax.legend() # 显示图例


# 调用逻辑回归模型（加入L1正则化），得到参数估计
w_l1_reg = logistic_regression_l1_reg(x_train, y_train)
print("加入L1正则化的参数估计为：", w_l1_reg)

# 在测试集上计算准确率和混淆矩阵
y_pred_l1_reg = np.where(np.c_[np.ones(len(x_test)), x_test].dot(w_l1_reg) > 0, 1, 0)
acc_l1_reg = accuracy_score(y_test, y_pred_l1_reg)
cm_l1_reg = confusion_matrix(y_test, y_pred_l1_reg)
print("加入L1正则化的准确率为：", acc_l1_reg)
print("加入L1正则化的混淆矩阵为：\n", cm_l1_reg)

# 绘制数据点和决策边界
ax = fig.add_subplot(122, projection='3d') # 创建另一个三维子图
# 根据不同的类别，用不同的颜色绘制数据点
ax.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], x_test[y_test == 1, 2], c='r', label='category1')
ax.scatter(x_test[y_test == 2, 0], x_test[y_test == 2, 1], x_test[y_test == 2, 2], c='g', label='category2')
# 绘制决策边界，即w.dot(x) = 0的平面
xx, yy = np.meshgrid(np.linspace(0, 255, 10), np.linspace(0, 255, 10)) # 创建网格点
zz = -(w_l1_reg[0] + w_l1_reg[1] * xx + w_l1_reg[2] * yy) / w_l1_reg[3] # 计算z坐标
ax.plot_surface(xx, yy, zz, alpha=0.5) # 绘制平面
ax.set_xlabel('B') # 设置x轴标签
ax.set_ylabel('G') # 设置y轴标签
ax.set_zlabel('R') # 设置z轴标签
ax.set_title('There are regularized decision boundaries') # 设置标题
ax.legend() # 显示图例
plt.show()