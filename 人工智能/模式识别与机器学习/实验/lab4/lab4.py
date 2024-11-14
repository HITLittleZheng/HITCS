# 导入numpy库
import numpy as np
# 给train_test_split函数起一个别名
from sklearn.model_selection import train_test_split 


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义激活函数的导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 定义多层感知机类
class MLP:

    # 初始化参数
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size # 输入层大小
        self.hidden_size = hidden_size # 隐藏层大小
        self.output_size = output_size # 输出层大小
        self.learning_rate = learning_rate # 学习率
        # 随机初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) # 输入层到隐藏层的权重矩阵
        self.b1 = np.random.randn(hidden_size) # 隐藏层的偏置向量
        self.W2 = np.random.randn(hidden_size, output_size) # 隐藏层到输出层的权重矩阵
        self.b2 = np.random.randn(output_size) # 输出层的偏置向量

    # 前向传播函数
    def forward(self, X):
        # 计算隐藏层的输出
        self.Z1 = X.dot(self.W1) + self.b1 # 线性组合
        self.A1 = sigmoid(self.Z1) # 激活函数
        # 计算输出层的输出
        self.Z2 = self.A1.dot(self.W2) + self.b2 # 线性组合
        self.A2 = sigmoid(self.Z2) # 激活函数
        return self.A2 # 返回输出层的输出

    # 反向传播函数
    def backward(self, X, Y):
        # 计算输出层的误差
        error2 = Y - self.A2 # 期望输出与实际输出的差值
        delta2 = error2 * sigmoid_derivative(self.Z2) # 误差乘以激活函数的导数，得到输出层的梯度
        # 计算隐藏层的误差
        error1 = delta2.dot(self.W2.T) # 输出层的梯度乘以权重矩阵的转置，得到隐藏层的误差
        delta1 = error1 * sigmoid_derivative(self.Z1) # 误差乘以激活函数的导数，得到隐藏层的梯度
        # 更新权重和偏置
        self.W2 += self.learning_rate * self.A1.T.dot(delta2) # 隐藏层的输出乘以输出层的梯度，得到权重矩阵的更新量，并乘以学习率进行更新
        self.b2 += self.learning_rate * np.sum(delta2, axis=0) # 对输出层的梯度求和，得到偏置向量的更新量，并乘以学习率进行更新
        self.W1 += self.learning_rate * X.T.dot(delta1) # 输入层的输出乘以隐藏层的梯度，得到权重矩阵的更新量，并乘以学习率进行更新
        self.b1 += self.learning_rate * np.sum(delta1, axis=0) # 对隐藏层的梯度求和，得到偏置向量的更新量，并乘以学习率进行更新

    # 训练函数
    def train(self, X, Y, epochs):
        # 定义一个空列表losses，用于存储每次迭代的损失值
        losses = []
        # 迭代指定次数
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            # 反向传播
            self.backward(X, Y)
            # 计算损失函数
            loss = np.mean((Y - output) ** 2)
            # 将损失值添加到losses列表中
            losses.append(loss)
            # 打印训练信息
            print(f"Epoch {epoch + 1}, Loss: {loss}")
        # 调用plt.plot函数，传入range(epochs)和losses作为参数，绘制损失曲线
        plt.plot(range(epochs), losses)
        # 调用plt.xlabel和plt.ylabel函数，分别设置x轴和y轴的标签
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # 调用plt.show函数，显示图形
        plt.show()

    # 预测函数
    def predict(self, X):
        # 前向传播
        output = self.forward(X)
        # 将输出转换为类别标签
        labels = np.argmax(output, axis=1)
        return labels
# 定义线性分类器类
class LinearClassifier:

    # 初始化参数
    def __init__(self, input_size, output_size, learning_rate):
        self.input_size = input_size # 输入层大小
        self.output_size = output_size # 输出层大小
        self.learning_rate = learning_rate # 学习率
        # 随机初始化权重和偏置
        self.W = np.random.randn(input_size, output_size) # 输入层到输出层的权重矩阵
        self.b = np.random.randn(output_size) # 输出层的偏置向量

    # 前向传播函数
    def forward(self, X):
        # 计算输出层的输出
        self.Z = X.dot(self.W) + self.b # 线性组合
        self.A = sigmoid(self.Z) # 激活函数
        return self.A # 返回输出层的输出

    # 反向传播函数
    def backward(self, X, Y):
        # 计算输出层的误差
        error = Y - self.A # 期望输出与实际输出的差值
        delta = error * sigmoid_derivative(self.Z) # 误差乘以激活函数的导数，得到输出层的梯度
        # 更新权重和偏置
        self.W += self.learning_rate * X.T.dot(delta) # 输入层的输出乘以输出层的梯度，得到权重矩阵的更新量，并乘以学习率进行更新
        self.b += self.learning_rate * np.sum(delta, axis=0) # 对输出层的梯度求和，得到偏置向量的更新量，并乘以学习率进行更新

    # 训练函数
    def train(self, X, Y, epochs):
        # 迭代指定次数
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            # 反向传播
            self.backward(X, Y)
            # 计算损失函数
            loss = np.mean((Y - output) ** 2)
            # 打印训练信息
            print(f"Epoch {epoch + 1}, Loss: {loss}")

    # 预测函数
    def predict(self, X):
        # 前向传播
        output = self.forward(X)
        # 将输出转换为类别标签
        labels = np.argmax(output, axis=1)
        return labels

# 定义可视化函数，需要安装matplotlib库
import matplotlib.pyplot as plt

# 绘制数据点函数，输入为数据集X和标签Y，颜色列表colors和标记列表markers，输出为绘制好的图形对象ax
def plot_data(X, Y, colors=['r', 'g', 'b', 'y'], markers=['o', 's', '^', '*']):
    ax = plt.gca() # 获取当前图形对象的坐标轴对象ax
    for i in range(len(colors)): # 遍历颜色列表中的每个颜色值i，对应一个类别标签i
        ax.scatter(X[Y == i, 0], X[Y == i, 1], c=colors[i], marker=markers[i]) # 绘制数据集中标签为i的数据点，用相应的颜色和标记表示
    return ax # 返回图形对象ax

# 绘制分类结果函数，输入为数据集X，标签Y，模型model，颜色列表colors和标记列表markers，输出为绘制好的图形对象ax
def plot_result(X, Y, model, colors=['r', 'g', 'b', 'y'], markers=['o', 's', '^', '*']):
    ax = plt.gca() # 获取当前图形对象的坐标轴对象ax
    # 获取数据集的最大值和最小值，并留出一些边缘空间
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 生成网格点矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # 将网格点矩阵展平，并拼接成二维特征矩阵
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # 将预测结果Z调整为和网格点矩阵xx一样的形状
    Z = Z.reshape(xx.shape)
    # 使用等高线函数将不同类别的区域用颜色填充
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # 遍历颜色列表中的每个颜色值i，对应一个类别标签i
    for i in range(len(colors)):
        # 绘制数据集中标签为i的数据点，用相应的颜色和标记表示
        ax.scatter(X[Y == i, 0], X[Y == i, 1], c=colors[i], marker=markers[i])
    return ax # 返回图形对象ax
# 导入sklearn库中的datasets模块，用于生成模拟数据集
from sklearn import datasets

# 生成一个包含200个样本，2个特征，4个类别的数据集
X, Y = datasets.make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.0)

# 创建一个两层的多层感知机对象，输入层大小为2，隐藏层大小为10，输出层大小为4，学习率为0.01
mlp = MLP(input_size=2, hidden_size=10, output_size=4, learning_rate=0.01)

# 将标签Y转换为one-hot编码形式，方便计算交叉熵损失
Y_onehot = np.eye(4)[Y]

# 训练多层感知机模型，迭代次数为100
mlp.train(X, Y_onehot, epochs=1000)

# 创建一个线性分类器对象，输入层大小为2，输出层大小为4，学习率为0.01
linear = LinearClassifier(input_size=2, output_size=4, learning_rate=0.01)
# 训练线性分类器模型，迭代次数为100
linear.train(X, Y_onehot, epochs=1000)

# 绘制多层感知机和线性分类器的分类结果
plt.figure(figsize=(12, 6)) # 创建一个大小为12x6英寸的图形对象
plt.subplot(1, 2, 1) # 创建一个1行2列的子图，当前为第一个子图
plot_data(X, Y) # 绘制原始数据点
plot_result(X, Y, mlp) # 绘制多层感知机的分类结果
plt.title('MLP') # 设置子图的标题
plt.subplot(1, 2, 2) # 创建一个1行2列的子图，当前为第二个子图
plot_data(X, Y) # 绘制原始数据点
plot_result(X, Y, linear) # 绘制线性分类器的分类结果
plt.title('Linear') # 设置子图的标题
plt.show() # 显示图形


# 定义一个函数，根据样本量生成数据集，并训练多层感知机模型，返回最后的loss值
def train_mlp(n_samples):
    # 生成数据集
    X, Y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=4, cluster_std=1.0)
    # 转换标签为one-hot编码
    Y_onehot = np.eye(4)[Y]
    # 创建多层感知机对象
    mlp = MLP(input_size=2, hidden_size=10, output_size=4, learning_rate=0.01)
    # 训练模型
    mlp.train(X, Y_onehot, epochs=1000)
    # 计算最后的loss值
    output = mlp.forward(X)
    loss = np.mean((Y_onehot - output) ** 2)
    # 返回loss值
    return loss
# 定义一个列表，用来存储不同样本量对应的loss值
loss_list = []
# 用一个循环，从10到1010，间隔为100
for n in range(10, 1010, 100):
    # 调用上面定义的函数，得到每个样本量的loss值
    loss = train_mlp(n)
    # 添加到列表中
    loss_list.append(loss)
# 导入matplotlib库
import matplotlib.pyplot as plt
# 画出loss随样本量的变化曲线
plt.plot(range(10, 1010, 100), loss_list, color='blue', label='loss')
# 设置x轴和y轴的标签
plt.xlabel('n_samples')
plt.ylabel('loss')
# 显示图例
plt.legend()
# 显示图像
plt.show()
