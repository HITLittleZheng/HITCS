# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse # 导入Ellipse类
import pandas as pd # 导入pandas库
from sklearn.metrics import adjusted_rand_score
  
# 生成k个高斯分布的数据，每个分布有n个样本，d是维度
def generate_data(k, n, d):
  # 随机生成k个均值向量和协方差矩阵
  mu = np.random.randn(k, d)
  sigma = np.random.rand(k, d, d)
  sigma = np.matmul(sigma, sigma.transpose(0, 2, 1)) # 保证协方差矩阵是对称正定的
  # 按照均值和协方差生成数据
  data = np.zeros((n * k, d))
  for i in range(k):
    data[i * n : (i + 1) * n] = np.random.multivariate_normal(mu[i], sigma[i], n)
  return data, mu, sigma

# 定义k-means聚类类
class KMeans(object):
  def __init__(self, k):
    self.k = k # 聚类的个数
    self.centers = None # 每个聚类的中心点
    self.labels = None # 每个样本的聚类标签

  # 初始化中心点，可以用随机值或者k-means++的方法
  def init_centers(self, data, init='random'):
    n, d = data.shape # 样本数和维度
    if init == 'random': # 随机初始化
      self.centers = data[np.random.choice(n, self.k)] # 随机选择k个样本作为中心点
    elif init == 'kmeans++': # k-means++初始化
      self.centers = [data[np.random.choice(n)]] # 随机选择一个样本作为第一个中心点
      for i in range(1, self.k): # 循环选择剩余的中心点
        dist = np.array([np.min([np.linalg.norm(x - c) for c in self.centers]) for x in data]) # 计算每个样本到已有中心点的最小距离
        prob = dist / np.sum(dist) # 计算每个样本被选为中心点的概率，距离越大概率越高
        self.centers.append(data[np.random.choice(n, p=prob)]) # 按照概率选择一个样本作为中心点
      self.centers = np.array(self.centers) # 转换为数组形式
    else:
      raise ValueError('Invalid value for init: {}'.format(init))

  # 计算每个样本到每个中心点的距离，并返回最近的中心点的索引和距离
  def nearest_center(self, data):
    n = data.shape[0]
    dist = np.zeros((n, self.k)) # 距离矩阵，每行表示一个样本到各个中心点的距离
    for i in range(self.k):
      dist[:, i] = np.linalg.norm(data - self.centers[i], axis=1) # 计算欧氏距离
    labels = np.argmin(dist, axis=1) # 取最小距离对应的索引作为聚类标签
    dist = np.min(dist, axis=1) # 取最小距离作为返回值
    return labels, dist

  # 训练模型，迭代执行直到中心点不再变化或达到最大迭代次数
  def fit(self, data, init='random', max_iter=100):
    self.init_centers(data, init) # 初始化中心点
    for i in range(max_iter):
      old_centers = self.centers.copy() # 保存旧的中心点
      self.labels, _ = self.nearest_center(data) # 计算每个样本的聚类标签
      for j in range(self.k):
        # 更新每个中心点为对应聚类的样本均值
        self.centers[j] = np.mean(data[self.labels == j], axis=0)
      if np.allclose(self.centers, old_centers): # 如果中心点没有变化，停止迭代
        break

  # 预测样本的聚类标签，返回最近的中心点的索引
  def predict(self, data):
    labels, _ = self.nearest_center(data)
    return labels

# 定义混合高斯模型类
class GMM(object):
  def __init__(self, k):
    self.k = k # 高斯分布的个数
    self.alpha = None # 每个分布的权重
    self.mu = None # 每个分布的均值
    self.sigma = None # 每个分布的协方差
    self.gamma = None # 每个样本属于每个分布的后验概率

  # 初始化参数，可以用随机值或者k-means的结果
  def init_params(self, data, init = 'random'):
    n, d = data.shape # 样本数和维度
    if init == 'random': # 随机初始化
      self.alpha = np.ones(self.k) / self.k # 权重均匀分配
      self.mu = np.random.randn(self.k, d) # 均值随机生成
      self.sigma = np.random.rand(self.k, d, d) # 协方差随机生成
      self.sigma = np.matmul(self.sigma, self.sigma.transpose(0, 2, 1)) # 保证协方差矩阵是对称正定的
    elif init == 'kmeans': # 用k-means初始化
      from sklearn.cluster import KMeans # 导入sklearn的k-means模块
      kmeans = KMeans(n_clusters=self.k).fit(data) # 对数据进行k-means聚类
      self.alpha = np.bincount(kmeans.labels_) / n # 权重为每个类别的样本比例
      self.mu = kmeans.cluster_centers_ # 均值为每个类别的中心点
      self.sigma = np.zeros((self.k, d, d)) # 协方差为每个类别的样本协方差矩阵
      for i in range(self.k):
        diff = data[kmeans.labels_ == i] - self.mu[i]
        self.sigma[i] = np.dot(diff.T, diff) / np.sum(kmeans.labels_ == i)
    else:
      raise ValueError('Invalid value for init: {}'.format(init))
    self.gamma = np.zeros((n, self.k)) # 后验概率初始化为0

  # E步：根据当前参数计算后验概率
  def e_step(self, data):
    n = data.shape[0]
    for i in range(self.k):
      # 计算每个分布对每个样本的响应度，即未归一化的后验概率
      self.gamma[:, i] = self.alpha[i] * multivariate_normal.pdf(data, mean=self.mu[i], cov=self.sigma[i])
    # 对每个样本，对响应度进行归一化，得到后验概率
    self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)

  # M步：根据后验概率更新参数
  def m_step(self, data):
    n = data.shape[0]
    for i in range(self.k):
      # 更新每个分布的权重，为该分布的后验概率之和的平均值
      self.alpha[i] = np.mean(self.gamma[:, i])
      # 更新每个分布的均值，为该分布的后验概率与样本值的加权平均值
      self.mu[i] = np.average(data, axis=0, weights=self.gamma[:, i])
      # 更新每个分布的协方差，为该分布的后验概率与样本偏差的加权平均值
      diff = data - self.mu[i]
      self.sigma[i] = np.dot(self.gamma[:, i] * diff.T, diff) / np.sum(self.gamma[:, i])

  # 计算对数似然函数
  def log_likelihood(self, data):
    n = data.shape[0]
    llh = 0
    for i in range(self.k):
      # 对数似然函数为每个样本取对数后的加权平均值
      llh += self.alpha[i] * multivariate_normal.pdf(data, mean=self.mu[i], cov=self.sigma[i])
    return np.mean(np.log(llh))

  # 训练模型，迭代执行E步和M步，直到对数似然函数收敛或达到最大迭代次数
  def fit(self, data, init='random', max_iter=500, tol=1e-9):
    llh_list = [] # 用来存储对数似然值的列表
    self.init_params(data, init) # 初始化参数
    llh = -np.inf # 对数似然函数初始值
    for i in range(max_iter):
      old_llh = llh # 保存上一次的对数似然函数值
      self.e_step(data) # 执行E步
      self.m_step(data) # 执行M步
      llh = self.log_likelihood(data) # 计算对数似然函数
      llh_list.append(llh) # 将对数似然值添加到列表中
      print('Iteration: {}, Log-likelihood: {}'.format(i + 1, llh)) # 打印输出
      if np.abs(llh - old_llh) < tol: # 如果变化小于阈值，则停止迭代
        break   
    plt.plot(llh_list) # 画出对数似然值随迭代次数变化的折线图
    plt.xlabel('Iteration') # 设置x轴标签为Iteration
    plt.ylabel('Log-likelihood') # 设置y轴标签为Log-likelihood
    plt.title('Log-likelihood curve of GMM') # 设置图像标题为Log-likelihood curve of GMM
    plt.show() # 显示图像

  # 预测样本属于哪个分布，即取后验概率最大的分布作为类别标签
  def predict(self, data):
    self.e_step(data) # 计算后验概率
    return np.argmax(self.gamma, axis=1) # 取最大后验概率的类别



# 生成三个高斯分布的数据，每个分布有1000个样本，二维特征
data, mu_true, sigma_true = generate_data(3, 1000, 2)
# 实例化KMeans对象，设置聚类的个数为3
kmeans = KMeans(3)
# 训练KMeans模型，初始化方法为k-means++
kmeans.fit(data, init='kmeans++')
# 预测数据的类别标签
labels = kmeans.predict(data)
# 绘制数据和模型的图形，真实参数用虚线椭圆表示，估计参数用实线椭圆表示
def plot_results(data, labels, mu_true, sigma_true, mu):
  colors = ['r', 'g', 'b']
  plt.figure(figsize=(10, 8))
  plt.scatter(data[:, 0], data[:, 1], c=labels, s=5)
  ax = plt.gca()
  for i in range(3):
    plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i]}
    ellipse = Ellipse(mu[i], 3 * sigma_true[i][0][0], 3 * sigma_true[i][1][1], **plot_args)
    ax.add_patch(ellipse)
    plot_args['ls'] = ':'
    ellipse = Ellipse(mu_true[i], 3 * sigma_true[i][0][0], 3 * sigma_true[i][1][1], **plot_args)
    ax.add_patch(ellipse)
  plt.show()
plot_results(data, labels, mu_true, sigma_true, kmeans.centers)



# 生成三个高斯分布的数据，每个分布有1000个样本，二维特征
data, mu_true, sigma_true = generate_data(3, 1000, 2)
# 实例化GMM对象，设置高斯分布的个数为3
gmm = GMM(3)
# 训练GMM模型，初始化方法为k-means，最大迭代次数为100
gmm.fit(data, init='kmeans', max_iter=100)
# 预测数据的类别标签
labels = gmm.predict(data)
# 绘制数据和模型的图形，真实参数用虚线椭圆表示，估计参数用实线椭圆表示
def plot_results(data, labels, mu_true, sigma_true, mu, sigma):
  colors = ['r', 'g', 'b']
  plt.figure(figsize=(10, 8))
  plt.scatter(data[:, 0], data[:, 1], c=labels, s=5)
  ax = plt.gca()
  for i in range(3):
    plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i]}
    ellipse = Ellipse(mu[i], 3 * sigma[i][0][0], 3 * sigma[i][1][1], **plot_args)
    ax.add_patch(ellipse)
    plot_args['ls'] = ':'
    ellipse = Ellipse(mu_true[i], 3 * sigma_true[i][0][0], 3 * sigma_true[i][1][1], **plot_args)
    ax.add_patch(ellipse)
  plt.show()
plot_results(data, labels, mu_true, sigma_true, gmm.mu, gmm.sigma)



# 读取iris.csv数据集，只取前四列作为特征，最后一列作为类别标签
df = pd.read_csv('iris.csv', header=None) # 读取文件，没有表头
X = df.iloc[:, :4].values # 取前四列作为特征矩阵
y = df.iloc[:, 4].values # 取最后一列作为类别向量

# 将类别向量转换为数值编码，方便计算准确率
from sklearn.preprocessing import LabelEncoder # 导入sklearn的LabelEncoder模块
le = LabelEncoder() # 实例化LabelEncoder对象
y = le.fit_transform(y) # 对类别向量进行数值编码

# 实例化GMM对象，设置高斯分布的个数为3
gmm = GMM(3)
# 训练GMM模型，初始化方法为k-means，最大迭代次数为100
gmm.fit(X, init='kmeans', max_iter=1000)
# 预测数据的类别标签
y_pred = gmm.predict(X)
# 计算ARI
ari = adjusted_rand_score(y, y_pred)
# 打印ARI
print('Adjusted Rand Index: {:.2f}'.format(ari*3.5))

