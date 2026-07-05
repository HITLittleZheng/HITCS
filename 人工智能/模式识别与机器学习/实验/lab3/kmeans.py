import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ================================
# 辅助函数：生成高斯数据
# ================================
def generate_gaussian_data(k=3, n_per_cluster=200, random_seed=42):
    """生成k个高斯分布的数据点，返回X, y_true, true_params"""
    np.random.seed(random_seed)
    d = 2
    means = [np.array([2, 2]), np.array([8, 3]), np.array([5, 8])] if k == 3 else \
            [np.array([1, 1]), np.array([5, 2]), np.array([8, 8]), np.array([2, 7])][:k]
    covs = [np.array([[1, 0.5], [0.5, 1]]),
            np.array([[1, -0.2], [-0.2, 1]]),
            np.array([[0.5, 0], [0, 1.5]])]
    covs = covs[:k]
    pis = [1/k] * k#创建一个长度为 k 的列表，其中每个元素都是 1/k。
    X_list, y_list = [], []
    for i in range(k):
        X_i = np.random.multivariate_normal(means[i], covs[i], n_per_cluster)
        X_list.append(X_i)#X 的每一行是一个数据样本（即一个观测点）
        y_list.append([i] * n_per_cluster)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    true_params = [{'mean': means[i], 'cov': covs[i], 'pi': pis[i]} for i in range(k)]
    return X, y, true_params

# ================================
# 层次聚类初始化（纯NumPy实现）
# ================================
def compute_distance_matrix(X):
    """计算欧氏距离平方矩阵（对称）"""
    n = X.shape[0]
    D = np.zeros((n, n))
    # 向量化计算距离矩阵
    for i in range(n):
        diff = X[i] - X#X[i]是第i个行向量
        D[i] = np.sqrt(np.sum(diff * diff, axis=1))#diff * diff 是逐元素平方，因为axis=1所以sum是对每一行求和
    return D

def average_linkage_distance(cluster1, cluster2, D):
    """计算两个簇的平均链接距离（使用距离矩阵）"""
    n1, n2 = len(cluster1), len(cluster2)
    total = 0.0
    for i in cluster1:
        for j in cluster2:
            total += D[i, j]
    return total / (n1 * n2)

def hierarchical_init_centroids(X, k):
    """
    使用凝聚层次聚类（平均链接）初始化质心
    返回centroids (k, d)
    """
    n = X.shape[0]
    # 初始每个样本作为一个簇
    clusters = [[i] for i in range(n)]
    # 距离矩阵
    D = compute_distance_matrix(X)
    
    # 当簇数量大于k时继续合并
    while len(clusters) > k:
        min_dist = np.inf
        merge_idx = (0, 0)
        # 找到距离最小的两个簇
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = average_linkage_distance(clusters[i], clusters[j], D)
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)
        # 合并簇
        i, j = merge_idx
        clusters[i].extend(clusters[j])
        del clusters[j]
    
    # 计算每个簇的质心
    centroids = []
    for cluster in clusters:
        centroids.append(np.mean(X[cluster], axis=0))
    return np.array(centroids)

# ================================
# K-means 算法
# ================================
def kmeans_fit(X, k, max_iters=100, tol=1e-4):
    """K-means聚类，返回labels, centroids"""
    centroids = hierarchical_init_centroids(X, k)
    for _ in range(max_iters):
        # 分配点
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)#X是nx1xd，centroids是1xkxd
        #np.linalg.norm 是 NumPy 中计算范数的函数，默认情况下计算的是 欧几里得范数（L2 范数）
        labels = np.argmin(distances, axis=1)
        # 更新质心
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(k)])#np.any(labels == i)的意思是第k个簇非空
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    return labels, centroids

# ================================
# 混合高斯模型（EM算法）
# ================================
def gmm_init_from_kmeans(X, k, kmeans_labels):
    """用K-means结果初始化GMM参数"""
    n_samples, n_features = X.shape
    means = np.array([X[kmeans_labels == i].mean(axis=0) for i in range(k)])
    covs = []
    reg_covar = 1e-6
    for i in range(k):
        X_i = X[kmeans_labels == i]
        if len(X_i) > 1:
            cov = np.cov(X_i.T)
        else:
            cov = np.eye(n_features)#单位阵
        cov += reg_covar * np.eye(n_features)#添加正则化项以保证协方差矩阵正定
        covs.append(cov)
    pis = np.array([np.sum(kmeans_labels == i) / n_samples for i in range(k)])#计算每个簇的先验概率
    return means, covs, pis

def gmm_e_step(X, means, covs, pis):
    """E步：计算responsibilities"""
    n_samples, n_features = X.shape
    k = len(means)
    gamma = np.zeros((n_samples, k))
    for i in range(k):
        diff = X - means[i]
        inv_cov = np.linalg.inv(covs[i])
        det_cov = np.linalg.det(covs[i])
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n_features * det_cov))
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        pdf_vals = norm_const * np.exp(exponent)
        gamma[:, i] = pis[i] * pdf_vals
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma = gamma / (gamma_sum + 1e-12)
    return gamma

def gmm_m_step(X, gamma, reg_covar=1e-6):
    """M步：更新参数"""
    n_samples, n_features = X.shape
    k = gamma.shape[1]
    Nk = gamma.sum(axis=0)
    pis = Nk / n_samples
    means = np.dot(gamma.T, X) / Nk[:, np.newaxis]
    covs = []
    for i in range(k):
        diff = X - means[i]
        weighted_diff = gamma[:, i][:, np.newaxis] * diff
        cov = np.dot(weighted_diff.T, diff) / Nk[i]
        cov += reg_covar * np.eye(n_features)
        covs.append(cov)
    return means, covs, pis

def gmm_compute_log_likelihood(X, means, covs, pis, reg_covar=1e-6):
    """计算对数似然"""
    n_samples, n_features = X.shape
    k = len(means)
    ll = np.zeros(n_samples)
    for i in range(k):
        diff = X - means[i]
        cov_i = covs[i] + reg_covar * np.eye(n_features)
        inv_cov = np.linalg.inv(cov_i)
        det_cov = np.linalg.det(cov_i)
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n_features * det_cov))
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        pdf_vals = norm_const * np.exp(exponent)
        ll += pis[i] * pdf_vals
    ll = np.log(ll + 1e-12)
    return np.sum(ll)

def gmm_fit(X, k, kmeans_labels, max_iters=100, tol=1e-5, reg_covar=1e-6):
    """训练GMM，返回估计的means, covs, pis, log_likelihood_history"""
    means, covs, pis = gmm_init_from_kmeans(X, k, kmeans_labels)
    log_likelihood_history = []
    prev_ll = -np.inf
    for _ in range(max_iters):
        gamma = gmm_e_step(X, means, covs, pis)
        means, covs, pis = gmm_m_step(X, gamma, reg_covar)
        cur_ll = gmm_compute_log_likelihood(X, means, covs, pis, reg_covar)
        log_likelihood_history.append(cur_ll)
        if np.abs(cur_ll - prev_ll) < tol:
            break
        prev_ll = cur_ll
    return means, covs, pis, log_likelihood_history

def gmm_predict(X, means, covs, pis):
    """预测标签"""
    gamma = gmm_e_step(X, means, covs, pis)
    return np.argmax(gamma, axis=1)

# ================================
# 可视化函数
# ================================
def plot_clustering(X, labels, title, true_labels=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    if true_labels is not None:
        plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='plasma', alpha=0.2, marker='x')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def compare_params(true_params, means, covs, pis):
    print("\n===== 参数比较 =====")
    for i in range(len(true_params)):
        print(f"\n簇 {i+1}:")
        print(f"  真实均值: {true_params[i]['mean']}")
        print(f"  估计均值: {means[i]}")
        print(f"  真实协方差:\n{true_params[i]['cov']}")
        print(f"  估计协方差:\n{covs[i]}")
        print(f"  真实混合系数: {true_params[i]['pi']:.3f}")
        print(f"  估计混合系数: {pis[i]:.3f}")

# ================================
# 加载UCI Iris数据（不使用任何外部库，尝试读取本地或网络）
# ================================
def load_iris_data():
    """从UCI仓库读取Iris数据，若失败则生成模拟数据"""
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    try:
        response = urllib.request.urlopen(url)
        data = np.genfromtxt(response, delimiter=',', dtype=str)
        X = data[:, :4].astype(float)
        y_str = data[:, 4]
        unique, y = np.unique(y_str, return_inverse=True)
        return X, y
    except Exception as e:
        print(f"无法从网络加载Iris数据: {e}")
        print("使用生成的4维高斯混合数据代替UCI数据集")
        np.random.seed(42)
        k = 3
        n_per_cluster = 50
        d = 4
        means = [np.array([1,1,1,1]), np.array([5,5,5,5]), np.array([9,9,9,9])]
        covs = [np.eye(d), np.eye(d), np.eye(d)]
        X_list, y_list = [], []
        for i in range(k):
            X_i = np.random.multivariate_normal(means[i], covs[i], n_per_cluster)
            X_list.append(X_i)
            y_list.append([i]*n_per_cluster)
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        return X, y

# ================================
# 主实验
# ================================
if __name__ == "__main__":
    # 1. 生成高斯数据
    print("实验1：生成高斯混合数据并进行K-means与GMM聚类")
    X, y_true, true_params = generate_gaussian_data(k=3, n_per_cluster=200)
    print(f"生成数据形状: {X.shape}, 真实簇数: {len(np.unique(y_true))}")

    # K-means
    kmeans_labels, kmeans_centroids = kmeans_fit(X, k=3)
    plot_clustering(X, kmeans_labels, "K-means Clustering (Hierarchical Init)", true_labels=y_true)

    # GMM + EM
    means_est, covs_est, pis_est, ll_history = gmm_fit(X, k=3, kmeans_labels=kmeans_labels)
    # 绘制似然曲线
    plt.figure(figsize=(8,5))
    plt.plot(ll_history, marker='o')
    plt.title("Log-Likelihood during EM Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.show()

    # 参数比较
    compare_params(true_params, means_est, covs_est, pis_est)

    # GMM预测并可视化
    gmm_labels = gmm_predict(X, means_est, covs_est, pis_est)
    plot_clustering(X, gmm_labels, "GMM Clustering (EM)")

    # 2. UCI Iris数据集应用
    print("\n实验2：UCI Iris数据集聚类")
    X_iris, y_iris = load_iris_data()
    print(f"Iris数据形状: {X_iris.shape}, 真实类别数: {len(np.unique(y_iris))}")

    # 标准化（手动实现）
    X_iris_scaled = (X_iris - X_iris.mean(axis=0)) / X_iris.std(axis=0)

    # K-means
    kmeans_labels_iris, _ = kmeans_fit(X_iris_scaled, k=3)
    # GMM
    means_iris, covs_iris, pis_iris, ll_iris = gmm_fit(X_iris_scaled, k=3, kmeans_labels=kmeans_labels_iris)
    gmm_labels_iris = gmm_predict(X_iris_scaled, means_iris, covs_iris, pis_iris)

    # 可视化（使用前两个特征）
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X_iris_scaled[:,0], X_iris_scaled[:,1], c=gmm_labels_iris, cmap='viridis', alpha=0.7)
    plt.title("GMM Clustering on Iris (first two features)")
    plt.xlabel("Sepal length (scaled)")
    plt.ylabel("Sepal width (scaled)")
    plt.subplot(1,2,2)
    plt.scatter(X_iris_scaled[:,0], X_iris_scaled[:,1], c=y_iris, cmap='plasma', alpha=0.7)
    plt.title("True Labels")
    plt.xlabel("Sepal length (scaled)")
    plt.ylabel("Sepal width (scaled)")
    plt.tight_layout()
    plt.show()

    print("\n实验完成。")