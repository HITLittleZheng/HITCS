import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows
from scipy.signal import convolve2d

class PCANet:
    """
    PCANet: 一种基于PCA的卷积神经网络，用于图像特征提取。
    """
    def __init__(self, patch_size=(7, 7), num_filters=(8, 8), block_size=(7, 7), stride=1):
        """
        参数:
            patch_size (tuple): 滤波器尺寸 (h, w)
            num_filters (tuple): 两个阶段滤波器数量 (L1, L2)
            block_size (tuple): 直方图池化块尺寸 (h, w)
            stride (int): 卷积步长，此处固定为1（滑动窗口步长）
        """
        self.patch_size = patch_size
        self.num_filters = num_filters
        self.block_size = block_size
        self.stride = stride
        self.W1 = None  # 第一阶段滤波器组
        self.W2 = None  # 第二阶段滤波器组

    def _extract_patches(self, images):
        """
        从图像列表中提取所有尺寸为 patch_size 的块，并去均值。
        输入: images - 列表或数组，形状 (N, H, W)
        返回: 所有块组成的矩阵 (每个块展平为一列), 形状 (patch_size[0]*patch_size[1], N_patches)
        """
        h, w = self.patch_size
        patches = []
        for img in images:
            # 提取所有滑动窗口块
            windows = view_as_windows(img, (h, w), step=self.stride)  # 形状 (H-h+1, W-w+1, h, w)
            windows = windows.reshape(-1, h * w)  # 展平每个块
            patches.append(windows)
        patches = np.vstack(patches)  # 合并所有图像的块
        patches = patches.T  # 转置，使每列为一个块
        # 去均值 (每个块减去其均值)
        patches = patches - patches.mean(axis=0, keepdims=True)
        return patches

    def _pca_filters(self, patches, num_filters):
        """
        对块矩阵进行PCA，返回前 num_filters 个主成分（滤波器）。
        输入: patches - 形状 (patch_dim, num_patches)
        返回: 滤波器组，形状 (patch_dim, num_filters)
        """
        pca = PCA(n_components=num_filters)
        pca.fit(patches.T)  # PCA 期望样本在行，特征在列
        # 主成分是 pca.components_ 的行，需要转置并 reshape 为滤波器
        filters = pca.components_.T  # 形状 (patch_dim, num_filters)
        return filters

    def _convolve(self, images, filters):
        """
        用一组滤波器对每个图像进行卷积，返回特征图列表。
        输入:
            images - 形状 (N, H, W)
            filters - 形状 (patch_dim, num_filters)
        返回: 列表，每个元素形状 (num_filters, H-h+1, W-w+1)
        """
        h, w = self.patch_size
        # 将每个滤波器 reshape 为 2D
        filters_2d = [f.reshape(h, w) for f in filters.T]  # 每个滤波器形状 (h, w)
        feature_maps = []
        for img in images:
            maps = []
            for f in filters_2d:
                # 使用 valid 卷积，输出尺寸为 (H-h+1, W-w+1)
                conv = convolve2d(img, f, mode='valid')
                maps.append(conv)
            feature_maps.append(np.array(maps))  # (num_filters, H', W')
        return feature_maps

    def _hash_coding(self, feature_maps):
        """
        将第二阶段的响应二值化并编码为整数。
        输入: feature_maps - 列表，每个元素形状 (L2, H'', W'')，来自第二阶段卷积
        返回: 编码图列表，每个元素形状 (H'', W'')
        """
        coded_maps = []
        for maps in feature_maps:  # maps 形状 (L2, H'', W'')
            # 二值化：大于0为1，否则0
            binary = (maps > 0).astype(np.uint8)
            # 编码：将每个像素的 L2 位组合成整数
            # 权重为 2^0, 2^1, ... 按通道顺序
            code = np.zeros(binary.shape[1:], dtype=np.uint8)
            for i in range(binary.shape[0]):
                code += binary[i] * (1 << i)
            coded_maps.append(code)
        return coded_maps

    def _block_histograms(self, coded_maps, num_bins):
        """
        对每个编码图进行分块直方图统计。
        输入:
            coded_maps - 编码图列表，每个形状 (H'', W'')
            num_bins - 直方图区间数（2^L2）
        返回: 特征向量列表，每个对应一个输入图像
        """
        features = []
        for code in coded_maps:
            h, w = self.block_size
            # 只取能被块尺寸整除的部分
            H = code.shape[0] // h
            W = code.shape[1] // w
            hist = []
            for i in range(H):
                for j in range(W):
                    block = code[i*h:(i+1)*h, j*w:(j+1)*w]
                    # 计算直方图
                    block_hist, _ = np.histogram(block.ravel(), bins=num_bins, range=(0, num_bins-1))
                    hist.append(block_hist)
            features.append(np.hstack(hist))
        return np.array(features)

    def fit(self, images):
        """
        训练阶段：学习两个阶段的 PCA 滤波器。
        输入: images - 形状 (N, H, W) 的训练图像数组
        """
        # 第一阶段：从原始图像提取块，学习 PCA 滤波器
        patches1 = self._extract_patches(images)
        self.W1 = self._pca_filters(patches1, self.num_filters[0])

        # 第一阶段卷积，得到特征图
        feature_maps1 = self._convolve(images, self.W1)  # 列表，每个元素 (L1, H1, W1)

        # 第二阶段：从第一阶段的特征图中提取块，学习 PCA 滤波器
        # 收集所有特征图的块
        all_maps = []  # 存储所有第一阶段特征图（展平）
        for maps in feature_maps1:
            # maps 形状 (L1, H1, W1) -> 将每个通道视为独立图像
            for ch in range(maps.shape[0]):
                all_maps.append(maps[ch])  # 每个形状 (H1, W1)
        patches2 = self._extract_patches(all_maps)
        self.W2 = self._pca_filters(patches2, self.num_filters[1])

    def transform(self, images):
        """
        特征提取：对输入图像提取 PCANet 特征。
        输入: images - 形状 (N, H, W)
        返回: 特征矩阵，形状 (N, feature_dim)
        """
        # 第一阶段卷积
        feature_maps1 = self._convolve(images, self.W1)  # 列表，每个 (L1, H1, W1)

        # 第二阶段卷积
        # 对每个第一阶段特征图的每个通道，与第二阶段滤波器卷积
        second_maps = []  # 每个输入图像对应一个列表，每个列表元素为 (L1, L2, H2, W2)
        for maps in feature_maps1:
            # maps: (L1, H1, W1)
            per_img_maps = []
            for ch in range(maps.shape[0]):
                # 将每个通道视为独立图像，输入到第二阶段卷积
                # 注意：_convolve 需要输入列表，每个元素为单张图像
                maps_ch = maps[ch:ch+1, :, :]  # 保持形状 (1, H1, W1)
                conv_res = self._convolve(maps_ch, self.W2)  # 返回列表，只有一个元素，形状 (L2, H2, W2)
                per_img_maps.append(conv_res[0])  # 形状 (L2, H2, W2)
            # 合并所有通道：堆叠成 (L1, L2, H2, W2)
            second_maps.append(np.stack(per_img_maps, axis=0))  # 形状 (L1, L2, H2, W2)

        # 哈希编码与池化
        all_coded = []  # 每个输入图像对应一个编码图列表，列表长度为 L1
        for maps in second_maps:  # maps: (L1, L2, H2, W2)
            coded = []
            for i in range(maps.shape[0]):  # 遍历 L1
                # 每个通道 (L2, H2, W2) 进行哈希编码
                coded.append(self._hash_coding([maps[i]])[0])  # 返回编码图 (H2, W2)
            all_coded.append(coded)  # 每个元素是 L1 个编码图的列表

        # 直方图池化，并拼接所有 L1 通道的特征
        features = []
        num_bins = 1 << self.num_filters[1]  # 2^L2
        for coded_list in all_coded:  # coded_list 长度 L1
            # 对每个编码图计算直方图，然后拼接
            feats = self._block_histograms(coded_list, num_bins)  # 形状 (L1, feat_per_channel)
            features.append(feats.flatten())
        return np.array(features)

# 示例：在 MNIST 上使用 PCANet + SVM 分类
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    # 加载 MNIST 数据（前100个样本用于演示）
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
    X = mnist.data[:100].astype(np.float32)          # 100张图片，每张784维
    y = mnist.target[:100].astype(int)
    # 将数据 reshape 为图像 (28, 28) 并归一化到 [0,1]
    X = X.reshape(-1, 28, 28) / 255.0
    # 划分训练集和测试集（80训练，20测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 PCANet 实例（参数可根据需要调整）
    pcanet = PCANet(patch_size=(7, 7), num_filters=(8, 8), block_size=(7, 7))

    # 训练 PCANet（学习滤波器）
    pcanet.fit(X_train)

    # 提取特征
    train_features = pcanet.transform(X_train)
    test_features = pcanet.transform(X_test)

    # 使用 SVM 分类器
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(train_features, y_train)
    y_pred = clf.predict(test_features)

    # 输出准确率
    acc = accuracy_score(y_test, y_pred)
    print(f"PCANet + SVM 测试准确率: {acc:.4f}")