import numpy as np

def cross_correlation_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    二维互相关操作，步长=1，采用'same'填充（输出尺寸与输入相同）。
    直接滑动核，不翻转。

    参数:
        image: 2D 输入数组 (H, W)
        kernel: 2D 卷积核 (Kh, Kw)，通常为奇数尺寸
    返回:
        2D 输出数组，尺寸与 image 相同
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    # 计算填充大小，使得输出尺寸等于输入尺寸
    pad_h = Kh // 2
    pad_w = Kw // 2
    # 零填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((H, W), dtype=np.float32)
    
    # 滑动窗口
    for i in range(H):
        for j in range(W):
            # 提取当前窗口
            window = padded[i:i+Kh, j:j+Kw]
            # 点积（互相关，不翻转）
            output[i, j] = np.sum(window * kernel)
    return output


def convolution_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    二维卷积操作（数学严格定义），步长=1，采用'same'填充（输出尺寸与输入相同）。
    先翻转核（旋转180°），再执行互相关。

    参数:
        image: 2D 输入数组 (H, W)
        kernel: 2D 卷积核 (Kh, Kw)
    返回:
        2D 输出数组，尺寸与 image 相同
    """
    # 翻转核：上下颠倒并左右颠倒（等价于旋转180°）
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    # 卷积 = 用翻转后的核做互相关
    return cross_correlation_2d(image, flipped_kernel)


# ------------------- 示例与验证 -------------------
if __name__ == "__main__":
    # 简单的5x5图像
    img = np.array([[ 1, 0, 1, 1, 1],
                    [ 0, 2, 0, 3, 1],
                    [ 1, 0, 1, 0, 0],
                    [ 0, 4, 0, 5, 1],
                    [ 1, 1, 0, 0, 1]], dtype=np.float32)
    # 3x3核，例如边缘检测核
    kernel = np.array([[ 1, 2, 3],
                       [ 4, 5, 6],
                       [ 7, 8, 9]], dtype=np.float32)

    print("原始图像:\n", img)
    print("\n核:\n", kernel)

    # 计算互相关
    corr = cross_correlation_2d(img, kernel)
    print("\n互相关结果（无翻转）:\n", corr)

    # 计算卷积（数学定义）
    conv = convolution_2d(img, kernel)
    print("\n卷积结果（核先翻转再互相关）:\n", conv)

    # 验证：卷积结果应该等于用翻转后的核做互相关的结果
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    conv_by_flip = cross_correlation_2d(img, flipped_kernel)
    print("\n直接使用翻转核做互相关得到的结果:\n", conv_by_flip)
    print("\n两个结果是否一致？", np.allclose(conv, conv_by_flip))

    # 额外：如果核是中心对称的，互相关和卷积的结果应相同
    sym_kernel = np.array([[0, 1, 0],
                           [1, 4, 1],
                           [0, 1, 0]], dtype=np.float32)
    corr_sym = cross_correlation_2d(img, sym_kernel)
    conv_sym = convolution_2d(img, sym_kernel)
    print("\n中心对称核下互相关与卷积是否相等？", np.allclose(corr_sym, conv_sym))