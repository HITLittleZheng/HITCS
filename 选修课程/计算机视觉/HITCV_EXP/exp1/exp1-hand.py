"""
实验一：图像处理基础综合实验（手写实现核心算法）
功能涵盖：点操作（线性拉伸、Gamma校正、直方图均衡化）、手工2D卷积、
均值/高斯/中值滤波、边缘检测（Sobel/Prewitt/Laplacian/Canny）、
高斯与拉普拉斯金字塔构建及多尺度图像融合，以及可选扩展（CLAHE/双边滤波）。
所有核心图像处理算法均手写实现，仅保留图像读取、灰度转换、尺寸调整、
简单绘图及翻转等辅助操作。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 手写图像处理核心函数 ====================

# ---------- 1. 基础2D卷积（步长1，零填充） ----------
def my_filter2D(image, kernel, padding='same'):
    """
    手工实现2D卷积
    image: 2D灰度图 (H, W)
    kernel: 2D卷积核 (kh, kw)
    padding: 'same' 输出同尺寸；'valid' 输出有效区域
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 零填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float32) if padding == 'same' else \
             np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    # 滑动窗口卷积
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(window * kernel)

    return output

# ---------- 2. 均值滤波 ----------
def my_mean_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return my_filter2D(image, kernel, padding='same').astype(np.uint8)

# ---------- 3. 高斯核生成 ----------
def my_gaussian_kernel(size, sigma=1.0):
    """生成二维高斯核 (size x size)"""
    kernel_1d = np.array([np.exp(-(x - (size-1)/2)**2 / (2*sigma**2)) for x in range(size)])
    kernel_1d = kernel_1d / np.sum(kernel_1d)  # 归一化
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

# ---------- 4. 高斯滤波 ----------
def my_gaussian_filter(image, kernel_size=3, sigma=1.0):
    kernel = my_gaussian_kernel(kernel_size, sigma)
    return my_filter2D(image, kernel, padding='same').astype(np.uint8)

# ---------- 5. 中值滤波 ----------
def my_median_filter(image, kernel_size=3):
    h, w = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(window)
    return output

# ---------- 6. 边缘检测算子 ----------
def my_sobel(image):
    """Sobel算子"""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    grad_x = my_filter2D(image, kernel_x, padding='same')
    grad_y = my_filter2D(image, kernel_y, padding='same')
    mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def my_prewitt(image):
    """Prewitt算子"""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = my_filter2D(image, kernel_x, padding='same')
    grad_y = my_filter2D(image, kernel_y, padding='same')
    mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def my_laplacian(image):
    """Laplacian算子 (3x3)"""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = my_filter2D(image, kernel, padding='same')
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)

# ---------- 7. Canny边缘检测（完整手写） ----------
def my_canny(image, low_thresh=50, high_thresh=150):
    """
    完整Canny算法：
    1. 高斯平滑
    2. Sobel梯度计算
    3. 非极大值抑制 (NMS)
    4. 双阈值 + 边缘连接
    """
    # 1. 高斯平滑
    smooth = my_gaussian_filter(image, kernel_size=5, sigma=1.4)

    # 2. Sobel梯度
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    grad_x = my_filter2D(smooth, kernel_x, padding='same')
    grad_y = my_filter2D(smooth, kernel_y, padding='same')
    magnitude = np.hypot(grad_x, grad_y)
    magnitude = np.clip(magnitude, 0, 255)  # 限制范围
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle[angle < 0] += 180  # 转换到 [0, 180)

    # 3. 非极大值抑制
    h, w = image.shape
    nms = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            a = angle[i, j]
            # 量化到四个方向：0, 45, 90, 135度
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                # 水平方向
                left = magnitude[i, j-1]
                right = magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                # 对角线 45°
                left = magnitude[i-1, j+1]
                right = magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                # 垂直方向
                left = magnitude[i-1, j]
                right = magnitude[i+1, j]
            else:  # 112.5 ~ 157.5
                # 对角线 135°
                left = magnitude[i-1, j-1]
                right = magnitude[i+1, j+1]

            if magnitude[i, j] >= left and magnitude[i, j] >= right:
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    # 4. 双阈值和边缘连接
    strong = 255
    weak = 50
    strong_thresh = high_thresh
    weak_thresh = low_thresh

    strong_pixels = (nms >= strong_thresh)
    weak_pixels = (nms >= weak_thresh) & (nms < strong_thresh)

    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong_pixels] = strong
    result[weak_pixels] = weak

    # 边缘连接：弱像素如果与强像素8连通则保留
    for i in range(1, h-1):
        for j in range(1, w-1):
            if result[i, j] == weak:
                if (result[i-1:i+2, j-1:j+2] == strong).any():
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

# ---------- 8. 图像金字塔 ----------
def my_pyrDown(image):
    """下采样：高斯模糊 + 隔行隔列"""
    # 使用5x5高斯核（标准pyrDown核）
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float32) / 256.0
    blurred = my_filter2D(image.astype(np.float32), kernel, padding='same')
    # 下采样：每隔一行一列取一个像素
    down = blurred[::2, ::2]
    return np.clip(down, 0, 255).astype(np.uint8)

def my_pyrUp(image):
    """上采样：插零 + 高斯模糊（使用相同高斯核）"""
    h, w = image.shape
    up_h, up_w = h * 2, w * 2
    up = np.zeros((up_h, up_w), dtype=np.float32)
    up[::2, ::2] = image.astype(np.float32)

    # 使用与pyrDown相同的高斯核
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float32) / 256.0
    blurred = my_filter2D(up, kernel, padding='same')
    return np.clip(blurred, 0, 255).astype(np.uint8)

# ---------- 9. 直方图均衡化（灰度图） ----------
def my_hist_equalize(image):
    """灰度直方图均衡化"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(np.float64)
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized = cdf_normalized[image].reshape(image.shape)
    return equalized.astype(np.uint8)

# ---------- 10. CLAHE 自适应直方图均衡化（扩展） ----------
def my_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    """
    手写CLAHE算法
    支持灰度图和彩色图（彩色图需先转LAB，仅对L通道处理）
    由于保留cv2.cvtColor，彩色图处理时直接调用cv2.cvtColor，但CLAHE核心手写
    """
    if len(image.shape) == 3:
        # 彩色图使用OpenCV的cvtColor（保留）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
        l_eq = _clahe_gray(l, clip_limit, grid_size)
        # 合并通道（使用numpy组合）
        result_lab = np.stack([l_eq, a, b], axis=2)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    else:
        return _clahe_gray(image, clip_limit, grid_size)

def _clahe_gray(gray, clip_limit, grid_size):
    """单通道CLAHE核心实现"""
    h, w = gray.shape
    tile_h = h // grid_size[0]
    tile_w = w // grid_size[1]

    # 计算每个分块的直方图均衡化映射
    mappings = []
    for i in range(grid_size[0]):
        row_maps = []
        for j in range(grid_size[1]):
            tile = gray[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=[0,256])
            hist = hist.astype(np.float64)
            # 裁剪直方图
            clip_val = clip_limit * tile_h * tile_w / 256
            excess = np.maximum(hist - clip_val, 0)
            total_excess = np.sum(excess)
            hist[hist > clip_val] = clip_val
            # 将多余像素均匀分布
            hist += total_excess // 256
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            row_maps.append(cdf.astype(np.uint8))
        mappings.append(row_maps)

    # 双线性插值生成每个像素的映射值
    result = np.zeros_like(gray, dtype=np.uint8)
    for i in range(h):
        # 定位所属块
        ti = min(i // tile_h, grid_size[0]-1)
        ti_next = min(ti+1, grid_size[0]-1)
        for j in range(w):
            tj = min(j // tile_w, grid_size[1]-1)
            tj_next = min(tj+1, grid_size[1]-1)

            # 像素在块内的相对位置 (0~1)
            y_ratio = (i % tile_h) / tile_h
            x_ratio = (j % tile_w) / tile_w

            # 四个相邻块的映射值
            v00 = mappings[ti][tj][gray[i,j]]
            v01 = mappings[ti][tj_next][gray[i,j]]
            v10 = mappings[ti_next][tj][gray[i,j]]
            v11 = mappings[ti_next][tj_next][gray[i,j]]

            # 双线性插值
            val = (1 - x_ratio) * (1 - y_ratio) * v00 + \
                  x_ratio * (1 - y_ratio) * v01 + \
                  (1 - x_ratio) * y_ratio * v10 + \
                  x_ratio * y_ratio * v11
            result[i,j] = np.clip(val, 0, 255).astype(np.uint8)
    return result

# ---------- 11. 双边滤波（扩展） ----------
def my_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    手写双边滤波
    image: 灰度图或彩色图（彩色图分别对每个通道处理）
    d: 直径（窗口大小，必须为奇数）
    sigma_color: 值域标准差
    sigma_space: 空间域标准差
    """
    if len(image.shape) == 3:
        # 彩色图：每个通道单独处理再合并
        channels = cv2.split(image)  # 使用OpenCV split简化，也可用numpy切片
        filtered_channels = [my_bilateral_filter(ch, d, sigma_color, sigma_space) for ch in channels]
        return cv2.merge(filtered_channels)
    else:
        # 灰度图
        h, w = image.shape
        pad = d // 2
        padded = np.pad(image, pad, mode='edge')
        result = np.zeros_like(image, dtype=np.float32)

        # 预计算空间高斯权重
        half = pad
        space_kernel = np.zeros((d, d))
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                space_kernel[i+half, j+half] = np.exp(-(i*i + j*j) / (2 * sigma_space*sigma_space))

        for i in range(h):
            for j in range(w):
                window = padded[i:i+d, j:j+d]
                # 值域权重：基于中心像素与邻域像素的亮度差
                center_val = window[half, half]
                range_weight = np.exp(-(window - center_val)**2 / (2 * sigma_color*sigma_color))
                weights = space_kernel * range_weight
                result[i,j] = np.sum(window * weights) / (np.sum(weights) + 1e-8)

        return np.clip(result, 0, 255).astype(np.uint8)


# ==================== 原实验函数（修改为使用手写函数） ====================

# 1. 点操作（无OpenCV依赖，保持不变）
def linear_stretch(img, low_percent=2, high_percent=98):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten()
    low_val = np.percentile(flat, low_percent)
    high_val = np.percentile(flat, high_percent)
    stretched = np.clip((img - low_val) * 255.0 / (high_val - low_val + 1e-6), 0, 255)
    return stretched.astype(np.uint8)

def gamma_transformation(img, gamma=1.0):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = img / 255.0
    corrected = np.power(img_norm, gamma)
    return (corrected * 255).astype(np.uint8)

def hist_equalization(img):
    """使用手写直方图均衡化"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return my_hist_equalize(img)

# 2. 卷积（替换cv2.filter2D）
def conv2d(image, kernel, padding='same'):
    """使用手写卷积"""
    return my_filter2D(image, kernel, padding=padding)

# 3. 添加噪声、PSNR（无OpenCV依赖，保持不变）
def add_noise(img, noise_type='gaussian', var=0.01):
    img_float = img.astype(np.float32) / 255.0
    if noise_type == 'gaussian':
        noise = np.random.normal(0, var**0.5, img.shape)
        noisy = np.clip(img_float + noise, 0, 1)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = 0.05
        noisy = img_float.copy()
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1]] = 1
        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1]] = 0
        noisy = np.clip(noisy, 0, 1)
    else:
        raise ValueError("Unsupported noise type")
    return (noisy * 255).astype(np.uint8)

def psnr(original, compressed):
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# 4. 降噪（使用手写滤波）
def mean_filter(img, kernel_size=3):
    return my_mean_filter(img, kernel_size)

def gaussian_kernel(size, sigma=1.0):
    return my_gaussian_kernel(size, sigma)

def gaussian_filter(img, kernel_size=3, sigma=1.0):
    return my_gaussian_filter(img, kernel_size, sigma)

def median_filter(img, kernel_size=3):
    return my_median_filter(img, kernel_size)

# 5. 边缘检测（使用手写算子）
def sobel_edge(img):
    return my_sobel(img)

def prewitt_edge(img):
    return my_prewitt(img)

def laplacian_edge(img):
    return my_laplacian(img)

def canny_edge(img, low_thresh=50, high_thresh=150):
    return my_canny(img, low_thresh, high_thresh)

# 6. 金字塔与融合（使用手写pyrDown/pyrUp，保留cv2.resize）
def gaussian_pyramid(img, levels=4):
    pyramid = [img]
    for i in range(levels-1):
        img = my_pyrDown(img)
        pyramid.append(img)
    return pyramid

def laplacian_pyramid(gauss_pyr):
    lap_pyr = []
    for i in range(len(gauss_pyr)-1):
        up = my_pyrUp(gauss_pyr[i+1])
        h, w = gauss_pyr[i].shape[:2]
        up = up[:h, :w]   # 尺寸对齐
        lap = gauss_pyr[i].astype(np.float32) - up.astype(np.float32)
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1].astype(np.float32))
    return lap_pyr

def pyramid_fusion(img1, img2, mask, levels=4):
    gauss_mask = gaussian_pyramid(mask, levels)
    lap1 = laplacian_pyramid(gaussian_pyramid(img1, levels))
    lap2 = laplacian_pyramid(gaussian_pyramid(img2, levels))
    fused_pyr = []
    for i in range(levels):
        curr_mask = cv2.resize(gauss_mask[i], (lap1[i].shape[1], lap1[i].shape[0]))
        fused = lap1[i] * (curr_mask / 255.0) + lap2[i] * (1.0 - curr_mask / 255.0)
        fused_pyr.append(fused)
    result = fused_pyr[-1]
    for i in range(levels-2, -1, -1):
        result = my_pyrUp(result.astype(np.uint8)).astype(np.float32)
        h, w = fused_pyr[i].shape[:2]
        result = result[:h, :w]
        result = result + fused_pyr[i]
    return np.clip(result, 0, 255).astype(np.uint8)

# 7. 扩展功能（CLAHE和双边滤波使用手写版本）
def clahe_equalization(img, clip_limit=2.0, grid_size=(8,8)):
    return my_clahe(img, clip_limit, grid_size)

def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return my_bilateral_filter(img, d, sigmaColor, sigmaSpace)


# ==================== 主演示函数 ====================
def main():
    # 创建输出目录
    os.makedirs('./output_hand_hand', exist_ok=True)

    # 读取测试图像（保留cv2.imread）
    img = cv2.imread('/home/zxx/HITCV/图像素材/OriginalImage.png')   
    if img is None:
        print("Image not found, generating a test image (256x256 gradient + checkerboard)")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img[i,j] = [i//2, j//2, (i+j)//4]
        # 保留cv2.rectangle
        cv2.rectangle(img, (100,100), (150,150), (255,255,255), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("="*50)
    print("Experiment 1: Fundamentals of Image Processing (Handwritten algorithms)")
    print("="*50)

    # 任务一：点操作
    print("\n[Task 1] Point operations...")
    linear = linear_stretch(gray)
    gamma_0_5 = gamma_transformation(gray, gamma=0.5)
    gamma_2_0 = gamma_transformation(gray, gamma=2.0)
    hist_eq = hist_equalization(gray)

    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1), plt.imshow(gray, cmap='gray'), plt.title('Original')
    plt.subplot(2,3,2), plt.imshow(linear, cmap='gray'), plt.title('Linear Stretch')
    plt.subplot(2,3,3), plt.imshow(gamma_0_5, cmap='gray'), plt.title('Gamma=0.5')
    plt.subplot(2,3,4), plt.imshow(gamma_2_0, cmap='gray'), plt.title('Gamma=2.0')
    plt.subplot(2,3,5), plt.imshow(hist_eq, cmap='gray'), plt.title('Histogram Equalization')
    plt.tight_layout()
    plt.savefig('./output_hand/point_operations.png')
    plt.show()
    
    # 绘制四张图片的直方图
    plt.figure(figsize=(12, 8))
    # 选择原图、线性拉伸、Gamma变换（取gamma=0.5）、直方图均衡化
    hist_images = [gray, linear, gamma_0_5, hist_eq]
    hist_titles = ['Original', 'Linear Stretch', 'Gamma (γ=0.5)', 'Histogram Equalization']
    for i, (img_hist, title) in enumerate(zip(hist_images, hist_titles), 1):
        plt.subplot(2, 2, i)
        plt.hist(img_hist.ravel(), bins=256, range=(0, 256), density=False, color='gray', alpha=0.7)
        plt.title(title)
        plt.xlabel('Pixel value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./output_hand/histograms.png')
    plt.show()
    # ------------------------------------------------

    # 任务二：卷积与降噪
    print("\n[Task 2] Convolution and denoising...")
    noisy_gaussian = add_noise(gray, 'gaussian', var=0.01)
    noisy_sp = add_noise(gray, 'salt_pepper')

    mean_my = mean_filter(noisy_gaussian, kernel_size=3)
    # 用于对比的OpenCV结果（注释掉，只用手写）
    # mean_cv = cv2.blur(noisy_gaussian, (3,3))

    gauss_my = gaussian_filter(noisy_gaussian, kernel_size=3, sigma=1.0)
    # gauss_cv = cv2.GaussianBlur(noisy_gaussian, (3,3), 1.0)

    median_res = median_filter(noisy_sp, kernel_size=3)

    # 计算PSNR（与原图比较）
    psnr_mean_my = psnr(gray, mean_my)
    psnr_gauss_my = psnr(gray, gauss_my)
    psnr_median = psnr(gray, median_res)

    print("Denoising PSNR (handwritten filters):")
    print(f"Mean filter: {psnr_mean_my:.2f} dB")
    print(f"Gaussian filter: {psnr_gauss_my:.2f} dB")
    print(f"Median filter (salt & pepper): {psnr_median:.2f} dB")

    plt.figure(figsize=(14,6))
    plt.subplot(2,4,1), plt.imshow(noisy_gaussian, cmap='gray'), plt.title('Gaussian Noise')
    plt.subplot(2,4,2), plt.imshow(mean_my, cmap='gray'), plt.title('Mean Filter')
    plt.subplot(2,4,3), plt.imshow(gauss_my, cmap='gray'), plt.title('Gaussian Filter')
    plt.subplot(2,4,4), plt.imshow(noisy_sp, cmap='gray'), plt.title('Salt & Pepper Noise')
    plt.subplot(2,4,5), plt.imshow(median_res, cmap='gray'), plt.title('Median Filter')
    # 不再显示OpenCV对比结果
    plt.tight_layout()
    plt.savefig('./output_hand/denoising.png')
    plt.show()

    # 任务三：边缘检测
    print("\n[Task 3] Edge detection...")
    sobel = sobel_edge(gray)
    prewitt = prewitt_edge(gray)
    laplacian = laplacian_edge(gray)
    canny = canny_edge(gray, 50, 150)

    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1), plt.imshow(gray, cmap='gray'), plt.title('Original')
    plt.subplot(2,3,2), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
    plt.subplot(2,3,3), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
    plt.subplot(2,3,4), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
    plt.subplot(2,3,5), plt.imshow(canny, cmap='gray'), plt.title('Canny')
    plt.tight_layout()
    plt.savefig('./output_hand/edge_detection.png')
    plt.show()

    # 任务四：金字塔融合（保留cv2.flip）
    print("\n[Task 4] Image pyramid fusion...")
    gray2 = cv2.flip(gray, 1)   # 保留
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[:, :mask.shape[1]//2] = 255
    fused = pyramid_fusion(gray, gray2, mask, levels=4)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.title('Image 1')
    plt.subplot(1,3,2), plt.imshow(cv2.cvtColor(gray2, cv2.COLOR_BGR2RGB)), plt.title('Image 2')
    plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)), plt.title('Pyramid Fusion Result')
    plt.tight_layout()
    plt.savefig('./output_hand/pyramid_fusion.png')
    plt.show()

    # 扩展功能演示
    print("\n[Extension] CLAHE and Bilateral Filter...")
    clahe_img = clahe_equalization(img, clip_limit=2.0)
    bilateral_img = bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(1,3,2), plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)), plt.title('CLAHE')
    plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2RGB)), plt.title('Bilateral Filter')
    plt.tight_layout()
    plt.savefig('./output_hand/extensions.png')
    plt.show()

    print("\nExperiment completed! All results saved in './output_hand' folder.")

if __name__ == "__main__":
    main()
