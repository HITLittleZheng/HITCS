"""
实验一：图像处理基础综合实验
功能涵盖：点操作（线性拉伸、Gamma校正、直方图均衡化）、手工2D卷积、
均值/高斯/中值滤波、边缘检测（Sobel/Prewitt/Laplacian/Canny）、
高斯与拉普拉斯金字塔构建及多尺度图像融合，以及可选扩展（CLAHE/双边滤波）。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#  1. 点操作 
def linear_stretch(img, low_percent=2, high_percent=98):
    """线性拉伸：将指定百分位区间映射到[0,255]"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten()
    low_val = np.percentile(flat, low_percent)
    high_val = np.percentile(flat, high_percent)
    stretched = np.clip((img - low_val) * 255.0 / (high_val - low_val + 1e-6), 0, 255)
    return stretched.astype(np.uint8)

def gamma_transformation(img, gamma=1.0):
    """Gamma变换：输出 = 255 * (输入/255)^gamma"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = img / 255.0
    corrected = np.power(img_norm, gamma)
    return (corrected * 255).astype(np.uint8)

def hist_equalization(img):
    """直方图均衡化（使用OpenCV）"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)

#  2. 手工2D卷积 （步长为1）
def my_2d_conv(image, kernel, padding='same'):
    """手工实现2D卷积，支持'same'（有填充，输入和输出同尺寸）和'valid'模式（无填充）"""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if padding == 'same':
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        out_h, out_w = h, w
    else:  # valid
        pad_img = image
        out_h, out_w = h - kh + 1, w - kw + 1

    result = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = pad_img[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result

def add_noise(img, noise_type='gaussian', var=0.01):
    """添加高斯噪声或椒盐噪声"""
    img_float = img.astype(np.float32) / 255.0
    if noise_type == 'gaussian':
        noise = np.random.normal(0, var**0.5, img.shape)
        noisy = np.clip(img_float + noise, 0, 1)
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5    #盐噪声（白点）占总噪声的比例
        amount = 0.05   #总噪声密度，即受噪声影响的像素（或元素）占总像素数的比例
        noisy = img_float.copy()
        # 盐噪声
        num_salt = np.ceil(amount * img.size * s_vs_p)#盐噪声的数量，向上取整。
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]#随机坐标
        noisy[coords[0], coords[1]] = 1
        # 胡椒噪声
        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1]] = 0
        noisy = np.clip(noisy, 0, 1)
    else:
        raise ValueError("Unsupported noise type")
    return (noisy * 255).astype(np.uint8)

def psnr(original, compressed):    #original是原始图像， compressed是处理后的图像
    """计算PSNR"""
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

#  3. 图像降噪：均值/高斯/中值滤波 
def mean_filter(img, kernel_size=3):
    """均值滤波，使用手工卷积"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    return my_2d_conv(img, kernel).astype(np.uint8)

def gaussian_kernel(size, sigma=1.0):
    """生成高斯核"""
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_filter(img, kernel_size=3, sigma=1.0):
    """高斯滤波，使用手工卷积"""
    kernel = gaussian_kernel(kernel_size, sigma)
    return my_2d_conv(img, kernel).astype(np.uint8)

def median_filter(img, kernel_size=3):
    """中值滤波（调用OpenCV快速实现）"""
    return cv2.medianBlur(img, kernel_size)

#  4. 边缘检测 
def sobel_edge(img):
    """Sobel边缘检测"""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(np.clip(sobel, 0, 255))

def prewitt_edge(img):
    """Prewitt算子手工实现"""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(img.astype(np.float32), -1, kernel_y)
    prewitt = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(np.clip(prewitt, 0, 255))

def laplacian_edge(img):
    """Laplacian边缘检测"""
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    return np.uint8(np.clip(np.abs(lap), 0, 255))

def canny_edge(img, low_thresh=50, high_thresh=150):
    """Canny边缘检测"""
    return cv2.Canny(img, low_thresh, high_thresh)

#  5. 图像金字塔与融合 
def gaussian_pyramid(img, levels=4):
    """构建高斯金字塔"""
    pyramid = [img]
    for i in range(levels-1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def laplacian_pyramid(gauss_pyr):
    """从高斯金字塔构建拉普拉斯金字塔"""
    lap_pyr = []
    for i in range(len(gauss_pyr)-1):
        up = cv2.pyrUp(gauss_pyr[i+1])
        # 确保尺寸一致
        h, w = gauss_pyr[i].shape[:2]
        up = up[:h, :w]
        lap = cv2.subtract(gauss_pyr[i].astype(np.float32), up.astype(np.float32))
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1].astype(np.float32))  # 最后一层为高斯残差
    return lap_pyr

def pyramid_fusion(img1, img2, mask, levels=4):
    """
    拉普拉斯金字塔融合
    mask: 二值或灰度掩码（与原始图像同尺寸），表示融合权重
    """
    # 构建高斯金字塔用于mask
    gauss_mask = gaussian_pyramid(mask, levels)
    # 构建两幅图像的拉普拉斯金字塔
    lap1 = laplacian_pyramid(gaussian_pyramid(img1, levels))
    lap2 = laplacian_pyramid(gaussian_pyramid(img2, levels))
    # 融合每一层
    fused_pyr = []
    for i in range(levels):
        # 将mask缩放到当前层尺寸
        curr_mask = cv2.resize(gauss_mask[i], (lap1[i].shape[1], lap1[i].shape[0]))
        if len(curr_mask.shape) == 2:
            curr_mask = np.expand_dims(curr_mask, axis=2)
        # RGB三个通道分别融合
        fused = lap1[i] * (curr_mask / 255.0) + lap2[i] * (1.0 - curr_mask / 255.0)
        fused_pyr.append(fused)
    # 重建图像
    result = fused_pyr[-1]
    for i in range(levels-2, -1, -1):
        result = cv2.pyrUp(result)
        h, w = fused_pyr[i].shape[:2]
        result = result[:h, :w]
        result = result + fused_pyr[i]
    return np.clip(result, 0, 255).astype(np.uint8)

#  6. 扩展功能（加分项） 
def clahe_equalization(img, clip_limit=2.0, grid_size=(8,8)):
    """CLAHE自适应直方图均衡化"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)

def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    """双边滤波"""
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

#  主演示函数 
def main():
    # 创建输出目录
    os.makedirs('./output', exist_ok=True)

    # 读取测试图像
    img = cv2.imread('/home/zxx/HITCV/图像素材/OriginalImage.png')   
    if img is None:
        print("Image not found, generating a test image (256x256 gradient + checkerboard)")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img[i,j] = [i//2, j//2, (i+j)//4]
        cv2.rectangle(img, (100,100), (150,150), (255,255,255), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("="*50)
    print("Experiment 1: Fundamentals of Image Processing")
    print("="*50)

    # - 任务一：点操作 -
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
    plt.savefig('./output/point_operations.png')
    plt.show()

    # - 任务二：卷积与降噪 -
    print("\n[Task 2] Convolution and denoising...")
    noisy_gaussian = add_noise(gray, 'gaussian', var=0.01)
    noisy_sp = add_noise(gray, 'salt_pepper')

    mean_my = mean_filter(noisy_gaussian, kernel_size=3)
    mean_cv = cv2.blur(noisy_gaussian, (3,3))

    gauss_my = gaussian_filter(noisy_gaussian, kernel_size=3, sigma=1.0)
    gauss_cv = cv2.GaussianBlur(noisy_gaussian, (3,3), 1.0)

    median_res = median_filter(noisy_sp, kernel_size=3)

    psnr_mean_my = psnr(gray, mean_my)
    psnr_mean_cv = psnr(gray, mean_cv)
    psnr_gauss_my = psnr(gray, gauss_my)
    psnr_gauss_cv = psnr(gray, gauss_cv)
    psnr_median = psnr(gray, median_res)

    print("Denoising PSNR comparison (original vs denoised):")
    print(f"Manual mean filter: {psnr_mean_my:.2f} dB | OpenCV mean filter: {psnr_mean_cv:.2f} dB")
    print(f"Manual Gaussian filter: {psnr_gauss_my:.2f} dB | OpenCV Gaussian filter: {psnr_gauss_cv:.2f} dB")
    print(f"Median filter (salt & pepper): {psnr_median:.2f} dB")

    plt.figure(figsize=(14,6))
    plt.subplot(2,4,1), plt.imshow(noisy_gaussian, cmap='gray'), plt.title('Gaussian Noise')
    plt.subplot(2,4,2), plt.imshow(mean_my, cmap='gray'), plt.title('Mean Filter (Manual)')
    plt.subplot(2,4,3), plt.imshow(gauss_my, cmap='gray'), plt.title('Gaussian Filter (Manual)')
    plt.subplot(2,4,4), plt.imshow(noisy_sp, cmap='gray'), plt.title('Salt & Pepper Noise')
    plt.subplot(2,4,5), plt.imshow(median_res, cmap='gray'), plt.title('Median Filter')
    plt.subplot(2,4,6), plt.imshow(mean_cv, cmap='gray'), plt.title('Mean Filter (OpenCV)')
    plt.subplot(2,4,7), plt.imshow(gauss_cv, cmap='gray'), plt.title('Gaussian Filter (OpenCV)')
    plt.tight_layout()
    plt.savefig('./output/denoising.png')
    plt.show()

    # - 任务三：边缘检测 -
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
    plt.savefig('./output/edge_detection.png')
    plt.show()

    # - 任务四：金字塔融合 -
    print("\n[Task 4] Image pyramid fusion...")
    img2 = cv2.flip(img, 1)  # horizontal flip
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[:, :mask.shape[1]//2] = 255   # left half from img1, right half from img2
    fused = pyramid_fusion(img, img2, mask, levels=4)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Image 1')
    plt.subplot(1,3,2), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image 2')
    plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)), plt.title('Pyramid Fusion Result')
    plt.tight_layout()
    plt.savefig('./output/pyramid_fusion.png')
    plt.show()

    # - 扩展功能演示 -
    print("\n[Extension] CLAHE and Bilateral Filter...")
    clahe_img = clahe_equalization(img, clip_limit=2.0)
    bilateral_img = bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(1,3,2), plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)), plt.title('CLAHE')
    plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2RGB)), plt.title('Bilateral Filter')
    plt.tight_layout()
    plt.savefig('./output/extensions.png')
    plt.show()

    print("\nExperiment completed! All results saved in './output' folder.")

if __name__ == "__main__":
    main()
