import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 辅助函数：2D DCT 和 IDCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# 辅助函数：生成 8x8 的 Zig-Zag 掩码
def get_zigzag_mask(keep_k):
    
    #保留前 keep_k 个系数，其余置 0。返回一个布尔型掩码。
    
    mask = np.zeros((8, 8), dtype=bool)
    # 8x8 Zig-Zag 扫描位置对应的索引矩阵 (数值为 0 到 63)
    zigzag_order = np.array([
        [ 0,  1,  5,  6, 14, 15, 27, 28],
        [ 2,  4,  7, 13, 16, 26, 29, 42],
        [ 3,  8, 12, 17, 25, 30, 41, 43],
        [ 9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])
    mask[zigzag_order < keep_k] = True
    return mask

def display_dct_basis():
    #显示 2 维 8x8 DCT 变换的基函数图像
    basis_images = np.zeros((8 * 8, 8 * 8))
    for u in range(8):
        for v in range(8):
            # 只有 (u, v) 处为 1，其余为 0
            coeff = np.zeros((8, 8))
            coeff[u, v] = 1
            # 对其进行逆 DCT 得到基函数
            basis = idct2(coeff)
            # 填入大图显示
            basis_images[u*8:(u+1)*8, v*8:(v+1)*8] = basis

    plt.figure(figsize=(6, 6))
    plt.imshow(basis_images, cmap='gray')
    plt.title('2D 8x8 DCT base')
    plt.axis('off')
    plt.show()

def process_dct_experiment(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("请检查图像路径是否正确！")
        return
    img_gray = img_gray.astype(np.float32)
    
    # 填充尺寸
    h, w = img_gray.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    img_padded = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='constant')
    new_h, new_w = img_padded.shape

    # 显示基函数
    display_dct_basis()

    #DCT 变换及能量聚集特性分析
    keep_coeffs = [1, 2, 4, 6, 8, 10] # 按要求保留的系数个数
    results = {}

    for k in keep_coeffs:
        mask = get_zigzag_mask(k)
        reconstructed = np.zeros_like(img_padded)

        # 分块处理
        for i in range(0, new_h, 8):
            for j in range(0, new_w, 8):
                block = img_padded[i:i+8, j:j+8]
                # 正变换
                dct_block = dct2(block)
                # 截断系数 (掩码外的数据置 0)
                dct_block_truncated = dct_block * mask
                # 逆变换
                reconstructed[i:i+8, j:j+8] = idct2(dct_block_truncated)
        
        # 裁剪掉 padding 部分用于客观质量评估
        recon_crop = reconstructed[:h, :w]
        orig_crop = img_gray[:h, :w]
        
        # 计算 PSNR 和 SSIM 
        # data_range=255 表示像素最大差值范围
        val_psnr = psnr(orig_crop, recon_crop, data_range=255)
        val_ssim = ssim(orig_crop, recon_crop, data_range=255)
        
        results[k] = (recon_crop, val_psnr, val_ssim)
        print(f"保留系数个数 {k:2d} -> PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")

    # 可视化重建结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, k in enumerate(keep_coeffs):
        img_res, p, s = results[k]
        axes[idx].imshow(img_res, cmap='gray')
        axes[idx].set_title(f'{k} \PSNR:{p:.2f}, SSIM:{s:.4f}')
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()


process_dct_experiment('D:\\study\\AI_Math\\experiment\\1\\DFT\\beauty1440x1440.jpeg')
