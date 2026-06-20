import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def process_dft_experiment(image_path):
    # 1. 读取图像并转换为 YCbCr，提取 Y 分量
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("请检查图像路径是否正确！")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y_channel = img_ycbcr[:, :, 0].astype(np.float32)

    # 将图像尺寸填充为 8 的整数倍 (不足部分补 0)
    h, w = Y_channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    Y_padded = np.pad(Y_channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    new_h, new_w = Y_padded.shape

    # 对 Y 分量做分块 8x8 二维 DFT
    dft_result = np.zeros_like(Y_padded, dtype=np.complex128)
    
    start_time = time.time()
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = Y_padded[i:i+8, j:j+8]
            # 进行 2D FFT 并将其移到中心
            f_transform = np.fft.fft2(block)
            f_shift = np.fft.fftshift(f_transform)
            dft_result[i:i+8, j:j+8] = f_shift
    end_time = time.time()
    print(f"8x8 分块 DFT 计算耗时: {end_time - start_time:.4f} 秒")

    # 计算幅度谱、相位谱、能量谱
    magnitude_spectrum = np.abs(dft_result)
    phase_spectrum = np.angle(dft_result)
    power_spectrum = magnitude_spectrum ** 2

    # 为了可视化幅度谱，通常需要取对数
    vis_magnitude = np.log(1 + magnitude_spectrum)

    # 逆变换：分别只保留幅度 或 只保留相位
    recon_magnitude_only = np.zeros_like(Y_padded)
    recon_phase_only = np.zeros_like(Y_padded)

    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            # 获取当前块的幅度和相位
            mag_block = magnitude_spectrum[i:i+8, j:j+8]
            phase_block = phase_spectrum[i:i+8, j:j+8]

            # 只保留幅度 (相位设为 0)
            complex_mag_only = mag_block * np.exp(1j * 0)
            f_ishift_mag = np.fft.ifftshift(complex_mag_only)
            recon_magnitude_only[i:i+8, j:j+8] = np.abs(np.fft.ifft2(f_ishift_mag))

            # 只保留相位 (幅度设为 1)
            complex_phase_only = 1 * np.exp(1j * phase_block)
            f_ishift_phase = np.fft.ifftshift(complex_phase_only)
            recon_phase_only[i:i+8, j:j+8] = np.abs(np.fft.ifft2(f_ishift_phase))

    #可视化
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(Y_padded, cmap='gray'); axes[0, 0].set_title('原始 Y 分量 (补零后)')
    axes[0, 1].imshow(vis_magnitude, cmap='gray'); axes[0, 1].set_title('幅度谱 (对数变换)')
    axes[0, 2].imshow(phase_spectrum, cmap='gray'); axes[0, 2].set_title('相位谱')
    axes[1, 0].imshow(np.log(1 + power_spectrum), cmap='gray'); axes[1, 0].set_title('能量谱 (对数变换)')
    axes[1, 1].imshow(recon_magnitude_only, cmap='gray'); axes[1, 1].set_title('仅保留幅度重建图像')
    axes[1, 2].imshow(recon_phase_only, cmap='gray'); axes[1, 2].set_title('仅保留相位重建图像')

    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()

process_dft_experiment('D:\\study\\AI_Math\\experiment\\1\\DFT\\beauty1440x1440.jpeg')
#process_dft_experiment('D:\\study\\AI_Math\\experiment\\1\\DFT\\letter_A_8x8.png')