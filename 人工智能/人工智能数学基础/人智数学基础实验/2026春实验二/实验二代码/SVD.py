from PIL import Image
import numpy as np

base_path = "D:\\study\\AI_Math\\experiment\\2\\"
INPUT_IMAGE = base_path+"beauty1440x1440.jpeg"  # 输入图片路径
# 打开并处理图片
img = Image.open(INPUT_IMAGE)
img_gray = img.convert('L')

# 转换为矩阵
matrix = np.array(img_gray)
print(f"\n矩阵信息:")
print(f"形状: {matrix.shape} (高度:{matrix.shape[0]}, 宽度:{matrix.shape[1]})")
print(f"元素数量: {matrix.size}")
print(f"像素值范围: {matrix.min()} ~ {matrix.max()}")

# SVD分解
U, S, Vt = np.linalg.svd(matrix, full_matrices=True)
print(f"SVD分解完成!")

# 将S向量转换为对角矩阵
m, n = matrix.shape  
Sigma= np.zeros((m, n))
Sigma[:len(S), :len(S)] = np.diag(S)
print(f"U形状: {U.shape}, Σ形状: {Sigma.shape}, Vt形状: {Vt.shape}")

# 使用 matplotlib 展示 原始图像、U、Σ、Vt
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ['original', 'U', 'Σ', 'Vᵀ']
images = [img_gray, U, Sigma, Vt]   # img_gray 是之前转换好的灰度图像

for ax, title, mat in zip(axes, titles, images):
    data = mat
    ax.imshow(data, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

def truncate_sigma_matrix(sigma_matrix, threshold_ratio):
    """
    从Sigma矩阵提取奇异值并截断
    
    参数:
    sigma_matrix: Sigma矩阵 (m×n)
    threshold_ratio: 阈值比例
    """
    # 提取对角线元素（奇异值）
    sigma_vector = np.diag(sigma_matrix)
    
    print(f"从Sigma矩阵提取奇异值:")
    print(f"矩阵形状: {sigma_matrix.shape}")
    print(f"奇异值数量: {len(sigma_vector)}")
    print(f"奇异值范围: {sigma_vector[0]:.6f} ~ {sigma_vector[-1]:.6f}")
    
    # 计算阈值
    max_sigma = sigma_vector[0]
    threshold = max_sigma * threshold_ratio
    
    # 应用截断
    truncated = sigma_vector.copy()
    truncated[truncated < threshold] = 0
    
    # 统计
    kept_count = np.sum(truncated > 0)
    zero_count = np.sum(truncated == 0)
    
    print(f"\n截断阈值: {threshold:.6e} (最大值的{threshold_ratio:.1%})")
    print(f"保留的奇异值: {kept_count}/{len(truncated)}")
    print(f"置零的奇异值: {zero_count}/{len(truncated)}")
    
    return truncated, sigma_vector

truncated, original = truncate_sigma_matrix(Sigma, 0.01)
# 将截断后的向量truncated转换为对角矩阵Sigma_new
m, n = Sigma.shape
Sigma_new= np.zeros((m, n))
Sigma_new[:len(truncated), :len(truncated)] = np.diag(truncated)
new_matrix = U @ Sigma_new @ Vt

# 使用 matplotlib 展示 原始图像、去除噪声的图像
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2, figsize=(20, 5))
titles = ['original', 'SVD']
images = [img_gray,new_matrix]   # img_gray 是之前转换好的灰度图像

for ax, title, mat in zip(axes, titles, images):
    data = mat
    ax.imshow(data, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()