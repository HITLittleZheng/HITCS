import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Optional

# 基础工具函数
def read_image(path: str) -> np.ndarray:
    """读取图像，若为彩色则转为灰度图"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_pixel(img: np.ndarray, x: int, y: int) -> int:
    """获取图像(x, y)处的像素值（x为列坐标，y为行坐标）"""
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return img[y, x]
    else:
        raise IndexError(f"坐标 ({x}, {y}) 超出图像范围 ({w}x{h})")

def pad_image(img: np.ndarray, block_h: int, block_w: int, pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """将图像填充至可被块大小整除，返回填充图像和原始尺寸"""
    h, w = img.shape[:2]
    pad_h = (block_h - h % block_h) % block_h
    pad_w = (block_w - w % block_w) % block_w
    if len(img.shape) == 3:  # 彩色图（本项目中不会出现，但保留兼容）
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=pad_value)
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
    return padded, (h, w)

def split_blocks(img: np.ndarray, block_h: int, block_w: int) -> Tuple[List[np.ndarray], int, int]:
    """将图像划分为块列表，返回(块列表, 行块数, 列块数)"""
    h, w = img.shape[:2]
    rows = h // block_h
    cols = w // block_w
    blocks = []
    for i in range(rows):
        for j in range(cols):
            block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            blocks.append(block)
    return blocks, rows, cols

def merge_blocks(blocks: List[np.ndarray], rows: int, cols: int, block_h: int, block_w: int) -> np.ndarray:
    """将块列表合并为图像"""
    # 确定图像维度和类型
    sample = blocks[0]
    if len(sample.shape) == 3:
        channels = sample.shape[2]
        img = np.zeros((rows*block_h, cols*block_w, channels), dtype=sample.dtype)
    else:
        img = np.zeros((rows*block_h, cols*block_w), dtype=sample.dtype)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = blocks[idx]
            idx += 1
    return img

def shuffle_blocks(blocks: List[np.ndarray], seed: int) -> Tuple[List[np.ndarray], List[int]]:
    """打乱块顺序，返回(打乱后的块, 置换索引)"""
    random.seed(seed)
    indices = list(range(len(blocks)))
    random.shuffle(indices)
    shuffled = [blocks[i] for i in indices]
    return shuffled, indices

def unshuffle_blocks(shuffled_blocks: List[np.ndarray], indices: List[int]) -> List[np.ndarray]:
    """根据置换索引恢复原始块顺序"""
    inv = [0] * len(indices)
    for i, idx in enumerate(indices):
        inv[idx] = i
    return [shuffled_blocks[i] for i in inv]

# 全局置乱与恢复 
def PermutationFun(inputImage, blockwidth: int, blockheight: int, seed: int, pad_value: int = 0):
    """
    对图像进行分块置乱并显示
    inputImage: 图像路径或numpy数组
    blockwidth: 块宽度
    blockheight: 块高度
    seed: 随机种子
    pad_value: 填充值（0或255）
    return: (置乱后的填充图像, 原始尺寸, 置换索引, (块高,块宽), 填充值)
    """
    # 读取图像
    if isinstance(inputImage, str):
        img = read_image(inputImage)
    else:
        img = inputImage.copy()

    # 填充
    padded, orig_shape = pad_image(img, blockheight, blockwidth, pad_value)

    # 分块
    blocks, rows, cols = split_blocks(padded, blockheight, blockwidth)

    # 置乱
    shuffled_blocks, perm = shuffle_blocks(blocks, seed)

    # 合并
    shuffled_padded = merge_blocks(shuffled_blocks, rows, cols, blockheight, blockwidth)

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(shuffled_padded, cmap='gray')
    plt.title(f'Shuffled (blocks {blockwidth}x{blockheight})')
    plt.show()

    return shuffled_padded, orig_shape, perm, (blockheight, blockwidth), pad_value

def recover_shuffled(shuffled_padded: np.ndarray, original_shape: Tuple[int, int],
                     perm: List[int], block_size: Tuple[int, int]) -> np.ndarray:
    """
    恢复置乱图像
    shuffled_padded: 置乱后的填充图像
    original_shape: 原始图像尺寸 (h, w)
    perm: 置换索引
    block_size: (块高, 块宽)
    return: 恢复的原始图像
    """
    block_h, block_w = block_size
    # 重新分块
    blocks, rows, cols = split_blocks(shuffled_padded, block_h, block_w)
    # 逆置乱
    original_blocks = unshuffle_blocks(blocks, perm)
    # 合并
    padded_original = merge_blocks(original_blocks, rows, cols, block_h, block_w)
    # 裁剪回原始尺寸
    h_orig, w_orig = original_shape

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(padded_original, cmap='gray')
    plt.title(f'Recover_Shuffled(blocks {block_w}x{block_h})')
    plt.show()

    return padded_original[:h_orig, :w_orig]

#指定区域置乱
def region_shuffle(inputImage, roi_x: int, roi_y: int, roi_w: int, roi_h: int,
                   blockwidth: int, blockheight: int, seed: int, pad_value: int = 0) -> np.ndarray:
    """
    对图像指定区域进行分块置乱并显示
    :param inputImage: 图像路径或numpy数组
    :param roi_x, roi_y: 区域左上角坐标
    :param roi_w, roi_h: 区域宽高
    :param blockwidth: 块宽度
    :param blockheight: 块高度
    :param seed: 随机种子
    :param pad_value: 填充值
    :return: 置乱后的图像
    """
    if isinstance(inputImage, str):
        img = read_image(inputImage)
    else:
        img = inputImage.copy()

    # 提取ROI
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

    # 填充ROI
    padded_roi, (roi_h_orig, roi_w_orig) = pad_image(roi, blockheight, blockwidth, pad_value)

    # 分块置乱
    blocks, rows, cols = split_blocks(padded_roi, blockheight, blockwidth)
    shuffled_blocks, _ = shuffle_blocks(blocks, seed)
    shuffled_padded_roi = merge_blocks(shuffled_blocks, rows, cols, blockheight, blockwidth)

    # 裁剪回原ROI大小（去掉填充边缘）
    shuffled_roi = shuffled_padded_roi[:roi_h, :roi_w]

    # 放回原图
    result = img.copy()
    result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = shuffled_roi

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original with ROI')
    rect = plt.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Region Shuffled')
    plt.show()

    return result

# 使用
if __name__ == "__main__":
    image_path = "D:/study/AI_Math/experiment/1/DFT/beauty1440x1440.jpeg"  

    # 读取图像
    img = read_image(image_path)

    # 测试获取像素值
    try:
        val = get_pixel(img, 50, 30)
        print(f"像素(50,30) = {val}")
    except IndexError as e:
        print(e)

    # 测试全局置乱（多种块大小）
    for bw, bh in [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]:
        shuffled, orig_shape, perm, block_size, pad_val = PermutationFun(
            img, bw, bh, seed=42, pad_value=0
        )
        # 恢复并验证
        recovered = recover_shuffled(shuffled, orig_shape, perm, block_size)
        if np.array_equal(img, recovered):
            print(f"块大小 {bw}x{bh}: 恢复成功")
        else:
            print(f"块大小 {bw}x{bh}: 恢复失败")
            # 显示差异（可选）
            diff = np.abs(img.astype(np.int16) - recovered.astype(np.int16))
            print(f"最大差异: {np.max(diff)}")

    # 测试区域置乱
    h, w = img.shape
    roi_x, roi_y = 100, 100
    roi_w, roi_h = 740, 740
    if roi_x + roi_w <= w and roi_y + roi_h <= h:
        region_shuffle(img, roi_x, roi_y, roi_w, roi_h, 16, 16, seed=123, pad_value=255)
    else:
        print("ROI超出图像范围，跳过区域置乱测试")