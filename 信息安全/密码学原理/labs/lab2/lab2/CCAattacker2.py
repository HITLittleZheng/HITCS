# CCA加密方案下的CCA敌手攻击
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
from PIL import Image
import numpy as np
from CCA import CCA
__author__ = 'Zephyr369'
# 解密oracle
def decrypt_oracle(input_file, output_file):
    cca = CCA()
    return cca.decrypt_image(input_file, output_file)

def tamper_encrypted_image(enc_path):
    above = r"img\CCAattack\上半部分篡改CCA.png"
    below = r"img\CCAattack\下半部分篡改CCA.png"
    result = r"img\CCAattack\破解结果CCA.png"
    # 加载加密后的图片
    with Image.open(enc_path) as img:
        r, g, b = img.split()
    # 将每个通道转换为numpy数组以便篡改
    r_array = np.array(r, dtype=np.uint8)
    g_array = np.array(g, dtype=np.uint8)
    b_array = np.array(b, dtype=np.uint8)
    # 篡改上半部分
    process_image(r_array.copy(), g_array.copy(), b_array.copy(), above, 1, r.size, enc_path)
    # 篡改下半部分
    process_image(r_array.copy(), g_array.copy(), b_array.copy(), below, 2, r.size, enc_path)
    # 二者拼合
    compose(above, below, result, r.size, enc_path)


def process_image(r, g, b, output_file, choice, size, input_file):
    # 计算图片高度
    height = r.shape[0]
    # 分半攻击
    tamper_height = height // 2
    # 上半部分
    if choice == 1:
        r[:tamper_height] ^= 0xFF
        g[:tamper_height] ^= 0xFF
        b[:tamper_height] ^= 0xFF
    # 下半部分
    if choice == 2:
        r[tamper_height:] ^= 0xFF
        g[tamper_height:] ^= 0xFF
        b[tamper_height:] ^= 0xFF
    
    r_new = Image.fromarray(r, mode='L')
    g_new = Image.fromarray(g, mode='L')
    b_new = Image.fromarray(b, mode='L')
    img_new = Image.merge("RGB", (r_new, g_new, b_new))
    img_new.save(output_file)
    decrypt_oracle(output_file,output_file)

def compose(above, below, result_path, size, input_file):
    with Image.open(above) as img:
        r, g, b = img.split()
    
    # 将每个通道转换为numpy数组以便篡改
    r_above = np.array(r, dtype=np.uint8)
    g_above = np.array(g, dtype=np.uint8)
    b_above = np.array(b, dtype=np.uint8)
    height = (r_above.shape[0]) // 2
    # 留下半部分
    r_above = r_above[height:]
    g_above = g_above[height:]
    b_above = b_above[height:]
    with Image.open(below) as img:
        r, g, b = img.split()
    # 将每个通道转换为numpy数组以便篡改
    r_below = np.array(r, dtype=np.uint8)
    g_below = np.array(g, dtype=np.uint8)
    b_below = np.array(b, dtype=np.uint8)
    # 留上半部分
    r_below = r_below[:height]
    g_below = g_below[:height]
    b_below = b_below[:height]
    # 合并
    r = np.concatenate((r_below, r_above), axis=0)
    g = np.concatenate((g_below, g_above), axis=0)
    b = np.concatenate((b_below, b_above), axis=0)

    r_new = Image.fromarray(r, mode='L')
    g_new = Image.fromarray(g, mode='L')
    b_new = Image.fromarray(b, mode='L')
    img_new = Image.merge("RGB", (r_new, g_new, b_new))
    img_new.save(result_path)

tamper_encrypted_image(r'img\avatarencCCA.png')
