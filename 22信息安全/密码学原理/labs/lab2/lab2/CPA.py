import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
from PIL import Image
import numpy as np
__author__ = 'Zephyr369'

'''生成一个秘钥和iv，在本次实验中，我们假设密钥和iv在通信双方已经通过安全手段事先准备好'''
# 读取秘钥

# 封装为一个class，这样可以抽象的调用加密解密接口，而且可以封装一个oracle
class CPA(object):
    # 初始 决定好秘钥和iv
    def __init__(self, key_path=r"key\key.bin", iv_path=r"key\iv.bin"):
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        # 生成或读取密钥和IV
        if not os.path.exists(key_path) or not os.path.exists(iv_path):
            self.key = os.urandom(32)  # AES-256要求的密钥长度
            self.iv = os.urandom(16)   # OFB模式的IV大小
            with open(key_path, "wb") as key_file:
                key_file.write(self.key)
            with open(iv_path, "wb") as iv_file:
                iv_file.write(self.iv)
            self.cipher = Cipher(algorithms.AES(self.key), modes.OFB(self.iv), backend=default_backend())
        else:
            with open(key_path, "rb") as key_file:
                self.key = key_file.read()
            with open(iv_path, "rb") as iv_file:
                self.iv = iv_file.read()
            self.cipher = Cipher(algorithms.AES(self.key), modes.OFB(self.iv), backend=default_backend())

    def _process_image(self, input_path, output_path, mode='encrypt'):
        with Image.open(input_path) as img:
            r, g, b = img.split()
        # 按照 r g b 通道对图片分别进行加解密
        if mode == 'encrypt': # 加密
            r_processed = self._encrypt_schema(np.array(r, dtype=np.uint8).tobytes())
            g_processed = self._encrypt_schema(np.array(g, dtype=np.uint8).tobytes())
            b_processed = self._encrypt_schema(np.array(b, dtype=np.uint8).tobytes())
        else:  # 解密
            r_processed = self._decrypt_schema(np.array(r, dtype=np.uint8).tobytes())
            g_processed = self._decrypt_schema(np.array(g, dtype=np.uint8).tobytes())
            b_processed = self._decrypt_schema(np.array(b, dtype=np.uint8).tobytes())

        # 重建图像
        r_new = Image.frombytes('L', r.size, r_processed)
        g_new = Image.frombytes('L', g.size, g_processed)
        b_new = Image.frombytes('L', b.size, b_processed)
        img_new = Image.merge("RGB", (r_new, g_new, b_new))
        img_new.save(output_path)

        # 加密路线
    def _encrypt_schema(self, image):

        encryptor = self.cipher.encryptor()
        return encryptor.update(image) + encryptor.finalize()
        # 解密路线
    def _decrypt_schema(self, image):
        decryptor = self.cipher.decryptor()
        return decryptor.update(image) + decryptor.finalize()

    def encrypt_image(self, input_path, output_path):
        self._process_image(input_path, output_path, mode='encrypt')

    def decrypt_image(self, input_path, output_path):
        self._process_image(input_path, output_path, mode='decrypt')