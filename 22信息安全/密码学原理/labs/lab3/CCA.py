__author__ = 'Zephyr369'
import os
import numpy as np
from PIL import Image, PngImagePlugin
import hmac
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
import hashlib 

class CCA(object):
    def __init__(self, key_path="lab3/key/key.bin", iv_path="lab3/key/iv.bin", mac_key_path="lab3/key/mac_key.bin"):
        self.backend = default_backend()
        # 生成或读取密钥和IV
        if not os.path.exists(key_path) or not os.path.exists(iv_path) or not os.path.exists(mac_key_path):
            self.key = os.urandom(32)  # AES-256要求的密钥长度
            self.iv = os.urandom(16)   # OFB模式的IV大小
            self.mac_key = os.urandom(32) # HMAC密钥长度
            with open(key_path, "wb") as key_file:
                key_file.write(self.key)
            with open(iv_path, "wb") as iv_file:
                iv_file.write(self.iv)
            with open(mac_key_path, "wb") as mac_key_file:
                mac_key_file.write(self.mac_key)
        else:
            with open(key_path, "rb") as key_file:
                self.key = key_file.read()
            with open(iv_path, "rb") as iv_file:
                self.iv = iv_file.read()
            with open(mac_key_path, "rb") as mac_key_file:
                self.mac_key = mac_key_file.read()
        
        self.cipher = Cipher(algorithms.AES(self.key), modes.OFB(self.iv), backend=self.backend)

    def _add_mac_chunk(self, image, mac):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("MAC", mac.hex())
        image.save("temp.png", pnginfo=metadata)
        return Image.open("temp.png")

    def _get_mac_from_chunk(self, image):
        metadata = image.info
        return bytes.fromhex(metadata.get("MAC", ""))

    def _encrypt_schema(self, plaintext):
        encryptor = self.cipher.encryptor()
        encrypted_data = encryptor.update(plaintext) + encryptor.finalize()

        return encrypted_data

    def _decrypt_schema(self, encrypted):
        decryptor = self.cipher.decryptor()
        plaintext = decryptor.update(encrypted) + decryptor.finalize()

        return plaintext

    def _generate_image_mac(self, image_data):
        # 生成图像内容的 MAC
        h = hmac.new(self.mac_key, msg=image_data, digestmod=hashlib.sha256)
        return h.digest()

    def _process_image(self, input_path, output_path, mode='encrypt'):
        with Image.open(input_path) as img:
            r, g, b = img.split()

        if mode == 'encrypt':
            # 加密 RGB 通道
            r_processed = self._encrypt_schema(np.array(r, dtype=np.uint8).tobytes())
            g_processed = self._encrypt_schema(np.array(g, dtype=np.uint8).tobytes())
            b_processed = self._encrypt_schema(np.array(b, dtype=np.uint8).tobytes())

            # 合并 RGB 通道为新的图像
            img_new = Image.merge("RGB", (Image.frombytes('L', r.size, r_processed),
                                        Image.frombytes('L', g.size, g_processed),
                                        Image.frombytes('L', b.size, b_processed)))

            # 生成图像内容的 MAC
            image_data = img_new.tobytes()
            image_mac = self._generate_image_mac(image_data)

            # 添加 MAC 到 PNG 图像中
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("MAC", image_mac.hex())
            img_new.save(output_path, format='PNG', pnginfo=metadata)  # 确保指定 pnginfo 参数和 format 参数
        else:
            # 提取出图像中的 MAC
            image_mac = self._get_mac_from_chunk(img)

            # 对图片的 RGB 内容部分重新生成 MAC
            image_data = img.tobytes()
            rgb_mac = self._generate_image_mac(image_data)

            # 比较原有的 MAC 和重新生成的 MAC
            if image_mac != rgb_mac:
                raise ValueError("MAC校验失败，密文不可以修改的哦亲！")

            # 解密 RGB 通道
            r_processed = self._decrypt_schema(np.array(r, dtype=np.uint8).tobytes())
            g_processed = self._decrypt_schema(np.array(g, dtype=np.uint8).tobytes())
            b_processed = self._decrypt_schema(np.array(b, dtype=np.uint8).tobytes())

            # 合并 RGB 通道为新的图像
            img_new = Image.merge("RGB", (Image.frombytes('L', r.size, r_processed),
                                          Image.frombytes('L', g.size, g_processed),
                                          Image.frombytes('L', b.size, b_processed)))

            img_new.save(output_path)

    def encrypt_image(self, input_path, output_path):
        self._process_image(input_path, output_path, mode='encrypt')

    def decrypt_image(self, input_path, output_path):
        self._process_image(input_path, output_path, mode='decrypt')
