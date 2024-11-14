__author__ = 'Zephyr369'
# 自己试了试wav
import os
import numpy as np
import wave
import hmac
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
import hashlib 

class CCAudio(object):
    def __init__(self, key_path="key/key.bin", iv_path="key/iv.bin", mac_key_path="key/mac_key.bin"):
        self.backend = default_backend()
        # 生成或读取密钥和IV
        if not os.path.exists(key_path) or not os.path.exists(iv_path) or not os.path.exists(mac_key_path):
            self.key = os.urandom(32)  # AES-256要求的密钥长度
            self.iv = os.urandom(16)   # OFB模式的IV大小
            self.mac_key = os.urandom(32) # HMAC密钥长度
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
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

    def _encrypt_audio(self, audio_data):
        encryptor = self.cipher.encryptor()
        encrypted_data = encryptor.update(audio_data) + encryptor.finalize()
        return encrypted_data

    def _decrypt_audio(self, encrypted_data):
        decryptor = self.cipher.decryptor()
        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()
        return plaintext

    def _generate_audio_mac(self, audio_data):
        h = hmac.new(self.mac_key, msg=audio_data, digestmod=hashlib.sha256)
        return h.digest()

    def encrypt_wav(self, input_path, output_path):
        with wave.open(input_path, 'rb') as infile:
            params = infile.getparams()
            audio_data = infile.readframes(params.nframes)

        encrypted_data = self._encrypt_audio(audio_data)
        audio_mac = self._generate_audio_mac(encrypted_data)

        with wave.open(output_path, 'wb') as outfile:
            outfile.setparams(params)
            outfile.writeframes(encrypted_data + audio_mac)

    def decrypt_wav(self, input_path, output_path):
        with wave.open(input_path, 'rb') as infile:
            params = infile.getparams()
            encrypted_data = infile.readframes(params.nframes)
            audio_data, audio_mac = encrypted_data[:-32], encrypted_data[-32:]

        recalculated_mac = self._generate_audio_mac(audio_data)

        if audio_mac != recalculated_mac:
            raise ValueError("MAC校验失败，密文不可以修改的哦亲！")

        decrypted_data = self._decrypt_audio(audio_data)

        with wave.open(output_path, 'wb') as outfile:
            outfile.setparams(params)
            outfile.writeframes(decrypted_data)
