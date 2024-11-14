from CCA import CCA
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import os

__author__ = "Zephyr369"

class RSACCA(CCA):
    # 继承上次实验的CCA类，调用用super()
    rsa_key_path = r"key\rsa_keys"
    
    def __init__(self, *args, **kwargs):
        # 先顺手调用一下CCA的构造函数吧CCA初始化了
        super().__init__(*args, **kwargs)
        self._ensure_rsa_keys()

    def _ensure_rsa_keys(self):
        os.makedirs(os.path.dirname(self.rsa_key_path), exist_ok=True)
        # 首先读取存不存在RSA的公钥或者私钥，如果存在，读取，否则，生成
        if not os.path.exists(f"{self.rsa_key_path}_private.pem") or not os.path.exists(f'{self.rsa_key_path}_public.pem'):
            self._generate_and_save_rsa_keys()

    def _generate_and_save_rsa_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size = 2048,
            backend = default_backend()
        )
        public_key = private_key.public_key()

        # 保存私钥和公钥
        with open(f"{self.rsa_key_path}_private.pem", 'w') as file:
            file.write(
                private_key.private_bytes(
                    encoding = serialization.Encoding.PEM, # 指定输出格式为PEM
                    format = serialization.PrivateFormat.PKCS8, # 指定私钥的格式为PKCS#8
                     encryption_algorithm = serialization.NoEncryption()
                ).decode('utf-8')
            )
        with open(f'{self.rsa_key_path}_public.pem', 'w') as file:
            file.write(
                public_key.public_bytes(
                    encoding = serialization.Encoding.PEM,
                    format = serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8')
            )
            
    # 用RSA公钥加密对称密钥
    def encrypt_symmetric_key_with_rsa(self):
        with open(f'{self.rsa_key_path}_public.pem', 'rb') as file:
            public_key = serialization.load_pem_public_key(
                file.read(),
                backend = default_backend()
            )

        encrypted_key = public_key.encrypt(
            self.key,
            padding.OAEP(
                mgf = padding.MGF1(algorithm = hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            )
        )
        return encrypted_key
    
    def decrypt_symmetric_key_with_rsa(self, encrypted_key):
        with open(f'{self.rsa_key_path}_private.pem', 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

        decrypted_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_key
    
    # 重写父类CCA的加密图片方法
    def encrypt_image(self, input_path, output_path):
        # 用公钥加密对称密钥
        encrypted_symmetric_key = self.encrypt_symmetric_key_with_rsa()

        with open("key/encrypted_key.bin", "wb") as file:
            file.write(encrypted_symmetric_key)
        
        super().encrypt_image(input_path, output_path)

    # 解密
    def decrypt_image(self, input_path, output_path):
        with open("key/encrypted_key.bin", "rb") as file:
            encrypted_symmetric_key = file.read()
        self.key = self.decrypt_symmetric_key_with_rsa(encrypted_symmetric_key)
        super().decrypt_image(input_path, output_path)