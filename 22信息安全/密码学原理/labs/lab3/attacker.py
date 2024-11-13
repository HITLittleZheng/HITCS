# https://paper.seebug.org/727/
__author__ = "Zephyr369"
from Crypto.PublicKey import RSA
from Crypto.Util.number import GCD
from Crypto.Util.Padding import unpad
from Crypto.Cipher import AES, PKCS1_OAEP
import base64
from PIL import Image
import numpy as np
# 把该定义的函数都定义定义

# 提取两个公钥的共同因子
def find_common_factor(n1, n2):
    return GCD(n1, n2)

def compute_private_key(n, e, p):
    q = n // p
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)
    return RSA.construct((n, e, d, p, q))

def load_public_key(filename):
    with open(filename, 'rb') as file:
        key = RSA.import_key(file.read()) 
    # 返回key的模数n和指数e
    return key.n, key.e

# 实现实验中的对称加密方案
# 加载图片
def load_encrypted_image(path):
    # 使用Pillow读取图像
    img = Image.open(path).convert("RGBA")
    img_data = np.array(img)
    
    # 将图像数据转换为字节序列
    img_bytes = img_data.tobytes()

    # 提取IV（前16个字节）
    iv = img_bytes[:16]
    
    # 解析自定义填充
    # 最后一个像素（4字节）大端序表示填充长度（以像素为单位）
    padding_indicator = img_bytes[-4:]
    padding_length_pixels = int.from_bytes(padding_indicator, "big")
    
    # 计算去除自定义填充后的密文长度
    # 每个像素RGBA占用4字节
    encrypted_content_length = len(img_bytes) - 16 - padding_length_pixels * 4
    encrypted_content = img_bytes[16:16 + encrypted_content_length]
    
    return iv, encrypted_content
# 解密
def decrypt_image(encrypted_image, symmetric_key, iv):
    cipher = AES.new(symmetric_key, AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(encrypted_image)
    # # 使用unpad去除PKCS#7填充
    decrypted_data_no_pkcs7_padding = unpad(decrypted_data, AES.block_size, style='pkcs7')
    return decrypted_data_no_pkcs7_padding

# 保存
def save_decrypted_image(dec_data_no_padding, output_path, image_size):
    image = Image.frombytes("RGBA", image_size, dec_data_no_padding)
    image.save(output_path)

pub1_path = r"lab3\attacks\pub1.pem"
pub2_path = r"lab3\attacks\pub2.pem"

n1, e1 = load_public_key(pub1_path)
n2, e2 = load_public_key(pub2_path)

# 共同因子p
p = find_common_factor(n1, n2)
print("共同因子为:" + str(p))

# 计算第一个公钥对应的私钥
private_key = compute_private_key(n1, e1, p)
# 加密的对称密钥(先给他来一套美汁汁base64解码)
encrypted_symmetric_key = base64.b64decode("MzhKNQx+U8ltsj5is29pSwu7yqdgoWPWIhgEwUTz3ywE84ue99Z7T/AISGOuyud6ET4E8xXFS/7wadzwYj3yL6dQrw+F9KFPJRNkTDQll0Re+3kkGt2+M68HJRvmIcJaD1/0PNTv9gek5PdL59TNq/VerwqXusAIIOdclwhb+U1EGJzJ0RS+8Wyp/+PU4J5P2mtFSak5SKNzDB8yg00uyhRBZGriQzw+QQRZanWJYs45UFYIP+9ZMUK3lOkf3b8CT+qGW/HcDFwG59hn59PUvN8UFER3PcOTIRD/+RBSKoi1Sdr7uxvQ3XTBvFJKlDMp1es4yzewmOgluBY2DtGV+aAbLzu5Sy6EfF7tJgid8V9T9ZQ8nqW9vtWkt6Y2okRhdkpX+E+y240gU1BEHOUNglM6oJ1b0nGiAL5cjUtX0IknEAsZR/U2ztsMQRzvy10xJpIgipKB52aNh6BnYzFH4DYndfehKh1NjVckcJOK+krTiUNwQMNhRYSZ8v1pZH6jR96TuDPib1KcJopjaGdf9zNa2bkdJ7NSWTe9j1jHMPJYjrP6XCefsixRTWp5dEz3KgzWEgGBHmIhz2SYYWLcy0SKb3ljYFUrY6tDwVRC+Srkk4GOeS09OvxT3r9E/JdaiA9BXuRjrV7LeCAW18AwbpZEaTHxjrVcoZ5sWpNasCI=")
# 处以解密极刑，先实例化一个pkcs1的oaep标准下的rsa
cipher_rsa = PKCS1_OAEP.new(private_key)
symmetric_key_base64 = cipher_rsa.decrypt(encrypted_symmetric_key)
# 题目要求，先对对称密钥进行base64 加密 然后给出的也是base64 因此我们需要先base64解开 然后解密 然后再base64 方能得出最终答案
symmetric_key = base64.b64decode(symmetric_key_base64)

print(f"解密出来的对称密钥{symmetric_key}")

iv, encrypted_content = load_encrypted_image(r"lab3\attacks\target.png")
decrypted_data = decrypt_image(encrypted_content, symmetric_key, iv)
# 查看一下图片信息 1920*1080
width = 1920
height = 1080
image_size = (width, height)
save_decrypted_image(decrypted_data, r"lab3\attacks/dec.png", image_size)