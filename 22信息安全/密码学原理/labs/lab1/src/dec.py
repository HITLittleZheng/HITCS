# 只需要最简单的异或 只要密钥保密 就足以实现perfect secret
def xor_decrypt_hex(encrypted_text_hex, key):
    encrypted_bytes = bytes.fromhex(encrypted_text_hex)
    extended_key = key * (len(encrypted_bytes) // len(key)) + key[:len(encrypted_bytes) % len(key)]
    decrypted_text = ''.join(chr(b ^ ord(k)) for b, k in zip(encrypted_bytes, extended_key))
    return decrypted_text
encrypted_text_hex = input("请输入密文") 
key = input("请输入秘钥")
decrypted_text = xor_decrypt_hex(encrypted_text_hex, key)
print(f"明文(netease163): {decrypted_text}")
