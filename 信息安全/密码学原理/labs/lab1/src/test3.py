def xor_encrypt_decrypt_hex(input_text, key):
    # 确保密钥至少与文本长度一致
    extended_key = key * (len(input_text) // len(key)) + key[:len(input_text) % len(key)]
    # 执行异或操作并生成16进制表示
    encrypted_bytes = bytes(ord(c) ^ ord(k) for c, k in zip(input_text, extended_key))
    return encrypted_bytes.hex()

# 待加密的文本
text = "after hours (the weeknd) 5:08-5:44"
# 密钥
key = "sinokcilsinokcilsinokcilsinokcil"

# 加密并以16进制输出
encrypted_text_hex = xor_encrypt_decrypt_hex(text, key)
print(f"Encrypted text in hex: {encrypted_text_hex}")
