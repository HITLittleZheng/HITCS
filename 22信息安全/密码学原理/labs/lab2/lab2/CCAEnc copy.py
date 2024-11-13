from audio import CCAudio
__author__ = 'Zephyr369'

cca = CCAudio()
cca.encrypt_wav(r'img\未命名 1.wav', r'img\导出.wav')
cca.decrypt_wav(r'img\导出.wav', r'img\导出解密.wav')