# -*- coding: utf-8 -*-
import os
import wave
import numpy as np


def get_wav_filepath(directory):
    """ 读取一个目录下所有的 WAV 文件，并返回其文件路径列表 """
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files

def read_wav_file(file_name):
    """ 从wav文件中读入原始数据 """
    file = open(file_name, "rb")
    str_data = (file.read())[44:]                                               # 过滤掉前面的帧
    wavedata = np.frombuffer(str_data, dtype = np.short)
    return wavedata  

def quantizer(x, num_bits, factor):
    """ 量化器 """
    quantization_levels = 2 ** (num_bits - 1)
    high = quantization_levels - 1
    low = -1 * quantization_levels
    bias = quantization_levels
    if x > high * factor:
        return high + bias                                                      # 加上bias，变为非负值，方便压缩
    elif x < low * factor:
        return low + bias
    else:
        i = int(np.ceil(x / factor))                                            # 向上取整
        return i + bias

def compress(encoded_signal):
    """ 压缩 """
    num = len(encoded_signal)
    compressed_data = np.zeros(int(num/2), dtype=np.int8)                       # 创建一个与编码信号一半长度相同的数组，用于存储压缩后的数据
    for i in range(0, num-1, 2):
        # 将奇数位置的值放在高四位，偶数位置的值放在低四位
        compressed_data[int(i/2)] = (np.int8(encoded_signal[i]) << 4) + (np.int8(encoded_signal[i + 1]))
    return compressed_data

def decompress(compressed_data):
    """ 解压 """
    num_compressed = len(compressed_data)
    x = np.zeros(num_compressed * 2, dtype=np.int8)                             # x 用来存还原后的量化误差
    for i in range(num_compressed):
        # 解压缩，将高四位和低四位分别放入还原后的数组中
        x[2 * i + 1] = compressed_data[i] & np.int8(int('00001111', 2))
        x[2 * i] = (compressed_data[i] >> 4) & np.int8(int('00001111', 2))
    return x

def DPCM_encode_quantized(signal, num_bits, factor):
    """ DPCM编码器(量化因子为factor) """
    quantization_levels = 2 ** (num_bits - 1)
    encoded_signal = [signal[0]]                                                # 初始化编码信号列表，将第一个样本直接添加到列表中
    prediction = signal[0]                                                      # 初始化预测值为第一个样本值

    for sample in signal[1:]:                                                   # 对信号中的每个样本进行编码
        difference = sample - prediction                                        # 计算当前样本与预测值之间的差异
        quantized_difference = quantizer(difference, num_bits, factor)          # 量化差异值
        encoded_signal.append(quantized_difference)                             # 将量化后的差异添加到编码信号列表中
        prediction += (quantized_difference - quantization_levels) * factor     # 更新预测值

    return signal[0], encoded_signal  

def DPCM_decode_quantized(begin, encoded_signal, num_bits, factor):
    """ DPCM解码器(量化因子为factor) """
    quantization_levels = 2 ** (num_bits - 1)
    data = np.zeros(len(encoded_signal) + 1, dtype=np.int16)
    data[0] = begin.item()                                                      # 第一个点一直保持不变
    for i in range(1, len(encoded_signal)):
        data[i] = data[i - 1] + (encoded_signal[i] - quantization_levels) * factor  # 预测下一个点
    return data

def calculate_SNR(wavData, data):
    """ 计算原语音和解码后语音的信噪比 """
    signal_power = np.linalg.norm(data) ** 2                                        # 信号功率
    noise_power = np.linalg.norm(wavData - data) ** 2                               # 噪声功率
    snr = signal_power / noise_power                                                # 信噪比
    return 10 * np.log10(snr)                                                       # 将信噪比转换为分贝

def save_encoded_data_to_file(begin, encoded_signal, filename):
    with open('./dpc/' + filename + '_4bit.dpc', 'wb') as file:
        file.write(begin)
        file.write(encoded_signal)
            
def save_decoded_data_to_pcm(decoded_signal, filename, framerate=44100):
    with wave.open('./pcm/' + filename + '_4bit.dpc', 'w') as pcmfile:
        pcmfile.setparams((1, 2, framerate, 0, 'NONE', 'not compressed'))           # 设置PCM文件的参数
        pcmfile.writeframes(np.array(decoded_signal).astype(np.int16).tobytes())    # 将解码后的信号写入PCM文件

def DPCM(file_name, factor):
    """ DPCM编码解码 """
    file_base_name = os.path.basename(file_name).split('.')[0]
    wav_data = read_wav_file(file_name) 
    begin, encoded_data = DPCM_encode_quantized(wav_data, 4, factor)  
    compressed_data = compress(encoded_data)  
    save_encoded_data_to_file(begin, compressed_data, file_base_name)

    dpc_file = open('./dpc/' + file_base_name + '_4bit.dpc', "rb")

    begin = np.frombuffer(dpc_file.read(2), dtype = np.short)
    str_data = dpc_file.read()  
    dpc_data = np.frombuffer(str_data, dtype=np.int8)
    decompressed_data = decompress(dpc_data)
    decoded_data = DPCM_decode_quantized(begin, decompressed_data, 4, factor) 
    save_decoded_data_to_pcm(decoded_data, file_base_name)
    legth = len(decoded_data) if len(wav_data) > len(decoded_data) else len(wav_data)
    print("{:.3f}".format(calculate_SNR(wav_data[:legth], decoded_data[:legth])))
    


if __name__ == '__main__':
    factor = 800
    directory = "语料"
    wav_files = get_wav_filepath(directory)
    for (i, wav_file) in enumerate(wav_files):
        DPCM(wav_file, factor)

