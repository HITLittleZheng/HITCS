import os
import wave
import numpy as np
def get_wav_filepath(directory):
    """读取一个目录下所有的 WAV 文件，并返回其文件路径列表"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files

def read_wav_file(wav_file):
    """读取一个 WAV 文件，并返回其中的数据和采样率"""
    f = wave.open(wav_file, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype = np.short)
    f.close()
    return wave_data, framerate

def quantize(sample, num_bits=8):
    quantization_levels = 2 ** (num_bits - 1)           # 计算量化范围
    quantized_sample = max(-quantization_levels, min(quantization_levels - 1, round(sample)))
    return quantized_sample

def DPCM_encode_quantized(signal, num_bits=8):
    encoded_signal = [signal[0]]                        # 初始化编码信号列表，将第一个样本直接添加到列表中
    prediction = signal[0]                              # 初始化预测值为第一个样本值

    for sample in signal[1:]:                           # 对信号中的每个样本进行编码
        difference = sample - prediction                # 计算当前样本与预测值之间的差异
        quantized_difference = quantize(difference, num_bits)     # 量化差异值
        encoded_signal.append(quantized_difference)     # 将量化后的差异添加到编码信号列表中
        prediction += quantized_difference              # 更新预测值为当前样本值

    return encoded_signal

def DPCM_decode_quantized(encoded_signal, num_bits=8):
    decoded_signal = [encoded_signal[0]]                # 初始化解码信号列表，将第一个样本直接添加到列表中
    prediction = encoded_signal[0]                      # 初始化预测值为第一个样本值

    for quantized_difference in encoded_signal[1:]:     # 对编码信号中的每个量化差异进行解码
        difference = quantized_difference               # 解码量化差异值
        sample = prediction + difference                # 计算当前样本值
        decoded_signal.append(sample)                   # 将样本值添加到解码信号列表中
        prediction = sample                             # 更新预测值为当前样本值

    return decoded_signal

def calculate_SNR(original_signal, decoded_signal):
    
    snr = np.linalg.norm(original_signal) ** 2
    snr /= np.linalg.norm(original_signal - decoded_signal) ** 2
    return 10 * np.log10(snr)

if __name__ == "__main__":
    directory = "语料"
    wav_files = get_wav_filepath(directory)
    print("Found {} WAV files:".format(len(wav_files)))
    for wav_file in wav_files:
        wave_data, framerate = read_wav_file(wav_file)
        encoded_signal_8bit = DPCM_encode_quantized(wave_data, num_bits=8)
        decoded_signal_8bit = DPCM_decode_quantized(encoded_signal_8bit, num_bits=8)
        SNR_8bit = calculate_SNR(wave_data, decoded_signal_8bit)
        print("{} 8-bit: SNR = {:.2f} dB".format(wav_file, SNR_8bit))
        encoded_signal_4bit = DPCM_encode_quantized(wave_data, num_bits=4)
        decoded_signal_4bit = DPCM_decode_quantized(encoded_signal_4bit, num_bits=4)
        SNR_4bit = calculate_SNR(wave_data, decoded_signal_4bit)
        print("{} 4-bit: SNR = {:.2f} dB".format(wav_file, SNR_4bit))