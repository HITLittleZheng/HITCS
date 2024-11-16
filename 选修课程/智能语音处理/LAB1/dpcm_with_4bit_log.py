from dpcm_with_4bit import *
import os
import wave
import math

def log_quantizer(difference, num_bits):
    """ 对数量化器 """
    quantize_levels = 2 ** (num_bits - 1)
    # c(n) = log(|d(n)|)
    # c(n) = min(c(n), quantize_levels - 1) + sgn(-d(n)) * quantize_levels
    if difference == 0:
        difference = 1
    log_difference = math.log(abs(difference)) 
    quantized_difference = min(max(round(log_difference), 0), quantize_levels - 1)
    # (-quantize_levels, quantize_levels - 1)
    quantized_difference = quantized_difference + quantize_levels if difference > 0 else quantized_difference

    return quantized_difference

def DPCM_encode_log_quantized(signal, num_bits):
    """DPCM 编码器"""
    quantize_levels = 2 ** (num_bits - 1) 
    encoded_signal = [signal[0]]
    prediction = signal[0]
    for sample in signal[1:]:
        difference = sample - prediction
        quantized_difference = log_quantizer(difference, num_bits)
        encoded_signal.append(quantized_difference)
        # x(n) = x(n-1) + ((-1)^sgn(c(n)- quantize_levels)) * (exp(c(n) & 7))
        sgn = 1 if quantized_difference >= quantize_levels else -1
        prediction += sgn * round(math.exp(quantized_difference & 7))

    return signal[0], encoded_signal

def DPCM_decode_log_quantized(begin, encoded_signal, num_bits):
    """DPCM 解码器"""
    # print(encoded_signal)
    # print(sgn)
    quantize_levels = 2 ** (num_bits - 1)
    data = np.zeros(len(encoded_signal) + 1, dtype=np.short)
    data[0] = begin.item()
    for i in range(1, len(encoded_signal)):
        sgn = 1 if encoded_signal[i] >= quantize_levels else -1
        data[i] = data[i - 1] + sgn * round(math.exp(encoded_signal[i] & (quantize_levels - 1)))
        # print(data[i])
    return data

def DPCM_log(file_name):
    """ DPCM编码解码 """
    file_base_name = os.path.basename(file_name).split('.')[0] + '_log'
    wav_data = read_wav_file(file_name) 
    begin, encoded_data = DPCM_encode_log_quantized(wav_data, 4)  
    compressed_data = compress(encoded_data)  
    save_encoded_data_to_file(begin, compressed_data, file_base_name)

    dpc_file = open('./dpc/' + file_base_name + '_4bit.dpc', "rb")

    begin = np.frombuffer(dpc_file.read(2), dtype = np.short)
    str_data = dpc_file.read()  
    dpc_data = np.frombuffer(str_data, dtype=np.int8)
    decompressed_data = decompress(dpc_data)
    decoded_data = DPCM_decode_log_quantized(begin, decompressed_data, 4) 
    # print(decoded_data[:10])
    save_decoded_data_to_pcm(decoded_data, file_base_name)
    legth = len(decoded_data) if len(wav_data) > len(decoded_data) else len(wav_data)
    print("{} {:.3f}".format(file_name, calculate_SNR(wav_data[:legth], decoded_data[:legth])))

if __name__ == '__main__':
    directory = "语料"
    wav_files = get_wav_filepath(directory)
    for (i, wav_file) in enumerate(wav_files):
        DPCM_log(wav_file)