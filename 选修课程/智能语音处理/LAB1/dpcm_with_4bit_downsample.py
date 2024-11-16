from dpcm_with_4bit import *
import numpy as np
import os
import wave

def downsample(data, factor):
    """ 平均下采样 """
    downsampled_data = []
    for i in range(0, len(data), factor):
        if i + 1 >= len(data):
            downsampled_data.append(np.short(data[i]))
            break
        downsampled_data.append(np.short(np.mean(data[i:i+factor])))
    return downsampled_data

def DPCM_downsample(file_name, factor):
    """ DPCM编码解码 """
    file_base_name = os.path.basename(file_name).split('.')[0] + '_downsample'
    wav_data = read_wav_file(file_name)
    # 下采样
    downsampled_data = downsample(wav_data, 2)
    
    begin, encoded_data = DPCM_encode_quantized(downsampled_data, 4, factor)
    compressed_data = compress(encoded_data)  
    save_encoded_data_to_file(begin, compressed_data, file_base_name)

    dpc_file = open('./dpc/' + file_base_name + '_4bit.dpc', "rb")

    begin = np.frombuffer(dpc_file.read(2), dtype = np.short)
    str_data = dpc_file.read()  
    dpc_data = np.frombuffer(str_data, dtype=np.int8)
    decompressed_data = decompress(dpc_data)
    decoded_data = DPCM_decode_quantized(begin, decompressed_data, 4, factor) 
    # 上采样
    upsampled_data = [decoded_data[i // 2] for i in range(len(decoded_data) * 2)]

    save_decoded_data_to_pcm(upsampled_data, file_base_name)
    legth = len(upsampled_data) if len(wav_data) > len(upsampled_data) else len(wav_data)
    print(file_name, calculate_SNR(wav_data[:legth], upsampled_data[:legth]))
    


if __name__ == '__main__':
    factor = 600
    directory = "语料"
    wav_files = get_wav_filepath(directory)
    for (i, wav_file) in enumerate(wav_files):
        DPCM_downsample(wav_file, factor)