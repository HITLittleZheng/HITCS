import torch
import soundfile as sf
from jiwer import wer
from tqdm import tqdm
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def map_to_pred(batch):
    input_values = processor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred)
wer_result = []

with tqdm(total=len(result)) as pbar:
    for i in range(len(result)):
        current_wer = wer(result["text"][i], result["transcription"][i])
        wer_result.append(current_wer)
        # print("WER:", current_wer)
        # 在 tqdm 对象上调用 set_postfix 方法更新后缀信息
        pbar.set_postfix(WER=current_wer)
        pbar.update(1)

print("average WER:", sum(wer_result)/len(wer_result))