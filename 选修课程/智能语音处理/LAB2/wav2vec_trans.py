import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)

for i in range(10):
    # load audio
    audio_input, sample_rate = sf.read(librispeech_samples_ds[i]["file"])

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    # INFERENCE
    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe
    transcription = processor.decode(predicted_ids[0])
    print(f"No.{i} Pred:\t", transcription)
    print(f"No.{i} Target:\t", librispeech_samples_ds[i]['text'])