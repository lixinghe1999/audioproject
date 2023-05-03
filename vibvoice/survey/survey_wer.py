from speechbrain.pretrained import EncoderDecoderASR
import os
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence
from evaluation import wer
sentences = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]]
def batch_ASR(audio_files, asr_model):
    sigs = []
    lens = []
    for audio_file in audio_files:
        snt, fs = torchaudio.load(audio_file)
        sigs.append(snt.squeeze())
        lens.append(snt.shape[1])
    batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
    lens = torch.Tensor(lens) / batch.shape[1]
    text = asr_model.transcribe_batch(batch, lens)[0]
    text = [t.split() for t in text]
    return text
if __name__ == "__main__":
    asr_model = EncoderDecoderASR.from_hparams(source="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="../pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device": "cuda"})
    audio_files = []
    for f in os.listdir('survey/model'):
        audio_files.append(os.path.join('survey/model', f))
    text = batch_ASR(audio_files, asr_model)
    WER = []
    for t in text:
        WER.append(wer(sentences[0], t))
    print(WER)
