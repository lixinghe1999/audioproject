import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation as separator
from DL.evaluation import wer

# This is baseline for 1. Noise suppression 2. beamforming: webrtc, coherence based 3. facebook denoiser 4. Speech brain
# basically this script is only for checking
sentences = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
                    ["WE", "WANT", "TO", "IMPROVE", "SPEECH", "QUALITY", "IN", "THIS", "PROJECT"],
                    ["BUT", "WE", "DON'T", "HAVE", "ENOUGH", "DATA", "TO", "TRAIN", "OUR", "MODEL"],
                    ["TRANSFER", "FUNCTION", "CAN", "BE", "A", "GOOD", "HELPER", "TO", "GENERATE", "DATA"]]
if __name__ == "__main__":

    asr_model = EncoderDecoderASR.from_hparams(source="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                               run_opts={"device":"cuda"})
    for earphone in ['apple']:
        directory = 'dataset/earphones/' + earphone
        # for f in os.listdir(directory):
        #     file = os.path.join(directory, f)
        #     if f[-3:] == 'm4a':
        #         wav_filename = os.path.join(directory, f[:-3] + 'wav')
        #         track = AudioSegment.from_file(file, format='m4a')
        #         file_handle = track.export(wav_filename, format='wav')
        #     elif f[-3:] == 'mp3':
        #         wav_filename = os.path.join(directory, f[:-3] + 'wav')
        #         sound = AudioSegment.from_mp3(file)
        #         sound.export(wav_filename, format="wav")
        for cls in ['clean', 'noise']:
            WER = []
            cls_directory = directory + '/' + cls
            for f in os.listdir(cls_directory):
                text = asr_model.transcribe_file(cls_directory + '/' + f).split()
                worst = 1000
                for s in sentences:
                    a = wer(s, text)
                    if a < worst:
                        worst = a
                #print(f, worst)
                WER.append(worst)
            print(earphone, cls, np.mean(WER))