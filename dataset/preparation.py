import os
import librosa
import numpy as np
import soundfile as sf

def mix(wav, index1):
    index2 = np.random.randint(0, len(wav))
    wav1, _ = librosa.load(wav[index1], sr=8000)
    wav1 = wav1 / np.max(wav1)
    wav2, _ = librosa.load(wav[index2], sr=8000)
    wav2 = wav2 / np.max(wav2)
    ratio = np.random.random() / 5 + 0.5
    if len(wav2) > len(wav1):
        mixture = wav1 + wav2[:len(wav1)] * ratio
    else:
        wav1[:len(wav2)] += wav2 * ratio
        mixture = wav1
    return wav1, wav2, mixture
def dataset(in_path, out_path):
    wav = []
    for f in os.listdir(in_path):
        wav.append(os.path.join(in_path, f))
    for i in range(len(wav)):
        wav1, wav2, mixture = mix(wav, i)
        sf.write(os.path.join(out_path, 's1', str(i) + '.wav'), wav1, 8000)
        sf.write(os.path.join(out_path, 's2', str(i) + '.wav'), wav2, 8000)
        sf.write(os.path.join(out_path, 'mix', str(i) + '.wav'), mixture, 8000)
    return wav

if __name__ == "__main__":

    wav = dataset('bss/raw', 'bss/tr')