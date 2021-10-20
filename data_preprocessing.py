import os
import librosa
import soundfile as sf
# basically use this script to change sample rate

dir = 'exp2/noisy'
file = os.listdir(dir)
for f in file:
    if f[-4:] == '.wav':
        y, s = librosa.load(dir + '/' + f, sr=16000)
        sf.write(dir + '/' + f, y, s, subtype='PCM_16')