import os
import librosa
import soundfile as sf
# basically use this script to change sample rate

dir = 'exp4'
id = ['he', 'hou', 'shi', 'wu']
cls = ['none', 'noise', 'clean', 'mix']
for i in id:
    for j in cls:
        path = os.path.join(dir, i, j)
        files = os.listdir(path)
        for f in files:
            if f[-4:] == '.wav':
                y, s = librosa.load(path + '/' + f, sr=16000)
                sf.write(path + '/' + f, y, 16000, subtype='PCM_16')