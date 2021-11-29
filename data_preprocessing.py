import os
import librosa
import soundfile as sf
# basically use this script to change sample rate

# dir = 'exp4'
# id = ['he', 'hou', 'shi', 'wu']
# cls = ['none', 'noise', 'clean', 'mix']

dir = 'exp5'
id = ['he', 'hou']
pose = ['pose1', 'pose2']
cls = ['noise', 'move', 'clean', 'mix', 'compare']
for i in id:
    for j in pose:
        for k in cls:
            path = os.path.join(dir, i, j, k)
            files = os.listdir(path)
            for f in files:
                if f[-4:] == '.wav':
                    y, s = librosa.load(path + '/' + f, sr=16000)
                    sf.write(path + '/' + f, y, 16000, subtype='PCM_16')