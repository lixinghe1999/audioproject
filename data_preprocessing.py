import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
# basically use this script to
# 1) change sample rate
# 2) synchronize audio
if __name__ == "__main__":
    # dir = 'exp4'
    # id = ['he', 'hou', 'shi', 'wu']
    # cls = ['none', 'noise', 'clean', 'mix']

    # dir = 'exp5'
    # id = ['he', 'hou']
    # pose = ['pose1', 'pose2']
    # cls = ['noise', 'move', 'clean', 'mix', 'compare']
    # for i in id:
    #     for j in pose:
    #         for k in cls:
    #             path = os.path.join(dir, i, j, k)
    #             files = os.listdir(path)
    #             for f in files:
    #                 if f[-4:] == '.wav':
    #                     y, s = librosa.load(path + '/' + f, sr=16000)
    #                     sf.write(path + '/' + f, y, 16000, subtype='PCM_16')

    # difficult to synchronize airpods and microphone
    #airpods, sr1 = sf.read('airpods_example.wav', dtype='int16')
    mic1, sr1 = sf.read('test.flac', dtype='int16')
    #airpods, sr1 = librosa.load('airpods_example.wav', sr=16000)
    mic2, sr2 = librosa.load('test.flac', sr=None)
    print(mic1.shape, sr1)
    print(mic2.shape, sr2)
    fig, axs = plt.subplots(2)
    #corr = np.convolve(airpods, mic, mode='same')
    axs[0].plot(mic1)
    axs[1].plot(mic2*32767)
    plt.show()