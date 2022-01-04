import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
# basically use this script to
# 1) change sample rate
# 2) synchronize audio
if __name__ == "__main__":
    # dir = 'exp4'
    # id = ['he', 'hou', 'shi', 'wu']
    # cls = ['none', 'noise', 'clean', 'mix']

    # dir = 'exp7'
    # id = ['s2']
    # for i in id:
    #     path = os.path.join(dir, i)
    #     files = os.listdir(path)
    #     for f in files:
    #         if f[-4:] == '.m4a':
    #             y, s = librosa.load(path + '/' + f, sr=16000)
    #             f = f[:-4] + '.wav'
    #             sf.write(path + '/' + f, y, 16000, subtype='PCM_16')

    # synchronize airpods and microphone
    sentences = ['s1', 's2']
    plt_good = []
    plt_bad = []
    PESQ = []
    for s in sentences:
        path = os.path.join('exp7\he', s)
        files = os.listdir(path)

        N = int(len(files) / 4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2 * N]
        files_mic1 = files[2 * N:3 * N]
        files_mic2 = files[3 * N:]
        files_mic2.sort(key=lambda x: int(x[4:-4]))
        for i in range(N):
            airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=None)[0]
            mic = librosa.load(os.path.join(path, files_mic1[i]), sr=None)[0]
            shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
            airpods = np.roll(airpods, shift)
            airpods = airpods[-80000:]
            sf.write(os.path.join('exp7/he_calibrated', s, files_mic2[i]), airpods, 16000)
            sf.write(os.path.join('exp7/he_calibrated', s, files_mic1[i]), mic, 16000)
            # fig, axs = plt.subplots(2, 1)
            # axs[0].plot(airpods)
            # axs[1].plot(mic)
            # plt.show()