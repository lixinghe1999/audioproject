import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from datetime import datetime

import imuplot
import micplot
from imuplot import read_data
from micplot import load_stereo
from shutil import copyfile
# basically use this script to
# 1) change sample rate
# 2) synchronize audio
if __name__ == "__main__":
    # dir = 'exp4'
    # id = ['he', 'hou', 'shi', 'wu']
    # cls = ['none', 'noise', 'clean', 'mix']
    #
    # dir = 'exp7'
    # id = ['galaxy', 'freebud', 'airpod', 'logitech']
    # sentence = ['s1', 's2', 's3', 's4']
    # for i in id:
    #     for s in sentence:
    #         path = os.path.join(dir, i, s)
    #         files = os.listdir(path)
    #         for f in files:
    #             if f[-4:] == '.m4a':
    #                 y, s = librosa.load(path + '/' + f, sr=16000)
    #                 f = f[:-4] + '.wav'
    #                 sf.write(path + '/' + f, y, 16000, subtype='PCM_16')

    #synchronize airpods and microphone
    sentences = ['s1', 's2', 's3', 's4']
    cls = ['noise_new']
    # sentences = ['train']
    # cls = ['']

    for s in sentences:
        for c in cls:
            path = os.path.join('exp7/hou_raw', s, c)
            files = os.listdir(path)

            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            files_mic2 = files[3 * N:]
            files_mic2.sort(key=lambda x: int(x[4:-4]))
            for i in range(N):
                copyfile(os.path.join(path, files_imu1[i]), os.path.join('exp7/hou', s, c, files_imu1[i]))
                copyfile(os.path.join(path, files_imu2[i]), os.path.join('exp7/hou', s, c, files_imu2[i]))

                mic = librosa.load(os.path.join(path, files_mic1[i]), sr=16000)[0]
                airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=16000)[0]
                shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
                airpods = np.roll(airpods, shift)[-80000:]
                airpods = np.pad(airpods, (0, 80000-len(airpods)))
                f1 = files_mic1[i][:-4] + '.wav'
                f2 = files_mic2[i][:-4] + '.wav'
                sf.write(os.path.join('exp7/hou', s, c, f1), mic, 16000)
                sf.write(os.path.join('exp7/hou', s, c, f2), airpods, 16000)



