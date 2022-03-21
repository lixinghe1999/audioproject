import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from imuplot import calibrate
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

    #synchronize airpods and microphone
    source = 'exp7/raw'
    target = 'exp7'
    sentences = ['s1', 's2', 's3', 's4']
    cls = ['noise', 'clean', 'mobile']
    T = 5
    # sentences = ['noise_train', 'train']
    # cls = ['']
    # T = 30
    for folder in ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]:
        print(folder)
        for s in sentences:
            for c in cls:
                path = os.path.join(source, folder, s, c)
                if not os.path.isdir(path):
                    continue
                else:
                    files = os.listdir(path)
                    N = int(len(files) / 4)
                    files_imu1 = files[:N]
                    files_imu2 = files[N:2 * N]
                    files_mic1 = files[2 * N:3 * N]
                    files_mic2 = files[3 * N:]
                    files_mic2.sort(key=lambda x: int(x[4:-4]))
                    for i in range(N):
                        time1 = float(files_imu1[i].split('_')[1][:-4])
                        time2 = float(files_imu2[i].split('_')[1][:-4])
                        time_mic = float(files_mic1[i].split('_')[1][:-4])
                        shift1 = int((time1 - time_mic) * 1600)
                        shift2 = int((time2 - time_mic) * 1600)
                        calibrate(os.path.join(path, files_imu1[i]), os.path.join(target, folder, s, c, files_imu1[i]), T, shift1)
                        calibrate(os.path.join(path, files_imu2[i]), os.path.join(target, folder, s, c, files_imu2[i]), T, shift2)

                        # mic = librosa.load(os.path.join(path, files_mic1[i]), sr=16000)[0]
                        # airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=16000)[0]
                        # shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
                        # airpods = np.roll(airpods, shift)[-T * 16000:]
                        # airpods = np.pad(airpods, (0, T*16000-len(airpods)))
                        # f1 = files_mic1[i][:-4] + '.wav'
                        # f2 = files_mic2[i][:-4] + '.wav'
                        # sf.write(os.path.join(target, folder, s, c, f1), mic, 16000)
                        # sf.write(os.path.join(target, folder, s, c, f2), airpods, 16000)

    # for source, target in zip(['exp7/freebud_raw', 'exp7/galaxy_raw'], ['exp7/freebud', 'exp7/galaxy']):
    #     for s in sentences:
    #         for c in cls:
    #             path = os.path.join(source, s, c)
    #             files = os.listdir(path)
    #             N = int(len(files) / 4)
    #             files_mic2 = files[:N]
    #             files_imu1 = files[N:2 * N]
    #             files_imu2 = files[2 * N:3 * N]
    #             files_mic1 = files[3 * N:]
    #             for i in range(N):
    #                 time1 = float(files_imu1[i].split('_')[1][:-4])
    #                 time2 = float(files_imu2[i].split('_')[1][:-4])
    #                 time_mic = float(files_mic1[i].split('_')[1][:-4])
    #                 shift1 = int((time1 - time_mic) * 1600)
    #                 shift2 = int((time2 - time_mic) * 1600)
    #                 calibrate(os.path.join(path, files_imu1[i]), os.path.join(target, s, c, files_imu1[i]), T, shift1)
    #                 calibrate(os.path.join(path, files_imu2[i]), os.path.join(target, s, c, files_imu2[i]), T, shift2)
    #
    #                 mic = librosa.load(os.path.join(path, files_mic1[i]), sr=16000)[0]
    #                 airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=16000)[0]
    #                 shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
    #                 airpods = np.roll(airpods, shift)[-T * 16000:]
    #                 airpods = np.pad(airpods, (0, T * 16000 - len(airpods)))
    #                 f1 = files_mic1[i][:-4] + '.wav'
    #                 f2 = '新' + files_mic2[i][:-4] + '.wav'
    #                 sf.write(os.path.join(target, s, c, f1), mic, 16000)
    #                 sf.write(os.path.join(target, s, c, f2), airpods, 16000)




