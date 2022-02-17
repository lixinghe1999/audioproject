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
    source = 'exp7/raw'
    target = 'exp7'
    sentences = ['s1', 's2', 's3', 's4']
    cls = ['noise']
    T = 5
    # sentences = ['noise_train']
    # cls = ['']
    # T = 30
    # for source, target in zip(['exp7/he_raw', 'exp7/hou_raw', 'exp7/liang_raw', 'exp7/shuai_raw', 'exp7/shi_raw', 'exp7/wu_raw', 'exp7/yan_raw'],
    #                           ['exp7/he', 'exp7/hou', 'exp7/liang', 'exp7/shuai', 'exp7/shi', 'exp7/wu', 'exp7/yan']):
    #for source, target in zip(['exp7/1_raw', 'exp7/2_raw', 'exp7/3_raw', 'exp7/4_raw', 'exp7/5_raw', 'exp7/6_raw', 'exp7/7_raw', 'exp7/8_raw'],
    # ['exp7/1', 'exp7/2', 'exp7/3', 'exp7/4', 'exp7/5', 'exp7/6', 'exp7/7', 'exp7/8'],):
    for folder in ['office', 'stair', 'corridor']:
        for s in sentences:
            for c in cls:
                path = os.path.join(source, folder, s, c)
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

                    mic = librosa.load(os.path.join(path, files_mic1[i]), sr=16000)[0]
                    airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=16000)[0]
                    shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
                    airpods = np.roll(airpods, shift)[-T * 16000:]
                    airpods = np.pad(airpods, (0, T*16000-len(airpods)))
                    f1 = files_mic1[i][:-4] + '.wav'
                    f2 = files_mic2[i][:-4] + '.wav'
                    sf.write(os.path.join(target, folder, s, c, f1), mic, 16000)
                    sf.write(os.path.join(target, folder, s, c, f2), airpods, 16000)

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




