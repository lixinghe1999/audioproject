import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import librosa

rate_imu = 1600

def read_data(file, seg_len=256, overlap=224, rate=1600, mfcc=False, filter=True):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    if filter:
        b, a = signal.butter(4, 100, 'highpass', fs=rate)
        data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
        data[:, :3] = np.clip(data[:, :3], -0.05, 0.05)
    if mfcc:
        Zxx = []
        for i in range(3):
            Zxx.append(librosa.feature.melspectrogram(data[:, i], sr=rate, n_fft=seg_len, hop_length=seg_len-overlap, power=1))
        Zxx = np.array(Zxx)
        Zxx = np.linalg.norm(Zxx, axis=0)
    else:
        Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)[-1]
        Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return data, Zxx
def calibrate(file, T, shift):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]

    f = interpolate.interp1d(data[:, -1] - data[0, -1], data[:, :3], axis=0, kind='nearest')
    t = min((data[-1, -1] - data[0, -1]), T)
    num_sample = int(T * rate_imu)
    data = np.empty((num_sample, 3))
    xnew = np.linspace(0, t, num_sample)
    data[shift:num_sample, :3] = f(xnew)[:-shift, :]

    # data[:, :-1] /= 2 ** 14
    # b, a = signal.butter(4, 100, 'highpass', fs=1600)
    # data = signal.filtfilt(b, a, data, axis=0)
    # data = np.clip(data, -0.05, 0.05)
    # data /= np.max(data, axis=0)

    return data

if __name__ == "__main__":
    file = 'bmiacc1_1651662404.1797326.txt'
    data, Zxx = read_data(file)
    plt.imshow(Zxx)
    plt.show()


