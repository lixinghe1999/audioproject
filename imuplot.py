import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import librosa

rate_imu = 1600

def read_data(file,seg_len=256, overlap=224, rate=1600, mfcc=False):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
    data[:, :3] = np.clip(data[:, :3], -0.05, 0.05)
    if mfcc:
        Zxx = []
        for i in range(3):
            Zxx.append(librosa.feature.melspectrogram(data[:, i], sr=rate, n_fft=seg_len, hop_length=seg_len-overlap))
        Zxx = np.array(Zxx)
        Zxx = np.linalg.norm(Zxx, axis=0)
    else:
        Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)[-1]
        Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return data, Zxx
def calibrate(file, target, T, shift):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    f = interpolate.interp1d(data[:, -1] - data[0, -1], data[:, :3], axis=0, kind='nearest')
    t = min((data[-1, -1]-data[0, -1]), T)
    num_sample = int(t * rate_imu)
    xnew = np.linspace(0, t, num_sample)
    data = np.zeros((T * 1600, 4))
    data[shift:num_sample, :3] = f(xnew)[:-shift, :]
    data[shift:num_sample, -1] = xnew[:-shift]
    writer = open(target, 'w')
    for i in range(T * 1600):
        writer.write(str(data[i, 0]) + ' ' + str(data[i, 1]) + ' ' + str(data[i, 2]) + ' ' + str(data[i, 3]) + '\n')
    writer.close()
    return None
