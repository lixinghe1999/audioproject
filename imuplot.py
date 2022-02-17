import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import librosa

seg_len_mic = 640
overlap_mic = 320
rate_mic = 16000
seg_len_imu = 64
overlap_imu = 32
rate_imu = 1600
segment = 6
stride = 4
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1

def read_data(file,seg_len=256, overlap=224, rate=1600):
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

    Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)[-1]
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return data, np.abs(Zxx)
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
    data = np.zeros((num_sample, 4))
    data[shift:num_sample, :3] = f(xnew)[:-shift, :]
    data[shift:num_sample, -1] = xnew[:-shift]
    writer = open(target, 'w')
    for i in range(num_sample):
        writer.write(str(data[i, 0]) + ' ' + str(data[i, 1]) + ' ' + str(data[i, 2]) + ' ' + str(data[i, 3]) + '\n')
    writer.close()
    return None


if __name__ == "__main__":
    # check imu plot
    loc = 'exp7/he/train/'
    files = os.listdir(loc)
    N = int(len(files) / 4)
    files_imu1 = files[:N]
    files_imu2 = files[N:2 * N]
    files_mic1 = files[2 * N:3 * N]

    for i in range(N):
        _, imu1 = read_data(loc + files_imu1[i], seg_len=seg_len_imu, overlap=overlap_imu)
        _, imu2 = read_data(loc + files_imu2[i], seg_len=seg_len_imu, overlap=overlap_imu)
        wave = librosa.load(loc + files_mic1[i], sr=None)[0]
        b, a = signal.butter(4, 100, 'highpass', fs=16000)
        wave = signal.filtfilt(b, a, wave)
        Zxx = signal.stft(wave, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(imu1, aspect='auto')
        axs[1].imshow(imu2, aspect='auto')
        axs[2].imshow(np.abs(Zxx[:freq_bin_high, :]), aspect='auto')
        plt.show()

    # re-coordinate acc to global frame by acc & compass
    # acc_data = read_data('../test/acc.txt')
    # compass = read_data('../test/compass.txt')
    # #acc_data = re_coordinate(acc_data, compass)
    # #acc_data = interpolation(acc_data)
    #
    # for i in range(3):
    #     plt.plot(acc_data[:, i])
    # plt.show()