import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import librosa
import numpy as np
import scipy.signal as signal
from skimage.transform import resize
import time
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
T = 15
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1
path = 'exp6/he/compare/'
files = os.listdir(path)
N = int(len(files) / 4)
files_imu1 = files[:N]
files_imu2 = files[N:2 * N]
files_mic1 = files[2 * N:3 * N]
files_mic2 = files[3 * N:]
files_mic2.sort(key=lambda x: int(x[4:-4]))
def data_extract(i):
    wave, sr = librosa.load(path + files_mic1[i], sr=rate_mic)
    wave = wave[:T*rate_mic]
    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    wave = signal.filtfilt(b, a, wave)
    Zxx = np.abs(signal.stft(wave, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1])
    data1, imu1, t_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
    data2, imu2, t_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
    return Zxx, imu1, imu2, wave
if __name__ == "__main__":
    # location: Experiment/subject/position/classification
    Zxx1, _, _, wave1 = data_extract(1)
    Zxx3, _, _, wave3 = data_extract(3)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        Zxx, imu1, imu2, wave = data_extract(i)
        imu1 = resize(imu1, (freq_bin_high, time_bin))
        imu1 = np.vstack((imu1, np.zeros(np.shape(imu1))))
        fig, axs = plt.subplots(3, sharex=True, figsize=(5, 3))
        fig.subplots_adjust(top=0.97, right=0.97, bottom=0.15, wspace=0, hspace=0.1)
        axs[0].imshow(imu1, extent=[0, T, 1600, 100], aspect='auto')
        axs[1].imshow(Zxx3[:2*freq_bin_high, :], extent=[0, T, 1600, 0], aspect='auto')
        axs[2].imshow(Zxx[:2*freq_bin_high, :], extent=[0, T, 1600, 0], aspect='auto')
        axs[0].set_ylabel('IMU')
        axs[0].yaxis.set_label_coords(-0.07, 0.5)
        axs[1].set_ylabel('Clean')
        axs[1].yaxis.set_label_coords(-0.07, 0.5)
        axs[2].set_ylabel('Noise')
        axs[2].yaxis.set_label_coords(-0.07, 0.5)
        axs[2].set_xlabel('Time (Second)')
        fig.text(0.007, 0.5, 'Frequency (Hz)', va='center', rotation='vertical')
        plt.show()
        # axs[2].plot(wave1)
        # axs[3].plot(wave3)
        # axs[4].plot(wave)
        #np.savez('data.npz', imu=imu1, noise=Zxx[:2*freq_bin_high, :], clean=Zxx1[:2*freq_bin_high, :])



