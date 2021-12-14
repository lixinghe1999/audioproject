import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import processing
import numpy as np
import time
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
T = 5
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1
if __name__ == "__main__":
    # this scripts is for testing of new data
    path_IMU1 = 'bmiacc1_1639192317.839837.txt'
    path_IMU2 = 'bmiacc2_1639192317.8456254.txt'
    path_stereo = 'mic_1639192317.6397896.wav'
    time1 = float(path_stereo[:-4].split('_')[1])
    wave1, wave2, Zxx1, Zxx2, phase1, phase2 = micplot.load_stereo(path_stereo, T, seg_len_mic, overlap_mic, rate_mic, normalize=True, dtw=False)
    data1, imu1, t_imu1 = imuplot.read_data(path_IMU1, seg_len_imu, overlap_imu, rate_imu)
    data2, imu2, t_imu2 = imuplot.read_data(path_IMU2, seg_len_imu, overlap_imu, rate_imu)
    imu1 = processing.imu_resize(t_imu1 - time1, imu1)
    imu2 = processing.imu_resize(t_imu2 - time1, imu2)
    fig, axs = plt.subplots(3, tight_layout=True)
    axs[0].imshow(imu1, aspect='auto')
    axs[0].set_xlabel('t/second')
    axs[0].set_ylabel('frequency/Hz')
    axs[1].imshow(imu2, aspect='auto')
    axs[1].set_xlabel('t/second')
    axs[1].set_ylabel('frequency/Hz')
    axs[2].imshow(Zxx1[freq_bin_low:freq_bin_high, :], aspect='auto')
    axs[2].set_xlabel('t/second')
    axs[2].set_ylabel('frequency/Hz')
    # axs[3].imshow(Zxx2[freq_bin_low:freq_bin_high, :], extent=[0, 15, 800, 100], aspect='auto')
    plt.show()
