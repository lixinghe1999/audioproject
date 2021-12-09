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
T = 15
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1
if __name__ == "__main__":
    # location: Experiment/subject/position/classification
    path = 'exp5/hou/pose1/move/'
    files = os.listdir(path)
    N = int(len(files) / 4)
    files_imu1 = files[:N]
    files_imu2 = files[N:2 * N]
    files_mic1 = files[2 * N:3 * N]
    files_mic2 = files[3 * N:]
    for index in range(N):
        #i = np.random.randint(0, N)
        i = index
        time1 = float(files_mic1[i][:-4].split('_')[1])
        time2 = float(files_mic2[i][:-4].split('_')[1])
        wave1, wave2, Zxx1, Zxx2, phase1, phase2 = micplot.load_audio(path + files_mic1[i], path + files_mic2[i],
                                                                      seg_len_mic, overlap_mic, rate_mic, normalize=True, dtw=False)
        imu1, t_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
        imu2, t_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
        imu1 = processing.imu_resize(t_imu1 - time1, imu1)
        imu2 = processing.imu_resize(t_imu2 - time1, imu2)

        fig, axs = plt.subplots(3, tight_layout=True)
        axs[0].imshow(imu1, extent=[0, 15, 800, 100], aspect='auto')
        axs[0].set_xlabel('t/second')
        axs[0].set_ylabel('frequency/Hz')
        axs[1].imshow(imu2, extent=[0, 15, 800, 100], aspect='auto')
        axs[1].set_xlabel('t/second')
        axs[1].set_ylabel('frequency/Hz')
        axs[2].imshow(Zxx1[freq_bin_low:freq_bin_high, :], extent=[0, 15, 800, 100], aspect='auto')
        axs[2].set_xlabel('t/second')
        axs[2].set_ylabel('frequency/Hz')
        # axs[3].imshow(Zxx2[freq_bin_low:freq_bin_high, :], extent=[0, 15, 800, 100], aspect='auto')
        plt.show()

