import matplotlib.pyplot
import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import scipy.signal as signal
import numpy as np
import argparse
from skimage.transform import resize
from skimage import filters
import time
from dtw import *
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
T = 30
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(T*rate_mic/(seg_len_mic-overlap_mic)) + 1
def noise_extraction():
    noise_list = os.listdir('dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('dataset/noise/' + noise_list[index])
    return noise_clip
def imu_resize(time_diff, imu1):
    shift = round(time_diff * rate_mic / (seg_len_mic - overlap_mic))
    Zxx_resize = np.roll(resize(imu1, (freq_bin_high, time_bin)), shift, axis=1)
    Zxx_resize[:, :shift] = 0
    Zxx_resize = Zxx_resize[freq_bin_low:, :]
    return Zxx_resize
def estimate_response(Zxx, imu1, imu2):
    Zxx = Zxx[freq_bin_low: freq_bin_high, :]
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu1 > 1 * filters.threshold_otsu(imu1)
    select3 = imu2 > filters.threshold_otsu(imu2)
    select = select2 & select1
    select_noise = np.logical_not(select)
    Zxx_ratio1 = np.divide(imu1, Zxx, out=np.zeros_like(imu1), where=select)
    Zxx_ratio2 = np.divide(imu2, Zxx, out=np.zeros_like(imu1), where=select)
    Zxx_ratio = (Zxx_ratio1 + Zxx_ratio2)/2
    new_variance = np.zeros(freq_bin_high - freq_bin_low)
    new_response = np.zeros(freq_bin_high - freq_bin_low)
    for i in range(freq_bin_high-freq_bin_low):
        if np.sum(select[i, :]) > 0:
            new_response[i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            new_variance[i] = np.std(Zxx_ratio[i, :], where=select[i, :])
    noise_mean = np.mean(imu1, axis=(0, 1), where=select_noise)
    noise_variance = np.sqrt(np.mean(np.abs(imu1 - noise_mean)**2))
    # new_response += np.random.normal(0, new_variance, (freq_bin_high-freq_bin_low))
    # new_response = np.clip(new_response, 0, 2)
    # augmentedZxx = (Zxx * np.tile(np.expand_dims(new_response, axis=1), (1, time_bin)))
    # augmentedZxx += np.random.normal(noise_mean, noise_variance, (freq_bin_high - freq_bin_low, time_bin))
    # fig, axs = plt.subplots(5)
    # axs[0].imshow(imu1, aspect='auto')
    # axs[1].imshow(select2, aspect='auto')
    # axs[2].imshow(select3, aspect='auto')
    # axs[3].imshow(select, aspect='auto')
    # axs[4].plot(new_response, 'b')
    # axs[4].plot(new_variance, 'r')
    # plt.show()


    # fig, axs = plt.subplots(2)
    # axs[0].hist(noise_distribution, bins=100)
    # axs[1].hist(distribution, bins=100)
    # plt.show()
    return new_response, new_variance, noise_mean, noise_variance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False, help='mode of processing, 0-time segment, 1-mic denoising')
    # mode 0 transfer function estimation
    # mode 1 noise extraction

    args = parser.parse_args()
    if args.mode == 0:
        #for name in ["he", "hou", "wu", "shi"]:
            # path = 'exp4/' + name + '/clean/'
        #for name in ["liang", "shuai", "shen", "wu", "he", "hou", "zhao", "shi"]:
        for name in ["liang", "shuai", "shen", "wu"]:
            path = 'exp6/' + name + '/clean/'
            files = os.listdir(path)
            N = int(len(files)/4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2*N]
            files_mic1 = files[2*N:3*N]
            files_mic2 = files[3*N:]
            t_start = time.time()
            response = np.zeros(freq_bin_high-freq_bin_low)
            variance = np.zeros(freq_bin_high-freq_bin_low)
            noise_mean = 0
            noise_variance = 0
            for index in range(N):
                i = index
                time1 = float(files_mic1[i][:-4].split('_')[1])
                #time2 = float(files_mic2[i][:-4].split('_')[1])
                # wave1, wave2, Zxx1, Zxx2, phase1, phase2 = micplot.load_audio(path + files_mic1[i], path + files_mic2[i],
                #                                                               seg_len_mic, overlap_mic, rate_mic,
                #                                                               normalize=True, dtw=False)
                wave1, wave2, Zxx1, Zxx2, phase1, phase2 = micplot.load_stereo(path + files_mic1[i], T, seg_len_mic, overlap_mic, rate_mic,
                                                                               normalize=True, dtw=False)
                data1, imu1, t_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
                data2, imu2, t_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
                imu1 = imu_resize(t_imu1 - time1, imu1)
                imu2 = imu_resize(t_imu2 - time1, imu2)
                new_response, new_variance, new_noise_mean, new_noise_variance = estimate_response(Zxx1, imu1, imu2)
                response = 0.1 * new_response + 0.9 * response
                variance = 0.1 * new_variance + 0.9 * variance
                noise_mean = 0.1 * new_noise_mean + 0.9 * noise_mean
                noise_variance = 0.1 * new_noise_variance + 0.9 * noise_variance
                #
                fig, axs = plt.subplots(2, sharex=True)
                fig.tight_layout()
                axs[0].imshow(Zxx1[freq_bin_low: freq_bin_high, 100:200], aspect='auto')
                axs[1].imshow(imu1[:, 100:200], aspect='auto')
                #axs[2].plot(data1[:, 0])
                plt.show()

            # save transfer function
            #np.savez(name + '_transfer_function.npz', response=response, variance=variance, noise=(noise_mean, noise_variance))

            #show transfer function effectiveness
            # full_response = np.tile(np.expand_dims(response, axis=1), (1, time_bin))
            # for j in range(time_bin):
            #     full_response[:, j] += np.random.normal(0, variance, (freq_bin_high-freq_bin_low))
            # augmentedZxx = Zxx1[freq_bin_low: freq_bin_high, :] * full_response
            # #augmentedZxx += noise_extraction()
            # fig, axs = plt.subplots(4)
            # axs[0].imshow(Zxx1[freq_bin_low: freq_bin_high, :], aspect='auto')
            # axs[1].imshow(imu1, aspect='auto')
            # axs[2].imshow(augmentedZxx, aspect='auto')
            # #axs[3].imshow(full_response, aspect='auto')
            # axs[3].plot(response, 'b')
            # axs[3].plot(variance, 'r')
            # axs[3].plot(new_response, 'g')
            # plt.show()
    elif args.mode == 1:
        count = 0
        for path in ['exp4/he/none/', 'exp4/hou/none/', 'exp4/wu/none/', 'exp4/shi/none/', 'exp6/he/none/', 'exp6/hou/none/',
                     'exp6/liang/none/', 'exp6/shen/none/','exp6/shuai/none/', 'exp6/wu/none/']:
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            for index in range(N):
                i = index
                time1 = float(files_mic1[i][:-4].split('_')[1])
                imu1, Zxx_imu1, t_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
                imu2, Zxx_imu2, t_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
                Zxx_imu1 = imu_resize(t_imu1 - time1, Zxx_imu1)
                Zxx_imu2 = imu_resize(t_imu2 - time1, Zxx_imu2)
                np.save('dataset/noise/' + str(count) + '.npy', Zxx_imu1)
                count += 1
                np.save('dataset/noise/' + str(count) + '.npy', Zxx_imu2)
                count += 1
            # fig, axs = plt.subplots(3)
            # axs[0].plot(a)
            # axs[0].plot(peaks, a[peaks], "x")
            # axs[0].vlines(x=peaks, ymin=a[peaks] - properties["prominences"],
            #            ymax=a[peaks], color="C1")
            # axs[0].hlines(y=properties["width_heights"], xmin=properties["left_ips"],
            #            xmax=properties["right_ips"], color="C1")
            # axs[1].imshow(Zxx_imu1, aspect='auto', vmax=0.004, vmin=0)
            # axs[2].imshow(Zxx_imu1 * select, aspect='auto', vmax=0.004, vmin=0)
            # #axs[2].imshow(Zxx_imu1[:, select], aspect='auto', vmax=0.004, vmin=0)
            # plt.show()


