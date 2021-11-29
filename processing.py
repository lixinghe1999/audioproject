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
import cv2 as cv

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
def imu_resize(time_diff, imu1):
    shift = round(time_diff * rate_mic / (seg_len_mic - overlap_mic))
    Zxx_resize = np.roll(resize(imu1, (freq_bin_high, time_bin)), shift, axis=1)
    Zxx_resize[:, :shift] = 0
    Zxx_resize = Zxx_resize[freq_bin_low:, :]
    return Zxx_resize
def estimate_response(Zxx, imu1, imu2):
    Zxx = Zxx[freq_bin_low: freq_bin_high, :]
    select1 = Zxx > filters.threshold_otsu(Zxx)
    select2 = imu1 > filters.threshold_otsu(imu1)
    select = select1 & select2
    select_noise = np.logical_not(select)
    Zxx_ratio1 = np.divide(imu1, Zxx, out=np.zeros_like(imu1), where=select)
    Zxx_ratio2 = np.divide(imu2, Zxx, out=np.zeros_like(imu1), where=select)
    Zxx_ratio = (Zxx_ratio1 + Zxx_ratio2)/2
    new_variance = np.zeros(freq_bin_high - freq_bin_low)
    new_response = np.zeros(freq_bin_high - freq_bin_low)
    for i in range(freq_bin_high-freq_bin_low):
        if np.sum(select[i, :]) > 0:
            new_response[i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            new_variance[i] = np.var(Zxx_ratio[i, :], where=select[i, :])
    noise_mean = np.mean(imu1, axis=(0, 1), where=select_noise)
    noise_variance = np.mean(np.abs(imu1 - noise_mean))
    # new_response += new_variance * 2 * (np.random.rand(freq_bin_high-freq_bin_low) - 0.5)
    # augmentedZxx = Zxx * np.tile(np.expand_dims(new_response, axis=1), (1, time_bin))
    # augmentedZxx += noise_variance * 2 * (np.random.rand(freq_bin_high - freq_bin_low, time_bin) - 0.5)
    #
    # fig, axs = plt.subplots(5)
    # axs[0].imshow(imu1, aspect='auto')
    # axs[1].imshow(select, aspect='auto')
    # axs[2].imshow(select_noise, aspect='auto')
    # axs[3].plot(new_response, 'b')
    # axs[3].plot(new_variance, 'r')
    # axs[4].imshow(augmentedZxx, aspect='auto')
    # plt.show()


    # fig, axs = plt.subplots(4)
    # axs[0].imshow(imu1, aspect='auto')
    # axs[1].imshow(Zxx, aspect='auto')
    # axs[2].imshow(response, aspect='auto')
    # axs[3].imshow(augmentedZxx, aspect='auto')
    # plt.show()

    return new_response, new_variance, noise_mean, noise_variance
def static_noise(path_none):
    files_none = os.listdir(path_none)
    N = int(len(files_none) / 4)
    for i in [0]:
        files_none1 = files_none[2 * N + i]
        files_none2 = files_none[3 * N + i]
        wave_1, Zxx1, phase1 = micplot.get_wav(path_none + files_none1, normalize=True)
        wave_2, Zxx2, phase2 = micplot.get_wav(path_none + files_none2, normalize=True)
    return (np.mean(Zxx1[freq_bin_low:freq_bin_high, :]) + np.mean(Zxx2[freq_bin_low:freq_bin_high, :]))/2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False, help='mode of processing, 0-time segment, 1-mic denoising')
    # mode 0 transfer function estimation
    # mode 1 IMU Mic fusion
    args = parser.parse_args()
    if args.mode == 0:
        path = 'exp4/wu/clean/'
        files = os.listdir(path)
        N = int(len(files)/4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2*N]
        files_mic1 = files[2*N:3*N]
        files_mic2 = files[3*N:]
        t_start = time.time()
        response = np.zeros(freq_bin_high-freq_bin_low)
        variance = np.zeros(freq_bin_high-freq_bin_low)
        num = np.zeros(freq_bin_high-freq_bin_low)
        noise_mean = 0
        noise_variance = 0
        r = []
        #fig, axs = plt.subplots(2)
        for index in range(N):
            #i = np.random.randint(0, N)
            i = index
            time1 = float(files_mic1[i][:-4].split('_')[1])
            time2 = float(files_mic2[i][:-4].split('_')[1])
            wave1, wave2, Zxx1, Zxx2, phase1, phase2 = micplot.load_audio(path + files_mic1[i], path + files_mic2[i],
                                                                          seg_len_mic, overlap_mic, rate_mic,
                                                                          normalize=True, dtw=False)
            imu1, t_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
            imu2, t_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
            imu1 = imu_resize(t_imu1 - time1, imu1)
            imu2 = imu_resize(t_imu2 - time1, imu2)
            new_response, new_variance, new_noise_mean, new_noise_variance = estimate_response(Zxx1, imu1, imu2)
            response = 0.1 * new_response + 0.9 * response
            variance = 0.1 * new_variance + 0.9 * variance
            noise_mean = 0.1 * new_noise_mean + 0.9 * noise_mean
            noise_variance = 0.1 * new_noise_variance + 0.9 * noise_variance
            if index % (N-1) == 0 and index > 0:
                print(noise_mean, noise_variance)
                response += np.random.normal(0, variance, (freq_bin_high-freq_bin_low))
                response = np.clip(response, 0, 2)
                Zxx1 = Zxx1[freq_bin_low: freq_bin_high, :]
                #select1 = Zxx1 > filters.threshold_otsu(Zxx1)
                #augmentedZxx = np.zeros((freq_bin_high-freq_bin_low, time_bin))
                augmentedZxx = (Zxx1 * np.tile(np.expand_dims(response, axis=1), (1, time_bin)))
                # augmentedZxx += np.tile(np.expand_dims(noise_mean, axis=1), (1, time_bin)) + np.tile(np.expand_dims(noise_variance, axis=1), (1, time_bin)) \
                #                 * 2 * (np.random.rand(freq_bin_high-freq_bin_low, time_bin)-0.5)
                augmentedZxx += np.random.normal(noise_mean, noise_variance, (freq_bin_high-freq_bin_low, time_bin))
                fig, axs = plt.subplots(4)
                axs[0].imshow(Zxx1[freq_bin_low: freq_bin_high, :], aspect='auto', vmax=0.002, vmin=0)
                axs[1].imshow(imu1, aspect='auto',  vmax=0.002, vmin=0)
                axs[2].imshow(augmentedZxx, aspect='auto',  vmax=0.002, vmin=0)
                axs[3].plot(response, 'b')
                axs[3].plot(variance, 'r')
                axs[3].plot(new_response, 'g')
                plt.show()
            #     response = np.zeros(freq_bin_high - freq_bin_low)
            #     variance = np.zeros(freq_bin_high - freq_bin_low)
        #         axs[0].plot(response)
        #         r.append(response)
        #         response = np.zeros(freq_bin_high - freq_bin_low)
        # corr = axs[1].imshow(np.corrcoef(r))
        # fig.colorbar(corr)
        # plt.show()

    else:
        path_noise = 'exp2/HE/noisy1/'
        path_clean = 'exp2/HE/mic2/'
        path_fusion = 'exp2/HE/imu1/'
        path_imu = 'exp2/HE/imu/'
        T = 5
        # path_noise = 'exp3/noisy1/'
        # path_clean = 'exp3/mic1/'
        # path_fusion = 'exp3/imu/'
        # path_imu = 'exp3/acc/'
        # T = 15
        files_noise = os.listdir(path_noise)
        files_clean = os.listdir(path_clean)
        files_imu = os.listdir(path_imu)
        freq_bin_high = int(rate_imu / rate_mic * (int(seg_len_mic / 2) + 1))
        freq_bin_low = int(200 / rate_mic * (int(seg_len_mic / 2) + 1))
        index = range(1)
        for i in index:
            file_noise = files_noise[i]
            file_clean = files_clean[i]
            file_imu = files_imu[i]
            time1 = float(file_clean[:-4].split('_')[1])
            wave_1, Zxx1, phase1 = micplot.get_wav(path_noise + file_noise, normalize=True)
            wave_2, Zxx2, phase2 = micplot.get_wav(path_clean + file_clean, normalize=True)

            imu = imuplot.read_data(path_imu + file_imu)
            b, a = signal.butter(4, 100, 'highpass', fs=rate_imu)
            imu[:, :3] = signal.filtfilt(b, a, imu[:, :3], axis=0)
            imu[:, :3] = np.clip(imu[:, :3], -0.015, 0.015)
            # dominant_axis = np.argmax(np.sum(imu[:, :3], axis=0))

            f, t, Zxx = signal.stft(imu[:, :3], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu,
                                    window="hamming", axis=0)
            Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
            time_bin = np.shape(Zxx1)[1]
            time_imu = imu[0, 3]
            time_diff = time_imu - time1
            shift = round(time_diff * rate_mic / (seg_len_mic - overlap_mic))
            Zxx_resize = np.roll(resize(Zxx, (freq_bin_high, time_bin)), shift, axis=1)
            Zxx_resize[:, :shift] = 0

            Zxx_final = np.sqrt(Zxx1[:freq_bin_high, :] * (np.where(Zxx_resize > np.mean(Zxx_resize, axis=(0, 1)), Zxx_resize, 0)), dtype=np.complex128)
            Zxx_final *= phase1[:freq_bin_high, :]
            Zxx_final = np.pad(Zxx_final, [(0, np.shape(Zxx1)[0]-freq_bin_high), (0, 0)], mode='constant')
            _, audio = signal.istft(Zxx_final, fs=rate_mic, window='hamming', nperseg=seg_len_mic, noverlap=overlap_mic)
            #audio = micplot.GLA(10, Zxx_final)

            micplot.save_wav(audio, path_fusion + file_clean)

            fig, axs = plt.subplots(4)
            axs[0].imshow(Zxx2[:freq_bin_high, :], extent=[0, T, rate_imu / 2, 0], aspect='auto')
            axs[1].imshow(Zxx1[:freq_bin_high, :], extent=[0, T, rate_imu / 2, 0], aspect='auto')
            axs[2].imshow(Zxx_resize, extent=[0, T, rate_imu / 2, 0], aspect='auto')
            axs[3].imshow(np.abs(Zxx_final[:freq_bin_high, :]), extent=[0, T, rate_imu / 2, 0], aspect='auto')
            plt.show()

