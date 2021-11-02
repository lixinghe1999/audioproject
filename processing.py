import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import scipy.signal as signal
import numpy as np
import argparse
from skimage.transform import resize
from PIL import Image

seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=1, required=False, help='mode of processing, 0-time segment, 1-mic denoising')
    args = parser.parse_args()
    if args.mode == 0:
        fig, axs = plt.subplots(2, 2)
        # path = 'test/bmi160/'
        count = 0
        for path in ['exp2/HOU/']:
            index = [18]
            #index = range(27)
            files = os.listdir(path)
            for i in index:
                data = imuplot.read_data(path + files[i])
                time_start = data[0, 3]
                # compass = read_data(path + 'compass_still.txt')
                mic1_index, mic2_index = i + 27, i + 54
                mic1, Zxx1, phase1 = micplot.get_wav(path + files[mic1_index], normalize=True)
                time1 = float(files[mic1_index][:-4].split('_')[1])
                p1, l1, r1 = micplot.peaks_mic(Zxx1)
                mic2, Zxx2, phase2 = micplot.get_wav(path + files[mic2_index], normalize=True)
                time2 = float(files[mic2_index][:-4].split('_')[1])
                p2, l2, r2 = micplot.peaks_mic(Zxx2)
                p1 = (p1 + 1) * overlap_mic / rate_mic
                p2 = (p2 + 1) * overlap_mic / rate_mic

                axs[1, 1].imshow(Zxx1[:int(rate_imu/rate_mic * (overlap_mic + 1)),:], extent=[0, 5, rate_imu/2, 0], aspect='auto')
                b, a = signal.butter(8, 50, 'highpass', fs=rate_imu)
                for j in range(3):
                    data[:, j] = signal.filtfilt(b, a, data[:, j])
                dominant_axis = (np.argmax(np.sum(np.abs(data[:, :-1]), axis=0)))
                for j in range(3):
                    f, t, Zxx = signal.stft(data[:, j], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu)
                    # phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
                    # phase_acc = np.exp(1j * np.angle(Zxx))
                    # acc, Zxx = GLA(0, Zxx, phase_acc)
                    Zxx = np.abs(Zxx)
                    axs[j//2, j%2].imshow(Zxx, extent=[0, 5, rate_imu/2, 0], aspect='auto')
                    # if j == dominant_axis:
                    #     p3 = imuplot.peaks_imu(Zxx)
                    #     p3 = (p3 + 1) * overlap_imu / rate_imu
                    #     p1 = p1 + time1 - time_start
                    #     p2 = p2 + time2 - time_start
                    #     match_tone = []
                    #     m1, m2 = imuplot.time_match(p1, p3), imuplot.time_match(p2, p3)
                    #     for k in range(len(p3)):
                    #         match_tone.append(m1[0][k] and m2[0][k])
                    #     l1 = l1[m1[1][match_tone]]
                    #     r1 = r1[m1[1][match_tone]]
                    #     count = imuplot.saveaswav(mic1, l1, r1, count)
                    #     l2 = l2[m2[1][match_tone]]
                    #     r2 = r2[m2[1][match_tone]]
                    #     count = imuplot.saveaswav(mic2, l2, r2, count)
                    #     p3 = p3[match_tone]
                    #     print(p1, p2, p3)
                plt.show()
    elif args.mode == 1:
        # path_noise = 'exp2/HOU/noisy2/'
        # path_clean = 'exp2/HOU/mic2/'
        # path_fusion = 'exp2/HOU/imu2/'
        # path_imu = 'exp2/HOU/imu/'

        path_noise = 'exp3/noisy1/'
        path_clean = 'exp3/mic1/'
        path_fusion = 'exp3/imu/'
        path_imu = 'exp3/acc/'

        files_noise = os.listdir(path_noise)
        files_clean = os.listdir(path_clean)
        files_imu = os.listdir(path_imu)
        #index = range(50)
        index = [0]
        for i in index:
            file_noise = files_noise[i]
            file_clean = files_clean[i]
            file_imu = files_imu[i]
            time1 = float(file_clean[:-4].split('_')[1])
            wave_1, Zxx1, phase1 = micplot.get_wav(path_noise + file_noise, normalize=True)
            wave_2, Zxx2, phase2 = micplot.get_wav(path_clean + file_clean, normalize=True)

            imu = imuplot.read_data(path_imu + file_imu)
            b, a = signal.butter(4, 100, 'highpass', fs= rate_imu)
            imu[:, :3] = signal.filtfilt(b, a, imu[:, :3], axis=0)
            imu[:, :3] = np.clip(imu[:, :3], -0.01, 0.01)
            #dominant_axis = np.argmax(np.sum(imu[:, :3], axis=0))

            f, t, Zxx = signal.stft(imu[:, :3], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu, window="hamming", axis=0)
            Zxx = np.abs(Zxx)
            Zxx = Zxx[:, 0, :] * Zxx[:, 1, :] * Zxx[:, 2, :]
            Zxx = np.cbrt(Zxx)
            # axs[0].imshow(Zxx)
            # freq_bin = int(rate_imu/rate_mic * (int(seg_len_mic/2) + 1))
            # axs[1].imshow(Zxx1[:freq_bin, :])
            # plt.show()
            # dominant_axis = (np.argmax(np.sum(np.abs(imu[:, :-1]), axis=0)))
            # chunk = rate_imu
            # for i in range(15):
            #     print(np.corrcoef(imu[i*chunk:(i+1)*chunk, :3].T))


            #Zxx = np.where(Zxx > 0.08, 0, Zxx)
            #Zxx = Zxx - np.mean(Zxx, axis=0)
            #plt.imshow(Zxx)
            #
            # imu = imuplot.interpolation(imu, rate_imu)
            # shift = round(time_diff * rate_imu)+45
            # imu = np.roll(imu, shift, axis=0)
            # imu = imu[:, dominant_axis]
            # imu = imu/imu.max()
            # for k in range(len(imu)):
            #     wave_1[k*10:(k+1)*10] *= imu[k]
            # wave_1 = np.sqrt(np.abs(wave_1))
            # fig, axs = plt.subplots(3, 1)
            # span = [2200, 2300]
            # axs[0].plot(wave_2[span[0]*10: span[1]*10])
            # axs[1].plot(imu[span[0]: span[1]])
            # axs[2].plot(wave_1[span[0]*10: span[1]*10])
            # plt.show()






            #
            # f, t, Zxx = signal.stft(imu[:, dominant_axis], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu, window="hamming")
            # Zxx = np.abs(Zxx)
            print(Zxx.shape, Zxx1.shape, imu.shape, wave_1.shape)
            freq_bin = int(rate_imu/rate_mic * (int(seg_len_mic/2) + 1))
            time_bin = np.shape(Zxx1)[1]
            time_imu = imu[0, 3]
            time_diff = time_imu - time1
            shift = round(time_diff * rate_mic / overlap_mic)
            Zxx_resize = np.roll(resize(Zxx, (freq_bin, time_bin)), shift, axis=1)
            Zxx_resize[:, :shift] = 0

            Zxx_phase = Zxx1
            Zxx_final = np.zeros(np.shape(Zxx1), dtype='complex64')
            #Zxx_final[:freq_bin, :] = Zxx_phase * (Zxx_resize > np.mean(Zxx_resize, axis=(0, 1)))
            #Zxx_final[:freq_bin, :] = np.sqrt(Zxx_phase * (np.where(Zxx_resize > np.mean(Zxx_resize, axis=(0, 1)), Zxx_resize, 0)))
            Zxx_final[:freq_bin, :] = Zxx_resize
            #Zxx_final[:freq_bin, :] *= phase1[:freq_bin, :]

            #_, audio = signal.istft(Zxx_final, fs=rate_mic, window='hamming', nperseg=seg_len_mic, noverlap=overlap_mic)
            audio = micplot.GLA(10, Zxx_final)

            micplot.save_wav(audio[:(len(wave_1))], path_fusion + file_clean)

            fig, axs = plt.subplots(4)
            axs[0].imshow(Zxx2[:freq_bin, :], extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[1].imshow(Zxx1[:freq_bin, :], extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[2].imshow(Zxx_resize, extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[3].imshow(np.abs(Zxx_final[:freq_bin, :]), extent=[0, 5, rate_imu / 2, 0], aspect='auto')
            plt.show()

