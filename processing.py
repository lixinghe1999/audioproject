import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import scipy.signal as signal
import numpy as np
import argparse
from skimage.transform import resize
import wave

seg_len_mic = 2048
overlap_mic = 1024
rate_mic = 44100
seg_len_imu = 256
overlap_imu = 128
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
        path_noise = 'exp2/noisy/'
        path_clean = 'exp2/HE/'
        files_noise = os.listdir(path_noise)
        files_clean = os.listdir(path_clean)
        index = [1]
        for i in index:
            file_noise1 = files_noise[i]
            file_noise2 = files_noise[i + 27]
            file_clean1 = files_clean[i + 27]
            file_clean2 = files_clean[i + 54]
            time1 = float(file_clean1[:-4].split('_')[1])
            time2 = float(file_clean2[:-4].split('_')[1])
            wave_1, Zxx1, phase1 = micplot.get_wav(path_noise + file_noise1, normalize=True)
            wave_2, Zxx2, phase2 = micplot.get_wav(path_noise + file_noise2, normalize=True)
            wave_3, Zxx3, phase3 = micplot.get_wav(path_clean + file_clean1, normalize=False)

            imu = imuplot.read_data(path_clean + files_clean[i])
            b, a = signal.butter(8, 0.05, 'highpass')
            for j in range(3):
                imu[:, j] = signal.filtfilt(b, a, imu[:, j])
            dominant_axis = (np.argmax(np.sum(np.abs(imu[:, :-1]), axis=0)))
            f, t, Zxx = signal.stft(imu[:, dominant_axis], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu)
            Zxx = np.abs(Zxx)
            Zxx_share = np.where(np.abs(Zxx1 - Zxx2) < 0.003, Zxx1, 0)

            freq_bin = int(rate_imu/rate_mic * (overlap_mic + 1))
            time_bin = np.shape(Zxx1)[1]
            time_imu = imu[0, 3]
            time_diff = time_imu - time1
            shift = int(time_diff * rate_mic / overlap_mic)
            Zxx_resize = np.roll(resize(Zxx, (freq_bin, time_bin)), shift, axis=1)
            Zxx_resize[:, :shift] = 0
            Zxx_both = Zxx_share[:freq_bin, :] * phase1[:freq_bin, :]
            Zxx_both = Zxx_both * (Zxx_resize > np.mean(Zxx_resize, axis=(0, 1)))

            fig, axs = plt.subplots(4)
            axs[0].imshow(Zxx3[:freq_bin, :], extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[1].imshow(Zxx1[:freq_bin, :], extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[2].imshow(Zxx_share[:freq_bin, :], extent=[0, 5, rate_imu/2, 0], aspect='auto')
            axs[3].imshow(Zxx_resize, extent=[0, 5, rate_imu / 2, 0], aspect='auto')
            plt.show()

            Zxx_final = np.zeros(np.shape(Zxx1), dtype='complex64')
            Zxx_final[:freq_bin, :] = Zxx_both
            wf = wave.open('test.wav', 'wb')
            _, audio = signal.istft(Zxx_final, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes((audio*32767).astype('int16'))
            wf.close()
