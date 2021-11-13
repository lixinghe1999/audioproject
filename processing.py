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
def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())
def response2filter(response, pad1, pad2):
    response = np.repeat(np.pad(response, (pad1, pad2), mode='constant', constant_values=(1, 0)), 2)
    print(response.shape)
    filter = np.fft.ifft(response)
    return filter
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=1, required=False, help='mode of processing, 0-time segment, 1-mic denoising')
    # mode 0 audio separation
    # mode 1 transfer function estimation
    # mode 2 IMU Mic fusion
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
        # path_noise = 'exp2/HE/noisy1/'
        # path_clean = 'exp2/HE/mic2/'
        # path_fusion = 'exp2/HE/imu1/'
        # path_imu = 'exp2/HE/imu/'
        # T = 5
        path_noise = 'exp3/noisy1/'
        path_clean = 'exp3/mic1/'
        path_fusion = 'exp3/imu/'
        path_imu = 'exp3/acc/'
        T = 15
        files_noise = os.listdir(path_noise)
        files_clean = os.listdir(path_clean)
        files_imu = os.listdir(path_imu)
        freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
        freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
        response = np.zeros(freq_bin_high-freq_bin_low)
        r = []
        fig, axs = plt.subplots(2)
        for index in range(100):
            i = np.random.randint(0, len(files_imu))
            #i = index
            file_noise = files_noise[i]
            file_clean = files_clean[i]
            file_imu = files_imu[i]
            time1 = float(file_clean[:-4].split('_')[1])
            wave_1, Zxx1, phase1 = micplot.get_wav(path_noise + file_noise, seg_len_mic, overlap_mic, rate_mic, normalize=True)
            wave_2, Zxx2, phase2 = micplot.get_wav(path_clean + file_clean, seg_len_mic, overlap_mic, rate_mic, normalize=True)

            imu = imuplot.read_data(path_imu + file_imu, seg_len_imu, overlap_imu, rate_imu)
            b, a = signal.butter(4, 100, 'highpass', fs= rate_imu)
            imu[:, :3] = signal.filtfilt(b, a, imu[:, :3], axis=0)
            imu[:, :3] = np.clip(imu[:, :3], -0.01, 0.01)
            # dominant_axis = np.argmax(np.sum(imu[:, :3], axis=0))
            f, t, Zxx = signal.stft(imu[:, :3], nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu, window="hamming", axis=0)
            Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
            time_bin = np.shape(Zxx2)[1]
            time_imu = imu[0, 3]
            time_diff = time_imu - time1
            shift = round(time_diff * rate_mic / (seg_len_mic - overlap_mic))
            Zxx_resize = np.roll(resize(Zxx, (freq_bin_high, time_bin)), shift, axis=1)
            Zxx_resize[:, :shift] = 0
            Zxx_resize = Zxx_resize[freq_bin_low:, :]
            #Zxx_ratio = Zxx_resize / Zxx2[freq_bin_low:freq_bin_high, :]

            Zxx_ratio = Zxx2[freq_bin_low:freq_bin_high, :] / (Zxx_resize + 0.0001)
            #Zxx_ratio = np.abs(np.diff(Zxx_ratio, axis=1))
            #diff_Zxx = np.abs(np.diff(Zxx2[freq_bin_low:freq_bin_high, :], axis=1))
            new_response = np.zeros(freq_bin_high - freq_bin_low)
            count = np.zeros(freq_bin_high - freq_bin_low)
            m = np.mean(Zxx_ratio, axis=(0, 1))
            n = np.var(Zxx_ratio, axis=(0, 1))
            for j in range(time_bin):
                freq_temp = Zxx_ratio[:, j]
                select_freq = freq_temp > 1 * m
                count += select_freq
                select_val = Zxx_ratio[select_freq, j]
                new_response[select_freq] += select_val
            # axs[0].plot(count)
            # axs[1].plot(new_response)

            for j in range(freq_bin_high - freq_bin_low):
                if count[j] != 0:
                    new_response[j] = new_response[j]/count[j]
            #new_response = np.mean(Zxx_ratio, axis=1, where=Zxx_ratio > 1 * np.mean(Zxx_ratio, axis=(0, 1)))
            # axs[2].plot(new_response)
            # plt.show()
            response = 0.5 * new_response + 0.5 * response


            if index % 5 == 0 and index > 0:
                axs[0].plot(response)
                # axs[1].plot(response2filter(response, freq_bin_low, np.shape(Zxx2)[0] - freq_bin_high))
                r.append(response)
                response = np.zeros(freq_bin_high - freq_bin_low)
        corr = axs[1].imshow(np.corrcoef(r))
        fig.colorbar(corr, boundaries=np.linspace(0, 1, 5))
        plt.show()

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

