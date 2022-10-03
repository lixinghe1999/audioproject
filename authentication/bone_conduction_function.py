import math
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.fft
import scipy.signal as signal
from skimage import filters


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600

freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
def noise_extraction(time_bin):
    noise_list = os.listdir('dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]
def synchronization(audio, imu):
    in1 = np.sum(audio[:freq_bin_high, :], axis=0)
    in2 = np.sum(imu, axis=0)
    shift = np.argmax(signal.correlate(in1, in2)) - len(in2)
    return np.roll(imu, shift, axis=1)

def moving_filter(signal, window):
    pad = math.ceil(len(signal) / window) * window - len(signal)
    signal = np.pad(signal, (0, pad))
    signal_2D = np.reshape(signal, (-1, window))
    average = np.mean(signal_2D, axis=1).repeat(window)
    variance = np.std(signal_2D, axis=1).repeat(window)
    signal = np.clip(signal, a_min=average - 3 * variance, a_max=average + 3 * variance)
    return signal, average, variance
def matching_features(audio, imu):
    imu_raw = np.linalg.norm(imu, axis=1)
    audio_raw = np.abs(audio[::10])
    # corr = signal.correlate(imu_intensity, audio_intensity)
    # shift = np.argmax(corr) - len(imu_intensity)
    # print(shift)
    # imu = np.roll(imu, shift, axis=0)
    # imu_intensity = np.roll(imu_intensity, shift, axis=0)

    N = 200
    audio_intensity = np.convolve(audio_raw, np.ones(N) / N, mode='valid')
    imu_intensity = np.convolve(imu_raw, np.ones(N) / (N), mode='valid')
    peaks1, properties1 = signal.find_peaks(audio_intensity, height=0.02, width=N/2)
    peaks2, properties2 = signal.find_peaks(imu_intensity, height=0.01, width=N/2)

    fig, axs = plt.subplots(2)
    axs[0].plot(audio_intensity)
    axs[0].plot(peaks1, audio_intensity[peaks1], "x")
    axs[1].plot(imu_intensity)
    axs[1].plot(peaks2, imu_intensity[peaks2], "x")
    plt.show()

    freq_shift = []
    segment_corr = []
    for i, p in enumerate(peaks2):
        deviation = np.abs(peaks1 - p)
        if np.min(deviation) < 100:
            index = np.argmin(deviation)
            left = int(min(properties1['left_ips'][index], properties2['left_ips'][i]))
            right = int(max(properties1['right_ips'][index], properties2['right_ips'][i]))
            corr = signal.correlate(imu_intensity[left: right], audio_intensity[left: right])
            # f0_audio = np.mean(librosa.yin(audio[left * 10: right * 10], fmin=65, fmax=500, sr=rate_mic, frame_length=400))
            # for j in range(3):
            #     f0 = librosa.yin(imu[left: right, j], fmin=65, fmax=500, sr=rate_imu, frame_length=40)
            #     freq_shift.append((np.mean(f0) - f0_audio) / f0_audio)
            # if np.max(corr) > 0.1:
                # print(np.max(corr))
            max_index = np.argmax(corr)
            left_pad = max(100 - max_index, 0)
            right_pad = max(100 - len(corr) + max_index, 0)
            corr = np.pad(corr, (left_pad, right_pad))
            normalized_corr = corr[left_pad + max_index - 100: left_pad + max_index + 100]
            segment_corr.append(normalized_corr)
    if len(segment_corr) == 0:
        segment_corr = np.zeros((200, 1))
    else:
        segment_corr = np.stack(segment_corr, axis=1)
    return segment_corr

def update_phones(token, imu1, imu2, audio, start, stop, dataset):
    for t in token:
        t = t.split()
        t[0] = float(t[0])
        if t[0] + 0.045 > stop or t[0] < start:
            continue
        t[0] = t[0] - start
        t[1] = float(t[1])
        if t[-1] in dataset:
            dataset[t[-1]].append([audio[int(t[0] * rate_mic):int((t[0]) * rate_mic) + 720].tolist(),
                                   imu1[int(t[0] * rate_imu):int((t[0]) * rate_imu) + 72].tolist(),
                                   imu2[int(t[0] * rate_imu):int((t[0]) * rate_imu) + 72].tolist()])
        else:
            dataset[t[-1]] = [[audio[int(t[0] * rate_mic):int((t[0]) * rate_mic) + 720].tolist(),
                              imu1[int(t[0] * rate_imu):int((t[0]) * rate_imu) + 72].tolist(),
                              imu2[int(t[0] * rate_imu):int((t[0]) * rate_imu) + 72].tolist()]]
    return dataset
def estimate_response(audio, imu):
    # first high-pass and transform to time-frequency
    f, t, audio = signal.stft(audio, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)
    audio = np.abs(audio[:freq_bin_high, :])
    f, t, imu = signal.stft(imu, nperseg=seg_len_imu, noverlap=overlap_imu, fs=rate_imu, axis=0)
    imu = np.linalg.norm(np.abs(imu), axis=1)
    imu = np.roll(imu, 3, axis=1)
    imu[:4, :] = 0
    select = audio > 1 * filters.threshold_otsu(audio)
    Zxx_ratio = np.divide(imu, audio, out=np.zeros_like(imu), where=select)

    response = np.zeros(freq_bin_high)
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            response[i] = np.mean(Zxx_ratio[i, :], where=select[i, :])

    fig, axs = plt.subplots(3)
    axs[0].imshow(audio)
    axs[1].imshow(imu)
    axs[2].imshow(Zxx_ratio)
    plt.show()
    # response for spectrum
    # N = 4800
    # audio = np.abs(scipy.fft.fft(audio[::10]))[:N//2]
    # imu = np.abs(scipy.fft.fft(imu))[:N//2]
    # imu = np.linalg.norm(np.abs(imu), axis=1)
    # response = imu/audio[:, np.newaxis]
    # fig, axs = plt.subplots(3)
    # axs[0].plot(audio)
    # axs[1].plot(imu)
    # axs[2].plot(response)
    # plt.show()

    return response

def spectrogram_recover(audio, imu, response):
    freq, time_bin = np.shape(audio)
    full_response = np.tile(np.expand_dims(response[0, :], axis=1), (1, time_bin))
    for j in range(time_bin):
        full_response[:, j] += np.random.normal(0, response[1, :], (freq_bin_high))
    augmentedZxx = audio * full_response
    e = np.mean(np.abs(augmentedZxx - imu)) / np.max(audio)
    print(e)
    # augmentedZxx += noise_extraction()
    # fig, axs = plt.subplots(2, figsize=(4, 2))
    # plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.98)
    # axs[0].imshow(augmentedZxx, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
    # axs[0].set_xticks([])
    # axs[1].imshow(clip1, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
    # fig.text(0.44, 0.022, 'Time (Sec)', va='center')
    # fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
    # plt.savefig('synthetic_compare.pdf')
    # plt.show()
    return None


