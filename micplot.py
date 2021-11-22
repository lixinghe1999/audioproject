import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
from librosa.feature import melspectrogram
import librosa.display
import librosa
import soundfile as sf

def get_wav(name, seg_len=2560, overlap=2240, rate=16000,  normalize = False, fft=False):
    wave_data, r = sf.read(name, dtype='int16')
    if normalize:
        b, a = signal.butter(4, 100, 'highpass', fs=rate)
        wave_data = signal.filtfilt(b, a, wave_data)
        wave_data = wave_data/32767
    if fft:
        Zxx = np.fft.fft(wave_data)
    else:
        f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
    phase = np.exp(1j * np.angle(Zxx))
    Zxx = np.abs(Zxx)
    return wave_data, Zxx, phase
def peaks_mic(Zxx):
    m, n = 0.05 * rate /(seg_len - overlap), 1 * rate /(seg_len - overlap)
    sum_Zxx = np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0)
    peaks, properties = find_peaks(sum_Zxx, distance=m, width=[m, n], prominence=3*np.var(sum_Zxx))
    return peaks, properties['left_ips'], properties['right_ips']
def save_wav(audio, name):
    sf.write(name, audio.astype('int16'), 16000, subtype='PCM_16')
def GLA(iter, Zxx, i = 'random'):
    audio = librosa.griffinlim(Zxx, iter, seg_len-overlap, seg_len, "hamming", init= i )
    return audio

if __name__ == "__main__":
    path_noise = 'exp4/wu/none/'
    path_none = 'exp4/he/clean/'
    files_noise = os.listdir(path_noise)
    files_none = os.listdir(path_none)
    index = [0]
    freq_bin = int(1600 / 16000 * (int(2560/2) + 1))
    N1 = int(len(files_noise)/4)
    N2 = int(len(files_none) / 4)
    for i in index:
        file_noise1 = files_noise[2*N1 + i]
        file_noise2 = files_noise[3*N1 + i]
        wave_1, Zxx1, phase1 = get_wav(path_noise + file_noise1, normalize=True)
        Zxx1[23:26, :] = 0
        wave_2, Zxx2, phase2 = get_wav(path_noise + file_noise2, normalize=True)
        Zxx2[23:26, :] = 0
        file_none1 = files_none[2*N2 + i]
        file_none2 = files_none[3*N2 + i]
        wave_3, Zxx3, phase3 = get_wav(path_none + file_none1, normalize=True)
        Zxx3[23:26, :] = 0
        wave_4, Zxx4, phase4 = get_wav(path_none + file_none2, normalize=True)
        Zxx4[23:26, :] = 0
        fig, axs = plt.subplots(4)
        # axs[0].plot(wave_1)
        # axs[1].plot(wave_2)
        # axs[0].imshow(Zxx1[:freq_bin, :])
        # axs[1].imshow(Zxx2[:freq_bin, :])
        axs[0].plot(np.mean(Zxx1[:freq_bin, :], axis=1))
        axs[1].plot(np.mean(Zxx2[:freq_bin, :], axis=1))
        axs[2].plot(np.mean(Zxx3[:, :], axis=1))
        axs[3].plot(np.mean(Zxx4[:, :], axis=1))
        plt.show()


