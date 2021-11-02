import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
from librosa.feature import melspectrogram
import librosa.display
import librosa
import soundfile as sf
seg_len = 2560
overlap = 2240
rate = 16000
def get_wav(name, normalize = False, mfcc=False):
    wave_data, r = librosa.load(name, sr=rate, mono=True)
    if normalize:
        b, a = signal.butter(4, 100, 'highpass', fs = rate)
        wave_data = signal.filtfilt(b, a, wave_data)
        wave_data = wave_data/wave_data.max()
        if mfcc:
            Zxx = melspectrogram(wave_data.astype(np.float32), sr=44100, n_fft=seg_len, hop_length=overlap, n_mels=128)
            phase = None
        else:
            f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
            phase = np.exp(1j * np.angle(Zxx))
            Zxx = np.abs(Zxx)
    else:
        if mfcc:
            Zxx = melspectrogram(wave_data.astype(np.float32), sr=44100, n_fft=seg_len, hop_length=overlap, n_mels=128)
            phase = None
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
    t = 32767 / audio.max()
    audio = audio * t
    sf.write(name, audio.astype('int16'), rate, subtype='PCM_16')
def GLA(iter, Zxx, i = 'random'):
    audio = librosa.griffinlim(Zxx, iter, seg_len-overlap, seg_len, "hamming", init= i )
    return audio

if __name__ == "__main__":
    path_noise = 'exp2/noisy/'
    path_clean = 'exp2/HE/'
    path_conversation = 'reference/conversation.wav'
    files_noise = os.listdir(path_noise)
    files_clean = os.listdir(path_clean)
    index = [1]
    freq_bin = int(1600 / rate * (int(seg_len/2) + 1))
    for i in index:
        file_noise1 = files_noise[i]
        file_noise2 = files_noise[i + 27]
        file_clean = files_clean[i+27]
        # wave_1, Zxx1, phase = get_wav(path_noise + file_noise1, normalize=False)
        # wave_2, Zxx2, phase = get_wav(path_noise + file_noise2, normalize=True)
        wave_3, Zxx3, phase = get_wav(path_clean + file_clean, normalize=False)
        #wave, Zxx, phase = get_wav(path_conversation, normalize=False)
        fig, ax = plt.subplots()
        #Zxx_dB = librosa.power_to_db(Zxx, ref=np.max)
        plt.imshow(Zxx3[:freq_bin, :])
        #img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=44100, ax=ax)
        # f, t, Zxx = signal.stft(imu[:, dominant_axis], nperseg=256, noverlap=128, fs=1600)
        # Zxx = melspectrogram(imu[:, dominant_axis], sr=44100, n_fft=256, hop_length=128, n_mels=128)
        # Zxx = np.abs(Zxx)
        plt.show()


