import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
from dtw import *
import random
import librosa
# contain many audio related function

def add_noise(wave_noise, wave_clean, ratio=1):
    random_index = random.randint(0, len(wave_noise) - len(wave_clean))
    clip = wave_noise[random_index: random_index + len(wave_clean)]
    # keep the same volume
    noisy = wave_clean + ratio * clip
    return noisy

def load_audio(name1, name2, seg_len=2560, overlap=2240, rate=16000, normalize=False, dtw=False):
    wave1 = get_wav(name1, normalize=normalize)
    wave2 = get_wav(name2, normalize=normalize)
    if dtw:
        wave1 = wave1[DTW(wave1, wave2)]
    Zxx1, phase1 = frequencydomain(wave1, seg_len=seg_len, overlap=overlap, rate=rate)
    Zxx2, phase2 = frequencydomain(wave2, seg_len=seg_len, overlap=overlap, rate=rate)
    return wave1, wave2, Zxx1, Zxx2, phase1, phase2
def get_wav(name, rate=16000, normalize = False):
    wave_data, sr = librosa.load(name, sr=None)
    #wave_data, r = sf.read(name, dtype='int16')
    if normalize:
        b, a = signal.butter(4, 100, 'highpass', fs=rate)
        wave_data = signal.filtfilt(b, a, wave_data)
        #wave_data = wave_data/32767
        wave_data = np.pad(wave_data, (0, 240000-len(wave_data)), 'constant', constant_values=(0, 0))
    return wave_data
def frequencydomain(wave_data, seg_len=2560, overlap=2240, rate=16000):
    f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
    phase = np.exp(1j * np.angle(Zxx))
    Zxx = np.abs(Zxx)
    return Zxx, phase
def DTW(wave_1, wave_2):
    alignment = dtw(wave_1[::20], wave_2[::20])
    print(wave_1[::20].shape, wave_2[::20].shape)
    wq = warp(alignment, index_reference=False)
    new_index = 20 * np.repeat([wq], 20) + np.tile(np.linspace(0, 19, 20, dtype='int'), len(wq))
    return new_index
def save_wav(audio, name):
    sf.write(name, audio.astype('int16'), 16000, subtype='PCM_16')

if __name__ == "__main__":
    # path1 = '前.m4a'
    # path2 = '后.m4a'
    # wave1, wave2, Zxx1, Zxx2, phase1, phase2 = load_audio(path1, path2, normalize=False)
    # plt.plot(wave1, 'b')
    # plt.plot(wave2, 'r')
    # plt.show()
    path = 'mic_1639035480.4962566.wav'
    wave, sr = librosa.load(path, mono=False, sr=None)
    wave1 = wave[0, :]
    wave2 = wave[1, :]
    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    wave1 = signal.filtfilt(b, a, wave1)
    wave1 = np.pad(wave1, (0, 80000 - len(wave2)), 'constant', constant_values=(0, 0))
    wave2 = signal.filtfilt(b, a, wave2)
    wave2 = np.pad(wave2, (0, 80000 - len(wave2)), 'constant', constant_values=(0, 0))
    fig, axs = plt.subplots(2)
    wave1 = np.clip(wave1, -0.01, 0.01)
    wave2 = np.clip(wave2, -0.01, 0.01)
    axs[0].plot(wave1, 'r')
    axs[1].plot(wave2, 'b')
    plt.show()



