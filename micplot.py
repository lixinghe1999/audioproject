import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
import audioread
from pesq import pesq
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
def normalization(wave_data, rate=16000, T=15):
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    wave_data = signal.filtfilt(b, a, wave_data)
    if len(wave_data) >= T * rate:
        return wave_data
    wave_data = np.pad(wave_data, (0, T * rate - len(wave_data)), 'constant', constant_values=(0, 0))
    return wave_data
def frequencydomain(wave_data, seg_len=2560, overlap=2240, rate=16000):
    f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
    phase = np.exp(1j * np.angle(Zxx))
    Zxx = np.abs(Zxx)
    return Zxx, phase
def load_audio(name1, name2, T, seg_len=2560, overlap=2240, rate=16000, normalize=False, dtw=False):
    wave1, sr = librosa.load(name1, sr=rate)
    wave2, sr = librosa.load(name2, sr=rate)
    if normalize:
        wave1 = normalization(wave1, rate, T)
        wave2 = normalization(wave2, rate, T)
    if dtw:
        wave1 = wave1[DTW(wave1, wave2)]

    Zxx1, phase1 = frequencydomain(wave1, seg_len=seg_len, overlap=overlap, rate=rate)
    Zxx2, phase2 = frequencydomain(wave2, seg_len=seg_len, overlap=overlap, rate=rate)
    return wave1, wave2, Zxx1, Zxx2, phase1, phase2

def load_stereo(name, T, seg_len=2560, overlap=2240, rate=16000, normalize=False, dtw=False):
    wave, sr = librosa.load(name, mono=False, sr=rate)
    wave1 = wave[0]
    wave2 = wave[1]
    if normalize:
        wave1 = normalization(wave[0], rate, T)
        wave2 = normalization(wave[1], rate, T)
    if dtw:
        wave1 = wave1[DTW(wave1, wave2)]
    Zxx1, phase1 = frequencydomain(wave1, seg_len=seg_len, overlap=overlap, rate=rate)
    Zxx2, phase2 = frequencydomain(wave2, seg_len=seg_len, overlap=overlap, rate=rate)
    return wave1, wave2, Zxx1, Zxx2, phase1, phase2

def offset(in1, in2):
    # generally in1 will be longer than in2
    corr = signal.correlate(in1, in2, 'valid')
    shift = np.argmax(corr)
    return shift
def save_wav(audio, name):
    sf.write(name, audio.astype('int16'), 16000, subtype='PCM_16')

if __name__ == "__main__":
    # test directional microphone
    # path1 = '前.m4a'
    # path2 = '后.m4a'
    # wave1, wave2, Zxx1, Zxx2, phase1, phase2 = load_audio(path1, path2, normalize=False)
    # plt.plot(wave1, 'b')
    # plt.plot(wave2, 'r')
    # plt.show()

    # test the time synchronization
    path1 = '录音 (26).m4a'
    wave1, wave2, Zxx1, Zxx2, phase1, phase2 = load_stereo(path1, 30, normalize=True)
    path2 = 'mic_1639207419.358194.wav'
    wave3, wave4, Zxx3, Zxx4, phase3, phase4 = load_stereo(path2, 30, normalize=True)
    wave3 = np.clip(wave3, -0.02, 0.02)
    start = 10000
    length = 30000
    wave3 = wave3[start: start+length]
    shift = offset(wave1, wave3)
    print(shift)
    print(pesq(16000, wave1[shift:shift+length], wave3, 'nb'))
    fig, axs = plt.subplots(3)
    axs[0].plot(wave1, 'r')
    axs[1].plot(wave3, 'b')
    axs[2].plot(wave1[shift:shift+length], 'r')
    plt.show()



