
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
from pesq import pesq
from dtw import *
import random
import librosa
# contain many audio related function

def add_noise(wave_noise, wave_clean, ratio=1):
    random_index = random.randint(0, len(wave_noise) - len(wave_clean))
    clip = wave_noise[random_index: random_index + len(wave_clean)]
    # keep the same volume
    noisy = wave_clean + ratio * (wave_clean.max()/clip.max()) * clip
    return noisy
def normalization(wave_data, rate=16000, T=5):
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    wave_data = signal.filtfilt(b, a, wave_data)
    if len(wave_data) >= T * rate:
        return wave_data[:T * rate]
    wave_data = np.pad(wave_data, (0, T * rate - len(wave_data)), 'constant', constant_values=(0, 0))
    return wave_data
def frequencydomain(wave_data, seg_len=2560, overlap=2240, rate=16000, mfcc=False):
    if mfcc:
        Zxx = librosa.feature.melspectrogram(wave_data, sr=rate, n_fft=seg_len, hop_length=seg_len-overlap, power=1)
        return Zxx, None
    else:
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

def load_stereo(name, T, seg_len=2560, overlap=2240, rate=16000, normalize=False, mfcc=False):
    wave, sr = librosa.load(name, sr=rate)
    if normalize:
        wave = normalization(wave, rate, T)
    Zxx, phase = frequencydomain(wave, seg_len=seg_len, overlap=overlap, rate=rate, mfcc=mfcc)
    return wave, Zxx, phase

def save_wav(audio, name):
    sf.write(name, audio.astype('int16'), 16000, subtype='PCM_16')
def MFCC(Zxx, rate):
    mels = librosa.feature.melspectrogram(S=Zxx, sr=rate, n_mels=64)
    log_mels = librosa.core.amplitude_to_db(mels, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mels, sr=rate, n_mfcc=20)
    return mfcc


