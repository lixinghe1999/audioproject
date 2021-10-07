import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
seg_len = 2048
overlap = 1024
rate = 44100
def get_wav(name):
    f = wave.open(name, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype = np.short)
    f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
    Zxx = np.abs(Zxx)
    return wave_data, Zxx
def peaks(Zxx):
    m, n = 0.03 * rate /(seg_len - overlap), 0.5 * rate /(seg_len - overlap)
    sum_Zxx = np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0)
    peaks, dict = find_peaks(sum_Zxx, distance=m, width=[m,n], prominence=1*np.mean(sum_Zxx))
    return dict['left_ips'], dict['right_ips']
if __name__ == "__main__":
    path = '../exp1/HE/70db/3/'
    files = os.listdir(path)
    fig, axs = plt.subplots(2, 2)
    for file in files:
        if file[3] == '1':
            wave_1, Zxx = get_wav(path + file)
            print(wave_1.max())
            axs[0, 0].plot(np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0))
            axs[0, 1].imshow(Zxx[:int(seg_len/40), :], extent=[0, 5, rate/40/2, 0])
            axs[0, 1].set_aspect(1/200)
            print(peaks(Zxx))
        if file[3] == '2':
            wave_2, Zxx = get_wav(path + file)
            axs[1, 0].plot(np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0))
            axs[1, 1].imshow(Zxx[:int(seg_len/40), :], extent=[0, 5, rate/40/2, 0])
            axs[1, 1].set_aspect(1/200)
            print(peaks(Zxx))
    plt.show()
