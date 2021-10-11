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
def peaks_mic(Zxx):
    m, n = 0.03 * rate /(seg_len - overlap), 0.5 * rate /(seg_len - overlap)
    sum_Zxx = np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0)
    peaks, properties = find_peaks(sum_Zxx, distance=m, width=[m,n], prominence=0.5*np.mean(sum_Zxx))
    return peaks, properties['left_ips'], properties['right_ips']
if __name__ == "__main__":
    path = '../exp2/HE/'
    files = os.listdir(path)
    index = 3
    fig, axs = plt.subplots(2, 2)
    file1 = files[index + 27]
    wave_1, Zxx = get_wav(path + file1)
    axs[0, 0].plot(np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0))
    axs[0, 1].imshow(Zxx[:int(seg_len/40), :], extent=[0, 5, rate/40/2, 0])
    axs[0, 1].set_aspect(1/200)
    print(peaks_mic(Zxx))
    file2 = files[index + 54]
    wave_2, Zxx = get_wav(path + file2)
    axs[1, 0].plot(np.sum(Zxx[int(seg_len/220):int(seg_len/40), :], axis=0))
    axs[1, 1].imshow(Zxx[:int(seg_len/40), :], extent=[0, 5, rate/40/2, 0])
    axs[1, 1].set_aspect(1/200)
    print(peaks_mic(Zxx))
    plt.show()
