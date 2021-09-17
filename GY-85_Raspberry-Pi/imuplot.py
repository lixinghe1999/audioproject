import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
def read_data(file):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i]
        l = line.split(' ')
        for j in range(4):
            data[i, j] = float(l[j])
    return data

def interpolation(data):
    start_time = data[0, 3]
    t_norm = data[:, 3] - start_time
    f_linear = interpolate.interp1d(t_norm, data[:, :3], axis=0)
    t_new = np.linspace(0, max(t_norm), int(max(t_norm) * 2500))
    return f_linear(t_new)

def phase_estimation(Zxx, phase):
    Zxx = Zxx * phase
    t, audio = signal.istft(Zxx, fs=2500)
    f, t, next_data = signal.stft(audio, fs=2500)
    phase = np.exp(1j*np.angle(next_data))
    return phase

def GLA(iter, Zxx, phase):
    for j in range(iter):
        phase = phase_estimation(Zxx, phase)
    Zxx = Zxx * phase
    t, audio = signal.istft(Zxx, fs=2500)
    return audio, Zxx
if __name__ == "__main__":
    fig ,axs = plt.subplots(3,2)
    data = read_data('../data/acc.txt')
    data = interpolation(data)
    for i in range(3):
        f, t, Zxx = signal.stft(data[:, i], fs=2500)
        phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
        phase_acc = np.exp(1j * np.angle(Zxx))
        Zxx = np.abs(Zxx)
        Zxx[:, :10] = 0
        audio, Zxx_new = GLA(10, Zxx, phase_random)
        axs[i, 0].plot(data[:, i])
        axs[i, 1].plot(audio)
        #t, audio = signal.istft(Zxx, fs=2500)
        #axs[i].plot(audio)
        #axs[i].plot(t, np.convolve(audio, np.ones(100)/100, mode='same'))
    plt.show()