import os
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import scipy.interpolate as interpolate

rate_imu = 1600
T = 3
def calibrate(file, T, shift):
    data = np.loadtxt(file)
    timestamp = data[:, -1]
    data = data[:, :3] / 2 ** 14
    data = data - np.mean(data, axis=0)
    f = interpolate.interp1d(timestamp - timestamp[0], data, axis=0, kind='nearest')
    t = min((timestamp[-1] - timestamp[0]), T)
    num_sample = int(T * rate_imu)
    data = np.zeros((num_sample, 3))
    xnew = np.linspace(0, t, num_sample)
    data[shift:num_sample, :3] = f(xnew)[:-shift, :]
    return data
def directory_decompose(files):
    import itertools
    dict = {}
    for k, v in itertools.groupby(files, key=lambda x: x.split('_')[0]):
        dict[k] = list(v)
    return dict
def synchronize():
    return
if __name__ == '__main__':
    '''
    imu1 left, imu2 right
    '''
    directory = '../dataset/touch'
    location = ["eyebrow_left", "eyebrow_right", "cheek_left", "cheek_right", "jaw_left", "jaw_right", "nose"]
    cls = ["knock", "slide", "rub", "touch",]
    dict = {}
    for l in location:
        for c in cls:
            path = os.path.join(directory, l, c)
            file_list = os.listdir(path)
            dict = directory_decompose(file_list)
            imu1 = dict['bmiacc1']
            imu2 = dict['bmiacc2']
            audio = dict['mic']
            for i in range(len(imu1)):
                time1 = float(imu1[i].split('_')[1][:-4])
                time2 = float(imu2[i].split('_')[1][:-4])
                time_mic = float(audio[i].split('_')[1][:-4])
                shift1 = int((time1 - time_mic) * 1600)
                shift2 = int((time2 - time_mic) * 1600)

                data1 = calibrate(os.path.join(path, imu1[i]), T, shift1)
                data2 = calibrate(os.path.join(path, imu2[i]), T, shift2)
                data3 = librosa.load(os.path.join(path, audio[i]), sr=None, mono=False)[0].T

                data3 = data3[50000:, :]
                data3 = data3 / np.max(data3, axis=0)
                data3 = data3 - np.mean(data3, axis=0)

                data1 = np.abs(signal.stft(data1, nperseg=256, noverlap=224, fs=1600, axis=0)[-1])
                #data1 = np.linalg.norm(np.abs(data1), axis=1)
                data2 = np.abs(signal.stft(data2, nperseg=256, noverlap=224, fs=1600, axis=0)[-1])
                #data2 = np.linalg.norm(np.abs(data2), axis=1)

                data3 = np.abs(signal.stft(data3, nperseg=1024, noverlap=512, fs=44100, axis=0)[-1])
                #data3 = np.linalg.norm(data3, axis=1)

                fig, axs = plt.subplots(2, 4)
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.02, wspace=0.02)
                for j in range(3):
                    axs[0, j].imshow(data1[:50, j, :], aspect='auto', origin='lower')
                for j in range(3):
                    axs[1, j].imshow(data2[:50, j, :], aspect='auto', origin='lower')
                axs[0, -1].imshow(data3[:50, 0, :], aspect='auto', origin='lower')
                axs[1, -1].imshow(data3[:50, 1, :], aspect='auto', origin='lower')
                # axs[0, -1].plot(data3[:, 0])
                # axs[1, -1].plot(data3[:, 1])
                #plt.show()
                plt.savefig('images/' + l + '_' + c + '_' + str(i) + '.png', dpi=300)
                plt.close()