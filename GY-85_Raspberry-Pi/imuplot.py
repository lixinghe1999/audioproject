import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from micplot import get_wav, peaks
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy


seg_len = 256
overlap = 128
rate = 3000
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
    t_new = np.linspace(0, max(t_norm), int(max(t_norm) * rate))
    return f_linear(t_new)

def phase_estimation(Zxx, phase):
    Zxx = Zxx * phase
    t, audio = signal.istft(Zxx, fs=rate)
    f, t, next_data = signal.stft(audio, fs=rate)
    phase = np.exp(1j*np.angle(next_data))
    return phase

def GLA(iter, Zxx, phase):
    for j in range(iter):
        phase = phase_estimation(Zxx, phase)
    Zxx = Zxx * phase
    t, audio = signal.istft(Zxx, nperseg=seg_len, noverlap=overlap, fs=rate)
    return audio, Zxx

def euler(accel, compass):
    accelX, accelY, accelZ = accel
    compassX, compassY, compassZ = compass
    pitch = 180 * math.atan2(accelX, math.sqrt(accelY*accelY + accelZ*accelZ))/math.pi
    roll = 180 * math.atan2(accelY, math.sqrt(accelX*accelX + accelZ*accelZ))/math.pi
    mag_x = compassX * math.cos(pitch) + compassY * math.sin(roll) * math.sin(pitch) + compassZ*math.cos(roll)*math.sin(pitch)
    mag_y = compassY * math.cos(roll) - compassZ * math.sin(roll)
    yaw = 180 * math.atan2(-mag_y, mag_x)/math.pi
    return [roll, pitch, yaw]
def smooth(data, width):
    for i in range(3):
        data[:, i] = np.convolve(data[:, i], np.ones(width)/width, mode='same')
    return data
def re_coordinate(acc, compass):
    compass_num = 0
    max_num = np.shape(compass)[0] - 1
    rotationmatrix = np.eye(3)
    smooth_acc = smooth(acc.copy(), 500)
    compass = smooth(compass, 2)
    for i in range(np.shape(acc)[0]):
        data1 = acc[i, :]
        data2 = compass[min(compass_num, max_num), :]
        if data1[-1] > data2[-1] and compass_num <= max_num:
            compass_num = compass_num + 1
            #print(euler(smooth_acc[i, :-1], data2[:-1]))
            #plt.scatter(compass_num, euler(smooth_acc[i, :-1], data2[:-1])[-1])
            rotationmatrix = R.from_euler('xyz', euler(smooth_acc[i, :-1], data2[:-1]), degrees=True).as_matrix()
        acc[i, :-1] = np.dot(rotationmatrix, data1[:-1].transpose())
    return acc
def noise_reduce(Zxx):

    shape = Zxx.shape
    low = int((shape[0] - 1) * 100 / 1500)
    high = int((shape[0] - 1) * 800 / 1500)
    Zxx[:low, :] = 0
    Zxx[high:, :] = 0
    Zxx = np.abs(Zxx)

    # left, right = p
    # left = ((left - 1) * 8 + 6) / 14.7
    # right = ((right - 1) * 8 + 6) / 14.7
    # noise = np.zeros((shape[0]))
    # j = 0
    # n = 1
    # for i in range(shape[1]):
    #     if n == 1:
    #         noise = noise * 0.8 + 0.2 * Zxx[:, i]
    #         #Zxx[:, i] = 0
    #     else:
    #         Zxx[:, i] = Zxx[:, i] - noise
    #     if i > left[j]:
    #         n = 0
    #     if i > right[j]:
    #         j = min(j + 1, len(left) - 1)
    #         n = 1
    #         noise = np.zeros((shape[0]))
    return Zxx
def normalize(Zxx):
    Zmax, Zmin = Zxx.max(), Zxx.min()
    Zxx = (Zxx - Zmin) / (Zmax - Zmin)
    #Zxx = np.clip(Zxx, 0.4, 1)
    Zxx = np.exp(Zxx)
    # Zxx_smooth = np.mean(Zxx, axis=1)
    # # Zxx_smooth = np.zeros(shape)
    # # for i in range(shape[0]):
    # #     Zxx_smooth[i, :] = np.convolve(Zxx[i, :], np.ones(5)/5, mode='same')
    # for i in range(shape[1]):
    #     Zxx[:, i] = np.abs(Zxx[:, i] - Zxx_smooth)
    entro = entropy(np.abs(Zxx), axis=0)
    return Zxx, entro

if __name__ == "__main__":
    # read data, get absolute value, estimate phase == reconstruction from time-frequency
    fig, axs = plt.subplots(3, 2)
    path = '../exp1/HE/70db/1/'
    #path = '../test/'
    data = read_data(path + 'acc.txt')
    compass = read_data(path + 'compass.txt')
    files = os.listdir(path)
    # for file in files:
    #     if file[3] == '1':
    #         mic1, Zxx1 = get_wav(path + file)
    #         time1 = float(file.split('.')[0].split('_')[1])
    #         p1 = peaks(Zxx1)[0], peaks(Zxx1)[1]
    #     if file[3] == '2':
    #         mic2, Zxx2 = get_wav(path + file)
    #         time2 = float(file.split('.')[0].split('_')[1])
    #         p2 = peaks(Zxx1)[0], peaks(Zxx1)[1]
    #data = re_coordinate(data, compass)
    #data = interpolation(data)
    for i in range(3):
        f, t, Zxx = signal.stft(data[:, i], nperseg=seg_len, noverlap=overlap, fs=rate)
        # phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
        # phase_acc = np.exp(1j * np.angle(Zxx))
        Zxx = noise_reduce(Zxx)
        Zxx, entro = normalize(Zxx)
        #audio, Zxx_new = GLA(0, Zxx, phase_acc)
        #t, audio = signal.istft(Zxx, nperseg=seg_len, noverlap=overlap, fs=rate)
        axs[i, 0].imshow(Zxx, extent=[0, 5, 1500, 0])
        axs[i, 0].set_aspect(1/300)
        # axs[i, 1].imshow(entro, extent=[0, 5, 1, 0])
        # axs[i, 1].set_aspect(5)
        axs[i, 1].plot(data[:, i])
    plt.show()

    # re-coordinate acc to global frame by acc & compass
    # acc_data = read_data('../test/acc.txt')
    # compass = read_data('../test/compass.txt')
    # #acc_data = re_coordinate(acc_data, compass)
    # #acc_data = interpolation(acc_data)
    #
    # for i in range(3):
    #     plt.plot(acc_data[:, i])
    # plt.show()