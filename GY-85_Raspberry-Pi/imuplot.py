import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

def euler(accel, compass):
    accelX, accelY, accelZ = accel
    compassX, compassY, compassZ = compass
    pitch = 180 * math.atan2(accelX, math.sqrt(accelY*accelY + accelZ*accelZ))/math.pi
    roll = 180 * math.atan2(accelY, math.sqrt(accelX*accelX + accelZ*accelZ))/math.pi
    mag_x = compassX * math.cos(pitch) + compassY * math.sin(roll)*math.sin(pitch) + compassZ*math.cos(roll)*math.sin(pitch)
    mag_y = compassY * math.cos(roll) - compassZ * math.sin(roll)
    # print(pitch, roll)
    plt.scatter(mag_x, mag_y)

    yaw = 180 * math.atan2(-mag_y, mag_x)/math.pi
    #print(yaw)
    return [roll, pitch, yaw]
def smooth(acc_data):
    for i in range(3):
        acc_data[:, i] = np.convolve(acc_data[:, i], np.ones(100)/100, mode='same')
    return acc_data
def re_coordinate(acc_data, compass):
    compass_num = 0
    max_num = np.shape(compass)[0] - 1
    rotationmatrix = np.eye(3)
    smooth_acc = smooth(acc_data)
    for i in range(np.shape(acc_data)[0]):
        data1 = acc_data[i, :]
        data2 = compass[min(compass_num, max_num), :]
        if data1[-1] > data2[-1] and compass_num <= max_num:
            compass_num = compass_num + 1
            #print(euler(smooth_acc[i, :-1], data2[:-1]))
            #plt.scatter(compass_num, euler(smooth_acc[i, :-1], data2[:-1])[-1])
            rotationmatrix = R.from_euler('xyz', euler(smooth_acc[i, :-1], data2[:-1]), degrees=True).as_matrix()
        acc_data[i, :-1] = np.dot(rotationmatrix, data1[:-1].transpose())
    return acc_data
if __name__ == "__main__":
    # read data, get absolute value, estimate phase == reconstruction from time-frequency
    # fig ,axs = plt.subplots(3,2)
    # data = read_data('../data/acc.txt')
    # data = interpolation(data)
    # for i in range(3):
    #     f, t, Zxx = signal.stft(data[:, i], fs=2500)
    #     phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
    #     phase_acc = np.exp(1j * np.angle(Zxx))
    #     Zxx = np.abs(Zxx)
    #     Zxx[:, :10] = 0
    #     audio, Zxx_new = GLA(10, Zxx, phase_random)
    #     axs[i, 0].plot(data[:, i])
    #     axs[i, 1].plot(audio)
    #     #t, audio = signal.istft(Zxx, fs=2500)
    #     #axs[i].plot(audio)
    #     #axs[i].plot(t, np.convolve(audio, np.ones(100)/100, mode='same'))
    # plt.show()

    # re-coordinate acc to global frame by acc & compass
    acc_data = read_data('../acc.txt')
    compass = read_data('../compass.txt')
    acc_data = re_coordinate(acc_data, compass)
    acc_data = interpolation(acc_data)

    #fig, axs = plt.subplots(3, 1)
    # for i in range(3):
    #     plt.plot(acc_data[:, i])
    plt.show()