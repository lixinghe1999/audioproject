import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import soundfile as sf

seg_len = 256
overlap = 128
rate = 1600
def read_data(file):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i]
        l = line.split(' ')
        for j in range(4):
            data[i, j] = float(l[j])
    data[:, :-1] /= 2**14
    return data

def interpolation(data):
    start_time = data[0, 3]
    t_norm = data[:, 3] - start_time
    f_linear = interpolate.interp1d(t_norm, data[:, :3], axis=0)
    t_new = np.linspace(0, max(t_norm), int(max(t_norm) * rate))
    return f_linear(t_new)


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
def peaks_imu(Zxx):
    m, n = 0.05 * rate/(seg_len - overlap), 1 * rate/(seg_len - overlap)
    sum_Zxx = np.sum(Zxx, axis=0)
    smooth_Zxx = np.convolve(sum_Zxx , np.ones(10)/10, mode='same')
    peaks, dict = signal.find_peaks(smooth_Zxx, distance=m, width=[m, n], prominence=3*np.var(smooth_Zxx))
    return peaks
def time_match(p1, p2):
    P1 = np.tile(p1.reshape((len(p1), 1)), (1, len(p2)))
    P2 = np.tile(p2, (len(p1), 1))
    diff = np.abs(P1 - P2)
    m = np.min(diff, axis = 0) < 0.2
    n = np.argmin(diff, axis = 0)
    return m, n
def saveaswav(data, l, r, count):
    l = (l + 1) * 1024
    r = (r + 1) * 1024
    for i in range(len(l)):
        data_seg = data[int(l[i])-5000: int(r[i])+5000]
        sf.write('exp2/audio_seg/' + str(count)+'.wav', data_seg, 16000, subtype='PCM_16')
        count = count + 1
    return count

if __name__ == "__main__":
    # check imu plot
    test_path = 'exp2/HE/bmiacc_1633585703.4800572.txt'
    data = read_data(test_path)
    length = data.shape[0]
    fig, axs = plt.subplots(3, 1)
    b, a = signal.butter(8, 0.02, 'highpass')
    for j in range(3):
        data[:, j] = signal.filtfilt(b, a, data[:, j])
    for j in range(3):
        f, t, Zxx = signal.stft(data[:, j], nperseg=seg_len, noverlap=overlap, fs=rate)
        # phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
        # phase_acc = np.exp(1j * np.angle(Zxx))
        # acc, Zxx = GLA(0, Zxx, phase_acc)
        Zxx = np.abs(Zxx)
        #Zxx[:, 200:] = 0
        axs[j].imshow(Zxx, extent=[0, 5, rate/2, 0], aspect='auto')
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