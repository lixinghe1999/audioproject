import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import soundfile as sf


def read_data(file, seg_len=256, overlap=224, rate=1600):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    time_imu = data[0, 3]
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
    data[:, :3] = np.clip(data[:, :3], -0.02, 0.02)
    #data = np.linalg.norm(data[:, :3], axis=1)
    #data = signal.filtfilt(b, a, data)
    # data = np.clip(data, -0.05, 0.05)
    #f, t, Zxx = signal.stft(data, nperseg=seg_len, noverlap=overlap, fs=rate, window="hamming")
    f, t, Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, window="hamming", axis=0)
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return data, np.abs(Zxx), time_imu

def interpolation(data, target_rate):
    start_time = data[0, 3]
    t_norm = data[:, 3] - start_time
    f_linear = interpolate.interp1d(t_norm, data[:, :3], axis=0)
    t_new = np.linspace(0, 5, 5 * target_rate)
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
    test_path = 'bmiacc2_1639035480.7018366.txt'
    #test_path = "exp2/HE/imu/bmiacc_1633585810.4177535.txt"
    #test_path = 'test/bmiacc_1635249068.8786418.txt'
    #test_path = 'test/bmigryo_1635249068.8956292.txt'
    data, Zxx, time_imu = read_data(test_path)
    plt.imshow(Zxx, aspect='auto')
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