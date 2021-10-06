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
import wave

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
def time_correlate(data, p, time_start_audio, time_start_imu ):
    left, right = p
    left = (left) * 1024 / 44100 + time_start_audio
    right = (right + 2) * 1024 / 44100 + time_start_audio
    left = ((left - time_start_imu) * 3000).astype(int)
    right = ((right - time_start_imu) * 3000).astype(int)
    print((left + right)/2)
    for i in range(1, len(left)):
        corr = np.convolve(data, data[left[i]: right[i]], mode='valid')
        print((left[i]+right[i])/2,(-corr).argsort()[:5])
def noise_reduce(Zxx):
    shape = Zxx.shape
    low = int((shape[0] - 1) * 50 / 1500)
    high = int((shape[0] - 1) * 800 / 1500)
    Zxx[:low, :] = 0
    Zxx[high:, :] = 0
    Zxx[:, :3] = 0
    Zxx[:, -2:] = 0
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
    #         Zxx[:, i] = 0
    #     else:
    #         Zxx[:, i] = Zxx[:, i] - noise
    #         #pass
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
    #Zxx = 1000**(Zxx)
    # Zxx_smooth = n p.mean(Zxx, axis=1)
    # # Zxx_smooth = np.zeros(shape)
    # # for i in range(shape[0]):
    # #     Zxx_smooth[i, :] = np.convolve(Zxx[i, :], np.ones(5)/5, mode='same')
    # for i in range(shape[1]):
    #     Zxx[:, i] = np.abs(Zxx[:, i] - Zxx_smooth)
    entro = entropy(np.abs(Zxx), axis=0)
    return Zxx

if __name__ == "__main__":
    # read data, get absolute value, estimate phase == reconstruction from time-frequency
    fig, axs = plt.subplots(1, 2)
    #path = '../exp1/HE/70db/3/'
    path = '../test/'
    data = read_data(path + 'bmi_acc' + '.txt')
    #compass = read_data(path + 'compass_still.txt')
    files = os.listdir(path)
    for file in files:
        if file[3] == '1':
            mic1, Zxx1 = get_wav(path + file)
            time1 = float(file.split('.')[0].split('_')[1])
            p1 = peaks(Zxx1)[0], peaks(Zxx1)[1]
        if file[3] == '2':
            mic2, Zxx2 = get_wav(path + file)
            time2 = float(file.split('.')[0].split('_')[1])
            p2 = peaks(Zxx1)[0], peaks(Zxx1)[1]
    #data = re_coordinate(data, compass)
    #data = interpolation(data)
    #time_correlate(data, p1, time1)

    data_norm = np.linalg.norm(data[:, :-1], axis=1)
    f, t, Zxx = signal.stft( data_norm, nperseg=seg_len, noverlap=overlap, fs=rate)
    phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
    phase_acc = np.exp(1j * np.angle(Zxx))
    #Zxx = noise_reduce(Zxx)
    Zxx = normalize(Zxx)
    acc, Zxx = GLA(0, Zxx, phase_acc)
    #time_correlate(audio, p1, time1, data[0, 3])

    axs[0].imshow(np.abs(Zxx), extent=[0, 5, rate/2, 0])
    axs[0].set_aspect(10/rate)
    # axs[0].plot(np.abs(np.fft.fft(data_norm)[5:]))
    axs[1].plot(acc)
    plt.show()

    # save IMU as audio file
    # wf = wave.open('vibration.wav', 'wb')
    # f_linear = interpolate.interp1d(np.linspace(0, 5, 15104), acc, axis=0)
    # t_new = np.linspace(0, 5, 5 * 44100)
    # audio = f_linear(t_new)
    # wf.setnchannels(1)
    # wf.setsampwidth(2)
    # wf.setframerate(44100)
    # wf.writeframes(audio.astype('int16'))
    # wf.close()


    # re-coordinate acc to global frame by acc & compass
    # acc_data = read_data('../test/acc.txt')
    # compass = read_data('../test/compass.txt')
    # #acc_data = re_coordinate(acc_data, compass)
    # #acc_data = interpolation(acc_data)
    #
    # for i in range(3):
    #     plt.plot(acc_data[:, i])
    # plt.show()