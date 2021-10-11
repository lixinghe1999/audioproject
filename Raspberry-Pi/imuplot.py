import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from micplot import get_wav, peaks_mic
import os
import matplotlib.pyplot as plt
import wave

seg_len = 128
overlap = 64
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

def noise_reduce(Zxx):
    shape = Zxx.shape
    low = int((shape[0] - 1) * 60 / rate * 2)
    high = int((shape[0] - 1) * 800 / rate * 2)
    Zxx[:low, :] = 0
    Zxx[high:, :] = 0
    Zxx[:, :2] = 0
    Zxx[:, -2:] = 0
    return Zxx
def normalize(Zxx):
    Zxx /= 2**14
    # Zmax, Zmin = Zxx.max(), Zxx.min()
    # Zxx = (Zxx - Zmin) / (Zmax - Zmin)
    #Zxx = np.clip(Zxx, 0.4, 1)
    #Zxx = 1000**(Zxx)
    # Zxx_smooth = n p.mean(Zxx, axis=1)
    # # Zxx_smooth = np.zeros(shape)
    # # for i in range(shape[0]):
    # #     Zxx_smooth[i, :] = np.convolve(Zxx[i, :], np.ones(5)/5, mode='same')
    # for i in range(shape[1]):
    #     Zxx[:, i] = np.abs(Zxx[:, i] - Zxx_smooth)
    return Zxx
def peaks_imu(Zxx):
    sum_Zxx = np.sum(Zxx, axis=0)
    m, n = 0.1 * rate/(seg_len - overlap), 1 * rate/(seg_len - overlap)
    smooth_Zxx = np.convolve(sum_Zxx , np.ones(10)/10, mode='same')
    peaks, dict = signal.find_peaks(smooth_Zxx, distance=m, width=[m, n], prominence=1*np.var(smooth_Zxx))
    return peaks
def time_match(p1, p2):
    P1 = np.tile(p1.reshape((len(p1), 1)), (1, len(p2)))
    P2 = np.tile(p2, (len(p1), 1))
    diff = np.abs(P1 - P2)
    #return np.sum(np.min(diff, axis = 0) < 0.2)/len(p2)
    m = np.min(diff, axis = 0) < 0.2
    n = np.argmin(diff, axis = 0)
    return m, n
def saveaswav(data, l, r, count):
    l = (l + 1) * 1024
    r = (r + 1) * 1024
    for i in range(len(l)):
        data_seg = data[int(l[i])-5000: int(r[i])+5000]
        wf = wave.open('../exp2/audio_seg/' + str(count)+'.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data_seg.astype('int16'))
        wf.close()
        count = count + 1
    return count

if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)
    # path = '../test/bmi160/'
    count = 0
    for path in ['../exp2/HE/', '../exp2/HOU/']:
        #index = [3]
        index = range(27)
        files = os.listdir(path)
        for i in index:
            data = read_data(path + files[i])
            time_start = data[0, 3]
            # compass = read_data(path + 'compass_still.txt')
            mic1_index, mic2_index = i + 27, i + 54
            mic1, Zxx1 = get_wav(path + files[mic1_index])
            time1 = float(files[mic1_index][:-4].split('_')[1])
            p1, l1, r1 = peaks_mic(Zxx1)
            mic2, Zxx2 = get_wav(path + files[mic2_index])
            time2 = float(files[mic2_index][:-4].split('_')[1])
            p2, l2, r2 = peaks_mic(Zxx2)
            p1 = (p1 + 1) * 1024 / 44100
            p2 = (p2 + 1) * 1024 / 44100
            #data = interpolation(data)
            #data = re_coordinate(data, compass)

            #data_norm = np.linalg.norm(data[:, :-1], axis=1)
            b, a = signal.butter(8, 0.07, 'highpass')
            for j in range(3):
                data[:, j] = signal.filtfilt(b, a, data[:, j])
            dominant_axis = (np.argmax(np.sum(np.abs(data[:, :-1]),axis=0)))
            for j in range(3):
                f, t, Zxx = signal.stft(data[:, j], nperseg=seg_len, noverlap=overlap, fs=rate)
                Zxx = normalize(Zxx)
                # phase_random = np.exp(2j * np.pi * np.random.rand(*Zxx.shape))
                # phase_acc = np.exp(1j * np.angle(Zxx))
                # acc, Zxx = GLA(0, Zxx, phase_acc)
                Zxx = np.abs(Zxx)
                # axs[j//2, j%2].imshow(Zxx, extent=[0, 5, rate/2, 0])
                # axs[j//2, j%2].set_aspect(10/rate)
                if j == dominant_axis:
                    p3 = peaks_imu(Zxx)
                    p3 = (p3 + 1) * 64/1600
                    p1 = p1 + time1 - time_start
                    p2 = p2 + time2 - time_start
                    match_tone = []
                    m1, m2 = time_match(p1, p3), time_match(p2, p3)

                    for k in range(len(p3)):
                        match_tone.append(m1[0][k] and m2[0][k])
                    l1 = l1[m1[1][match_tone]]
                    r1 = r1[m1[1][match_tone]]

                    count = saveaswav(mic1, l1, r1, count)
                    l2 = l2[m2[1][match_tone]]
                    r2 = r2[m2[1][match_tone]]
                    count = saveaswav(mic2, l2, r2, count)
                    p3 = p3[match_tone]
                    #print(p1, p2, p3)

            # axs[1, 1].imshow(Zxx1[:75],  extent=[0, 5, 800, 0])
            # axs[1, 1].set_aspect(5 / 550)
            # plt.show()

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