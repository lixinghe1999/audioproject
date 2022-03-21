import matplotlib.pyplot as plt
import os
import imuplot
import micplot
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from skimage import filters
import scipy.signal as signal


seg_len_mic = 2560
overlap_mic = 2240
seg_len_imu = 256
overlap_imu = 224
rate_mic = 16000
rate_imu = 1600
T = 30
segment = 5
stride = 3
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
#freq_bin_high = 128
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
time_stride = int(stride * rate_mic/(seg_len_mic-overlap_mic))

def estimate_response(Zxx, imu):
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    select = select2 & select1
    Zxx_ratio1 = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
    Zxx_ratio = Zxx_ratio1
    new_variance = np.zeros(freq_bin_high)
    new_response = np.zeros(freq_bin_high)
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            new_response[i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            new_variance[i] = np.std(Zxx_ratio[i, :], where=select[i, :])
    return new_response, new_variance
def transfer_function(j, imu, Zxx, response, variance):
    clip1 = imu[:, j * time_stride:j * time_stride + time_bin]
    clip2 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
    new_response, new_variance = estimate_response(clip2, clip1)
    response = 0.25 * new_response + 0.75 * response
    variance = 0.25 * new_variance + 0.75 * variance
    return response, variance



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument('--mode', action="store", type=int, default=0, required=False, help='mode of processing, 0-time segment, 1-mic denoising')
    X1 = []
    X2 = []
    Y = []
    ratio = 0.2
    candidate = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]
    for i in range(len(candidate)):
        name = candidate[i]
        print(name)
        path = 'exp7/' + name + '/train/'
        files = os.listdir(path)
        N = int(len(files)/4)
        files_imu1 = files[:N]
        files_imu2 = files[N:2*N]
        files_mic1 = files[2*N:3*N]
        files_mic2 = files[3*N:]
        for index in range(N):
            r1, r2, v1, v2 = np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high)
            wave, Zxx, phase = micplot.load_stereo(path + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
            data1, imu1 = imuplot.read_data(path + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
            data2, imu2 = imuplot.read_data(path + files_imu2[index], seg_len_imu, overlap_imu, rate_imu)
            for j in range(int((T - segment) / stride) + 1):
                r1, v1 = transfer_function(j, imu1, Zxx, r1, v1)
                r2, v2 = transfer_function(j, imu2, Zxx, r2, v2)
            X1.append(r1/np.max(r1))
            X1.append(r2/np.max(r2))
            X2.append(v1/np.max(r1))
            X2.append(r2/np.max(r2))
            Y = Y + [i, i]
    X1, X2, Y = np.array(X1), np.array(X2), np.array(Y)
    print(X1.shape, Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=ratio, stratify=Y, random_state=1)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))

