import os

import matplotlib.pyplot as plt

from imuplot import calibrate, read_data
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import argparse
import micplot
from skimage import filters, morphology
# basically use this script to
# 1) change sample rate
# 2) synchronize audio
def synchronization(Zxx, imu):
    in1 = np.sum(Zxx, axis=0)
    in2 = np.sum(imu, axis=0)
    corr = signal.correlate(in1, in2)
    shift = int(np.argmax(corr) - len(in2))
    imu = np.roll(imu, shift, axis=1)
    return imu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()
    if args.mode == 0:
        #synchronize airpods and microphone
        source = 'exp7/raw'
        target = 'exp7'
        sentences = ['s1', 's2', 's3', 's4']
        cls = ['noise', 'clean', 'mobile']
        T = 5
        # sentences = ['noise_train', 'train']
        # cls = ['']
        # T = 30
        for folder in ['he', 'yan', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8", 'jiang', '9']:
            print(folder)
            for s in sentences:
                for c in cls:
                    path = os.path.join(source, folder, s, c)
                    if not os.path.isdir(path):
                        continue
                    else:
                        files = os.listdir(path)
                        N = int(len(files) / 4)
                        files_imu1 = files[:N]
                        files_imu2 = files[N:2 * N]
                        files_mic1 = files[2 * N:3 * N]
                        files_mic2 = files[3 * N:]
                        files_mic2.sort(key=lambda x: int(x[4:-4]))
                        for i in range(N):
                            time1 = float(files_imu1[i].split('_')[1][:-4])
                            time2 = float(files_imu2[i].split('_')[1][:-4])
                            time_mic = float(files_mic1[i].split('_')[1][:-4])
                            shift1 = int((time1 - time_mic) * 1600)
                            shift2 = int((time2 - time_mic) * 1600)
                            calibrate(os.path.join(path, files_imu1[i]), os.path.join(target, folder, s, c, files_imu1[i]), T, shift1)
                            calibrate(os.path.join(path, files_imu2[i]), os.path.join(target, folder, s, c, files_imu2[i]), T, shift2)

                            mic = librosa.load(os.path.join(path, files_mic1[i]), sr=16000)[0]
                            airpods = librosa.load(os.path.join(path, files_mic2[i]), sr=16000)[0]
                            shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
                            airpods = np.roll(airpods, shift)[-T * 16000:]
                            airpods = np.pad(airpods, (0, T * 16000 - len(airpods)))
                            f1 = files_mic1[i][:-4] + '.wav'
                            f2 = files_mic2[i][:-4] + '.wav'
                            sf.write(os.path.join(target, folder, s, c, f1), mic, 16000)
                            sf.write(os.path.join(target, folder, s, c, f2), airpods, 16000)
    elif args.mode == 1:
        # synchronize airpods and microphone
        source = 'replay/record'
        T = 5
        # source = 'attack/box'
        # T = 30
        for folder in os.listdir(source):
            path = os.path.join(source, folder)
            if not os.path.isdir(path):
                continue
            else:
                files = os.listdir(path)
                N = int(len(files) / 3)
                files_imu1 = files[:N]
                files_imu2 = files[N:2 * N]
                files_mic1 = files[2 * N:]
                for i in range(N):
                    calibrate(os.path.join(path, files_imu1[i]),
                              os.path.join(path, files_imu1[i]), T, 0)
                    calibrate(os.path.join(path, files_imu2[i]),
                              os.path.join(path, files_imu2[i]), T, 0)

    else:
        source = 'replay/record/he'
        target = 'replay/generate'
        T = 5
        path = source
        files = os.listdir(path)
        N = int(len(files) / 3)
        files_imu1 = files[:N]
        files_imu2 = files[N:2 * N]
        files_mic1 = files[2 * N:]
        for i in range(N):
            data1, imu1 = read_data(os.path.join(path, files_imu1[i]))
            data2, imu2 = read_data(os.path.join(path, files_imu2[i]))
            wave, mic, phase = micplot.load_stereo(os.path.join(path, files_mic1[i]), T, 2560, 2240, 16000, normalize=True)
            imu1 = synchronization(mic[:60, :], imu1)
            imu2 = synchronization(mic[:60, :], imu2)
            # imu1[imu1 < 0.5 * filters.threshold_otsu(imu1)] = 0
            # imu2[imu2 < 0.5 * filters.threshold_otsu(imu2)] = 0
            # imu1 = morphology.dilation(imu1, morphology.square(3))
            # imu2 = morphology.dilation(imu2, morphology.square(3))
            re1 = phase[:129, :] * imu1
            re2 = phase[:129, :] * imu2
            out = np.zeros((1281, 251), dtype='complex64')
            out[:129] = re1
            _, x1 = signal.istft(out, fs=16000, window='hann', nperseg=2560, noverlap=2240)
            out[:129] = re2
            _, x2 = signal.istft(out, fs=16000, window='hann', nperseg=2560, noverlap=2240)
            sf.write(os.path.join(target, '1_' + str(i) + '.wav'), 10 * x1, 16000)
            sf.write(os.path.join(target, '2_' + str(i) + '.wav'), 10 * x2, 16000)
            fig, axs = plt.subplots(3, figsize=(5, 3))
            axs[0].imshow(imu1, aspect='auto')
            axs[1].imshow(imu2, aspect='auto')
            axs[2].imshow(mic[:129, :], aspect='auto')
            plt.show()






