import matplotlib.pyplot as plt
import os
import imuplot
import micplot
import numpy as np
import argparse
import scipy.signal as signal
from skimage import filters

seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
# seg_len_mic = 2560
# overlap_mic = 2240
# seg_len_imu = 256
# overlap_imu = 224
rate_mic = 16000
rate_imu = 1600
T = 30
segment = 5
stride = 3
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
time_stride = int(stride * rate_mic/(seg_len_mic-overlap_mic))

def noise_extraction():
    noise_list = os.listdir('dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]
def synchronization(Zxx, imu):
    in1 = np.sum(Zxx, axis=0)
    in2 = np.sum(imu, axis=0)
    shift = np.argmax(signal.correlate(in1, in2)) - len(in2)
    return np.roll(imu, shift, axis=1)
def estimate_response(Zxx, imu):
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    select = select2 & select1
    Zxx_ratio = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
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
    # fig, axs = plt.subplots(2, figsize=(5, 3))
    # axs[0].imshow(clip1, aspect='auto')
    # axs[1].imshow(clip2, aspect='auto')
    # plt.show()
    new_response, new_variance = estimate_response(clip2, clip1)
    response = 0.2 * new_response + 0.8 * response
    variance = 0.2 * new_variance + 0.8 * variance
    return response, variance

def error(imu, Zxx, response, variance):
    num_time = np.shape(imu)[1]
    full_response = np.tile(np.expand_dims(response, axis=1), (1, num_time))
    for j in range(num_time):
        full_response[:, j] += np.random.normal(0, variance, (freq_bin_high))
    augmentedZxx = Zxx[:freq_bin_high, :num_time] * full_response
    e = np.mean(np.abs(augmentedZxx - imu)) / np.max(imu)
    return e

def filter_function(response):
    m = np.max(response)
    n1 = np.mean(response[-5:])
    n2 = np.mean(response)
    if m > 30 or n1 > 3 or n2 < 2:
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    # mode 0 transfer function estimation
    # mode 1 noise extraction
    # mode 2 id

    args = parser.parse_args()
    if args.mode == 0:
        candidate = ["liang", "wu", "he", "hou", "zhao", "shi"]
        # "shen", "shuai"
        for i in range(len(candidate)):
            name = candidate[i]
            count = 0
            error = []
            path = 'exp6/' + name + '/clean/'
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            files_mic2 = files[3 * N:]

            for index in range(N):
                response, variance = np.zeros(freq_bin_high), np.zeros(freq_bin_high)
                wave, Zxx, phase = micplot.load_stereo(path + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic,
                                                       normalize=True)
                data1, imu1 = imuplot.read_data(path + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
                # data2, imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
                for j in range(int((T - segment) / stride) + 1):
                    clip1 = imu1[:, j * time_stride:j * time_stride + time_bin]
                    # clip2 = imu2[:, j * time_stride:j * time_stride + time_  bin]
                    clip3 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
                    clip1 = synchronization(clip3, clip1)
                    # clip2 = synchronize(clip2, clip3)
                    new_response, new_variance = estimate_response(clip3, clip1)
                    # new_response2, new_variance2 = estimate_response(clip3, clip2)

                    if filter_function(new_response):
                        response = 0.5 * new_response + 0.5 * response
                        variance = 0.5 * new_variance + 0.5 * variance

                    full_response = np.tile(np.expand_dims(new_response, axis=1), (1, time_bin))
                    for j in range(time_bin):
                        full_response[:, j] += np.random.normal(0, variance, (freq_bin_high))
                    augmentedZxx = clip3 * full_response
                    e = np.mean(np.abs(augmentedZxx - clip1)) / np.max(clip1)
                    error.append(e)
                    # augmentedZxx += noise_extraction()
                    # fig, axs = plt.subplots(2, sharex=True, figsize=(5, 3))
                    # axs[0].imshow(augmentedZxx, extent=[0, 5, 800, 0], aspect='auto')
                    # axs[1].imshow(clip1, extent=[0, 5, 800, 0], aspect='auto')
                    # fig.text(0.45, 0.022, 'Time (Second)', va='center')
                    # fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
                    # plt.savefig('synthetic_compare.eps')
                    # plt.show()

                full_response = np.tile(np.expand_dims(response, axis=1), (1, 1501))
                for j in range(time_bin):
                    full_response[:, j] += np.random.normal(0, variance, (freq_bin_high))
                augmentedZxx = Zxx[:freq_bin_high, :] * full_response
                e = np.mean(np.abs(augmentedZxx - imu1)) / np.max(imu1)

                if e < 0.05 and filter_function(response):
                    # print(e)
                    # plt.plot(response)
                    # plt.plot(variance)
                    # plt.show()
                    np.savez('transfer_function/' + str(i) + '_' + str(count) + '.npz',
                             response=response, variance=variance)
                    count += 1
            print(sum(error)/len(error))
        plt.show()
    elif args.mode == 1:
        for name in ['he', 'yan', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]:
            print(name)
            count = 0
            path = 'exp7/' + name + '/noise_train/'
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            files_mic1 = files[2 * N:3 * N]
            files_mic2 = files[3 * N:]
            for index in range(N):
                i = index
                r1, r2, v1, v2 = np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high)
                wave, Zxx, phase = micplot.load_stereo(path + files_mic1[i], T, seg_len_mic, overlap_mic, rate_mic,normalize=True)
                data1, imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
                data2, imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
                for j in range(int((T - segment) / stride) + 1):
                    clip1 = imu1[:, j * time_stride:j * time_stride + time_bin]
                    #clip2 = imu2[:, j * time_stride:j * time_stride + time_bin]
                    clip3 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]

                    r1, v1 = transfer_function(j, imu1, Zxx, r1, v1)
                    #r2, v2 = transfer_function(j, imu2, Zxx, r2, v2)
                    fig, axs = plt.subplots(2, sharex=True, figsize=(5, 3))
                    axs[0].imshow(clip1, extent=[0, 5, 800, 0], aspect='auto')
                    axs[1].imshow(clip3, extent=[0, 5, 800, 0], aspect='auto')
                    plt.show()
                # num_time = np.shape(imu1)[1]
                # full_response = np.tile(np.expand_dims(response, axis=1), (1, num_time))
                # for j in range(time_bin):
                #     full_response[:, j] += np.random.normal(0, variance, (freq_bin_high))
                # augmentedZxx = Zxx[:freq_bin_high, :num_time] * full_response
                # e = np.mean(np.abs(augmentedZxx - imu1)) / np.max(imu1)
                # error.append(e)
                np.savez('transfer_function/' + str(count) + '_' + name + '_transfer_function.npz',
                         response=r1, variance=v1)
                count += 1
                # np.savez('transfer_function/' + str(count) + '_' + name + '_transfer_function.npz',
                #          response=r2, variance=v2)
        plt.show()
    elif args.mode == 2:
        candidate = ["he", "liang", "hou", "shi", "shuai", "wu", "yan", "jiang", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        for i in range(len(candidate)):
            count = 0
            e = []
            for folder in ['/train/', '/noise_train/']:
                path = 'exp7/' + candidate[i] + folder
                files = os.listdir(path)
                N = int(len(files)/4)
                files_imu1 = files[:N]
                files_imu2 = files[N:2*N]
                files_mic1 = files[2*N:3*N]
                files_mic2 = files[3*N:]
                r1, r2, v1, v2 = np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high)
                for index in range(N):
                    wave, Zxx, phase = micplot.load_stereo(path + files_mic2[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
                    data1, imu1 = imuplot.read_data(path + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
                    data2, imu2 = imuplot.read_data(path + files_imu2[index], seg_len_imu, overlap_imu, rate_imu)
                    imu1 = synchronization(Zxx, imu1)
                    imu2 = synchronization(Zxx, imu2)
                    for j in range(int((T - segment) / stride) + 1):
                        mfcc = np.mean(micplot.MFCC(Zxx[:, j * time_stride:j * time_stride + time_bin], rate_mic), axis=1)
                        r1, v1 = transfer_function(j, imu1, Zxx, r1, v1)
                        r2, v2 = transfer_function(j, imu2, Zxx, r2, v2)
                        e.append(error(imu1, Zxx, r1, v1))
                        e.append(error(imu2, Zxx, r2, v2))
                        np.savez('transfer_function/' + str(i) + '_' + str(count) + '.npz',
                                 r1=r1, r2=r2, v1=v1, v2=v2, mfcc=mfcc)
                        count += 1
            print(sum(e)/len(e))
    elif args.mode == 3:
        for loc in ['box', 'human']:
            candidate = os.listdir(os.path.join('attack', loc))
            for i in range(len(candidate)):
                count = 0
                path = os.path.join('attack', loc, candidate[i])
                files = os.listdir(path)
                N = int(len(files) / 3)
                files_imu1 = files[:N]
                files_imu2 = files[N:2 * N]
                files_mic1 = files[2 * N:]
                for index in range(N):
                    r1, r2, v1, v2 = np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high), np.zeros(freq_bin_high)
                    wave, Zxx, phase = micplot.load_stereo(path + '/' + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
                    data1, imu1 = imuplot.read_data(path + '/' + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
                    data2, imu2 = imuplot.read_data(path + '/' + files_imu2[index], seg_len_imu, overlap_imu, rate_imu)
                    # imu1 = synchronization(Zxx, imu1)
                    # imu2 = synchronization(Zxx, imu2)
                    for j in range(int((T - segment) / stride) + 1):
                        mfcc = np.mean(micplot.MFCC(Zxx[:, j * time_stride:j * time_stride + time_bin], rate_mic), axis=1)
                        r1, v1 = transfer_function(j, imu1, Zxx, r1, v1)
                        r2, v2 = transfer_function(j, imu2, Zxx, r2, v2)
                        np.savez('attack_transfer_function/' + loc + '/' + str(i) + '_' + str(count) + '.npz',
                                 r1=r1, r2=r2, v1=v1, v2=v2, mfcc=mfcc)
                        count += 1
    else:
        count = 0
        Mean = []
        for path in ['exp4/he/none/', 'exp4/hou/none/', 'exp4/wu/none/', 'exp4/shi/none/', 'exp6/he/none/', 'exp6/hou/none/',
                     'exp6/liang/none/', 'exp6/shen/none/','exp6/shuai/none/', 'exp6/wu/none/']:
            files = os.listdir(path)
            N = int(len(files) / 4)
            files_imu1 = files[:N]
            files_imu2 = files[N:2 * N]
            for index in range(N):
                i = index
                imu1, Zxx_imu1 = imuplot.read_data(path + files_imu1[i], seg_len_imu, overlap_imu, rate_imu)
                imu2, Zxx_imu2 = imuplot.read_data(path + files_imu2[i], seg_len_imu, overlap_imu, rate_imu)
                np.save('dataset/noise/' + str(count) + '.npy', Zxx_imu1)
                Mean.append(np.mean(Zxx_imu1))
                count += 1
                np.save('dataset/noise/' + str(count) + '.npy', Zxx_imu2)
                count += 1
                Mean.append(np.mean(Zxx_imu2))
        print(sum(Mean)/count)


