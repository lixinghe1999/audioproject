import librosa
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams.update({'font.size': 10})
import pickle
import os
import imuplot
import micplot
import numpy as np
import argparse
import scipy.signal as signal
import scipy.stats as stats
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
segment = 30
stride = 30
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
    in1 = np.sum(Zxx[:freq_bin_high, :], axis=0)
    in2 = np.sum(imu, axis=0)
    shift = np.argmax(signal.correlate(in1, in2)) - len(in2)
    return np.roll(imu, shift, axis=1)
def estimate_response(imu, Zxx):
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    select = select2 & select1
    Zxx_ratio = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
    response = np.zeros((2, freq_bin_high))
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            response[0, i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            response[1, i] = np.std(Zxx_ratio[i, :], where=select[i, :])
    return response
def transfer_function(clip1, clip2, response):
    new_response = estimate_response(clip1, clip2)
    #if filter_function(new_response[0, :]):
    response = 0.25 * new_response + 0.75 * response
    return response
def ratio_accumulate(Zxx, imu, Zxx_valid):
    Zxx = Zxx[:freq_bin_high, :]
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    select = select2 & select1
    Zxx_ratio = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            new_Zxx = Zxx_ratio[i, select[i, :]].tolist()
            Zxx_valid[i] = Zxx_valid[i] + new_Zxx
    return Zxx_valid

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
        candidate = ["he", "liang", "wu", "hou", "zhao", "shi", "shen", "shuai"]
        for i in range(len(candidate)):
            Zxx_valid = [[]] * freq_bin_high
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
                response = np.zeros((2, freq_bin_high))
                wave, Zxx, phase = micplot.load_stereo(path + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
                data1, imu1 = imuplot.read_data(path + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
                # data2, imu1 = imuplot.read_data(path + files_imu2[index], seg_len_imu, overlap_imu, rate_imu)
                # Zxx = librosa.feature.melspectrogram(S=Zxx, sr=rate_mic, n_mels=264)
                # imu1 = librosa.feature.melspectrogram(S=imu1, sr=rate_imu, n_mels=33)
                imu1 = synchronization(Zxx, imu1)
                for j in range(int((T - segment) / stride) + 1):

                    clip2 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
                    clip1 = imu1[:, j * time_stride:j * time_stride + time_bin]

                    # store gamma distribution parameters
                    # Zxx_valid = ratio_accumulate(clip2, clip1, Zxx_valid)
                    response = transfer_function(clip1, clip2, response)

                    full_response = np.tile(np.expand_dims(response[0, :], axis=1), (1, time_bin))
                    for j in range(time_bin):
                        full_response[:, j] += stats.norm.rvs(response[0, :], response[1, :])
                    augmentedZxx = clip2 * full_response
                    e = np.mean(np.abs(augmentedZxx - clip1)) / np.max(clip1)
                    error.append(e)
                    augmentedZxx += noise_extraction()
                    fig, axs = plt.subplots(2, figsize=(4, 2))
                    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.98)
                    axs[0].imshow(augmentedZxx, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
                    axs[0].set_xticks([])
                    axs[1].imshow(clip1, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
                    fig.text(0.44, 0.022, 'Time (Sec)', va='center')
                    fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
                    plt.savefig('synthetic_compare.pdf')
                    plt.show()

                # full_response = np.tile(np.expand_dims(response[0, :], axis=1), (1, 1501))
                # for j in range(time_bin):
                #     full_response[:, j] += np.random.normal(0, response[1, :], (freq_bin_high))
                # augmentedZxx = Zxx[:freq_bin_high, :] * full_response
                # e = np.mean(np.abs(augmentedZxx - imu1)) / np.max(imu1)
                if filter_function(response[0, :]):
                    # plt.plot(response[0, :])
                    # plt.show()
                    #np.savez('transfer_function/' + str(i) + '_' + str(count) + '.npz', response=response[0, :], variance=response[1, :])
                    count += 1
            # parameters = ratio_gamma(Zxx_valid)
            # my_file = 'gamma_transfer_function/' + str(i) + '.pkl'
            # with open(my_file, 'wb') as f:
            #     pickle.dump(parameters, f)
    elif args.mode == 1:
        directory = 'C://Users/HeLix/Downloads/EMSB'
        g = os.walk(directory)
        count = 0
        for path, dir_list, file_list in g:
            N = len(file_list)
            if N > 0:
                print(path)
                name = path.split('/')[-1]
                json_data = []
                for f in file_list:
                    data, _ = librosa.load(os.path.join(path, f), mono=False, sr=rate_mic)
                    b, a = signal.butter(4, 80, 'highpass', fs=rate_mic)

                    audio = data[0]
                    imu = data[1]
                    audio = signal.filtfilt(b, a, audio, axis=0)
                    imu = signal.filtfilt(b, a, imu, axis=0)

                    audio = np.abs(signal.stft(audio, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1])
                    imu = np.abs(signal.stft(imu, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1])
                    T = int(data.shape[1] / rate_mic)

                    for j in range(int((T - segment) / stride) + 1):

                        clip1 = imu[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
                        clip2 = audio[:freq_bin_high, j * time_stride:j * time_stride + time_bin]

                        response = estimate_response(clip1, clip2)
                        count += 1
                        plt.plot(response[0])
                        plt.show()
                        np.savez('transfer_function_EMSB/' + str(count) + '.npz', response=response[0, :], variance=response[1, :])
                        # full_response = np.tile(np.expand_dims(response[0, :], axis=1), (1, time_bin))
                        # for j in range(time_bin):
                        #     full_response[:, j] += stats.norm.rvs(response[0, :], response[1, :])
                        # augmentedZxx = clip2 * full_response

                        # fig, axs = plt.subplots(2, figsize=(4, 2))
                        # plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.98)
                        # axs[0].imshow(augmentedZxx, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
                        # axs[0].set_xticks([])
                        # axs[1].imshow(clip1, extent=[0, 5, 0, 800], aspect='auto', origin='lower')
                        # fig.text(0.44, 0.022, 'Time (Sec)', va='center')
                        # fig.text(0.01, 0.52, 'Frequency (Hz)', va='center', rotation='vertical')
                        # plt.savefig('synthetic_compare.pdf')
                        # plt.show()
                    break

    elif args.mode == 2:
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


