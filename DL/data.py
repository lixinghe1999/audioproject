# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss
import os
import json
import math
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torchaudio

import matplotlib.pyplot as plt
from A2net import A2net
from A2netcomplex import A2net_m, A2net_p
import scipy.signal as signal
import librosa
import pyroomacoustics as pra
# seg_len_mic = 2560
# overlap_mic = 2240
# seg_len_imu = 256
# overlap_imu = 224

seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32

rate_mic = 16000
rate_imu = 1600
length = 5
stride = 3
N = 75
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
def spectrogram(audio):
    Zxx = signal.stft(audio, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]
    Zxx = Zxx[:8 * freq_bin_high, :]
    Zxx = np.expand_dims(Zxx, 0)
    return Zxx

def read_transfer_function(path, include=['he', 'hou', 'liang', 'shen', 'shuai', 'shi', 'wu', 'zhao']):
    npzs = os.listdir(path)
    transfer_function = np.zeros((len(npzs), freq_bin_high))
    variance = np.zeros((len(npzs), freq_bin_high))
    for i in range(len(npzs)):
        if npzs[i].split('_')[1] in include:
            npz = np.load(path + '/' + npzs[i])
            transfer_function[i, :] = npz['response']
            variance[i, :] = npz['variance']
    return transfer_function, variance
def noise_extraction():
    noise_list = os.listdir('../dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('../dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]
def synthetic(clean, transfer_function, variance):
    index = np.random.randint(0, N)
    f = transfer_function[index, :]
    response = np.tile(np.expand_dims(f, axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, variance[index, :], (freq_bin_high))
    noisy = clean[:, :freq_bin_high, :] * response
    background_noise = noise_extraction()
    noisy += 1 * background_noise
    return noisy
def read_data(file, seg_len=256, overlap=224, rate=1600, offset=0, duration=length):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((duration * rate, 4))
    for i in range(duration * rate):
        line = lines[min(i + offset * rate, len(lines)-1)].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
    data[:, :3] = np.clip(data[:, :3], -0.05, 0.05)
    f, t, Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    Zxx = np.expand_dims(Zxx, 0)
    return Zxx
def synchronize(x, y):
    x_t = np.sum(np.abs(x[:, :freq_bin_high, :]), axis=(0, 1))
    y_t = np.sum(y[:, :freq_bin_high, :], axis=(0, 1))
    corr = signal.correlate(y_t, x_t, 'full')
    shift = np.argmax(corr) - time_bin
    new = np.roll(x, shift, axis=2)
    return new

class Audioset:
    def __init__(self, files=None, pad=True, imu=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.imu = imu
        self.length = length
        self.stride = stride
        if self.imu:
            self.sample_rate = rate_imu
            self.window = seg_len_imu
            self.overlap = overlap_imu
        else:
            self.sample_rate = rate_mic
            self.window = seg_len_mic
            self.overlap = overlap_mic
        for info in self.files:
            _, file_length = info
            if self.length is None:
                examples = 1
            elif file_length < self.length*self.sample_rate:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length * self.sample_rate) / self.stride/self.sample_rate) + 1)
            else:
                examples = (file_length - self.length * self.sample_rate) // (self.stride*self.sample_rate) + 1
            self.num_examples.append(examples)
    def __len__(self):
        return sum(self.num_examples)
    def __getitem__(self, index):
        for info, examples in zip(self.files, self.num_examples):
            file, _ = info

            if index >= examples:
                index -= examples
                continue
            duration = 0
            offset = 0
            if self.length:
                offset = self.stride * index
                duration = self.length
            if self.imu:
                out = read_data(file, seg_len=self.window, overlap=self.overlap, rate=self.sample_rate, offset=offset, duration=duration)
            else:

                out, sr = librosa.load(file, offset=offset, duration=duration, sr=rate_mic)
                b, a = signal.butter(4, 100, 'highpass', fs=rate_mic)
                if self.length:
                    out = np.pad(out, (0, duration * self.sample_rate - out.shape[-1]))
                out = signal.filtfilt(b, a, out)
            return out
class NoisyCleanSet:
    def __init__(self, transfer_function, variance, json_path1, json_path2, phase=False, alpha=(1, 1)):
        """__init__. n

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        with open(json_path1, 'r') as f:
            clean = json.load(f)
        with open(json_path2, 'r') as f:
            noise = json.load(f)
        self.clean_set = Audioset(clean)
        self.noise_set = Audioset(noise)
        self.transfer_function = transfer_function
        self.variance = variance
        self.alpha = alpha
        self.phase = phase
        self.length = len(self.noise_set)

    def __getitem__(self, index):
        speech = self.clean_set[index]
        noise = self.noise_set[np.random.randint(0, self.length)]
        # room = pra.ShoeBox([1, 1])
        # random_loc = np.random.rand(1, 2)
        # room.add_source(random_loc[0], signal=noise)
        # room.add_source(np.array([0.49, 0.51]), signal=speech)
        # room.add_microphone_array(np.array([[0.5], [0.5]]))
        # room.simulate()
        # noise = room.mic_array.signals[0, :rate_mic * length]
        noise = noise * (np.random.random()/2 + 0.5) + speech
        noise = spectrogram(noise)
        speech = spectrogram(speech)

        if self.phase:
            return torch.from_numpy(
                synthetic(np.abs(speech) / self.alpha[0], self.transfer_function, self.variance) / self.alpha[1]), \
                   torch.from_numpy(noise / self.alpha[2]), torch.from_numpy(speech/self.alpha[3])
        else:
            return torch.from_numpy(synthetic(np.abs(speech)/self.alpha[0], self.transfer_function, self.variance)/self.alpha[1]),\
                   torch.from_numpy(np.abs(noise)/self.alpha[2]), torch.from_numpy(np.abs(speech)/self.alpha[3])
    def __len__(self):
        return len(self.clean_set)
class IMUSPEECHSet:
    def __init__(self, imu_path, wav_path, noise_path, simulate=True, person=["liang", "wu", "he", "hou", "zhao", "shi", "shuai", "shen"], phase=False, minmax=(1, 1)):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        with open(imu_path, 'r') as f:
            imu = json.load(f)
        select = []
        for i in range(len(imu)):
            x1, _ = imu[i]
            if x1.split('\\')[1] in person:
                select.append(i)
        with open(wav_path, 'r') as f:
            wav = json.load(f)
        with open(noise_path, 'r') as f:
            noise = json.load(f)
        self.phase = phase
        self.minmax = minmax
        self.simulate = simulate
        self.imu_set = Audioset([imu[index] for index in select], imu=True)
        self.wav_set = Audioset([wav[index] for index in select], imu=False)
        self.noise_set = Audioset([noise[index] for index in select], imu=False)
        self.length = len(self.noise_set)

    def __getitem__(self, index):
        imu = self.imu_set[index]
        speech = self.wav_set[index]
        if self.simulate:
            noise = self.noise_set[np.random.randint(0, self.length, 1)]
            noise = noise * (np.random.random()/5 + 0.5) + speech
            # room = pra.ShoeBox([3, 3])
            # random_loc = np.random.rand(1, 2) * 3
            # room.add_source(random_loc[0], signal=noise)
            # room.add_source(np.array([1.45, 1.55]), signal=speech)
            # room.add_microphone_array(np.array([[1.5], [1.5]]))
            # room.simulate()
            # noise = room.mic_array.signals[0, :rate_mic * length]
        else:
            noise = self.noise_set[index]

        speech = spectrogram(speech)
        noise = spectrogram(noise)

        imu = synchronize(imu, np.abs(speech))
        if self.phase:
            return torch.from_numpy(imu / self.minmax[0]), torch.from_numpy(noise / self.minmax[1]), torch.from_numpy(
                speech / self.minmax[2])
        else:
            return torch.from_numpy(imu / self.minmax[0]), torch.from_numpy(np.abs(noise) / self.minmax[1]), torch.from_numpy(np.abs(speech)/ self.minmax[2])
    def __len__(self):
        return len(self.imu_set)
if __name__ == "__main__":
    #
    # audio_files = []
    # for path in [r"../dataset/test-clean", r"../dataset/background", r"../dataset/music"]:
    #     g = os.walk(path)
    #     for path, dir_list, file_list in g:
    #         for file_name in file_list:
    #             if file_name[-3:] not in ['txt', 'mp3']:
    #                 audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
    # json.dump(audio_files, open('background.json', 'w'), indent=4)


    # imu_files = []
    # wav_files = []
    # g = os.walk(r"../exp6")
    # folder = 'clean'
    # person = ["liang", "wu", "he", "hou", "zhao", "shi", "shuai", "shen"]
    # for path, dir_list, file_list in g:
    #     N = int(len(file_list) / 4)
    #     if path[-5:] != folder or path[8:-6] not in person:
    #         # only collect some type of data
    #         continue
    #     imu1 = file_list[: N]
    #     imu2 = file_list[N: 2 * N]
    #     wav = file_list[2 * N: 3 * N]
    #     for i in range(N-1, -1, -1):
    #         # have 2 IMUs, but now only use one
    #         imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i])).readlines())])
    #         wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #
    #         imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i])).readlines())])
    #         wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    # json.dump(imu_files, open(folder + '_imuexp6.json', 'w'), indent=4)
    # json.dump(wav_files, open(folder + '_wavexp6.json', 'w'), indent=4)

    # imu_files = []
    # wav_files = []
    # gt_files = []
    # g = os.walk(r"../exp7")
    # folder = ['train']
    # name = 'train'
    # person = ["liang", "wu", "he", "hou", "zhao", "shi", "shuai", "shen"]
    # for path, dir_list, file_list in g:
    #     N = int(len(file_list) / 4)
    #     if N > 0:
    #         p = path.split('\\')
    #         if p[-1] not in folder or p[1] not in person:
    #             # only collect some type of data
    #             continue
    #         imu1 = file_list[: N]
    #         imu2 = file_list[N: 2 * N]
    #         gt = file_list[2 * N: 3 * N]
    #         wav = file_list[3 * N:]
    #         wav.sort(key=lambda x: int(x[4:-5]))
    #         for i in range(N):
    #             # have 2 IMUs, but now only use one
    #             imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i])).readlines())])
    #             wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #             gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])
    #
    #             imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i])).readlines())])
    #             wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #             gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])
    #     json.dump(imu_files, open(name + '_imuexp7.json', 'w'), indent=4)
    #     json.dump(wav_files, open(name + '_wavexp7.json', 'w'), indent=4)
    #     json.dump(gt_files, open(name + '_gtexp7.json', 'w'), indent=4)



    transfer_function, variance = read_transfer_function('../transfer_function')
    BATCH_SIZE = 1
    #model = A2net()
    #model = A2netU()
    model = A2net_m()

    #dataset_train = IMUSPEECHSet('train_imuexp7.json', 'train_wavexp7.json', 'train_wavexp7.json', person=['shuai'], minmax=(1, 1, 1))
    #dataset_train = IMUSPEECHSet('clean_imuexp7.json', 'clean_wavexp7.json', 'clean_wavexp7.json', person=['shi'], minmax=(1, 1, 1))
    #dataset_train = IMUSPEECHSet('noise_imuexp7.json', 'noise_gtexp7.json', 'noise_wavexp7.json', simulate=False, person=['shi'], minmax=(1, 1, 1))

    #dataset_train = NoisyCleanSet(transfer_function, variance, 'speech100.json', 'background.json', alpha=(28, 0.04, 0.095, 0.095))
    dataset_train = NoisyCleanSet(transfer_function, variance, 'devclean.json', 'background.json', alpha=(28, 0.04, 0.095, 0.095))
    loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    L1loss = nn.L1Loss()
    L2loss = nn.MSELoss()
    x_mean = []
    noise_mean = []
    y_mean = []
    #mel_trans = torchaudio.transforms.MelScale(n_mels=256, f_min=100, f_max=100 + 8 * 700, n_stft=896)
    L = []
    with torch.no_grad():
        for step, (x, noise,  y) in enumerate(loader):



            # predict1 = model(x.to(dtype=torch.float), noise.to(dtype=torch.float))
            # print(L1loss(noise, y).item())
            fig, axs = plt.subplots(3, 1)
            axs[0].imshow(x[0, 0], vmin=0, vmax=x.max(), aspect='auto')
            axs[1].imshow(noise[0, 0, :2*freq_bin_high, :], vmin=0, vmax=noise.max(), aspect='auto')
            axs[2].imshow(y[0, 0, :2*freq_bin_high, :], vmin=0, vmax=y.max(), aspect='auto')
            plt.show()
        #
        #     x_mean.append(np.max(x.numpy(), axis=(0, 1, 2, 3)))
        #     noise_mean.append(np.max(noise.numpy(), axis=(0, 1, 2, 3)))
        #     y_mean.append(np.max(y.numpy(), axis=(0, 1, 2, 3)))
        #     # if step > 1000:
        #     #     break
        # print(np.mean(x_mean))
        # print(np.mean(noise_mean))
        # print(np.mean(y_mean))
