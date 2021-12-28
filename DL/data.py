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
import soundfile as sf
from torchvision import transforms
import scipy.signal as signal
from scipy.interpolate import interp2d
from skimage.transform import resize
import librosa
from pytorch_msssim import MS_SSIM
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
length = 4
stride = 1
N = 100
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
Loss_weight = torch.tile(torch.unsqueeze(np.exp(torch.linspace(0, 2, (freq_bin_high-freq_bin_low))), dim=1), (1, time_bin))
def weighted_loss(input, target, device='cpu'):
    #return (Loss_weight.to(device=device) * torch.abs(input - target)).mean()
    beta = 0.01
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return (Loss_weight.to(device=device) * loss).mean()
def transfer_function_generator(transfer_function):
    n, m = np.shape(transfer_function)
    x = np.linspace(1, m, m)
    y = np.linspace(1, n, n)
    x1 = np.linspace(1, m, m)
    y1 = np.linspace(1, n, N)
    f = interp2d(x, y, transfer_function, kind='cubic')
    Z = f(x1, y1)
    return np.clip(Z, 0, None)
def read_transfer_function(path):
    npzs = os.listdir(path)
    transfer_function = np.zeros((len(npzs), freq_bin_high - freq_bin_low))
    variance = np.zeros((len(npzs), freq_bin_high - freq_bin_low))
    noise = np.zeros((len(npzs), 2))
    for i in range(len(npzs)):
        npz = np.load(path + '/' + npzs[i])
        transfer_function[i, :] = npz['response']
        variance[i, :] = npz['variance']
        noise[i, :] = npz['noise']
    return transfer_function, variance, noise
def noise_extraction():
    noise_list = os.listdir('../dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('../dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]
def synthetic(clean, transfer_function, variance):
    index = np.random.randint(0, N)
    response = np.tile(np.expand_dims(transfer_function[index, :], axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, variance[index, :], (freq_bin_high-freq_bin_low))
    noisy = clean * response
    background_noise = noise_extraction()
    noisy += 1 * background_noise
    noisy = np.hstack((noisy, np.zeros(np.shape(noisy))))
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
    data[:, :3] = np.clip(data[:, :3], -0.02, 0.02)
    f, t, Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, window="hamming", axis=0)
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    #Zxx = resize(Zxx, (freq_bin_high, time_bin))
    Zxx = Zxx[freq_bin_low:, :]
    # Zxx = np.vstack((Zxx, np.zeros(np.shape(Zxx))))
    return Zxx
def synchronize(x, y):
    x_t = np.sum(x[:, :50, :], axis=(0, 1))
    y_t = np.sum(y[:, :50, :], axis=(0, 1))
    corr = signal.correlate(y_t, x_t, 'full')
    shift = np.argmax(corr) - time_bin
    new = np.roll(x, shift, axis=2)
    return new

class Audioset:
    def __init__(self, files=None, full=False, pad=True, imu=False, phase=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.imu = imu
        self.length = length
        self.stride = stride
        self.phase = phase
        self.full = full
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
                Zxx = read_data(file, seg_len=self.window, overlap=self.overlap, rate=self.sample_rate, offset=offset, duration=duration)
            else:
                out, sr = librosa.load(file, offset=offset, duration=duration, sr=rate_mic)
                if self.length:
                    out = np.pad(out, (0, duration * self.sample_rate - out.shape[-1]))
                Zxx = signal.stft(out, nperseg=self.window, noverlap=self.overlap, fs=sr)[-1]
                if not self.full:
                    Zxx = Zxx[freq_bin_low:freq_bin_high, :]
                else:
                    Zxx = Zxx[freq_bin_low:, :]
            if self.phase:
                out = np.expand_dims(Zxx, 0)
            else:
                out = np.expand_dims(np.abs(Zxx), 0)
            return out
class NoisyCleanSet:
    def __init__(self, transfer_function, variance, noise, json_path, alpha=(1, 1)):
        """__init__. n

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        with open(json_path, 'r') as f:
            clean = json.load(f)
        self.clean_set = Audioset(clean)
        self.transfer_function = transfer_function
        self.variance = variance
        self.noise = noise
        self.alpha = alpha
    def __getitem__(self, index):
        return torch.from_numpy(synthetic(self.clean_set[index]/self.alpha[0],
                                          self.transfer_function, self.variance)/self.alpha[1]),\
               torch.from_numpy(self.clean_set[index]/self.alpha[2])
    def __len__(self):
        return len(self.clean_set)
class IMUSPEECHSet:
    def __init__(self, imu_path, wav_path, full, phase=False, minmax=(1, 1)):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        with open(imu_path, 'r') as f:
            imu = json.load(f)
        with open(wav_path, 'r') as f:
            wav = json.load(f)
        self.phase = phase
        self.full = full
        self.imu_set = Audioset(imu, imu=True)
        self.wav_set = Audioset(wav, phase=self.phase, full=self.full, imu=False)
        self.minmax = minmax

    def __getitem__(self, index):
        speech = self.wav_set[index]
        audio = self.imu_set[index]
        audio = synchronize(audio, np.abs(speech))
        return torch.from_numpy(audio/self.minmax[0]), torch.from_numpy(speech/self.minmax[1])
    def __len__(self):
        return len(self.imu_set)

if __name__ == "__main__":
    # audio_files = []
    # g = os.walk(r"../dataset/train-clean-100")
    # for path, dir_list, file_list in g:
    #     for file_name in file_list:
    #         if file_name[-4:] == 'flac':
    #             audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
    # json.dump(audio_files, open('speech100.json', 'w'), indent=4)
    #
    #
    # imu_files = []
    # wav_files = []
    # gt_files = []
    # g = os.walk(r"../exp6")
    # folder = 'noise'
    # for path, dir_list, file_list in g:
    #     N = int(len(file_list) / 4)
    #     if path[-5:] != folder:
    #         # only collect some type of data
    #         continue
    #     imu1 = file_list[: N]
    #     imu2 = file_list[N: 2 * N]
    #     wav = file_list[2 * N: 3 * N]
    #     gt = file_list[3 * N:]
    #     gt.sort(key=lambda x: int(x[4:-5]))
    #     for i in range(N):
    #         # have 2 IMUs, but now only use one
    #         imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i])).readlines())])
    #         wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #         gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #
    #         imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i])).readlines())])
    #         wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    #         gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
    # json.dump(imu_files, open(folder + '_imuexp6.json', 'w'), indent=4)
    # json.dump(wav_files, open(folder + '_wavexp6.json', 'w'), indent=4)
    # json.dump(gt_files, open(folder + '_gtexp6.json', 'w'), indent=4)

    #
    # transfer_function, variance, noise = read_transfer_function('transfer_function')
    # transfer_function = transfer_function_generator(transfer_function)
    # variance = transfer_function_generator(variance)
    # # plt.plot(transfer_function[0])
    # plt.text(60, 0, 'Frequency', ha='center', fontsize=50)
    # plt.text(-5, 6, 'Response', va='center', rotation='vertical', fontsize=50)
    # plt.show()
    # for i in range(N):
    #     plt.plot(transfer_function[i])
    # plt.show()
    # fig, axs = plt.subplots(2, sharex=True)
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0.05)
    # axs[0].plot(transfer_function[0])
    # for i in range(8):
    #     axs[1].plot(transfer_function[i])
    # fig.text(0.5, 0.02, 'Frequency', ha='center', fontsize=50)
    # fig.text(0.02, 0.5, 'Response', va='center', rotation='vertical', fontsize=50)
    # plt.show()

    # for i in range(N):
    #     fig, axs = plt.subplots(2, 2)
    #     axs[0, 0].imshow(hist[i])
    #     axs[0, 1].imshow(bins[i])
    #     axs[1, 0].plot(noise_hist[i])
    #     axs[1, 1].plot(transfer_function[i])
    #     plt.show()

    #
    BATCH_SIZE = 1
    dataset_train = IMUSPEECHSet('clean_imuexp6.json', 'clean_wavexp6.json', minmax=(0.012, 0.002))
    #dataset_train = NoisyCleanSet(transfer_function, variance, noise, 'speech100.json', alpha=(6, 0.012, 0.0583))
    # dataset_train = NoisyCleanSet(transfer_function, variance,noise,'devclean.json', alpha=(31.53, 0.00185))
    loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    # try the best loss to describe
    Loss = nn.L1Loss()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(dtype=torch.float), y.to(dtype=torch.float)
        #print(weighted_loss(x[0, 0, :, :], y[0, 0, :, :]))
        fig, axs = plt.subplots(2)
        axs[0].imshow(x[0, 0, :, :], aspect='auto')
        axs[1].imshow(y[0, 0, :, :], aspect='auto')
        plt.show()
