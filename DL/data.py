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
segment = 4
stride = 1
T = 15
N = 50
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
Loss_weight = torch.tile(torch.unsqueeze(np.exp(torch.linspace(0, 1, freq_bin_high-freq_bin_low)), dim=1), (1, time_bin))
def weighted_loss(input, target, device='cpu'):
    return (Loss_weight.to(device=device) * torch.abs(input - target)).mean()
def transfer_function_generator(transfer_function):
    n, m = np.shape(transfer_function)
    x = np.linspace(1, m, m)
    y = np.linspace(1, n, n)
    x1 = np.linspace(1, m, m)
    y1 = np.linspace(1, n, N)
    f = interp2d(x, y, transfer_function, kind='cubic')
    Z = f(x1, y1)
    return np.clip(Z, 0, 2)
def histogram(hist, bins):
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)
    if cdf[-1] == 0:
        return bin_midpoints, cdf+1
    cdf = cdf / cdf[-1]
    return bin_midpoints, cdf
def read_transfer_function(path):
    npzs = os.listdir(path)
    transfer_function = np.zeros((len(npzs), freq_bin_high - freq_bin_low))
    variance = np.zeros((len(npzs), freq_bin_high - freq_bin_low))
    hist = np.zeros((len(npzs), freq_bin_high - freq_bin_low, 50))
    bins = np.zeros((len(npzs), freq_bin_high - freq_bin_low, 50))
    noise_hist = np.zeros((len(npzs), 50))
    noise_bins = np.zeros((len(npzs), 50))
    noise = np.zeros((len(npzs), 2))
    for i in range(len(npzs)):
        npz = np.load(path + '/' + npzs[i])
        transfer_function[i, :] = npz['response']
        variance[i, :] = npz['variance']
        for j in range(freq_bin_high - freq_bin_low):
            bins[i, j, :], hist[i, j, :] = histogram(npz['hist'][j, :], npz['bins'][j, :])
        noise_bins[i, :], noise_hist[i, :] = histogram(npz['noise_hist'], npz['noise_bins'])
        noise[i, :] = npz['noise']
    return transfer_function, variance, hist, bins, noise_hist, noise_bins, noise
def randomness_bins(bins):
    #print(np.diff(bins, axis=1).shape)
    bins_diff = np.concatenate((np.diff(bins, axis=1), np.zeros((freq_bin_high-freq_bin_low, 1))), axis=1)
    bins = bins + 2 * (np.random.rand(freq_bin_high-freq_bin_low, 50)-0.5) * bins_diff
    return bins
def noise_extraction():
    noise_list = os.listdir('../dataset/noise/')
    index = np.random.randint(0, len(noise_list))
    noise_clip = np.load('../dataset/noise/' + noise_list[index])
    index = np.random.randint(0, noise_clip.shape[1] - time_bin)
    return noise_clip[:, index:index + time_bin]
def transfer_function_sampler(transfer_function, variance, hist, bins, index):
    response = np.tile(np.expand_dims(transfer_function[index, :], axis=1), (1, time_bin))
    # for i in range(freq_bin_high-freq_bin_low):
    #     response[i, :] += bins[index, i, np.searchsorted(hist[index, i, :], np.random.rand(time_bin))]
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, variance[index, :], (freq_bin_high-freq_bin_low))
    return response
def synthetic(clean, transfer_function, variance, hist, bins, noise_hist, noise_bins, noise):
    index = np.random.randint(0, N)
    #bins[index] = randomness_bins(bins[index])
    noisy = clean * transfer_function_sampler(transfer_function, variance, hist, bins, index)
    #noisy = clean
    # bins_diff = np.append(np.diff(noise_bins[index]), 0)
    # for j in range(time_bin):
    #     new_noise_bins = noise_bins[index, :] + 2 * (np.random.rand(50) - 0.5) * bins_diff
    #
    #     select = np.searchsorted(noise_hist[index, :], np.random.rand((freq_bin_high-freq_bin_low)))
    #     print(new_noise_bins[select])
    #     noisy[:, j] += new_noise_bins[select]
    #noisy += np.random.normal(noise[index, 0], noise[index, 1], (freq_bin_high-freq_bin_low, time_bin))
    background_noise = noise_extraction()
    #noisy += torch.max(clean) / np.max(background_noise) * background_noise * 0.4
    noisy += 1 * background_noise
    return noisy

def read_data(file, seg_len=256, overlap=224, rate=1600, offset=None, duration=None):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    if offset and duration:
        data = data[offset*rate:(offset + duration) * rate, :]
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
    data[:, :3] = np.clip(data[:, :3], -0.02, 0.02)
    f, t, Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, window="hamming", axis=0)
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return Zxx
def imu_resize(time_diff, imu1):
    shift = round(time_diff * rate_mic / (seg_len_mic - overlap_mic))
    Zxx_resize = np.roll(resize(imu1, (freq_bin_high, time_bin)), shift, axis=1)
    #Zxx_resize[:, :0] = 0
    Zxx_resize = Zxx_resize[freq_bin_low:, :]
    return Zxx_resize

class Audioset:
    def __init__(self, files=None, with_path=False, pad=True, imu=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.imu = imu
        if self.imu:
            self.length = segment
            self.stride = stride
            self.with_path = with_path
            self.sample_rate = rate_imu
            self.window = seg_len_imu
            self.overlap = overlap_imu
        else:
            self.length = segment
            self.stride = stride
            self.with_path = with_path
            self.sample_rate = rate_mic
            self.window = seg_len_mic
            self.overlap = overlap_mic

        for file, file_length, _ in self.files:
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
        for (file, _, time_diff), examples in zip(self.files, self.num_examples):
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
                Zxx = imu_resize(time_diff, Zxx)
                out = torch.unsqueeze(torch.from_numpy(np.abs(Zxx)), 0)
            else:
                out, sr = librosa.load(file, offset=offset, duration=duration, sr=None)
                #out *= 32767
                #out, sr = sf.read(file, start=offset*rate_mic, stop=(offset+duration)*rate_mic, dtype='int16')
                #out = out / 32767
                if self.length:
                    out = np.pad(out, (0, duration * self.sample_rate - out.shape[-1]))
                Zxx = signal.stft(out, nperseg=self.window, noverlap=self.overlap, fs=sr)[-1]
                Zxx = Zxx[freq_bin_low:freq_bin_high, :]
                out = torch.unsqueeze(torch.from_numpy(np.abs(Zxx)), 0)
            if self.with_path:
                return out, file
            else:
                return out
class NoisyCleanSet:
    def __init__(self, transfer_function, variance, hist, bins, noise_hist, noise_bins, noise, json_path, alpha=(1, 1)):
        """__init__. n

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        #noisy_json = os.path.join(json_dir, 'noisy.json')
        # with open(noisy_json, 'r') as f:
        #     noisy = json.load(f)
        # self.noisy_set = Audioset(noisy, **kw)
        with open(json_path, 'r') as f:
            clean = json.load(f)
        self.clean_set = Audioset(clean)
        self.transfer_function = transfer_function
        self.variance = variance
        self.hist = hist
        self.bins = bins
        self.noise_hist = noise_hist
        self.noise_bins = noise_bins
        self.noise = noise
        self.alpha = alpha
    def __getitem__(self, index):
        return synthetic(self.clean_set[index]/self.alpha[0], self.transfer_function, self.variance, self.hist, self.bins, self.noise_hist,
                         self.noise_bins, self.noise)/self.alpha[1], self.clean_set[index]/self.alpha[0]/self.alpha[1]
    def __len__(self):
        return len(self.clean_set)
class IMUSPEECHSet:
    def __init__(self, imu_path, wav_path, minmax=(1, 1)):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        #noisy_json = os.path.join(json_dir, 'noisy.json')
        # with open(noisy_json, 'r') as f:
        #     noisy = json.load(f)
        # self.noisy_set = Audioset(noisy, **kw)
        with open(imu_path, 'r') as f:
            imu = json.load(f)
        with open(wav_path, 'r') as f:
            wav = json.load(f)
        self.imu_set = Audioset(imu, imu=True)
        self.wav_set = Audioset(wav, imu=False)
        self.minmax = minmax
    def __getitem__(self, index):
        return self.imu_set[index]/self.minmax[0], self.wav_set[index]/self.minmax[1]
    def __len__(self):
        return len(self.imu_set)

if __name__ == "__main__":
    # audio_files = []
    # g = os.walk(r"../dataset/dev-clean")
    # for path, dir_list, file_list in g:
    #     for file_name in file_list:
    #         if file_name[-4:] == 'flac':
    #             audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames, 0])
    # json.dump(audio_files, open('devclean.json', 'w'), indent=4)


    # imu_files = []
    # wav_files = []
    # g = os.walk(r"../exp4")
    # for path, dir_list, file_list in g:
    #     N = int(len(file_list) / 4)
    #     if path[-5:] != "clean":
    #         # only collect clean data
    #         continue
    #     for i in range(N):
    #         time_imu = float(file_list[i][:-4].split('_')[1])
    #         time_wav = float(file_list[2*N + i][:-4].split('_')[1])
    #         imu_files.append([os.path.join(path, file_list[i]),
    #                           len(open(os.path.join(path, file_list[i])).readlines()), max(time_imu - time_wav, 0)])
    #         wav_files.append([os.path.join(path, file_list[2*N + i]),
    #                           torchaudio.info(os.path.join(path, file_list[2*N + i])).num_frames, max(time_wav - time_imu, 0)])
    # json.dump(imu_files, open('imuexp4.json', 'w'), indent=4)
    # json.dump(wav_files, open('wavexp4.json', 'w'), indent=4)

    #
    transfer_function, variance, hist, bins, noise_hist, noise_bins, noise = read_transfer_function('transfer_function')
    #
    transfer_function = transfer_function_generator(transfer_function)
    variance = transfer_function_generator(variance)
    fig, axs = plt.subplots(2)
    for i in range(N):
        axs[0].plot(transfer_function[i])
        axs[1].plot(variance[i])
    plt.show()
    # for i in range(N):
    #     fig, axs = plt.subplots(2, 2)
    #     axs[0, 0].imshow(hist[i])
    #     axs[0, 1].imshow(bins[i])
    #     axs[1, 0].plot(noise_hist[i])
    #     axs[1, 1].plot(transfer_function[i])
    #     plt.show()


    # BATCH_SIZE = 1
    # #device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # #dataset_train = IMUSPEECHSet('imuexp4.json', 'wavexp4.json', minmax=(1, 1))
    # #dataset_train = NoisyCleanSet(transfer_function, variance, hist, bins, noise_hist, noise_bins, noise, 'speech100.json', alpha=5.93)
    # dataset_train = NoisyCleanSet(transfer_function, variance, hist, bins, noise_hist, noise_bins, noise,
    #                               'devclean.json', alpha=(5.93, 0.01))
    # loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    # # X_list = np.empty((len(dataset_train), 2))
    # # Y_list = np.empty((len(dataset_train), 2))
    # #Loss = nn.SmoothL1Loss(beta=0.1)
    # #Loss = nn.MSELoss()
    # Loss = nn.L1Loss()
    # #Loss = MS_SSIM(data_range=1, win_size=3, channel=1)
    # for step, (x, y) in enumerate(loader):
    # #     X_list[step, :] = [torch.amax(x, dim=(0, 2, 3)), torch.amin(x, dim=(0, 2, 3))]
    # #     Y_list[step, :] = [torch.amax(y, dim=(0, 2, 3)), torch.amin(y, dim=(0, 2, 3))]
    # # print(np.mean(X_list[:, 0]), np.mean(X_list[:, 1]))
    # # print(np.mean(Y_list[:, 0]), np.mean(Y_list[:, 1]))
    #     fig, axs = plt.subplots(2)
    #     x, y = x.to(dtype=torch.float), y.to(dtype=torch.float)
    #     print(Loss(x, y))
    #     print(weighted_loss(x[0, 0, :, :], y[0, 0, :, :]))
    #     #generated = synthetic(y[0, 0, :, :], transfer_function, variance, hist, bins, noise_hist, noise_bins, noise)
    #     axs[0].imshow(x[0, 0, :, :], aspect='auto')
    #     axs[1].imshow(y[0, 0, :, :], aspect='auto')
    #     #axs[2].imshow(generated, aspect='auto')
    #     plt.show()