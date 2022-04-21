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
import pickle
import argparse
import torchaudio
#torchaudio.set_audio_backend("sox_io")
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
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
stride = 5
N = len(os.listdir('../transfer_function'))


freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(length * rate_mic/(seg_len_mic-overlap_mic)) + 1
def spectrogram(audio):
    Zxx = signal.stft(audio, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate_mic)[-1]
    Zxx = Zxx[: 8 * freq_bin_high, :]
    Zxx = np.expand_dims(Zxx, 0)
    return Zxx

def read_transfer_function(path):
    npzs = os.listdir(path)
    transfer_function = np.zeros((len(npzs), freq_bin_high))
    variance = np.zeros((len(npzs), freq_bin_high))
    for i in range(len(npzs)):
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
    f_norm = f / np.max(f)
    v_norm = variance[index, :] / np.max(f)
    response = np.tile(np.expand_dims(f_norm, axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, v_norm, (freq_bin_high))
    noisy = clean[:, :freq_bin_high, :] * response
    background_noise = noise_extraction()
    noisy += 2 * background_noise
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
    Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)[-1]
    Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    Zxx = np.expand_dims(Zxx, 0)
    return Zxx


class Audioset:
    def __init__(self, files=None, pad=False, imu=False):
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
            return out, file
class NoisyCleanSet:
    def __init__(self, transfer_function, variance, json_path1, json_path2, phase=False, alpha=(1, 1), ratio=1):
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
        self.ratio = ratio
        self.length = len(self.noise_set)

    def __getitem__(self, index):
        speech, _ = self.clean_set[index]
        noise, _ = self.noise_set[np.random.randint(0, self.length)]
        ratio = np.max(speech) / np.max(noise)
        noise = noise * ratio * (np.random.random()/5 + 0.5) + speech
        #noise = noise * (np.random.random()/5 + 0.5) + speech
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
        return int(len(self.clean_set) * self.ratio)
class IMUSPEECHSet:
    def __init__(self, imu_path, wav_path, noise_path, ratio=1, simulate=True, person=['he'], phase=False, minmax=(1, 1, 1)):
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
            if x1.replace('\\', '/').split('/')[2] in person:
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
        self.ratio = ratio
        self.length = len(self.noise_set)
    def __getitem__(self, index):
        imu, _ = self.imu_set[index]
        speech, file = self.wav_set[index]
        if self.simulate:
            noise, _ = self.noise_set[np.random.randint(0, self.length)]
            ratio = np.max(speech) / np.max(noise)
            noise = noise * ratio * (np.random.random() / 5 + 0.5) + speech
        else:
            noise, _ = self.noise_set[index]
        speech = spectrogram(speech)
        noise = spectrogram(noise)
        if self.phase:
            return int(file.split('\\')[2][-1]), torch.from_numpy(imu / self.minmax[0]), torch.from_numpy(noise / self.minmax[1]), torch.from_numpy(
                speech / self.minmax[2])
        else:
            return torch.from_numpy(imu / self.minmax[0]), torch.from_numpy(np.abs(noise) / self.minmax[1]), torch.from_numpy(np.abs(speech)/ self.minmax[2])
    def __len__(self):
        return int(len(self.imu_set) * self.ratio)
def norm(name, jsons, simulate, candidate):
    dict = {}
    for target in candidate:
        x_mean = []
        noise_mean = []
        y_mean = []
        dataset_train = IMUSPEECHSet(jsons[0], jsons[1], jsons[2], simulate=simulate, person=[target], minmax=(1, 1, 1))
        loader = Data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)
        for step, (x, noise, y) in enumerate(loader):
            x_mean.append(np.max(x.numpy(), axis=(0, 1, 2, 3)))
            noise_mean.append(np.max(noise.numpy(), axis=(0, 1, 2, 3)))
            y_mean.append(np.max(y.numpy(), axis=(0, 1, 2, 3)))
        dict[target] = (np.mean(x_mean), np.mean(noise_mean), np.mean(y_mean))
        print(target)
        print(dict[target])
    file = open(name, "wb")
    pickle.dump(dict, file)
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-generate pretrain json, 1-generate json, \
                             2-normalization, 3-visualzation')
    args = parser.parse_args()
    if args.mode == 0:
        audio_files = []
        for path in [r"../dataset/background", r"../dataset/music", r"../dataset/test-clean"]:
        #for path in [r"../dataset/train-clean-100"]:
        #for path in [r"../dataset/dev-clean"]:
            g = os.walk(path)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name[-3:] not in ['txt', 'mp3']:
                        audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
        json.dump(audio_files, open('background.json', 'w'), indent=4)
    elif args.mode == 1:
        person = ["liang", "he", "hou", "shi", "shuai", "wu", "yan", "jiang", "1", "2", "3", "4", "5", "6", "7", "8", "airpod", "galaxy", 'freebud']
        for folder, name in zip(['noise_train', 'train', 'mobile', 'clean', 'noise'],
                               ['noise_train', 'clean_train', 'mobile', 'clean', 'noise']):
        #person = ['office', 'corridor', 'stair']
        #for folder, name in zip(['noise'], ['field']):
            g = os.walk(r"../exp7")
            imu_files = []
            wav_files = []
            gt_files = []
            for path, dir_list, file_list in g:
                N = int(len(file_list) / 4)
                if N > 0:
                    p = path.replace('\\', '/').split('/')
                    if p[-1] != folder or p[2] not in person:
                        # only collect some type of data
                        continue
                    imu1 = file_list[: N]
                    imu2 = file_list[N: 2 * N]
                    gt = file_list[2 * N: 3 * N]
                    wav = file_list[3 * N:]

                    for i in range(N):
                        imu_files.append([os.path.join(path, imu1[i]), len(open(os.path.join(path, imu1[i])).readlines())])
                        wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
                        gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])

                        imu_files.append([os.path.join(path, imu2[i]), len(open(os.path.join(path, imu2[i])).readlines())])
                        wav_files.append([os.path.join(path, wav[i]), torchaudio.info(os.path.join(path, wav[i])).num_frames])
                        gt_files.append([os.path.join(path, gt[i]), torchaudio.info(os.path.join(path, gt[i])).num_frames])
                json.dump(imu_files, open(name + '_imuexp7.json', 'w'), indent=4)
                json.dump(wav_files, open(name + '_wavexp7.json', 'w'), indent=4)
                json.dump(gt_files, open(name + '_gtexp7.json', 'w'), indent=4)
    elif args.mode == 2:
        # field = ['office', 'corridor', 'stair']
        #
        # norm('field_paras.pkl', ['field_imuexp7.json', 'field_gtexp7.json', 'field_wavexp7.json'], False, field)
        #
        # norm('field_train_paras.pkl', ['field_train_imuexp7.json', 'field_train_gtexp7.json', 'field_train_wavexp7.json'],False, ['canteen', 'station'])

        candidate_all = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8", "airpod", "galaxy", 'freebud']

        candidate = ['yan', 'he', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8"]

        norm('noise_train_paras.pkl', ['noise_train_imuexp7.json', 'noise_train_gtexp7.json', 'noise_train_wavexp7.json'], False, candidate_all)

        norm('clean_train_paras.pkl', ['clean_train_imuexp7.json', 'clean_train_wavexp7.json', 'clean_train_wavexp7.json'], True, candidate)

        norm('noise_paras.pkl', ['noise_imuexp7.json', 'noise_gtexp7.json', 'noise_wavexp7.json'], False, candidate_all)

        norm('clean_paras.pkl', ['clean_imuexp7.json', 'clean_wavexp7.json', 'clean_wavexp7.json'], True, ['he', 'hou'])

        norm('mobile_paras.pkl', ['mobile_imuexp7.json', 'mobile_wavexp7.json', 'mobile_wavexp7.json'], True, ['he', 'hou'])
    elif args.mode == 3:
        transfer_function, variance = read_transfer_function('../transfer_function')
        dataset_train = NoisyCleanSet(transfer_function, variance, 'speech100.json', 'devclean.json', alpha=(1, 0.1, 0.1, 0.1))
        #dataset_train = NoisyCleanSet(transfer_function, variance, 'devclean.json', 'background.json', alpha=(1, 0.1, 0.1, 0.1))
        loader = Data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=False)
        x_mean = []
        noise_mean = []
        y_mean = []
        for step, (x, noise, y) in enumerate(loader):
            x = x.numpy()[0, 0]
            noise = noise.numpy()[0, 0]
            y = y.numpy()[0, 0]
            print(np.max(x), np.max(noise), np.max(y))
            fig, axs = plt.subplots(3, 1)
            axs[0].imshow(x, aspect='auto')
            axs[1].imshow(noise[:2 * freq_bin_high, :], aspect='auto')
            axs[2].imshow(y[:2 * freq_bin_high, :], aspect='auto')
            plt.show()
    else:
        transfer_function, variance = read_transfer_function('../transfer_function')
        for i in range(19):
            index = np.random.randint(0, N)
            plt.plot(transfer_function[index, :])
            plt.show()


