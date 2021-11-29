# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss
import os
import json
import torch.utils.data as Data
import math
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from skimage.transform import resize
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
segment = 5
stride = 1
T = 15
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
def synthetic(clean):
    response = np.tile(np.expand_dims(np.exp(np.linspace(5, 0, freq_bin_high - freq_bin_low)), axis=1), (1, time_bin))
    response += 10 * (np.random.rand(freq_bin_high - freq_bin_low, time_bin) - 0.5)
    noisy = clean * response + 0.1 * np.random.rand(freq_bin_high - freq_bin_low, time_bin)
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
    data[:, :3] = np.clip(data[:, :3], -0.01, 0.01)
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
                out, sr = sf.read(file, start=offset*rate_mic, stop=(offset+duration)*rate_mic, dtype='int16')
                out = out / 32767
                if self.length:
                    out = np.pad(out, (0, duration* self.sample_rate - out.shape[-1]))
                Zxx = signal.stft(out, nperseg=self.window, noverlap=self.overlap, fs=sr)[-1]
                Zxx = Zxx[freq_bin_low:freq_bin_high, :]
                out = torch.unsqueeze(torch.from_numpy(np.abs(Zxx)), 0)
            if self.with_path:
                return out, file
            else:
                return out
class NoisyCleanSet:
    def __init__(self, json_path):
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
        with open(json_path, 'r') as f:
            clean = json.load(f)
        self.clean_set = Audioset(clean)
    def __getitem__(self, index):
        return synthetic(self.clean_set[index]), self.clean_set[index]
    def __len__(self):
        return len(self.clean_set)

class IMUSPEECHSet:
    def __init__(self, imu_path, wav_path):
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
    def __getitem__(self, index):
        return self.imu_set[index], self.wav_set[index]
    def __len__(self):
        return len(self.imu_set)

if __name__ == "__main__":
    audio_files = []
    g = os.walk(r"../dataset/train-clean-360")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[-4:] == 'flac':
                audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames, 0])
    json.dump(audio_files, open('speech360.json', 'w'), indent=4)


    # imu_files = []
    # wav_files = []
    # g = os.walk(r"../exp4")
    # for path, dir_list, file_list in g:
    #     N = int(len(file_list) / 4)
    #     for i in range(N):
    #         time_imu = float(file_list[i][:-4].split('_')[1])
    #         time_wav = float(file_list[2*N + i][:-4].split('_')[1])
    #         imu_files.append([os.path.join(path, file_list[i]),
    #                           len(open(os.path.join(path, file_list[i])).readlines()), max(time_imu - time_wav, 0)])
    #         wav_files.append([os.path.join(path, file_list[2*N + i]),
    #                           torchaudio.info(os.path.join(path, file_list[2*N + i])).num_frames, max(time_wav - time_imu, 0)])
    # json.dump(imu_files, open('imuexp4.json', 'w'), indent=4)
    # json.dump(wav_files, open('wavexp4.json', 'w'), indent=4)


    # BATCH_SIZE = 1
    # lr = 0.001
    # pad = True
    # #device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # dataset_train = IMUSPEECHSet('imuexp4.json', 'wavexp4.json')
    # #dataset_train = NoisyCleanSet('speech100.json')
    # loader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    # for step, (x, y) in enumerate(loader):
    #     fig, axs = plt.subplots(2)
    #     axs[0].imshow(x[0, 0, :, :], aspect='auto')
    #     axs[1].imshow(y[0, 0, :, :], aspect='auto')
    #     plt.show()