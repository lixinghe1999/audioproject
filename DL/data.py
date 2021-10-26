# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os

import math
import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
import librosa
import scipy.signal as signal

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

class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None, window=2048, overlap=1024):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.window = window
        self.overlap = overlap
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length*self.sample_rate:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length * self.sample_rate) / self.stride/self.sample_rate) + 1)
            else:
                examples = (file_length - self.length * self.sample_rate) // (self.stride*self.sample_rate) + 1
            self.num_examples.append(examples)


    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            duration = 0
            offset = 0
            if self.length:
                offset = self.stride * index
                duration = self.length
            out, sr = librosa.load(file, sr=None, offset=offset, duration = duration, mono=True)
            if self.length:
                out = np.pad(out, (0, duration*self.sample_rate - out.shape[-1]))
            Zxx = signal.stft(out, nperseg=self.window, noverlap=self.overlap, fs=sr)[-1]
            freq_bin_max = int(1600 / self.sample_rate * (self.window/2 + 1))
            freq_bin_min = int(100 / self.sample_rate * (self.window / 2 + 1))
            Zxx = Zxx[freq_bin_min:freq_bin_max, :]
            #out = np.dstack((np.real(Zxx), np.imag(Zxx)))
            #out = torch.from_numpy(np.transpose(out,(2, 0, 1)))
            out = torch.unsqueeze(torch.from_numpy(np.abs(Zxx)), 0)
            if self.with_path:
                return out, file
            else:
                return out
class NoisyCleanSet:
    def __init__(self, json_dir, length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)
    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)

def find_audio_files(path):
    audio_files = []
    for file in os.listdir(path):
        if file[-4:] == '.txt':
            audio_files.append([path + file, len(open(path + file, 'r').readlines())])
        else:
            audio_files.append([path + file, torchaudio.info(path + file).num_frames])
    return audio_files

if __name__ == "__main__":
    meta = []
    for path in ['../exp2/HE/imu2/', '../exp2/HE/imu1/', '../exp2/HOU/imu2/', '../exp2/HOU/imu1/']:
    #for path in ['../exp2/HE/mic2/', '../exp2/HE/mic1/', '../exp2/HOU/mic2/', '../exp2/HOU/mic1/']:
        meta += find_audio_files(path)
    json.dump(meta, open('dataset/noisy.json', 'w'), indent=4)