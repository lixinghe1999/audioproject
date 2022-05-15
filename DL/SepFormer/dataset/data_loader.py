from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import json
import math


class MyDataset(Dataset):
    def __init__(self, data_dir, sr, duration=5, stride=5):

        super(MyDataset, self).__init__()

        self.data_dir = data_dir
        self.sr = sr
        self.duration = duration
        self.stride = stride

        file = ["mix.json", "s1.json", "s2.json"]

        self.mix_dir = os.path.join(data_dir, file[0])
        with open(self.mix_dir, 'r') as f:
            self.mix_list = json.load(f)

        self.s1_dir = os.path.join(data_dir, file[1])
        with open(self.mix_dir, 'r') as f:
            self.s1_list = json.load(f)

        self.s2_dir = os.path.join(data_dir, file[2])
        with open(self.mix_dir, 'r') as f:
            self.s2_list = json.load(f)

        self.num_examples = []
        for file in self.mix_list:
            if file[1] < duration * sr:
                examples = 0
            else:
                examples = (file[1] - self.duration * self.sr) // (self.stride *self.sr) + 1
            self.num_examples.append(examples)
    def __len__(self):
        #return sum(self.num_examples)
        return 20
    def __getitem__(self, index):
        for mix_path, s1_path, s2_path, examples in zip(self.mix_list, self.s1_list, self.s2_list, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            offset = self.stride * index
            duration = self.duration
            mix_data, _ = librosa.load(mix_path[0], offset=offset, duration=duration, sr=self.sr)
            length = len(mix_data)

            s1_data, _ = librosa.load(path=s1_path[0], offset=offset, duration=duration, sr=self.sr)
            s2_data, _ = librosa.load(path=s2_path[0], offset=offset, duration=duration, sr=self.sr)
            s_data = np.stack((s1_data, s2_data), axis=0)
            return mix_data, length, s_data


