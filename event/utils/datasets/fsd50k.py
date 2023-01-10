import os
import random
import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td
from typing import Optional
import torchvision as tv
class ToTensor1D(tv.transforms.ToTensor):
    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])

        return tensor_2d.squeeze_(0)
class FSD50K(td.Dataset):
    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 transform_audio=ToTensor1D(),
                 few_shot=None,
                 length=3,
                 **_):

        super(FSD50K, self).__init__()

        vocab = pd.read_csv(os.path.join(root, 'FSD50K.ground_truth', 'vocabulary.csv'), header=None)

        self.class_idx_to_label = dict()
        for idx, row in vocab.iterrows():
            idx, label, mids = row
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        self.length = length

        self.sample_rate = sample_rate
        self.train = train
        self.data = list()
        if self.train:
            meta = pd.read_csv(os.path.join(root, 'FSD50K.ground_truth', 'dev.csv'))
            self.load_data(meta, os.path.join(root, 'FSD50K.dev_audio'), few_shot)
        else:
            meta = pd.read_csv(os.path.join(root, 'FSD50K.ground_truth', 'eval.csv'))
            self.load_data(meta, os.path.join(root, 'FSD50K.eval_audio'), few_shot)
        self.transform = transform_audio

    def load_data(self, meta: pd.DataFrame, base_path: str, few_shot=None):
        class_count = dict()
        for idx, row in meta.iterrows():
            fname = row['fname']
            label = row['labels'].split(',')[0]
            target = self.label_to_class_idx[label]
            if few_shot is not None and target in class_count:
                if class_count[target] >= few_shot:
                    continue
            self.data.append({
                'audio': os.path.join(base_path, str(fname) + '.wav'),
                'sample_rate': self.sample_rate,
                'target': target,
                'category': label,
            })
            if target in class_count:
                class_count[target] += 1
            else:
                class_count[target] = 1

    def __getitem__(self, index: int):
        if not (0 <= index < len(self)):
            raise IndexError

        sample = self.data[index]
        filename: str = sample['audio']
        audio, sample_rate = librosa.load(filename, sr=sample['sample_rate'], mono=True)

        if len(audio) >= self.length * sample_rate:
            t_start = random.sample(range(len(audio) - self.length * sample_rate + 1), 1)[0]
            audio = audio[t_start: t_start + self.length * sample_rate]
        else:
            audio = np.pad(audio, (0, self.length * sample_rate - len(audio)))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        target = [self.data[index]['category']]

        if self.transform is not None:
            audio = self.transform(audio)
        return audio, None, target

    def __len__(self) -> int:
        return len(self.data)