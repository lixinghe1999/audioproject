import os
import warnings
import multiprocessing as mp

import tqdm
import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from utils.transforms import ToTensor1D

class ESC50(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 transform_audio=ToTensor1D(),
                 target_transform=None,
                 **_):

        super(ESC50, self).__init__()

        self.sample_rate = sample_rate

        meta = self.load_meta(os.path.join(root, 'meta', 'esc50.csv'))

        if fold is None:
            fold = 5

        self.folds_to_load = set(meta['fold'])

        if fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')
        self.train = train
        self.transform = transform_audio

        if self.train:
            self.folds_to_load -= {fold}
        else:
            self.folds_to_load -= self.folds_to_load - {fold}

        self.data = list()
        self.load_data(meta, os.path.join(root, 'audio'))
        self.class_idx_to_label = dict()
        for row in self.data:
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}

        # self.target_transform = target_transform

    @staticmethod
    def load_meta(path_to_csv: str) -> pd.DataFrame:
        meta = pd.read_csv(path_to_csv)

        return meta

    @staticmethod
    def _load_worker(idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]

        wav = wav.T * 32768.0

        return idx, sample_rate, wav.astype(np.float32)

    def load_data(self, meta: pd.DataFrame, base_path: str):
        items_to_load = dict()

        for idx, row in meta.iterrows():
            if row['fold'] in self.folds_to_load:
                items_to_load[idx] = os.path.join(base_path, row['filename']), self.sample_rate

        items_to_load = [(idx, path, sample_rate) for idx, (path, sample_rate) in items_to_load.items()]
        for item in items_to_load:
            idx, path, sample_rate = item
            row = meta.loc[idx]
            self.data.append({
                        'audio': path,
                        'sample_rate': sample_rate,
                        'target': row['target'],
                        'category': row['category'].replace('_', ' '),
                        'fold': row['fold'],
                        'esc10': row['esc10']
                    })
        return items_to_load

    def __getitem__(self, index: int):
        if not (0 <= index < len(self)):
            raise IndexError

        sample = self.data[index]
        filename: str = sample['audio']
        audio, sample_rate = librosa.load(filename, sr=sample['sample_rate'], mono=True)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        target = [self.data[index]['category']]

        if self.transform is not None:
            audio = self.transform(audio)
        return audio, None, target

    def __len__(self) -> int:
        #return len(self.data)
        return 16