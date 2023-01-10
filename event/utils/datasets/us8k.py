import os

import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td
import random
import sklearn.model_selection as skms

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value

class UrbanSound8K(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 mono: bool = False,
                 transform_audio=None,
                 length=5,
                 **_):

        super(UrbanSound8K, self).__init__()

        self.root = root
        self.sample_rate = sample_rate
        self.train = train
        self.random_split_seed = None

        if fold is None:
            fold = 1

        if not (1 <= fold <= 10):
            raise ValueError(f'Expected fold in range [1, 10], got {fold}')

        self.fold = fold
        self.folds_to_load = set(range(1, 11))

        if self.fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        if self.train:
            # if in training mode, keep all but test fold
            self.folds_to_load -= {self.fold}
        else:
            # if in evaluation mode, keep the test samples only
            self.folds_to_load -= self.folds_to_load - {self.fold}

        self.mono = mono

        self.transform = transform_audio

        self.data: Dict[str, Dict[str, Any]] = dict()
        self.indices = dict()
        self.load_data()

        self.class_idx_to_label = dict()
        for row in self.data.values():
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        self.length = length
    def load_data(self):
        # read metadata
        meta = pd.read_csv(
            os.path.join(self.root, 'metadata', 'UrbanSound8K.csv'),
            sep=',',
            index_col='slice_file_name'
        )
        for row_idx, (fn, row) in enumerate(meta.iterrows()):
            path = os.path.join(self.root, 'audio', 'fold{}'.format(row['fold']), fn)
            self.data[fn] = path, self.sample_rate, self.mono

        # by default, the official split from the metadata is used
        files_to_load = list()
        # if the random seed is not None, the random split is used
        if self.random_split_seed is not None:
            # given an integer random seed
            skf = skms.StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_split_seed)

            # split the US8K samples into 10 folds
            for fold_idx, (train_ids, test_ids) in enumerate(skf.split(
                    np.zeros(len(meta)), meta['classID'].values.astype(int)
            ), 1):
                # if this is the fold we want to load, add the corresponding files to the list
                if fold_idx == self.fold:
                    ids = train_ids if self.train else test_ids
                    filenames = meta.iloc[ids].index
                    files_to_load.extend(filenames)
                    break
        else:
            # if the random seed is None, use the official split
            for fn, row in meta.iterrows():
                if int(row['fold']) in self.folds_to_load:
                    files_to_load.append(fn)

        self.data = {fn: vals for fn, vals in self.data.items() if fn in files_to_load}
        self.indices = {idx: fn for idx, fn in enumerate(self.data)}

        for fn, (path, sr, mono) in self.data.items():
                self.data[fn] = {
                    'audio': path,
                    'sample_rate': sr,
                    'target': meta.loc[fn, 'classID'],
                    'category': meta.loc[fn, 'class'].replace('_', ' ').strip(' '),
                    'background': bool(meta.loc[fn, 'salience'] - 1)
                }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not (0 <= index < len(self)):
            raise IndexError
        data = self.data[self.indices[index]]
        audio = data['audio']
        audio, sample_rate = librosa.load(audio, sr=data['sample_rate'], mono=True)
        if len(audio) >= self.length * sample_rate:
            t_start = random.sample(range(len(audio) - self.length * sample_rate + 1), 1)[0]
            audio = audio[t_start: t_start + self.length * sample_rate]
        else:
            audio = np.pad(audio, (0, self.length * sample_rate - len(audio)))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        target: str = data['category']

        if self.transform is not None:
            audio = self.transform(audio)
        return audio, None, [target]

    def __len__(self) -> int:
        return len(self.data)
