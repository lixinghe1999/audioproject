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


class VGGSound(td.Dataset):
    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 transform_audio=ToTensor1D(),
                 transform_image=None,
                 few_shot=None,
                 length=5,
                 **_):

        super(VGGSound, self).__init__()

        self.sample_rate = sample_rate

        meta = self.load_meta('vggsound_small.csv')

        if fold is None:
            fold = 5

        self.folds_to_load = set(meta['fold'])

        if fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')
        self.train = train
        self.transform_audio = transform_audio
        self.transform_image = transform_image

        if self.train:
            self.folds_to_load -= {fold}
        else:
            self.folds_to_load -= self.folds_to_load - {fold}

        self.data = list()
        self.load_data(meta, os.path.join(root, 'audio'), few_shot)
        self.class_idx_to_label = dict()
        for row in self.data:
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        self.length = length

    @staticmethod
    def load_meta(path_to_csv: str) -> pd.DataFrame:
        meta = pd.read_csv(path_to_csv)
        return meta

    def load_data(self, meta: pd.DataFrame, base_path: str, few_shot=None):
        class_count = dict()
        for idx, row in meta.iterrows():
            if row['fold'] in self.folds_to_load:
                if few_shot is None:
                    pass
                elif row['target'] in class_count:
                    if class_count[row['target']] >= few_shot:
                        continue
                self.data.append({
                    'audio': os.path.join(base_path, row['filename'] + '.mp4'),
                    'vision': os.path.join(base_path, row['filename'] + '.flac'),
                    'sample_rate': self.sample_rate,
                    'target': row['target'],
                    'category': row['category'].replace('_', ' '),
                    'fold': row['fold'],
                    'esc10': row['esc10']
                })
                if row['target'] in class_count:
                    class_count[row['target']] += 1
                else:
                    class_count[row['target']] = 1

    def __getitem__(self, index: int):
        if not (0 <= index < len(self)):
            raise IndexError

        sample = self.data[index]
        filename_audio: str = sample['audio']
        filename_vision: str = sample['vision']
        audio, sample_rate = librosa.load(filename_audio, sr=sample['sample_rate'], mono=True)
        image = tv.io.read_video(filename_vision)

        if len(audio) >= self.length * sample_rate:
            t_start = random.sample(range(len(audio) - self.length * sample_rate + 1), 1)[0]
            audio = audio[t_start: t_start + self.length * sample_rate]
        else:
            audio = np.pad(audio, (0, self.length * sample_rate - len(audio)))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        target = [self.data[index]['category']]

        if self.transform_audio is not None:
            audio = self.transform_audio(audio)
        if self.transform_image is not None:
            image = self.transform_image(image)
        return audio, image, target

    def __len__(self) -> int:
        return len(self.data)

def csv_filter(limit=40):
    with open('vggsound.csv') as f:
        lines = f.readlines()
    dl_list = [print(line) for line in lines]
    num_class = dict()
    dl_list_new = []
    for s in dl_list:
        label = s[2]
        if label in num_class:
            if num_class[label] <= limit:
                dl_list_new.append(s)
            num_class[label] += 1
        else:
            num_class[label] = 1
    return dl_list_new, list(num_class.keys())
if __name__ == "__main__":
    # generate new csv to part of the dataset
    # format: filename, fold, target (number), category(string)
    data_dir = '../dataset/VggSound'
    data_list, label_list = csv_filter()
    # data_frame = {'filename': [], 'target': [], 'category': []}
    # for data in data_list:
    #     fname = data[0] + '_' + str(data[1])
    #     # if download success
    #     if os.path.isfile(data_dir + '/' + fname + '.mp4') and os.path.isfile(data_dir + '/' + fname + '.flac'):
    #         category = data[2]
    #         target = label_list.index(category)
    #         data_frame['filename'] += [fname]
    #         data_frame['target'] += [target]
    #         data_frame['category'] += [category]
    # df = pd.DataFrame(data=data_frame)
    #
    # df.to_csv('vggsound_small.csv')

