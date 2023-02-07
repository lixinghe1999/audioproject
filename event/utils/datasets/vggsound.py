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
                 root: str = '../dataset/VggSound',
                 transform_audio=ToTensor1D(),
                 transform_image=None,
                 few_shot=None,
                 length=5,
                 **_):

        super(VGGSound, self).__init__()


        self.transform_audio = transform_audio
        self.transform_image = transform_image

        self.data = list()
        meta = pd.read_csv('vggsound_small.csv')
        self.load_data(meta, root)
        self.class_idx_to_label = dict()
        for row in self.data:
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        self.length = length

    @staticmethod
    def load_data(self, meta: pd.DataFrame, base_path: str):
        for idx, row in meta.iterrows():
            self.data.append({
                'audio': os.path.join(base_path, row['filename'] + '.mp4'),
                'vision': os.path.join(base_path, row['filename'] + '.flac'),
                'sample_rate': 44100,
                'target': row['target'],
                'category': row['category'].replace('_', ' '),
                'fold': row['fold'],
                'esc10': row['esc10']
            })

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
    meta = pd.read_csv('vggsound.csv')
    num_class = dict()
    dl_list_new = []
    for idx, row in meta.iterrows():
        s = [row[0], row[1], row[2].replace(',', '')]
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
    data_frame = {'filename': [], 'target': [], 'category': []}
    for data in data_list:
        fname = data[0] + '_' + str(data[1])
        # if download success
        if os.path.isfile(data_dir + '/' + fname + '.mp4') and os.path.isfile(data_dir + '/' + fname + '.flac'):
            category = data[2]
            target = label_list.index(category)
            data_frame['filename'] += [fname]
            data_frame['target'] += [target]
            data_frame['category'] += [category]
    df = pd.DataFrame(data=data_frame)
    df.to_csv('vggsound_small.csv')

