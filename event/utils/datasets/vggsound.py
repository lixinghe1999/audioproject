import os
import random
import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
import ffmpeg
import torchvision as tv
class ToTensor1D(tv.transforms.ToTensor):
    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        return tensor_2d.squeeze_(0)
transform_image = tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
class VGGSound(td.Dataset):
    def __init__(self,
                 root: str = '../dataset/VggSound',
                 transform_audio=ToTensor1D(),
                 transform_image=transform_image,
                 length=5,
                 **_):

        super(VGGSound, self).__init__()
        self.transform_audio = transform_audio
        self.transform_image = transform_image

        self.data = list()
        meta = pd.read_csv('vggsound_small.csv')
        self.load_data(meta, root)
        self.class_idx_to_label = dict()
        self.label_to_class_idx = dict()
        idx = 0
        for row in self.data:
            label = row['category']
            if label not in self.label_to_class_idx:
                self.class_idx_to_label[idx] = label
                self.label_to_class_idx[label] = idx
                idx += 1
        self.length = length

    def load_data(self, meta: pd.DataFrame, base_path: str):
        for idx, row in meta.iterrows():
            self.data.append({
                'audio': os.path.join(base_path, row['filename'] + '.flac'),
                'vision': os.path.join(base_path, row['filename'] + '.mp4'),
                'name': row['filename'],
                'sample_rate': 44100,
                'category': row['category'],
            })

    def __getitem__(self, index: int):
        if not (0 <= index < len(self)):
            raise IndexError

        sample = self.data[index]
        filename_audio: str = sample['audio']
        filename_vision: str = sample['vision']

        vid = ffmpeg.probe(filename_vision)
        center = float(vid['streams'][0]['duration']) / 2

        audio, sample_rate = librosa.load(filename_audio, sr=None, offset=center - self.length / 2,
                                          duration=self.length)
        audio = np.pad(audio, (0, self.length * sample_rate - len(audio)))
        audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]

        image, _, _ = tv.io.read_video(filename_vision, start_pts=center, end_pts=center, pts_unit='sec')
        image = (image[0] / 255).permute(2, 0, 1)
        target = self.data[index]['category']
        if self.transform_audio is not None:
            audio = self.transform_audio(audio)
        if self.transform_image is not None:
            image = self.transform_image(image)
        return audio, image, target, sample['name']

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

