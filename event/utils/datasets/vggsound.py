import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
from tqdm import tqdm
import torchvision as tv
import subprocess
import multiprocessing as mp
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
                 transform_image=transform_image,
                 length=10,
                 **_):

        super(VGGSound, self).__init__()
        self.transform_image = transform_image
        meta = pd.read_csv('vggsound_small.csv')
        self.data = list()
        self.class_idx_to_label = dict()
        self.label_to_class_idx = dict()
        count = 0
        for idx, row in meta.iterrows():
            self.data.append({
                'audio': row['filename'] + '.flac',
                'vision': row['filename'] + '.mp4',
                'name': row['filename'],
                'category': row['category'],
            })
            if row['category'] not in self.label_to_class_idx:
                self.class_idx_to_label[count] = row['category']
                self.label_to_class_idx[row['category']] = count
                count += 1
        self.length = length

    def __getitem__(self, index: int):
        sample = self.data[index]
        filename_audio: str = sample['audio']
        filename_vision: str = sample['vision']

        audio, sample_rate = librosa.load(filename_audio, sr=16000, duration=self.length)
        assert len(audio) == 160000
        audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]

        image, _, _ = tv.io.read_video(filename_vision, start_pts=5, end_pts=5, pts_unit='sec')
        image = (image[0] / 255).permute(2, 0, 1)

        target = self.data[index]['category']
        target = self.label_to_class_idx[target]
        if self.transform_image is not None:
            image = self.transform_image(image)
        return audio, image, target, sample['name']

    def __len__(self) -> int:
        return len(self.data)
def csv_filter():
    meta = pd.read_csv('vggsound.csv')
    dl_list_new = []
    for idx, row in meta.iterrows():
        s = [row[0], row[1], row[2].replace(',', '')]
        dl_list_new.append(s)
    return dl_list_new
def stat(meta):
    num_class = dict()
    for idx, row in meta.iterrows():
        label = row[1]
        if label in num_class:
            num_class[label] += 1
        else:
            num_class[label] = 1
    keys = num_class.keys()
    values = sorted(num_class.values())
    print('Whole number of clips:', sum(values))
    print('Number of types:', len(keys))
    # plt.bar(range(len(keys)), values)
    # plt.show()
if __name__ == "__main__":
    def crop(data):
        idx, row = data
        name = row[0]
        input_file = name + '.mp4'
        output_file = name + '.mp4'
        subprocess.call(
            ['ffmpeg', '-y', '-i', input_file, '-filter:v', 'scale=640:-2', output_file])
    def check(data):
        data_dir = '../dataset/VggSound'
        name, time, category = data
        fname = data_dir + '/' + name + '_' + str(time)
        # if download success
        if os.path.exists(fname + '.mp4') and os.path.exists(fname + '.flac'):
            try:
                y, sr = librosa.load(fname + '.flac')
                duration = librosa.get_duration(y=y, sr=sr)
                if duration < 10 or np.max(y) < 0.1:
                    print('invalid data')
                else:
                    return fname, category
            except:
                print('invalid data')


    # data_list = csv_filter()
    # data_frame = {'filename': [], 'category': []}
    #
    # num_processes = os.cpu_count()
    # with mp.Pool(processes=num_processes) as p:
    #     vals = list(tqdm(p.imap(check, data_list), total=len(data_list)))
    # for val in vals:
    #     if val:
    #         data_frame['filename'] += [val[0]]
    #         data_frame['category'] += [val[1]]
    # df = pd.DataFrame(data=data_frame)
    # df.to_csv('vggsound_small.csv')

    meta = pd.read_csv('vggsound_small.csv', index_col=0)
    stat(meta)
    num_processes = os.cpu_count()
    for d in meta.iterrows():
        crop(d)
        break
    # with mp.Pool(processes=16) as p:
    #     vals = list(tqdm(p.imap(crop, meta.iterrows())))

