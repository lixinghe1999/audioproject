import os
import librosa
import numpy as np
import pandas as pd
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
        for idx, row in meta.iterrows():
            self.data.append({
                'audio': row['filename'] + '.flac',
                'vision': row['filename'] + '.mp4',
                'name': row['filename'],
                'category': row['category'],
            })
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

    def __getitem__(self, index: int):
        sample = self.data[index]
        filename_audio: str = sample['audio']
        filename_vision: str = sample['vision']

        # vid = ffmpeg.probe(filename_vision)
        # center = float(vid['streams'][0]['duration']) / 2

        audio, sample_rate = librosa.load(filename_audio, sr=16000, duration=self.length)
        assert len(audio) == 160000
        # if len(audio) > self.length * sample_rate:
        #     rand_start = np.random.randint(0, len(audio) - self.length * sample_rate)
        #     audio = audio[rand_start: rand_start + self.length * sample_rate]
        # else:
        #     audio = np.pad(audio, (0, self.length * sample_rate - len(audio)))
        audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]

        image, _, _ = tv.io.read_video(filename_vision, start_pts=5, end_pts=5, pts_unit='sec')
        image = (image[0] / 255).permute(2, 0, 1)
        target = self.data[index]['category']
        if self.transform_image is not None:
            image = self.transform_image(image)
        return audio, image, target, sample['name']

    def __len__(self) -> int:
        return len(self.data)
def csv_filter(limit=10):
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
    def crop(input_file, output_file):
        subprocess.call(
            ['ffmpeg', '-y', '-i', input_file, '-filter:v', 'scale=640:-2', output_file])
    def check(data):
        nonlocal data_frame
        name, time, category = data
        fname = data_dir + '/' + name + '_' + str(time)
        # if download success
        if os.path.isfile(fname + '.mp4') and os.path.isfile(fname + '.flac'):
            try:
                y, sr = librosa.load(fname + '.flac')
                duration = librosa.get_duration(y=y, sr=sr)
                if duration < 10 or np.max(y) < 0.1:
                    print('invalid data')
                else:
                    # crop(fname + '.mp4', fname + '.mp4')
                    # target = label_list.index(category)
                    data_frame['filename'] += [fname]
                    # data_frame['target'] += [target]
                    data_frame['category'] += [category]
            except:
                print('invalid data')
    # generate new csv to part of the dataset
    # format: filename, fold, category(string)
    data_dir = '../dataset/VggSound'
    data_list, label_list = csv_filter()
    data_frame = {'filename': [], 'target': [], 'category': []}

    num_processes = os.cpu_count()
    # with mp.Pool(processes=8) as pool:
    #     for _ in pool.starmap(
    #             func=check,
    #             iterable=tqdm(data_list),
    #     ):
    #         pass
    with mp.Pool(processes=8) as p:
        list(tqdm(p.imap(check, data_list), total=len(data_list)))
    df = pd.DataFrame(data=data_frame)
    df.to_csv('vggsound_small.csv')

