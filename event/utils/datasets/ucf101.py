'''
1. split video into vision + audio
3. Dataset Class for Epic_kitchen
'''
import pickle
import os
import random
import librosa
import numpy as np
from datetime import datetime
import argparse
import torch.utils.data as td
import torchvision as tv
import subprocess
transform_image = tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
class ToTensor1D(tv.transforms.ToTensor):
    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        return tensor_2d.squeeze_(0)

def ffmpeg_extraction(input_video, output_sound, sample_rate):
    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-ac', '1', '-ar', sample_rate,
                      output_sound]
    subprocess.call(ffmpeg_command)

class UCF101(td.Dataset):
    def __init__(self,
                 root: str = '../dataset/UCF101/',
                 sample_rate: int = 44100,
                 train = False,
                 transform_audio=ToTensor1D(),
                 transform_image=transform_image,
                 length=5,
                 **_):
        super(UCF101, self).__init__()
        if train:
            data = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']
        else:
            data = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt']
        self.data = []
        for d in data:
            with open(root + d, 'rb') as f:
                self.data += f.readlines()
        self.class_idx_to_label = dict()
        self.label_to_class_idx = dict()
        with open(root + 'classInd.txt', 'rb') as f:
            for line in f.readlines():
                self.class_idx_to_label[line[0]] = line[1]
                self.label_to_class_idx[line[1]] = line[0]
        self.root = root
        self.sample_rate = sample_rate
        self.length = length
        self.transform_audio = transform_audio
        self.transform_image = transform_image
        self.length = length

    def __getitem__(self, index: int):
        row = self.data[index]
        print(row)
        fname_video = self.root + row[0] + '.avi'
        fname_audio = self.root + row[1] + '.wav'
        target = [self.class_idx_to_label[row[1]]]
        reader = tv.io.VideoReader(fname_video, "video")
        metadata = reader.get_metadata()
        print(metadata)
        # reader.seek(center)
        start = (datetime.strptime(row['start_timestamp'], '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds()
        stop = (datetime.strptime(row['stop_timestamp'], '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds()
        center = (start + stop) / 2
        image, _, _ = tv.io.read_video(fname_video, start_pts=center, end_pts=center, pts_unit='sec')
        image = (image[0] / 255).permute(2, 0, 1)
        audio, sample_rate = librosa.load(fname_audio, sr=self.sample_rate, offset=center - self.length/2, duration=self.length)


        if len(audio) >= self.length * self.sample_rate:
            t_start = random.sample(range(len(audio) - self.length * self.sample_rate + 1), 1)[0]
            audio = audio[t_start: t_start + self.length * self.sample_rate]
        else:
            audio = np.pad(audio, (0, self.length * self.sample_rate - len(audio)))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        if self.transform_audio is not None:
            audio = self.transform_audio(audio)
        if self.transform_image is not None:
            image = self.transform_image(image)
        return audio, image, target

    def __len__(self) -> int:
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Directory of UCF videos with audio')
    parser.add_argument('output_dir', help='Directory of UCF videos with audio')
    parser.add_argument('--sample_rate', default='44100', help='Rate to resample audio')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.avi'):
                ffmpeg_extraction(os.path.join(root, f),
                                  os.path.join(args.output_dir,
                                               os.path.splitext(f)[0] + '.wav'),
                                  args.sample_rate)