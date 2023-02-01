'''
1. split video into vision + audio
2. downsample image to 224 by Bicubic
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
            #tv.transforms.ToTensor(),
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

class EPIC_Kitchen(td.Dataset):
    def __init__(self,
                 root: str = '../dataset/epic_kitchen/',
                 sample_rate: int = 44100,
                 transform_audio=ToTensor1D(),
                 transform_image=transform_image,
                 length=5,
                 **_):
        super(EPIC_Kitchen, self).__init__()
        with open('EPIC_train_action_labels.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data[self.data['participant_id'].isin(['P02'])]
        self.root = root
        self.sample_rate = sample_rate
        self.length = length
        self.transform_audio = transform_audio
        self.transform_image = transform_image
        self.class_idx_to_label = dict()
        self.label_to_class_idx = dict()
        idx = 0
        for _, row in self.data.iterrows():
            label = row['narration']
            if label not in self.label_to_class_idx:
                self.class_idx_to_label[idx] = label
                self.label_to_class_idx[label] = idx
                idx += 1
        self.length = length

    def __getitem__(self, index: int):
        row = self.data.iloc[index].to_dict()
        fname_video = self.root + row['participant_id'] + '/' + row['video_id'] + '.MP4'
        fname_audio = self.root + row['participant_id'] + '/' + row['video_id'] + '.wav'
        # target = [row['narration']]
        target = [row['noun'].replace(':', ' ')]
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
        return len(self.data.index)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Directory of EPIC videos with audio')
    parser.add_argument('output_dir', help='Directory of EPIC videos with audio')
    parser.add_argument('--sample_rate', default='44100', help='Rate to resample audio')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.MP4'):
                ffmpeg_extraction(os.path.join(root, f),
                                  os.path.join(args.output_dir,
                                               os.path.splitext(f)[0] + '.wav'),
                                  args.sample_rate)