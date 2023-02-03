'''
1. split video into vision + audio
3. Dataset Class for Epic_kitchen
'''
import os
import numpy as np
import argparse
import ffmpeg
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
            with open(root+d, "r") as f:
                self.data += f.readlines()
        self.class_idx_to_label = dict()
        self.label_to_class_idx = dict()
        with open(root + 'classInd.txt', 'r') as f:
            for line in f.readlines():
                idx, label = line.rstrip().split()
                self.class_idx_to_label[int(idx)-1] = label
                self.label_to_class_idx[label] = int(idx)-1
        self.root = root
        self.sample_rate = sample_rate
        self.length = length
        self.transform_audio = transform_audio
        self.transform_image = transform_image
        self.length = length

    def __getitem__(self, index: int):
        row = self.data[index].rstrip()
        fname_video = self.root + row
        fname_audio = self.root + row[:-3] + 'wav'
        target = [row.split('/')[0]]
        vid = ffmpeg.probe(fname_video)
        center = float(vid['streams'][0]['duration']) / 2
        image, _, _ = tv.io.read_video(fname_video, start_pts=center, end_pts=center, pts_unit='sec')
        image = (image[0] / 255).permute(2, 0, 1)
        # UCF-101 has unstable audio
        audio = None
        # if os.path.isfile(fname_audio):
        #     audio, sample_rate = librosa.load(fname_audio, sr=self.sample_rate, offset=center - self.length/2, duration=self.length)
        #     audio = np.pad(audio, (0, self.length * self.sample_rate - len(audio)))
        #     audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]
        # else:
        #     audio = np.zeros((1, self.sample_rate * self.length))
        if self.transform_audio is not None and audio is not None:
            audio = self.transform_audio(audio)
        if self.transform_image is not None and image is not None:
            image = self.transform_image(image)
        return audio, image, target, row
    def __len__(self) -> int:
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Directory of UCF videos with audio')
    parser.add_argument('--sample_rate', default='44100', help='Rate to resample audio')

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.avi'):
                ffmpeg_extraction(os.path.join(root, f),
                                  os.path.join(root,
                                               os.path.splitext(f)[0] + '.wav'),
                                  args.sample_rate)