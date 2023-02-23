import os
import time
import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
import torch.utils.data as td
from tqdm import tqdm
import torchvision as tv
import torchaudio as ta
import multiprocessing as mp
transform_audio = tv.transforms.Compose([])
transform_image = tv.transforms.Compose([
            tv.transforms.Resize(256, interpolation=tv.transforms.InterpolationMode.BICUBIC),
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
                'audio': row['filename'].replace('VggSound', 'VggSound_small') + '.flac',
                'vision': row['filename'].replace('VggSound', 'VggSound_small') + '.png',
                'name': row['filename'],
                'category': row['category'],
            })
            if row['category'] not in self.label_to_class_idx:
                self.class_idx_to_label[count] = row['category']
                self.label_to_class_idx[row['category']] = count
                count += 1
        self.length = length
    def preprocessing_audio(self, audio):
        fbank = ta.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0,
                                                  frame_shift=25)
        target_length = 384
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        return fbank
    def __getitem__(self, index: int):
        sample = self.data[index]
        filename_audio: str = sample['audio']
        filename_vision: str = sample['vision']
        audio, sr = ta.load(filename_audio)
        image = tv.io.read_image(filename_vision)/255
        target = self.data[index]['category']
        target = self.label_to_class_idx[target]
        if self.transform_image is not None:
            image = self.transform_image(image)
        audio = self.preprocessing_audio(audio)
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
if __name__ == "__main__":
    def cal_norm(dataset):
        loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=True,
                                             drop_last=True, pin_memory=False)
        mean = 0
        std = 0
        for idx, batch in enumerate(tqdm(loader)):
            audio, image, text, _ = batch
            mean += torch.mean(audio)
        mean = mean/len(loader)
        for idx, batch in enumerate(tqdm(loader)):
            audio, image, text, _ = batch
            std += ((audio - mean)**2).sum()/(audio.shape[-1] * audio.shape[-2])
        std = (std/len(loader))**0.5
        print(mean, std)
    def crop(data):
        resize = tv.transforms.Resize(size=480)
        idx, row = data
        name = row[0]
        input_file = name + '.flac'
        output_file = name.replace('VggSound', 'VggSound_small') + '.flac'
        audio, sample_rate = librosa.load(input_file, sr=16000, duration=10)
        assert len(audio) == 160000
        # audio = (audio * 32768.0).astype(np.float32)[np.newaxis, :]
        if os.path.exists(output_file):
            pass
        else:
            soundfile.write(output_file, audio, sample_rate)
        input_file = name + '.mp4'
        output_file = name.replace('VggSound', 'VggSound_small') + '.png'
        if os.path.exists(output_file):
            pass
        else:
            image, _, _ = tv.io.read_video(input_file, start_pts=5, end_pts=5, pts_unit='sec')
            image = (image[0] / 255).permute(2, 0, 1)
            image = resize(image)
            tv.utils.save_image(image, output_file)
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
    datast = VGGSound()
    cal_norm(datast)

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

    # meta = pd.read_csv('vggsound_small.csv', index_col=0)
    # num_processes = os.cpu_count()
    # with mp.Pool(processes=16) as p:
    #     vals = list(tqdm(p.imap(crop, meta.iterrows())))

