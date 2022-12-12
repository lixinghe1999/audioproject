'''
This is the script for
'''

import numpy as np
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
from scipy.io import wavfile
from preprocess import preprocess
import json

transformers = transforms.ToTensor()


class AudioDataset(Dataset):
    def __init__(self, file, croplen=48000, is_train=True):
        with open(file, 'r') as f:
            dataset = json.load(f)
        self.X = []
        self.Y = []
        for wav, id in dataset:
            self.X.append(wav)
            self.Y.append(int(id))
        self.is_train=is_train
        self.croplen=croplen

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        label=self.Y[idx]
        sr, audio=wavfile.read(self.X[idx])
        if(self.is_train):
            start = np.random.randint(0,audio.shape[0]-self.croplen+1)
            audio = audio[start:start+self.croplen]
        audio = preprocess(audio).astype(np.float32)
        audio = np.expand_dims(audio, 2)
        return transformers(audio), label
