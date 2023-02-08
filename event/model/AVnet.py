import torch.nn as nn
import torch
from torchvision.models import resnet18

class AVnet(nn.Module):
    def __init__(self, num_cls=10):

        super().__init__()
        self.audio = resnet18()
        self.audio.load_state_dict(torch.load('resnet18.pth'))
        self.audio.fc = torch.nn.Linear(512, num_cls)

        self.image = resnet18()
        self.image.load_state_dict(torch.load('resnet18.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)
        self.n_fft = 2048
        self.hop_length = 561
        self.win_length = 1654
        self.window = 'blackmanharris'
        self.normalized = True
        self.onesided = True

    def forward(self, audio, image):
        audio = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=torch.hann_window(self.win_length, device=audio.device),
                           pad_mode='reflect',normalized=self.normalized, onesided=True)
        print(audio.shape, image.shape)
        audio = self.audio(audio)
        image = self.image(image)
