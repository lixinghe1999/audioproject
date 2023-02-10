'''
We implement multi-modal dynamic network here
'''
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
from slim_model import MMTM, BasicBlock, ResNet

class AVnet_edge(nn.Module):
    def __init__(self, num_cls=309):
        super(AVnet_edge, self).__init__()
        self.audio = ResNet(img_channels=1, num_layers=34, block=BasicBlock, num_classes=num_cls)
        # self.audio.load_state_dict(torch.load('resnet34.pth'))

        self.image = ResNet(img_channels=3, num_layers=34, block=BasicBlock, num_classes=1000)
        self.image.load_state_dict(torch.load('resnet34.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)

        self.mmtm1 = MMTM(128, 128, 4)
        self.mmtm2 = MMTM(256, 256, 4)
        self.mmtm3 = MMTM(512, 512, 4)
        self.fc = nn.Linear(512 * 2, num_cls)

        self.n_fft = 512
        self.hop_length = 512
        self.win_length = 512
        self.normalized = True
        self.onesided = True

    def get_audio_params(self):
        parameters = [
            {'params': self.audio.parameters()},
            {'params': self.mmtm1.parameters()},
            {'params': self.mmtm2.parameters()},
            {'params': self.mmtm3.parameters()},
        ]
        return parameters
    def get_image_params(self):
        parameters = [
            {'params': self.image.parameters()},
            {'params': self.mmtm1.parameters()},
            {'params': self.mmtm2.parameters()},
            {'params': self.mmtm3.parameters()},
        ]
        return parameters
    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=torch.hann_window(self.win_length, device=audio.device),
                          pad_mode='reflect', normalized=self.normalized, onesided=True, return_complex=True)
        spec = torch.abs(spec)
        spec = torch.log(spec + 1e-7)
        mean = torch.mean(spec)
        std = torch.std(spec)
        spec = (spec - mean) / (std + 1e-9)
        return spec.unsqueeze(1)
    def forward(self, audio, image):
        audio = self.preprocessing_audio(audio)
        audio = self.audio.conv1(audio)
        audio = self.audio.bn1(audio)
        audio = self.audio.relu(audio)
        audio = self.audio.maxpool(audio)

        image = self.image.conv1(image)
        image = self.image.bn1(image)
        image = self.image.relu(image)
        image = self.image.maxpool(image)

        audio = self.audio.layer1(audio)
        image = self.image.layer1(image)

        audio = self.audio.layer2(audio)
        image = self.image.layer2(image)
        # audio, image = self.mmtm2(audio, image)
        audio = self.audio.layer3(audio)
        image = self.image.layer3(image)
        # audio, image = self.mmtm3(audio, image)
        audio = self.audio.layer4(audio)
        image = self.image.layer4(image)
        # audio, image = self.mmtm4(audio, image)

        audio = self.audio.avgpool(audio)
        image = self.audio.avgpool(image)
        audio = torch.flatten(audio, 1)
        image = torch.flatten(image, 1)
        output = (self.audio.fc(audio) + self.image.fc(image)) / 2
        return output
if __name__ == "__main__":
    num_cls = 100
    model = AVnet(num_cls=100)
    audio = torch.zeros(16, 1, 220500)
    image = torch.zeros(16, 3, 224, 224)
    output = model(audio, image)
    print(output.shape)