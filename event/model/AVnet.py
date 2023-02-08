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

    def forward(self, audio, image):
        audio = self.audio(audio)
        image = self.image(image)
