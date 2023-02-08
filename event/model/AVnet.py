import torch.nn as nn
import torch
from torchvision.models import resnet18

class AVnet(nn.Module):
    def __init__(self, num_cls=10):

        super().__init__()
        self.audio = resnet18()
        self.audio.load_state_dict(torch.load('resnet18.pth'))
        self.audio.fc = torch.nn.Linear(512, num_cls)
        self.n_fft = 2048
        self.hop_length = 561
        self.win_length = 1654
        self.window = 'blackmanharris'
        self.normalized = True
        self.onesided = True
        self.conv1_channel = 3
        self.spec_height = 224
        self.spec_width = 224

        self.image = resnet18()
        self.image.load_state_dict(torch.load('resnet18.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)

    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=torch.hann_window(self.win_length, device=audio.device),
                           pad_mode='reflect', normalized=self.normalized, onesided=True)
        spec_height_per_band = spec.shape[1] // self.conv1_channel
        spec_height_single_band = self.conv1_channel * spec_height_per_band
        spec = spec[:, :spec_height_single_band]

        spec = spec.reshape(spec.shape[0], -1, spec.shape[-3] // self.conv1_channel , *spec.shape[-2:])
        spec = spec[..., 0] ** 2 + spec[..., 1] ** 2
        if self.spec_height != spec.shape[-2] or self.spec_width != spec.shape[-1]:
            spec = torch.nn.functional.interpolate(
                input=spec, size=(self.spec_height, self.spec_width),
                mode='bilinear', align_corners=True
            )
        spec = torch.clamp(spec, 1e-18)
        spec = torch.log10(spec).mul(10.0)
        return spec
    def fusion(self, audio, image):
        return (audio + image)/2
    def forward(self, audio, image):
        audio = self.preprocessing_audio(audio)
        audio = self.audio(audio)
        image = self.image(image)
        predict = self.fusion(audio, image)
        return predict
