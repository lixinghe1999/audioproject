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
    def preprocessing_audio(self, audio):
        spec = torch.stft(audio.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=torch.hann_window(self.win_length, device=audio.device),
                           pad_mode='reflect', normalized=self.normalized, onesided=True)
        print(spec.shape)
        spec_height_per_band = spec.shape[1] // 16
        spec_height_single_band = 16 * spec_height_per_band
        spec = spec[:, :spec_height_single_band]

        spec = spec.reshape(spec.shape[0], -1, spec.shape[-3] // self.conv1.in_channels, *spec.shape[-2:])
        print(spec.shape)

        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width
        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2
        print(pow_spec.shape)
        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = torch.nn.functional.interpolate(
                pow_spec,
                size=(spec_height, spec_width),
                mode='bilinear',
                align_corners=True
            )
        # pow_spec_split_ch = torch.where(
        #     cast(torch.Tensor, pow_spec_split_ch > 0.0),
        #     pow_spec_split_ch,
        #     torch.full_like(pow_spec_split_ch, self.log10_eps)
        # )
        # pow_spec_split_ch = pow_spec_split_ch.reshape(
        #     x.shape[0], -1, self.conv1.in_channels, *pow_spec_split_ch.shape[-2:]
        # )
        # x_db = torch.log10(pow_spec_split_ch).mul(10.0)

        return spec
    def forward(self, audio, image):
        audio = self.preprocessing_audio(audio)
        print(audio.shape, image.shape)
        audio = self.audio(audio)
        image = self.image(image)
