'''
We implement multi-modal dynamic network here
'''
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
from dyn_op import DSConv2d, DSpwConv2d, DSdwConv2d, DSBatchNorm2d, DSLinear, DSAdaptiveAvgPool2d
import numpy as np
def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
class MMTM(nn.Module):
  def __init__(self, dim_1, dim_2, ratio):
    super(MMTM, self).__init__()
    dim = dim_1 + dim_2
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_1)
    self.fc_skeleton = nn.Linear(dim_out, dim_2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
            self,
            img_channels: int,
            layers: tuple,
            block: Type[BasicBlock],
            num_classes: int = 1000
    ) -> None:
        super(ResNet, self).__init__()
        self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
            self,
            block: Type[BasicBlock],
            out_channels: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
class AVnet(nn.Module):
    def __init__(self, edge=True, dynamic=False, train=False, layers=(3, 4, 6, 3), num_cls=309):
        '''
        :param edge: True - with preprocess, False - only ResBlock
        :param dynamic: True - get early exit for each block, False - don't get
        :param layers: resnet18: [2, 2, 2, 2] resnet34: [3, 4, 6, 3]
        :param num_cls: number of class
        '''
        super(AVnet, self).__init__()
        self.edge = edge
        self.dynamic = dynamic
        self.train = train
        self.audio = ResNet(img_channels=1, layers=layers, block=BasicBlock, num_classes=num_cls)
        self.n_fft = 512
        self.hop_length = 512
        self.win_length = 512
        self.normalized = True
        self.onesided = True

        self.image = ResNet(img_channels=3, layers=layers, block=BasicBlock, num_classes=1000)
        self.image.load_state_dict(torch.load('resnet34.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)

        self.mmtm1 = MMTM(128, 128, 4)
        self.mmtm2 = MMTM(256, 256, 4)
        self.mmtm3 = MMTM(512, 512, 4)
        self.fc = nn.Linear(512 * 2, num_cls)
        self.audio_gate = 0
        self.image_gate = 0
        if self.dynamic:
            # set gate to use the last feature by default
            self.projection_audio = nn.Sequential(*[DSpwConv2d(in_channels_list=(128, 256, 512), out_channels_list=512),
                                             DSBatchNorm2d([512]), nn.ReLU()])
            self.projection_image = nn.Sequential(*[DSpwConv2d(in_channels_list=(128, 256, 512), out_channels_list=512),
                                             DSBatchNorm2d([512]), nn.ReLU()])
    def set_gate(self, audio, image, level=0):
        # TODO: add gate network to make decision
        if not self.dynamic:
            self.audio_gate = level + 1
            self.image_gate = level + 1
        else:
            if self.audio_gate >= level:
                self.audio_gate = level + 1 if np.random.random() > 0.5 else level
                for n, m in self.projection_audio.named_modules():
                    set_exist_attr(m, 'channel_choice', self.audio_gate)
            if self.image_gate >= level:
                self.image_gate = level + 1 if np.random.random() > 0.5 else level
                for n, m in self.projection_image.named_modules():
                    set_exist_attr(m, 'channel_choice', self.image_gate)
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

        self.set_gate(audio, image, level=0)
        if self.audio_gate > 0:
            audio = self.audio.layer2(audio)
        if self.image_gate > 0:
            image = self.image.layer2(image)
        # audio, image = self.mmtm2(audio, image)

        self.set_gate(audio, image, level=1)
        if self.audio_gate > 1:
            audio = self.audio.layer3(audio)
        if self.image_gate > 1:
            image = self.image.layer3(image)
        # audio, image = self.mmtm3(audio, image)

        self.set_gate(audio, image, level=2)
        if self.audio_gate > 2:
            audio = self.audio.layer4(audio)
        if self.image_gate > 2:
            image = self.image.layer4(image)
        # audio, image = self.mmtm4(audio, image)

        if self.dynamic:
            audio = self.projection_audio(audio)
            image = self.projection_image(image)

        audio = self.audio.avgpool(audio)
        image = self.audio.avgpool(image)

        audio = torch.flatten(audio, 1)
        image = torch.flatten(image, 1)
        output = (self.audio.fc(audio) + self.image.fc(image)) / 2
        return output
if __name__ == "__main__":
    num_cls = 100
    model = AVnet(dynamic=False, num_cls=100)
    audio = torch.zeros(16, 1, 220500)
    image = torch.zeros(16, 3, 224, 224)
    output = model(audio, image)
    print(output.shape)