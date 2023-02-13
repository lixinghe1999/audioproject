'''
We implement multi-modal dynamic network here
'''
import torch.nn as nn
import torch

from model.modified_resnet import ModifiedResNet
from model.resnet34 import ResNet, BasicBlock
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
class AVnet(nn.Module):
    def __init__(self, exit=True, layers=(3, 4, 6, 3), num_cls=309):
        '''
        :param exit: True - with exit, normally for testing, False - no exit, normally for training
        :param train: True - get early exit for each block, False - don't get
        :param layers: resnet18: [2, 2, 2, 2] resnet34: [3, 4, 6, 3]
        :param num_cls: number of class
        '''
        super(AVnet, self).__init__()
        self.edge = exit
        self.audio = ResNet(img_channels=1, layers=layers, block=BasicBlock, num_classes=num_cls)
        self.n_fft = 512
        self.hop_length = 512
        self.win_length = 512
        self.spec_scale = 224
        self.normalized = True
        self.onesided = True

        self.image = ResNet(img_channels=3, layers=layers, block=BasicBlock, num_classes=1000)
        self.image.load_state_dict(torch.load('resnet34.pth'))
        self.image.fc = torch.nn.Linear(512, num_cls)

        self.mmtm1 = MMTM(128, 128, 4)
        self.mmtm2 = MMTM(256, 256, 4)
        self.mmtm3 = MMTM(512, 512, 4)
        self.fc = nn.Linear(512 * 2, num_cls)
        self.early_exit1 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(64 * 2, num_cls)])
        self.early_exit2 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(128 * 2, num_cls)])
        self.early_exit3 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1), nn.Linear(256 * 2, num_cls)])
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
        spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size=self.spec_scale, mode='bilinear')
        return spec
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
        early_output1 = self.early_exit1(torch.cat([audio, image], dim=1))

        audio = self.audio.layer2(audio)
        image = self.image.layer2(image)
        # audio, image = self.mmtm2(audio, image)
        early_output2 = self.early_exit2(torch.cat([audio, image], dim=1))

        audio = self.audio.layer3(audio)
        image = self.image.layer3(image)
        # audio, image = self.mmtm3(audio, image)
        early_output3 = self.early_exit3(torch.cat([audio, image], dim=1))

        audio = self.audio.layer4(audio)
        image = self.image.layer4(image)
        # audio, image = self.mmtm4(audio, image)

        audio = self.audio.avgpool(audio)
        image = self.audio.avgpool(image)

        audio = torch.flatten(audio, 1)
        image = torch.flatten(image, 1)
        output = (self.audio.fc(audio) + self.image.fc(image)) / 2
        return [early_output1, early_output2, early_output3, output]
if __name__ == "__main__":
    num_cls = 100
    model = AVnet(num_cls=100)
    audio = torch.zeros(16, 1, 220500)
    image = torch.zeros(16, 3, 224, 224)
    outputs = model(audio, image)
    for output in outputs:
        print(output.shape)