import torch
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, input, output, dilation):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv1d(input, output, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(output, output, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, S):
        super(EncoderBlock, self).__init__()
        self.re1 = ResUnit(input_channel, output_channel // 2, dilation=1)
        self.re2 = ResUnit(output_channel // 2, output_channel // 2, dilation=3)
        self.re3 = ResUnit(output_channel // 2, output_channel // 2, dilation=9)
        self.conv = nn.Conv1d(output_channel // 2, output_channel, kernel_size=2*S, stride=S, padding=S//2)
    def forward(self, x):
        x = self.re1(x)
        x = self.re2(x)
        x = self.re3(x)
        x = self.conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, S):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(input_channel, output_channel, kernel_size=2 * S, stride=S, padding=S//2)
        self.re1 = ResUnit(output_channel, output_channel, dilation=1)
        self.re2 = ResUnit(output_channel, output_channel, dilation=3)
        self.re3 = ResUnit(output_channel, output_channel, dilation=9)
    def forward(self, x):
        x = self.conv(x)
        x = self.re1(x)
        x = self.re2(x)
        x = self.re3(x)
        return x

class SEANet(nn.Module):
    def __init__(self):
        super(SEANet, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.E1 = EncoderBlock(32, 64, 2)
        self.E2 = EncoderBlock(64, 128, 2)
        self.E3 = EncoderBlock(128, 256, 8)
        self.E4 = EncoderBlock(256, 512, 8)

        self.conv2 = nn.Conv1d(512, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 512, kernel_size=7, padding=3)

        self.D1 = DecoderBlock(512, 256, 8)
        self.D2 = DecoderBlock(256, 128, 8)
        self.D3 = DecoderBlock(128, 64, 2)
        self.D4 = DecoderBlock(64, 32, 2)
        self.conv4 = nn.Conv1d(32, 4, kernel_size=7, padding=3)

    def forward(self, acc, audio):
        # down-sample
        acc = torch.nn.functional.interpolate(acc, dim=2, scale_factor=10)
        x1 = torch.cat([audio, acc], dim=1)
        x2 = self.conv1(x1)
        x3 = self.E1(x2)
        x4 = self.E2(x3)
        x5 = self.E3(x4)
        x6 = self.E4(x5)
        x = self.conv3(self.conv2(x6)) + x6
        # up-sample, may need padding if the duration is not * 256
        x = nn.functional.pad(self.D1(x), (0, 4)) + x5
        #x = self.D1(x) + x5
        x = self.D2(x) + x4
        x = self.D3(x) + x3
        x = self.D4(x) + x2
        x = self.conv4(x) + x1

        return x[:, 0, :], x[:, 1, :]

class SEANet_mapping(nn.Module):
    def __init__(self):
        super(SEANet_mapping, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.E1 = EncoderBlock(32, 64, 2)
        self.E2 = EncoderBlock(64, 128, 2)
        self.E3 = EncoderBlock(128, 256, 8)
        self.E4 = EncoderBlock(256, 512, 10)

        self.conv2 = nn.Conv1d(512, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 512, kernel_size=7, padding=3)

        self.D1 = DecoderBlock(512, 128, 8)
        self.D2 = DecoderBlock(128, 64, 2)
        self.D3 = DecoderBlock(64, 32, 2)
        self.conv4 = nn.Conv1d(32, 3, kernel_size=7, padding=3)
        self.m = nn.AvgPool1d(10, stride=10)

    def forward(self, audio):
        # down-sample
        # acc = torch.nn.functional.interpolate(acc, scale_factor=10)
        x1 = audio
        x2 = self.conv1(x1)
        x3 = self.E1(x2)
        x4 = self.E2(x3)
        x5 = self.E3(x4)
        x6 = self.E4(x5)
        x = self.conv3(self.conv2(x6)) + x6
        # up-sample, may need padding if the duration is not * 256
        x = self.D1(x) + self.m(x4)
        x = self.D2(x) + self.m(x3)
        x = self.D3(x) + self.m(x2)
        x = self.conv4(x) + self.m(x1)
        return x
def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

if __name__ == '__main__':
    model = SEANet()
    acc = torch.randn(4, 1, 8000)
    audio = torch.randn(4, 1, 80000)
    acc = model(acc, audio)
    print(model_size(model))