import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
from torch.nn.modules.utils import _pair

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding is None:
            padding = [int((kernel_size[i] -1) * dilation[i]) for i in range(len(kernel_size))]
        self.left_padding = _pair(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)
    def forward(self, inputs):
        inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super().forward(inputs)
        return output
class IMU_branch(nn.Module):
    def __init__(self):
        super(IMU_branch, self).__init__()
        self.conv1 = nn.Sequential(
            CausalConv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            CausalConv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            CausalConv2d(32, 64, kernel_size=5, stride=(1, 2), dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            CausalConv2d(64, 128, kernel_size=5, stride=(3, 1), dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        print(x.shape)
        return x
class Audio_branch(nn.Module):
    def __init__(self):
        super(Audio_branch, self).__init__()
        self.conv1 = nn.Sequential(
            CausalConv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CausalConv2d(16, 32, kernel_size=5, stride=(2, 1), dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
            CausalConv2d(32, 64, kernel_size=5, stride=(2, 1), dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            CausalConv2d(64, 128, kernel_size=5, stride=(2, 1), dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            CausalConv2d(128, 128, kernel_size=5, stride=(3, 2), dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        print(x.shape)
        return x

class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.r1 = nn.Sequential(
            CausalConv2d(in_channels, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        self.r2 = nn.Sequential(
            CausalConv2d(128, 128, kernel_size=5, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.r3 = nn.Sequential(
            CausalConv2d(64, 64, kernel_size=5, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.r4 = nn.Sequential(
            CausalConv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(3, 2))
        self.final = CausalConv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = self.up1(self.r1(x) + x)
        x = self.up2(self.r2(x) + x)
        x = self.up3(self.r3(x) + x)
        x = self.up4(self.r4(x) + x)
        return self.final(x)

class Fusion_branch(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_branch, self).__init__()
        self.r1 = nn.Sequential(
            CausalConv2d(in_channels, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        self.r2 = nn.Sequential(
            CausalConv2d(128, 128, kernel_size=5, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.r3 = nn.Sequential(
            CausalConv2d(64, 64, kernel_size=5, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.r4 = nn.Sequential(
            CausalConv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(3, 2))
        self.final = CausalConv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = self.up1(self.r1(x) + x)
        x = self.up2(self.r2(x) + x)
        x = self.up3(self.r3(x) + x)
        x = self.up4(self.r4(x) + x)
        return self.final(x)

class Causal_A2net(nn.Module):
    def __init__(self):
        super(Causal_A2net, self).__init__()
        self.IMU_branch = IMU_branch()
        self.Audio_branch = Audio_branch()
        self.Residual_block = Residual_Block(256)
    def forward(self, x1, x2):

        x = torch.cat([self.IMU_branch(x1), self.Audio_branch(x2)], dim=1)
        x = self.Residual_block(x) * x2
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

def model_save(model):
    model.eval()
    x = torch.rand((1, 1, 33, 151))
    noise = torch.rand((1, 1, 264, 151))
    scripted_module = torch.jit.trace(model, [x, noise])
    #scripted_module.save("inference.pt")
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("inference.ptl")

    save_image(x, 'input1.jpg')
    save_image(noise, 'input2.jpg')

def model_speed(model, input):
    t_start = time.time()
    step = 1000
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
if __name__ == "__main__":

    acc = torch.rand(1, 1, 33, 150)
    audio = torch.rand(1, 1, 264, 150)
    model = Causal_A2net()
    print(model(acc, audio).shape)

    #model_save(model)