import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
class IMU_branch(nn.Module):
    def __init__(self, inference=False):
        super(IMU_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.inference = inference
        if not inference:
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.conv6 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))
            self.conv7 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(3, 2)),
                nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True))


    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_mid = self.conv4(x)
        # up-sample with supervision
        if not self.inference:
            x = self.conv5(x_mid)
            x = self.conv6(x)
            x = self.conv7(x)
            x = F.pad(x, [0, 1, 0, 0])
            return x_mid, x
        else:
            return x_mid
class Audio_branch(nn.Module):
    def __init__(self):
        super(Audio_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=5, padding=6, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        self.r2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.r3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.r4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 2), stride=(3, 2))
        self.final = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = self.up1(self.r1(x) + x)
        x = self.up2(self.r2(x) + x)
        x = self.up3(self.r3(x) + x)
        x = self.up4(self.r4(x) + x)
        x = F.pad(x, [0, 1, 0, 0])
        return self.final(x)

class A2net(nn.Module):
    def __init__(self, inference=False):
        super(A2net, self).__init__()
        self.inference = inference
        self.IMU_branch = IMU_branch(self.inference)
        self.Audio_branch = Audio_branch()
        self.Residual_block = Residual_Block(256)
        # for p in self.IMU_branch.parameters():
        #     p.requires_grad = False
        # for p in self.Audio_branch.parameters():
        #     p.requires_grad = False
    def forward(self, x1, x2):
        x1 = self.IMU_branch(x1)
        if self.inference:
            x = torch.cat([x1, self.Audio_branch(x2)], dim=1)
            x = self.Residual_block(x) * x2
            return x
        else:
            x1, x_extra = x1
            x = torch.cat([x1, self.Audio_branch(x2)], dim=1)
            x = self.Residual_block(x) * x2
            return x, x_extra

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

    imu = torch.rand(1, 1, 33, 151)
    audio = torch.rand(1, 1, 264, 151)
    model = A2net(inference=True)

    # size_all_mb = model_size(model)
    # print('model size: {:.3f}MB'.format(size_all_mb))
    #
    # latency = model_speed(model, [imu, audio])
    # print('model latency: {:.3f}S'.format(latency))

    model_save(model)