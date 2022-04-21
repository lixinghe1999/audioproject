import torch.nn as nn
import torch
import torch.nn.functional as F
class IMU_branch(nn.Module):
    def __init__(self):
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
        x = self.conv5(x_mid)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.pad(x, [0, 1, 0, 0])
        return x_mid, x

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
    def __init__(self):
        super(A2net, self).__init__()
        self.IMU_branch = IMU_branch()
        self.Audio_branch = Audio_branch()
        self.Residual_block = Residual_Block(256)
    def forward(self, x1, x2):
        x1 = self.IMU_branch(x1)
        x1, x_extra = x1
        x = torch.cat([x1, self.Audio_branch(x2)], dim=1)
        x = self.Residual_block(x) * x2
        return x, x_extra
