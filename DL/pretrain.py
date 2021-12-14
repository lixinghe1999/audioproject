import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from data import NoisyCleanSet, IMUSPEECHSet, transfer_function_generator, read_transfer_function, weighted_loss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
seg_len_mic = 2560
overlap_mic = 2240
rate_mic = 16000
seg_len_imu = 256
overlap_imu = 224
rate_imu = 1600
segment = 4
stride = 1
T = 15
N = 4
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
freq_bin_low = int(200 / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class TinyUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TinyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.start = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256//factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.final = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.final(x)
        return logits
if __name__ == "__main__":

    EPOCH = 20
    BATCH_SIZE = 8
    lr = 0.2
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # transfer_function, variance, hist, bins, noise_hist, noise_bins, noise = read_transfer_function('transfer_function')
    # response = np.tile(np.expand_dims(transfer_function[0, :], axis=1), (1, time_bin))
    model = TinyUNet(1, 1).to(device)
    model.load_state_dict(torch.load("transferfunction_3_0.01977690916427009.pth"))
    Loss = nn.SmoothL1Loss(beta=0.05)
    IMU_dataset = IMUSPEECHSet('imuexp4.json', 'wavexp4.json', minmax=(0.01, 0.01))
    train_loader = Data.DataLoader(dataset=IMU_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # for epoch in range(EPOCH):
    #     for x, y in tqdm(train_loader):
    #         x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
    #         predict = model(y)
    #         loss = Loss(predict, x)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     with torch.no_grad():
    #         L = []
    #         for x, y in train_loader:
    #             x, y = x.to(device=device, dtype=torch.float), y.to(device=device,  dtype=torch.float)
    #             predict = model(y)
    #             L.append(Loss(predict, x).item())
    #     torch.save(model.state_dict(), 'transferfunction' + '_' + str(epoch) + '_' + str(np.mean(L)) + '.pth')
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
            predict = model(y)
            print(Loss(x, y), Loss(predict, x))
            fig, axs = plt.subplots(3)
            axs[0].imshow(x.cpu()[0, 0, :, :], aspect='auto')
            axs[1].imshow(y.cpu()[0, 0, :, :], aspect='auto')
            axs[2].imshow(predict.cpu()[0, 0, :, :], aspect='auto')
            plt.show()