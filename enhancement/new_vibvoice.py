import torch.nn as nn
import torch
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
class Conv_Block(nn.Module):
    def __init__(self, input, output):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=(3, 3))
        self.batch = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
class Res_Block(nn.Module):
    def __init__(self, input, output):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(input, input, kernel_size=(3, 3))
        self.batch = nn.BatchNorm2d(input)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(input, output, kernel_size=(3, 3))
    def forward(self, x):
        x = self.relu(self.batch(self.conv(x))) + x
        x = self.conv2(x)
        return x
class IMU_branch(nn.Module):
    def __init__(self):
        super(IMU_branch, self).__init__()
        self.conv1 = Conv_Block(1, 16)
        self.conv2 = Conv_Block(16, 32)
        self.conv3 = Conv_Block(32, 64)
        self.conv4 = Conv_Block(64, 128)
        self.downsample1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv5 = Conv_Block(128, 256)
        self.downsample2 = nn.MaxPool2d(kernel_size=(2, 1))
    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.downsample1(x)
        x = self.conv4(x)
        x = self.downsample2(x)
        x = self.conv5(x)
        return x
class Audio_branch(nn.Module):
    def __init__(self):
        super(Audio_branch, self).__init__()
        self.conv1 = Conv_Block(1, 16)
        self.conv2 = Conv_Block(16, 32)
        self.conv3 = Conv_Block(32, 64)
        self.conv4 = Conv_Block(64, 128)
        self.conv5 = Conv_Block(128, 256)
        self.downsample1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.downsample2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.downsample3 = nn.MaxPool2d(kernel_size=(2, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample1(x)
        x = self.conv3(x)
        x = self.downsample2(x)
        x = self.conv4(x)
        x = self.downsample3(x)
        x = self.conv5(x)
        return x

class Attention_block(nn.Module):
    def __init__(self):
        super(Attention_block, self).__init__()
        self.Pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 256 // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256 // 4, 256, bias=False),
            nn.Sigmoid()
        )
    def forward(self, acc, audio):
        z = self.Pooling(acc)
        z = self.fc(z.reshape((-1, 256))).reshape((-1, 256, 1, 1))
        return audio * z
class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.r1 = Res_Block(in_channels)
        self.upsample1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),
        self.r2 = Res_Block(128)
        self.r3 = Res_Block(in_channels)
        self.upsample1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),
        self.r4 = Res_Block(in_channels)
        self.upsample1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),
        self.r5 = Res_Block(in_channels)
        self.upsample1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=(2, 1), stride=(2, 1)),

        self.r2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=2, dilation=1),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.r3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, padding=4, dilation=2),
            nn.ConvTranspose2d(32, 32, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.r4 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=4, dilation=2),
            nn.ConvTranspose2d(16, 16, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 1), stride=(2, 1)),
            nn.Sigmoid()
            )
    def forward(self, features):

        [x1, x2, x3, x4, x5] = features
        x = torch.cat([self.r1(x5), x4], dim=1)
        x = torch.cat([self.r2(x), x3], dim=1)
        x = torch.cat([self.r3(x), x2], dim=1)
        x = torch.cat([self.r4(x), x1], dim=1)
        x = self.final(x)
        return x

class A2net(nn.Module):
    def __init__(self):
        super(A2net, self).__init__()
        self.IMU_branch = IMU_branch()
        self.Audio_branch = Audio_branch()
        self.Residual_block = Residual_Block(256*3)
        self.attention = Attention_block()
    def forward(self, acc, audio):
        acc = self.IMU_branch(acc)
        [x1, x2, x3, x4, x5] = self.Audio_branch(audio)
        x5 = self.attention(acc, x5)
        x = self.Residual_block([x1, x2, x3, x4, x5]) * audio
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
    x = torch.rand((1, 1, 32, 250))
    noise = torch.rand((1, 1, 256, 250))
    scripted_module = torch.jit.trace(model, [x, noise])
    #scripted_module.save("inference.pt")
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("vibvoice.ptl")

    # save_image(x, 'input1.jpg')
    # save_image(noise, 'input2.jpg')

def model_speed(model, input):
    t_start = time.time()
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
if __name__ == "__main__":

    acc = torch.rand(1, 1, 32, 250)
    audio = torch.rand(1, 1, 256, 250)
    model = A2net()
    audio = model(acc, audio)
    print(audio.shape)

    size_all_mb = model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))

    # latency = model_speed(model, [acc, audio])
    # print('model latency: {:.3f}S'.format(latency))

    #model_save(model)
